"""Generate leaderboard CSVs from per-task AUC / C-index summaries.

Scores (jun12 normalisation — both on a single [0,1] discrimination axis):
  classification : clipped ROC-AUC (chance 0.5 ; perfect 1.0)
  regression     : concordance index (Cindex.csv) — Harrell's C / Somers'-D
                   rescaled, the ordinal generalisation of ROC-AUC (0.5 = chance
                   ordering, 1.0 = perfect; equals macro AUC on a binary target).

This replaces the jun3 regression score ``max(0, 1 - MAE/(2*MAD))``, whose
denominator was the test-label MAD (a cohort property that drifts with the
split and is not comparable across tasks). The C-index needs no denominator and
is the *same statistic* as the AUC column, so the two finally average cleanly.
Build Cindex.csv with ``scripts/aggregate_cindex.py``.

Outputs in input_dir (or --output-dir), suffixed by --tag:
  leaderboard<tag>.csv             rows = models, cols = avg + per task/course
  leaderboard_by_category<tag>.csv encoder x (overall + per-category) score and rank
  leaderboard_by_rank<tag>.csv     encoder x reciprocal-rank per task

Filtering / metric flags:
  --drop a,b   exclude encoder columns before scoring (re-ranks the rest)
  --no-merge   keep the raw per-dataset tasks (skip the six merged courses)
  --mrr        write only the mean-reciprocal-rank board (avg = MRR) as
               leaderboard<tag>.csv
  --mae        score regression tasks by raw MAE (lower better) instead of the
               C-index (ranks them ascending); only with --mrr
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
# Current per-task summary (date-stamped; bump when a new summary is cut). Must
# contain AUC.csv and Cindex.csv (build the latter with scripts/aggregate_cindex.py).
DEFAULT_INPUT_DIR = REPO / "exps" / "_summary_jun3_full" / "single_task"

DATASET_TO_CATEGORY = {
    "edaic": "Affective", "iemocap": "Affective", "ravdess": "Affective",
    "aphasia": "Cognitive", "dbank": "Cognitive",
    "ksof": "Motor", "mdvr": "Motor", "torgo": "Motor", "uaspeech": "Motor",
    "avfad": "Respiratory", "c9s": "Respiratory", "coswara": "Respiratory",
}

CATEGORY_ORDER = ["Affective", "Cognitive", "Motor", "Respiratory"]
TASK_TYPE_ORDER = ["Binary", "Multiclass", "Multilabel", "Regression"]

TASK_TYPE = {
    "aphasia_pwaC":   "Binary",
    "avfad_pathC":    "Binary",
    "c9s_L_t1":       "Binary",
    "c9s_L_t2":       "Binary",
    "c9s_t1":         "Binary",
    "c9s_t2":         "Binary",
    "coswara_covidC": "Binary",
    "coswara_sympC":  "Binary",
    "dbank_adC":      "Binary",
    "edaic_depC":     "Binary",
    "iemocap_emoBC":  "Binary",
    "ksof_intC":      "Binary",
    "mdvr_parkC":     "Binary",
    "ravdess_emoBC":  "Binary",
    "torgo_dysC":     "Binary",
    "uaspeech_dysC":  "Binary",
    "iemocap_emoC":   "Multiclass",
    "ravdess_emoC":   "Multiclass",
    "c9s_sympL":      "Multilabel",
    "coswara_sympL":  "Multilabel",
    "ksof_stutL":     "Multilabel",
    "dbank_mmseR":    "Regression",
    "edaic_phqR":     "Regression",
    "mdvr_hyR":       "Regression",
    "mdvr_updrs18R":  "Regression",
    "mdvr_updrs5R":   "Regression",
    "torgo_sevR":     "Regression",
}

# Same six paper-faithful merges as build_report_card.py.
MERGES = {
    "emo_classify":   ["ravdess_emoC", "iemocap_emoC"],         # T3+T5
    "emo_neg":        ["ravdess_emoBC", "iemocap_emoBC"],       # T4+T6
    "dysarthria_det": ["torgo_dysC", "uaspeech_dysC"],          # T10+T12
    "symptomatic":    ["c9s_t1", "c9s_L_t1", "coswara_sympC"],  # T19+T20+T24
    "covid_det":      ["c9s_t2", "c9s_L_t2", "coswara_covidC"], # T21+T22+T25
    "symptom_multi":  ["c9s_sympL", "coswara_sympL"],           # T23+T26
}

# Category + type for merged tasks (their names don't carry a dataset prefix).
MERGED_TASK_META = {
    "emo_classify":   ("Affective",   "Multiclass"),
    "emo_neg":        ("Affective",   "Binary"),
    "dysarthria_det": ("Motor",       "Binary"),
    "symptomatic":    ("Respiratory", "Binary"),
    "covid_det":      ("Respiratory", "Binary"),
    "symptom_multi":  ("Respiratory", "Multilabel"),
}

# Jun2+ summaries label tasks "T2 (edaic_T2)"; the scoring tables above are keyed
# on canonical task ids. Map the T-number back to its canonical id (pairings
# verified against the may14 run: classification AUC mean |Δ|<0.002, regression
# MAE |Δ|<0.003, mdvr triplet resolved by the 3x3 MAE-distance matrix).
JUN2_TASK_MAP = {
    "T1":  "edaic_depC",   "T2":  "edaic_phqR",
    "T3":  "ravdess_emoC", "T4":  "ravdess_emoBC",
    "T5":  "iemocap_emoC", "T6":  "iemocap_emoBC",
    "T7":  "dbank_adC",    "T8":  "dbank_mmseR",
    "T9":  "aphasia_pwaC", "T10": "torgo_dysC",  "T11": "torgo_sevR",
    "T12": "uaspeech_dysC","T13": "mdvr_parkC",
    "T14": "mdvr_updrs5R", "T15": "mdvr_updrs18R", "T16": "mdvr_hyR",
    "T17": "ksof_intC",    "T18": "ksof_stutL",
    "T19": "c9s_t1",       "T20": "c9s_L_t1",
    "T21": "c9s_t2",       "T22": "c9s_L_t2",  "T23": "c9s_sympL",
    "T24": "coswara_sympC","T25": "coswara_covidC", "T26": "coswara_sympL",
    "T27": "avfad_pathC",
}


def canonical_task(label: str) -> str:
    """Normalise a summary task label to its canonical id.

    Jun2-style ``"T2 (edaic_T2)"`` -> ``"edaic_phqR"``; already-canonical labels
    (e.g. a may14-style ``"edaic_phqR"``) pass through unchanged.
    """
    m = re.match(r"\s*(T\d+)\b", label)
    if m and m.group(1) in JUN2_TASK_MAP:
        return JUN2_TASK_MAP[m.group(1)]
    return label


def task_category(task: str) -> str:
    if task in MERGED_TASK_META:
        return MERGED_TASK_META[task][0]
    prefix = task.split("_", 1)[0]
    cat = DATASET_TO_CATEGORY.get(prefix)
    if cat is None:
        raise ValueError(f"unknown category for task '{task}' (prefix '{prefix}')")
    return cat


def task_type(task: str) -> str:
    if task in MERGED_TASK_META:
        return MERGED_TASK_META[task][1]
    try:
        return TASK_TYPE[task]
    except KeyError as e:
        raise ValueError(f"unknown task_type for task '{task}'") from e


def apply_merges(scores: pd.DataFrame) -> pd.DataFrame:
    """Average raw-task scores into the six paper-faithful merged courses."""
    merged = scores.copy()
    for new, src in MERGES.items():
        present = [s for s in src if s in merged.index]
        if not present:
            continue
        if len(present) != len(src):
            missing = sorted(set(src) - set(present))
            raise SystemExit(f"merge '{new}' missing source tasks: {missing}")
        merged.loc[new] = merged.loc[present].mean(axis=0)
        merged = merged.drop(index=present)
    return merged.sort_index()


def normalize_scores(input_dir: Path, repo: Path,
                     regression: str = "cindex") -> pd.DataFrame:
    """Tasks x encoders score table.

    Classification = clipped ROC-AUC (higher better, [0,1], 0.5 = chance).
    Regression = ``cindex`` (concordance index, clipped [0,1], higher better) or
    ``mae`` (raw MAE.csv, lower better, NOT clipped — only meaningful for the
    per-task MRR ranking, which takes regression ranks ascending).
    """
    frames = []
    auc_path = input_dir / "AUC.csv"
    if auc_path.exists():
        auc = pd.read_csv(auc_path).set_index("task").rename(index=canonical_task)
        frames.append(auc.clip(lower=0, upper=1))
    reg_file = "MAE.csv" if regression == "mae" else "Cindex.csv"
    reg_path = input_dir / reg_file
    if reg_path.exists():
        reg = pd.read_csv(reg_path).set_index("task").rename(index=canonical_task)
        frames.append(reg if regression == "mae" else reg.clip(lower=0, upper=1))
    if not frames:
        raise SystemExit(f"no AUC.csv or {reg_file} in {input_dir}")
    encoders = sorted(set().union(*(f.columns for f in frames)))
    return pd.concat([f.reindex(columns=encoders) for f in frames], axis=0).sort_index()


# Regression tasks are scored lower-is-better under --mae (raw MAE), so their
# per-task ranks must be taken ascending; ROC-AUC / C-index stay higher-is-better.
REGRESSION_TASKS = frozenset(t for t, v in TASK_TYPE.items() if v == "Regression")


def reciprocal_rank(scores: pd.DataFrame,
                    lower_better: frozenset = frozenset()) -> pd.DataFrame:
    ranks = scores.rank(axis=1, method="min", ascending=False)
    lb = [t for t in scores.index if t in lower_better]
    if lb:
        ranks.loc[lb] = scores.loc[lb].rank(axis=1, method="min", ascending=True)
    return 1.0 / ranks


def build_task_leaderboard(scores: pd.DataFrame) -> pd.DataFrame:
    enc_cols = list(scores.columns)
    lb = scores.copy()
    lb.insert(0, "category",  [task_category(t) for t in lb.index])
    lb.insert(1, "task_type", [task_type(t)     for t in lb.index])
    lb.insert(2, "avg",       lb[enc_cols].mean(axis=1))
    lb.insert(3, "std",       lb[enc_cols].std(axis=1))
    cat_key  = lb["category"].map({c: i for i, c in enumerate(CATEGORY_ORDER)})
    type_key = lb["task_type"].map({t: i for i, t in enumerate(TASK_TYPE_ORDER)})
    lb = (lb.assign(_cat=cat_key, _type=type_key)
            .sort_values(["_cat", "_type", "avg"], ascending=[True, True, False])
            .drop(columns=["_cat", "_type"]))
    lb.index.name = "task"
    return lb


def build_category_summary(scores: pd.DataFrame) -> pd.DataFrame:
    enc_avg = scores.mean(axis=0).round(4)
    enc_rank = enc_avg.rank(method="min", ascending=False).astype("Int64")
    summary = pd.DataFrame({"overall_score": enc_avg, "overall_rank": enc_rank})

    cats_by_task = pd.Series({t: task_category(t) for t in scores.index})
    cat_score_cols = []
    for cat in CATEGORY_ORDER:
        cat_tasks = cats_by_task[cats_by_task == cat].index
        if len(cat_tasks) == 0:
            continue
        cat_score = scores.loc[cat_tasks].mean(axis=0, skipna=True).round(4)
        cat_rank = cat_score.rank(method="min", ascending=False).astype("Int64")
        summary[f"{cat}_score"] = cat_score.reindex(summary.index)
        summary[f"{cat}_rank"]  = cat_rank.reindex(summary.index)
        cat_score_cols.append(f"{cat}_score")

    eqcat_score = summary[cat_score_cols].mean(axis=1, skipna=True).round(4)
    eqcat_rank = eqcat_score.rank(method="min", ascending=False).astype("Int64")
    summary.insert(2, "eqcat_score", eqcat_score)
    summary.insert(2, "eqcat_rank", eqcat_rank)
    return summary.sort_values("overall_rank")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_dir", type=Path, nargs="?", default=DEFAULT_INPUT_DIR,
                    help=f"directory with AUC.csv and/or Cindex.csv (default: {DEFAULT_INPUT_DIR.relative_to(REPO)})")
    ap.add_argument("-o", "--output-dir", type=Path, default=None,
                    help="output directory (default: repo root)")
    ap.add_argument("--tag", default="",
                    help="filename suffix, e.g. --tag _jun12 -> leaderboard_jun12.csv")
    ap.add_argument("--no-merge", action="store_true",
                    help="keep the raw per-dataset tasks instead of averaging them "
                         "into the six paper-faithful merged courses")
    ap.add_argument("--drop", default="",
                    help="comma-separated encoder columns to exclude before scoring, "
                         "e.g. --drop wavlm_base,gemini")
    ap.add_argument("--mrr", action="store_true",
                    help="emit the mean-reciprocal-rank board (per-task reciprocal "
                         "rank + a leading avg = MRR) as the sole leaderboard<tag>.csv "
                         "output, instead of the three score/category/rank tables")
    ap.add_argument("--mae", action="store_true",
                    help="score regression tasks by raw MAE (lower better, from "
                         "MAE.csv) instead of the C-index; ranks those tasks ascending. "
                         "Only valid with --mrr (MAE is not averageable with AUC).")
    args = ap.parse_args()

    if args.mae and not args.mrr:
        raise SystemExit("--mae is only supported with --mrr")

    out_dir = args.output_dir or REPO
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag

    scores = normalize_scores(args.input_dir, REPO,
                              regression="mae" if args.mae else "cindex")
    if args.drop:
        drop = [e.strip() for e in args.drop.split(",") if e.strip()]
        missing = [e for e in drop if e not in scores.columns]
        if missing:
            raise SystemExit(f"--drop: encoder(s) not in scores: {missing}")
        scores = scores.drop(columns=drop)
    # Drop encoders with no data at all (every task NaN) — an all-empty row.
    scores = scores.loc[:, scores.notna().any(axis=0)]
    if not args.no_merge:
        scores = apply_merges(scores)

    if args.mrr:
        # Mean-reciprocal-rank board: rank the encoders within each task, take the
        # reciprocal, and average across tasks (the leading avg column = MRR). This
        # is the sole output in --mrr mode. Regression tasks rank ascending under --mae.
        lower_better = REGRESSION_TASKS if args.mae else frozenset()
        rranks = reciprocal_rank(scores, lower_better=lower_better).T
        rranks.insert(0, "avg", rranks.mean(axis=1, skipna=True))
        rranks = rranks[rranks["avg"].notna()].sort_values("avg", ascending=False)
        rranks.index.name = "encoder"
        rranks.round(4).to_csv(out_dir / f"leaderboard{tag}.csv")
        print(f"wrote {out_dir / f'leaderboard{tag}.csv'} (MRR board, "
              f"regression={'MAE' if args.mae else 'C-index'})")
        return

    lb = build_task_leaderboard(scores)
    # Model-oriented board: rows = models, leading cols = avg (all tasks) and
    # avg_no_rgs (classification/multilabel only — the regression-free headline),
    # then one column per task/course (grouped by category/type as sorted above).
    enc_cols = [c for c in lb.columns if c not in ("category", "task_type", "avg", "std")]
    model_lb = lb[enc_cols].T
    task_cols = list(model_lb.columns)
    non_rgs = [t for t in task_cols if lb.loc[t, "task_type"] != "Regression"]
    avg_all = model_lb[task_cols].mean(axis=1, skipna=True)
    avg_norgs = model_lb[non_rgs].mean(axis=1, skipna=True)
    model_lb.insert(0, "avg_no_rgs", avg_norgs)
    model_lb.insert(0, "avg", avg_all)
    model_lb = model_lb.sort_values("avg", ascending=False)
    model_lb.index.name = "encoder"
    model_lb.round(4).to_csv(out_dir / f"leaderboard{tag}.csv")

    rranks = reciprocal_rank(scores).T
    rranks.insert(0, "avg", rranks.mean(axis=1, skipna=True))
    rranks = rranks.sort_values("avg", ascending=False)
    rranks.index.name = "encoder"
    rranks.round(4).to_csv(out_dir / f"leaderboard_by_rank{tag}.csv")

    summary = build_category_summary(scores)
    summary.to_csv(out_dir / f"leaderboard_by_category{tag}.csv")

    print(f"wrote {out_dir / f'leaderboard{tag}.csv'}")
    print(f"wrote {out_dir / f'leaderboard_by_category{tag}.csv'}")
    print(f"wrote {out_dir / f'leaderboard_by_rank{tag}.csv'}")


if __name__ == "__main__":
    main()
