"""Generate leaderboard CSVs from per-task AUC/MAE summaries.

Scores follow the report-card normalisation (see scripts/build_report_card.py):
  classification : clipped AUC (chance 0.5 ; perfect 1.0)
  regression     : max(0, 1 - MAE / (2 * MAD)) with MAD = test-label MAD

Outputs in input_dir (or --output-dir):
  leaderboard.csv             rows = tasks, cols = category / task_type / avg / std / <encoders>
  leaderboard_by_category.csv encoder x (overall + per-category) score and rank
  leaderboard_by_rank.csv     encoder x reciprocal-rank per task
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
# Current per-task AUC/MAE summary (date-stamped; bump when a new summary is cut).
DEFAULT_INPUT_DIR = REPO / "exps" / "_summary_may14" / "single_task"

DATASET_TO_CATEGORY = {
    "edaic": "Affective", "iemocap": "Affective", "ravdess": "Affective",
    "aphasia": "Cognitive", "dbank": "Cognitive",
    "ksof": "Motor", "mvdr": "Motor", "torgo": "Motor", "uaspeech": "Motor",
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
    "mvdr_parkC":     "Binary",
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
    "mvdr_hyR":       "Regression",
    "mvdr_updrs18R":  "Regression",
    "mvdr_updrs5R":   "Regression",
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

LABEL_SOURCES = {
    "dbank_mmseR":   ("data/dementiabank/processed/dbank.csv", "mmse"),
    "edaic_phqR":    ("data/edaic/processed/edaic.csv",        "PHQ_Score"),
    "torgo_sevR":    ("data/torgo/processed/torgo.csv",        "severity"),
    "mvdr_hyR":      ("data/mvdr/processed/mvdr.csv",          "hy_rating"),
    "mvdr_updrs5R":  ("data/mvdr/processed/mvdr.csv",          "updrs_ii5"),
    "mvdr_updrs18R": ("data/mvdr/processed/mvdr.csv",          "updrs_iii18"),
}


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


def compute_mad(repo: Path) -> dict[str, float]:
    mad: dict[str, float] = {}
    for task, (rel, col) in LABEL_SOURCES.items():
        s = pd.to_numeric(pd.read_csv(repo / rel)[col], errors="coerce").dropna()
        mad[task] = float((s - s.mean()).abs().mean())
    return mad


def normalize_scores(input_dir: Path, repo: Path) -> pd.DataFrame:
    """Tasks x encoders score table, normalised report-card style."""
    frames = []
    auc_path = input_dir / "AUC.csv"
    if auc_path.exists():
        frames.append(pd.read_csv(auc_path).set_index("task").clip(lower=0, upper=1))
    mae_path = input_dir / "MAE.csv"
    if mae_path.exists():
        mae = pd.read_csv(mae_path).set_index("task")
        mad = compute_mad(repo)
        mae_score = pd.DataFrame(index=mae.index, columns=mae.columns, dtype=float)
        for task in mae.index:
            if task not in mad:
                raise SystemExit(f"missing MAD source for regression task '{task}'")
            mae_score.loc[task] = (1 - mae.loc[task] / (2 * mad[task])).clip(0, 1)
        frames.append(mae_score)
    if not frames:
        raise SystemExit(f"no AUC.csv or MAE.csv in {input_dir}")
    encoders = sorted(set().union(*(f.columns for f in frames)))
    return pd.concat([f.reindex(columns=encoders) for f in frames], axis=0).sort_index()


def reciprocal_rank(scores: pd.DataFrame) -> pd.DataFrame:
    return 1.0 / scores.rank(axis=1, method="min", ascending=False)


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
                    help=f"directory with AUC.csv and/or MAE.csv (default: {DEFAULT_INPUT_DIR.relative_to(REPO)})")
    ap.add_argument("-o", "--output-dir", type=Path, default=None,
                    help="output directory (default: repo root)")
    args = ap.parse_args()

    out_dir = args.output_dir or REPO
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = normalize_scores(args.input_dir, REPO)
    scores = apply_merges(scores)

    lb = build_task_leaderboard(scores)
    lb_t = lb.round(4).T
    lb_t.index.name = "encoder"
    lb_t.to_csv(out_dir / "leaderboard.csv")

    rranks = reciprocal_rank(scores).T
    rranks.insert(0, "avg", rranks.mean(axis=1, skipna=True))
    rranks = rranks.sort_values("avg", ascending=False)
    rranks.index.name = "encoder"
    rranks.round(4).to_csv(out_dir / "leaderboard_by_rank.csv")

    summary = build_category_summary(scores)
    summary.to_csv(out_dir / "leaderboard_by_category.csv")

    print(f"wrote {out_dir / 'leaderboard.csv'}")
    print(f"wrote {out_dir / 'leaderboard_by_category.csv'}")
    print(f"wrote {out_dir / 'leaderboard_by_rank.csv'}")


if __name__ == "__main__":
    main()
