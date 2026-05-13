"""Update the gemini and qwen3omni columns in the leaderboard summary CSVs.

Reads exps/<root>/T*/predictions.metrics.json, maps T-numbers to legacy task
names (e.g. T1 -> edaic_depC), and writes the headline metric into the
corresponding column of:

  exps/_summary_run2_all_mvdr3/single_task/AUC.csv
  exps/_summary_run2_all_mvdr3/single_task/MAE.csv

Headline metric per task type:

  binary       AUC  := auc_hard                   MAE := (n/a)
  multiclass   AUC  := auc_hard (macro OvR)       MAE := (n/a)
  multilabel   AUC  := auc_hard (per-class)       MAE := (n/a)
  regression   AUC  := (n/a)                      MAE := mae

For CV tasks the *_mean from fold_aggregate is used when present, else the
overall metric.

Usage:
    python scripts/update_summary_llm.py                  # both models
    python scripts/update_summary_llm.py --model gemini
    python scripts/update_summary_llm.py --model qwen3omni
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SUMMARY_DIR = REPO / "exps" / "_summary_run2_all_mvdr3" / "single_task"

# T-number -> legacy task name used as the row key in the leaderboard CSVs.
T_TO_LEGACY = {
    "T1":  "edaic_depC",   "T2":  "edaic_phqR",
    "T3":  "ravdess_emoC", "T4":  "ravdess_emoBC",
    "T5":  "iemocap_emoC", "T6":  "iemocap_emoBC",
    "T7":  "dbank_adC",    "T8":  "dbank_mmseR",
    "T9":  "aphasia_pwaC",
    "T10": "torgo_dysC",   "T11": "torgo_sevR",
    "T12": "uaspeech_dysC",
    "T13": "mvdr_parkC",   "T14": "mvdr_updrs5R",
    "T15": "mvdr_updrs18R","T16": "mvdr_hyR",
    "T17": "ksof_intC",    "T18": "ksof_stutL",
    "T19": "c9s_t1",       "T20": "c9s_L_t1",
    "T21": "c9s_t2",       "T22": "c9s_L_t2",
    "T23": "c9s_sympL",
    "T24": "coswara_sympC","T25": "coswara_covidC",
    "T26": "coswara_sympL",
    "T27": "avfad_pathC",
}

MODEL_ROOT = {
    "gemini":    "exps/gemini_3_1_pro",
    "qwen3omni": "exps/qwen3omni_all",
}

# Column name to write each model's value into.
MODEL_COL = {
    "gemini":    "gemini",
    "qwen3omni": "qwen3omni",
}

sys.path.insert(0, str(REPO / "scripts"))
from test_qwen3omni_all import TASKS  # noqa: E402

# Same registry fixes as gemini_batch.py
TASKS["T7"]["source"] = {"manifest": "exps/single_task/dbank_adC/manifest", "split": "test"}
TASKS["T8"]["source"] = {"manifest": "exps/single_task/dbank_mmseR/manifest", "split": "test"}


def _pick(metrics: dict, key: str):
    """Prefer fold_aggregate <key>_mean; fall back to overall <key>."""
    fold_agg = metrics.get("fold_aggregate") or {}
    if f"{key}_mean" in fold_agg:
        return fold_agg[f"{key}_mean"]
    overall = metrics.get("overall") or {}
    return overall.get(key)


def headline_value(task: str, metrics: dict, summary_kind: str):
    """Return the value to put in <summary_kind>.csv (AUC or MAE) for `task`.

    None means "leave the cell empty for this task" (e.g. AUC for a
    regression task).
    """
    cfg = TASKS[task]
    m = cfg["metric"]
    if summary_kind == "AUC":
        if m == "binary":      return _pick(metrics, "auc_hard")
        if m == "multiclass":  return _pick(metrics, "auc_hard")
        if m == "multilabel":  return _pick(metrics, "auc_hard")
        return None
    if summary_kind == "MAE":
        if m == "regression":  return _pick(metrics, "mae")
        return None
    raise ValueError(f"unknown summary kind: {summary_kind}")


def load_metrics(model_root: Path, task: str) -> dict | None:
    p = model_root / task / "predictions.metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def update_summary(model: str) -> None:
    model_root = REPO / MODEL_ROOT[model]
    col = MODEL_COL[model]

    for kind in ("AUC", "MAE"):
        path = SUMMARY_DIR / f"{kind}.csv"
        if not path.exists():
            print(f"[{model}] missing {path}; skip")
            continue
        df = pd.read_csv(path).set_index("task")
        if col not in df.columns:
            df[col] = pd.NA

        updates = 0
        for t, legacy in T_TO_LEGACY.items():
            if legacy not in df.index:
                continue
            metrics = load_metrics(model_root, t)
            if not metrics:
                continue
            val = headline_value(t, metrics, kind)
            if val is None:
                continue
            df.loc[legacy, col] = float(val)
            updates += 1

        df.reset_index().to_csv(path, index=False)
        print(f"[{model}] {kind}.csv: wrote {updates} cells -> {path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=list(MODEL_ROOT.keys()) + ["all"], default="all")
    args = p.parse_args()

    models = list(MODEL_ROOT.keys()) if args.model == "all" else [args.model]
    for m in models:
        update_summary(m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
