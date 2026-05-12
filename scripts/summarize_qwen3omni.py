"""Aggregate qwen3omni_all per-task metrics into a single CSV.

Reads exps/qwen3omni_all/T*/predictions.metrics.json (or the older
predictions_<jobid>.metrics.json) and emits a row per task with the
canonical primary metric (AUC for binary, MAE for regression,
accuracy/macro-F1 for multiclass, micro/macro F1 for multilabel),
plus n_parsed.

Usage:
    python scripts/summarize_qwen3omni.py
    python scripts/summarize_qwen3omni.py --out exps/qwen3omni_all/_summary.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from test_qwen3omni_all import TASKS  # noqa: E402


def _read_metrics(task_dir: Path) -> dict | None:
    primary = task_dir / "predictions.metrics.json"
    if primary.exists():
        return json.loads(primary.read_text())
    legacy = sorted(task_dir.glob("predictions_*.metrics.json"))
    if legacy:
        return json.loads(legacy[-1].read_text())
    return None


def _extract_row(task: str, metrics: dict) -> dict:
    cfg = TASKS[task]
    overall = metrics.get("overall", {})
    fold_agg = metrics.get("fold_aggregate", {})
    src = fold_agg or overall

    def g(k):
        if k in src:
            return src[k]
        agg = src.get(f"{k}_mean")
        return agg

    row = {
        "task": task,
        "metric_type": cfg["metric"],
        "n_total": overall.get("n_total"),
        "n_parsed": overall.get("n_parsed"),
        "n_unparsed": overall.get("n_unparsed"),
        "AUC": g("auc_hard"),
        "MAE": g("mae"),
        "accuracy": g("accuracy"),
        "f1": g("f1"),
        "f1_macro": g("f1_macro"),
        "f1_micro": g("f1_micro"),
        "f1_samples": g("f1_samples"),
        "rmse": g("rmse"),
        "pearson": g("pearson"),
        "is_cv": "fold_aggregate" in metrics,
    }
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="exps/qwen3omni_all")
    p.add_argument("--out", default=None,
                   help="CSV path (default: <root>/_summary.csv)")
    args = p.parse_args()

    root = (REPO / args.root).resolve()
    out_path = Path(args.out).resolve() if args.out else root / "_summary.csv"

    rows = []
    for task in sorted(TASKS, key=lambda t: int(t[1:])):
        task_dir = root / task
        if not task_dir.exists():
            rows.append({"task": task, "metric_type": TASKS[task]["metric"],
                         "n_total": None, "n_parsed": None})
            continue
        metrics = _read_metrics(task_dir)
        if not metrics:
            rows.append({"task": task, "metric_type": TASKS[task]["metric"],
                         "n_total": None, "n_parsed": None})
            continue
        rows.append(_extract_row(task, metrics))

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    # Compact stdout view
    cols = ["task", "metric_type", "n_parsed", "AUC", "MAE",
            "accuracy", "f1_macro", "is_cv"]
    print(df[cols].to_string(index=False))
    print(f"\nwrote -> {out_path}")


if __name__ == "__main__":
    main()
