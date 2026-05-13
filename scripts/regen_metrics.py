"""Regenerate metrics against canonical task manifests.

For each task in TASKS, load the canonical sample set from its manifest (or
CSV fallback), filter predictions.csv to only rows whose rowid is in the
canonical set, overwrite stale `true` and `fold` columns, and recompute
metrics.

This corrects:
  1. Spurious rows: the batch fetcher added <upload-failed> rows for every
     failed upload to every task in the dataset, even if the rowid didn't
     belong to that task (affects T19-T23 on the Gemini side).
  2. Coverage: surfaces any rowids that are in the manifest but missing from
     predictions.csv (added as <missing>).
  3. Stale fold IDs: Gemini's collect_samples inherited the first task's
     fold for shared audios, so T11/T14/T15/T16/T18 had folds from their
     sister tasks (T10/T13/T13/T13/T17). This pass overwrites `fold` from
     the canonical manifest.
  4. Wrong sample set on Qwen T11: qwen3omni_all/T11 has 9416 rowids (T10's
     set) instead of 3181. The rowid filter drops the extras.

Usage:
    python scripts/regen_metrics.py                       # gemini, all tasks
    python scripts/regen_metrics.py --task T19            # single task
    python scripts/regen_metrics.py --root exps/qwen3omni_all
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_qwen3omni_all import (  # noqa: E402
    TASKS, load_samples, normalize_true, compute_metrics, cv_aggregate,
)

# Same registry fixes as gemini_batch.py
TASKS["T7"]["source"] = {"manifest": "exps/single_task/dbank_adC/manifest", "split": "test"}
TASKS["T8"]["source"] = {"manifest": "exps/single_task/dbank_mmseR/manifest", "split": "test"}


def regen_task(task: str, pred_root: Path) -> None:
    preds_path = pred_root / task / "predictions.csv"
    if not preds_path.exists():
        print(f"[{task}] no predictions.csv at {preds_path}")
        return

    cfg = TASKS[task]
    samples, _ = load_samples(task)
    # Canonical rowid → sample
    canonical = {str(s["_rowid"]): s for s in samples}

    df = pd.read_csv(preds_path, dtype={"rowid": str})
    n_before = len(df)

    # Drop spurious rows
    df_canon = df[df["rowid"].isin(canonical)].copy()
    n_dropped = n_before - len(df_canon)

    # Find missing rowids and add <missing> rows
    have = set(df_canon["rowid"])
    missing = [rid for rid in canonical if rid not in have]
    if missing:
        rows = []
        for rid in missing:
            s = canonical[rid]
            try:
                tv = normalize_true(task, s.get(cfg["label_col"]))
            except Exception:
                tv = None
            rows.append({
                "rowid": rid, "fold": s.get("_fold"),
                "duration_sec": None,
                "true": tv if cfg["metric"] != "multilabel" else None,
                "pred": None,
                "true_vec": tv if cfg["metric"] == "multilabel" else None,
                "pred_vec": None,
                "response": "<missing>",
            })
        df_canon = pd.concat([df_canon, pd.DataFrame(rows)], ignore_index=True)

    # Overwrite ground-truth AND fold from the canonical per-task manifest.
    # Fold matters because gemini_batch dedup inherits the first task's fold
    # for shared audios (T11 from T10, T18 from T17, etc.).
    is_multi = cfg["metric"] == "multilabel"
    canon_true: dict[str, object] = {}
    canon_fold: dict[str, object] = {}
    for rid, s in canonical.items():
        try:
            canon_true[rid] = normalize_true(task, s.get(cfg["label_col"]))
        except Exception:
            canon_true[rid] = None
        canon_fold[rid] = s.get("_fold")

    if is_multi:
        df_canon["true_vec"] = df_canon["rowid"].map(canon_true)
    else:
        df_canon["true"] = df_canon["rowid"].map(canon_true)

    # Only overwrite fold if the manifest actually has folds (CV task);
    # otherwise leave whatever was there (likely all NaN).
    if any(v is not None for v in canon_fold.values()):
        df_canon["fold"] = df_canon["rowid"].map(canon_fold)

    df_canon.to_csv(preds_path, index=False)

    # Recompute metrics
    df_load = pd.read_csv(preds_path)
    for col in ("true_vec", "pred_vec"):
        df_load[col] = df_load[col].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x
        )
    try:
        metrics = {"overall": compute_metrics(task, df_load)}
        if df_load["fold"].notna().any():
            per_fold = []
            for fi in sorted(df_load["fold"].dropna().unique()):
                sub = df_load[df_load["fold"] == fi]
                m = compute_metrics(task, sub)
                m["fold"] = int(fi)
                per_fold.append(m)
            metrics["per_fold"] = per_fold
            metrics["fold_aggregate"] = cv_aggregate(per_fold)
        out = preds_path.with_suffix(".metrics.json")
        out.write_text(json.dumps(metrics, indent=2, default=str))
    except Exception as e:
        print(f"[{task}] metrics calc failed: {type(e).__name__}: {e}")
        return

    print(f"[{task}] n_canonical={len(canonical)}  before={n_before}  "
          f"after={len(df_canon)}  dropped={n_dropped}  added_missing={len(missing)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default=None, help="single task ID")
    p.add_argument("--root", default="exps/gemini_3_1_pro",
                   help="prediction root (default: exps/gemini_3_1_pro)")
    args = p.parse_args()

    pred_root = Path(args.root)
    tasks = [args.task] if args.task else sorted(TASKS.keys(), key=lambda t: int(t[1:]))
    print(f"Regenerating metrics in {pred_root}")
    for t in tasks:
        regen_task(t, pred_root)


if __name__ == "__main__":
    main()
