"""Interactive retry of failed predictions across all SpeechDx tasks.

For each row in exps/gemini_3_1_pro/<task>/predictions.csv whose response
starts with "<batch-error" or "<upload-failed", re-run via the default
generate_content API (interactive, standard tier), parse, and overwrite the row.

Dedups: if the same rowid is failed on multiple tasks within the same dataset,
the audio file is uploaded once and queried with each prompt.

Usage:
    python scripts/gemini_retry_interactive.py
    python scripts/gemini_retry_interactive.py --dataset c19sounds
    python scripts/gemini_retry_interactive.py --workers 8
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_qwen3omni_all import (  # noqa: E402
    TASKS, PROMPTS, load_samples, normalize_true,
    compute_metrics, cv_aggregate,
)
from gemini_batch import (  # noqa: E402
    DATASETS, PRED_ROOT, rewrite_path, audio_duration_sec,
    get_api_key, _parse_response,
)

TASKS["T7"]["source"] = {"manifest": "exps/single_task/dbank_adC/manifest", "split": "test"}
TASKS["T8"]["source"] = {"manifest": "exps/single_task/dbank_mmseR/manifest", "split": "test"}

MODEL = "gemini-3.1-pro-preview"
MAX_RETRIES = 8


def find_failed(dataset_filter: str | None = None) -> list[dict]:
    """[{dataset, rowid, tasks, sample}, ...]."""
    work = []
    for dataset, tasks in DATASETS.items():
        if dataset_filter and dataset != dataset_filter:
            continue
        per_task_samples = {t: {str(s["_rowid"]): s
                                for s in load_samples(t)[0]} for t in tasks}

        failed_by_rid: dict[str, list[str]] = {}
        for t in tasks:
            f = PRED_ROOT / t / "predictions.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f, dtype={"rowid": str})
            resp = df["response"].astype(str).fillna("")
            mask = resp.str.startswith("<batch-error") | resp.str.startswith("<upload-failed")
            for rid in df.loc[mask, "rowid"]:
                failed_by_rid.setdefault(str(rid), []).append(t)

        for rid, ts in failed_by_rid.items():
            # pick any task's sample (audio path is same; ground-truth resolved per-task later)
            sample = next((per_task_samples[t].get(rid) for t in ts
                           if rid in per_task_samples.get(t, {})), None)
            if sample:
                work.append({"dataset": dataset, "rowid": rid, "tasks": ts,
                             "sample": sample, "per_task_samples": per_task_samples})
    return work


def _retry_call(label: str, fn, max_retries: int = MAX_RETRIES, cap: float = 60.0):
    last = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last = e
            wait = random.uniform(0.5, min(cap, 2 ** attempt))
            time.sleep(wait)
    raise RuntimeError(f"{label}: max retries exceeded ({last!r})")


def process_one(client, types, item: dict) -> tuple[dict, list[tuple[str, dict]]]:
    """Upload audio (if exists), run each pending task, return rows to update."""
    sample = item["sample"]
    rowid = item["rowid"]
    path = rewrite_path(sample["path"])

    rows: list[tuple[str, dict]] = []
    duration = audio_duration_sec(path) if Path(path).exists() else None

    def fail_all(reason: str):
        for task in item["tasks"]:
            cfg = TASKS[task]
            spt = item["per_task_samples"].get(task, {}).get(rowid, sample)
            try:
                tv = normalize_true(task, spt.get(cfg["label_col"]))
            except Exception:
                tv = None
            rows.append((task, {
                "rowid": rowid, "fold": spt.get("_fold"),
                "duration_sec": duration,
                "true": tv if cfg["metric"] != "multilabel" else None,
                "pred": None,
                "true_vec": tv if cfg["metric"] == "multilabel" else None,
                "pred_vec": None,
                "response": reason,
            }))

    if not Path(path).exists():
        fail_all("<retry-load-failed>")
        return item, rows

    try:
        f = _retry_call("upload",
                        lambda: client.files.upload(file=path))
        # wait for ACTIVE
        deadline = time.time() + 300
        while f.state and f.state.name == "PROCESSING":
            if time.time() > deadline:
                raise RuntimeError("file processing timeout")
            time.sleep(2)
            f = client.files.get(name=f.name)
        if f.state and f.state.name != "ACTIVE":
            raise RuntimeError(f"file state {f.state.name}")
    except Exception as e:
        fail_all(f"<retry-upload-failed: {type(e).__name__}: {str(e)[:120]}>")
        return item, rows

    try:
        for task in item["tasks"]:
            cfg = TASKS[task]
            spt = item["per_task_samples"].get(task, {}).get(rowid, sample)
            try:
                tv = normalize_true(task, spt.get(cfg["label_col"]))
            except Exception:
                tv = None
            contents = [
                types.Part.from_uri(file_uri=f.uri, mime_type=f.mime_type),
                PROMPTS[task],
            ]
            config = types.GenerateContentConfig(service_tier=types.ServiceTier.STANDARD)
            try:
                resp = _retry_call(
                    "gen",
                    lambda: client.models.generate_content(
                        model=MODEL, contents=contents, config=config),
                )
                resp_text = resp.text or ""
            except Exception as e:
                resp_text = f"<retry-gen-failed: {type(e).__name__}: {str(e)[:120]}>"
            pred, pred_vec = _parse_response(task, resp_text)
            rows.append((task, {
                "rowid": rowid, "fold": spt.get("_fold"),
                "duration_sec": duration,
                "true": tv if cfg["metric"] != "multilabel" else None,
                "pred": pred,
                "true_vec": tv if cfg["metric"] == "multilabel" else None,
                "pred_vec": pred_vec,
                "response": resp_text,
            }))
    finally:
        try:
            client.files.delete(name=f.name)
        except Exception:
            pass

    return item, rows


def _save_metrics(task: str, preds_path: Path) -> None:
    df = pd.read_csv(preds_path)
    for col in ("true_vec", "pred_vec"):
        df[col] = df[col].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x
        )
    try:
        metrics = {"overall": compute_metrics(task, df)}
        if df["fold"].notna().any():
            per_fold = []
            for fi in sorted(df["fold"].dropna().unique()):
                sub = df[df["fold"] == fi]
                m = compute_metrics(task, sub)
                m["fold"] = int(fi)
                per_fold.append(m)
            metrics["per_fold"] = per_fold
            metrics["fold_aggregate"] = cv_aggregate(per_fold)
        out = preds_path.with_suffix(".metrics.json")
        out.write_text(json.dumps(metrics, indent=2, default=str))
    except Exception as e:
        print(f"  [warn] {task} metrics calc failed: {e}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=None, choices=list(DATASETS.keys()))
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    from google import genai
    from google.genai import types
    client = genai.Client(api_key=get_api_key())

    work = find_failed(args.dataset)
    n_calls = sum(len(w["tasks"]) for w in work)
    print(f"Failures: {len(work)} unique rowids, {n_calls} inference calls "
          f"({'all datasets' if not args.dataset else args.dataset})")
    if not work:
        return 0

    updates_by_task: dict[str, dict[str, dict]] = {}
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_one, client, types, w) for w in work]
        for fut in as_completed(futures):
            done += 1
            try:
                item, rows = fut.result()
                for task, row in rows:
                    updates_by_task.setdefault(task, {})[row["rowid"]] = row
            except Exception as e:
                import traceback
                print(f"  [error] {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
            if done % 10 == 0 or done == len(futures):
                elapsed = time.time() - t0
                rate = elapsed / done
                eta = rate * (len(futures) - done)
                print(f"  {done}/{len(futures)} done  elapsed={elapsed:.0f}s "
                      f"eta={eta:.0f}s", flush=True)

    print(f"\nWriting updates to {len(updates_by_task)} task CSVs ...")
    for task, rows in updates_by_task.items():
        preds_path = PRED_ROOT / task / "predictions.csv"
        df = pd.read_csv(preds_path, dtype={"rowid": str})
        for col in ("true_vec", "pred_vec"):
            if col in df.columns:
                df[col] = df[col].astype(object)
        n_updated = 0
        for rid, new in rows.items():
            idxs = df.index[df["rowid"] == rid].tolist()
            if idxs:
                for idx in idxs:
                    for col, val in new.items():
                        df.at[idx, col] = val
                n_updated += 1
            else:
                df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
                n_updated += 1
        df.to_csv(preds_path, index=False)
        _save_metrics(task, preds_path)
        print(f"  [{task}] updated {n_updated} rows")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
