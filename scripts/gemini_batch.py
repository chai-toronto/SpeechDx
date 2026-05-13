"""Gemini Batch API runner for SpeechDx benchmark.

Subcommands (operate on one or more datasets):
    submit   upload audio + JSONL request file, create batch job
    status   show batch state(s)
    fetch    download batch results, parse to per-task predictions.csv + metrics
    cleanup  delete uploaded audio files from the Gemini Files API

Persistent state at exps/gemini_batches/<dataset>/:
    audio_uris.jsonl   resumable per-file upload log
    batch_job.json     {job_name, input_file, dest_file, state, ...}
    requests.jsonl     batch input (audit trail)
    results.jsonl      downloaded raw batch output

Per-task predictions write to exps/gemini_3_1_pro/<task>/predictions.csv so
results live alongside the interactive E-DAIC run.

Usage:
    python scripts/gemini_batch.py submit  --dataset mvdr
    python scripts/gemini_batch.py submit  --all
    python scripts/gemini_batch.py status
    python scripts/gemini_batch.py fetch   --dataset mvdr
    python scripts/gemini_batch.py cleanup --dataset mvdr
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_qwen3omni_all import (  # noqa: E402
    TASKS, PROMPTS, load_samples, normalize_true,
    parse_yes_no, parse_label, parse_integer, parse_multilabel,
    compute_metrics, cv_aggregate,
)

# Repair stale registry paths
TASKS["T7"]["source"] = {"manifest": "exps/single_task/dbank_adC/manifest", "split": "test"}
TASKS["T8"]["source"] = {"manifest": "exps/single_task/dbank_mmseR/manifest", "split": "test"}

MODEL = "gemini-3.1-pro-preview"
REPO = Path(__file__).resolve().parents[1]
STATE_ROOT = REPO / "exps" / "gemini_batches"
PRED_ROOT = REPO / "exps" / "gemini_3_1_pro"

# Each dataset bundles the tasks that share its audio source.
DATASETS: dict[str, list[str]] = {
    "edaic":     ["T1", "T2"],
    "ravdess":   ["T3", "T4"],
    "iemocap":   ["T5", "T6"],
    "dbank":     ["T7", "T8"],
    "aphasia":   ["T9"],
    "torgo":     ["T10", "T11"],
    "uaspeech":  ["T12"],
    "mvdr":      ["T13", "T14", "T15", "T16"],
    "ksof":      ["T17", "T18"],
    "c19sounds": ["T19", "T20", "T21", "T22", "T23"],
    "coswara":   ["T24", "T25", "T26"],
    "avfad":     ["T27"],
}

# Manifest paths use legacy dir names; rewrite to current on-disk locations.
_PATH_REWRITES = {
    "/data/dbank/": "/data/dementiabank/",
    "/data/c9s/": "/data/c19sounds/",
}
_TORGO_PREFIX_RE = re.compile(r"^s\d+_")


def rewrite_path(p: str) -> str:
    for old, new in _PATH_REWRITES.items():
        if old in p:
            p = p.replace(old, new, 1)
            break
    if "/data/torgo/" in p and not Path(p).exists():
        path = Path(p)
        stripped = _TORGO_PREFIX_RE.sub("", path.name)
        if stripped != path.name:
            alt = path.with_name(stripped)
            if alt.exists():
                return str(alt)
    return p


def audio_duration_sec(path: str) -> float | None:
    try:
        info = sf.info(path)
        return info.frames / float(info.samplerate)
    except Exception:
        return None


def mime_for(path: str) -> str:
    ext = Path(path).suffix.lower()
    return {
        ".wav": "audio/wav", ".mp3": "audio/mp3", ".flac": "audio/flac",
        ".m4a": "audio/mp4", ".ogg": "audio/ogg", ".opus": "audio/opus",
        ".aac": "audio/aac",
    }.get(ext, "audio/wav")


def get_api_key() -> str | None:
    if k := os.environ.get("GEMINI_API_KEY"):
        return k
    try:
        out = subprocess.run(
            ["security", "find-generic-password", "-s", "gemini-api-key", "-w"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def make_client():
    from google import genai
    key = get_api_key()
    if not key:
        sys.exit("ERROR: no Gemini API key (env GEMINI_API_KEY or keychain 'gemini-api-key')")
    return genai.Client(api_key=key)


# ============================================================
# Sample collection (deduped audio across all tasks in a dataset)
# ============================================================

def collect_samples(dataset: str) -> tuple[list[dict], dict[str, list[str]]]:
    """Return (unique_samples, rowid -> [task_id, ...] map).

    A "sample" here is a dict keyed by audio path; multiple tasks may target
    the same audio. We dedup by rowid (CV folds collapse) and union across
    tasks in the dataset.
    """
    task_ids = DATASETS[dataset]
    by_rowid: dict[str, dict] = {}
    tasks_per_rowid: dict[str, list[str]] = {}
    for task in task_ids:
        samples, _ = load_samples(task)
        for s in samples:
            rid = str(s["_rowid"])
            if rid not in by_rowid:
                by_rowid[rid] = s
            tasks_per_rowid.setdefault(rid, [])
            if task not in tasks_per_rowid[rid]:
                tasks_per_rowid[rid].append(task)
    return list(by_rowid.values()), tasks_per_rowid


# ============================================================
# State files
# ============================================================

def state_dir(dataset: str) -> Path:
    d = STATE_ROOT / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_audio_uris(dataset: str) -> dict[str, dict]:
    """Read rowid -> {uri, name, mime_type} from audio_uris.jsonl."""
    p = state_dir(dataset) / "audio_uris.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[rec["rowid"]] = rec
    return out


def append_audio_uri(dataset: str, rec: dict) -> None:
    p = state_dir(dataset) / "audio_uris.jsonl"
    with p.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def save_batch_job(dataset: str, info: dict) -> None:
    (state_dir(dataset) / "batch_job.json").write_text(json.dumps(info, indent=2))


def load_batch_job(dataset: str) -> dict | None:
    p = state_dir(dataset) / "batch_job.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


# ============================================================
# Submit
# ============================================================

def _upload_one(client, path: str, rowid: str) -> dict:
    """Upload one audio file and wait for ACTIVE state."""
    f = client.files.upload(file=path)
    deadline = time.time() + 600
    while f.state and f.state.name == "PROCESSING":
        if time.time() > deadline:
            raise RuntimeError(f"file processing timed out for {path}")
        time.sleep(2)
        f = client.files.get(name=f.name)
    if f.state and f.state.name == "FAILED":
        raise RuntimeError(f"file processing FAILED for {path}")
    return {"rowid": rowid, "name": f.name, "uri": f.uri,
            "mime_type": f.mime_type, "local_path": path}


def _is_retryable_err(err: str) -> bool:
    """Network blips, 429s, 503s, and unknown-mime should be retried next time."""
    if not err:
        return False
    if "missing-on-disk" in err:
        return False
    markers = ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE",
               "ConnectError", "TimeoutError", "Unknown mime"]
    return any(m in err for m in markers)


def _rewrite_audio_uris(dataset: str, keep: dict[str, dict]) -> None:
    """Rewrite the append-only log to drop entries we're going to retry."""
    p = state_dir(dataset) / "audio_uris.jsonl"
    with p.open("w") as f:
        for rec in keep.values():
            f.write(json.dumps(rec) + "\n")


def upload_audio(client, dataset: str, samples: list[dict], workers: int) -> dict[str, dict]:
    """Upload all sample audio files (resumable). Returns rowid -> file rec map.

    Entries that previously failed with transient errors (429, network, 503,
    bad mime-type detection) are eligible for retry — drop them from the log
    so they re-appear in the to_upload queue.
    """
    existing = load_audio_uris(dataset)

    # Sweep: any retryable-failure entries get cleared so they go back in queue
    retryable = [rid for rid, rec in existing.items()
                 if not rec.get("uri") and _is_retryable_err(rec.get("error", ""))]
    if retryable:
        print(f"[{dataset}] clearing {len(retryable)} retryable-failure entries "
              f"for re-upload", flush=True)
        keep = {rid: rec for rid, rec in existing.items() if rid not in retryable}
        _rewrite_audio_uris(dataset, keep)
        existing = keep

    to_upload = []
    for s in samples:
        rid = str(s["_rowid"])
        if rid in existing:
            continue
        path = rewrite_path(s["path"])
        if not Path(path).exists():
            # Mark as failed-to-find; emit a fake record so batch can still
            # emit a per-task <load-failed> row later.
            append_audio_uri(dataset, {"rowid": rid, "name": None, "uri": None,
                                       "mime_type": None, "local_path": path,
                                       "error": "missing-on-disk"})
            existing[rid] = {"rowid": rid, "uri": None, "error": "missing-on-disk"}
            continue
        to_upload.append((rid, path))

    print(f"[{dataset}] {len(existing)} already uploaded, {len(to_upload)} new "
          f"(total {len(samples)})", flush=True)
    if not to_upload:
        return existing

    done_n = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_upload_one, client, p, r): (r, p) for r, p in to_upload}
        for fut in as_completed(futures):
            rid, path = futures[fut]
            try:
                rec = fut.result()
                append_audio_uri(dataset, rec)
                existing[rid] = rec
            except Exception as e:
                err = {"rowid": rid, "name": None, "uri": None,
                       "mime_type": None, "local_path": path,
                       "error": f"{type(e).__name__}: {str(e)[:200]}"}
                append_audio_uri(dataset, err)
                existing[rid] = err
                print(f"  [{dataset}] upload FAILED {rid}: {err['error']}", flush=True)
            done_n += 1
            if done_n % 20 == 0 or done_n == len(to_upload):
                rate = (time.time() - t0) / done_n
                eta = rate * (len(to_upload) - done_n)
                print(f"  [{dataset}] uploaded {done_n}/{len(to_upload)}  "
                      f"({rate:.1f}s/file, eta {eta:.0f}s)", flush=True)
    return existing


def build_requests_jsonl(dataset: str, samples: list[dict],
                         tasks_per_rowid: dict[str, list[str]],
                         audio_recs: dict[str, dict]) -> tuple[Path, list[dict]]:
    """Write the batch JSONL. Each line = one (rowid, task) inference call.

    Skip rowids whose audio failed to upload (we'll record per-task
    <upload-failed> rows during fetch).
    """
    out_path = state_dir(dataset) / "requests.jsonl"
    rows_meta = []  # parallel list for fetch parsing

    # Per-task fold lookup. The same rowid can land in different folds across
    # sister tasks (e.g. T11 vs T10 differ on 2760/3181 rowids), so we can't
    # use the deduped sample's `_fold` for every task — that gives the first
    # task's partition to all siblings.
    per_task_folds: dict[str, dict[str, object]] = {}
    for task in DATASETS[dataset]:
        per_task_folds[task] = {str(s["_rowid"]): s.get("_fold")
                                for s in load_samples(task)[0]}

    with out_path.open("w") as f:
        for s in samples:
            rid = str(s["_rowid"])
            rec = audio_recs.get(rid)
            if not rec or not rec.get("uri"):
                continue
            local = rewrite_path(s["path"])
            duration = audio_duration_sec(local)
            for task in tasks_per_rowid[rid]:
                key = f"{rid}__{task}"
                req = {
                    "key": key,
                    "request": {
                        "contents": [{
                            "parts": [
                                {"file_data": {"mime_type": rec["mime_type"],
                                               "file_uri": rec["uri"]}},
                                {"text": PROMPTS[task]},
                            ],
                        }],
                    },
                }
                f.write(json.dumps(req) + "\n")
                rows_meta.append({"key": key, "rowid": rid, "task": task,
                                  "duration_sec": duration,
                                  "fold": per_task_folds.get(task, {}).get(rid)})
    return out_path, rows_meta


def submit_dataset(client, dataset: str, workers: int = 32) -> None:
    existing_job = load_batch_job(dataset)
    if existing_job and existing_job.get("state") not in (None, "JOB_STATE_FAILED",
                                                         "JOB_STATE_EXPIRED",
                                                         "JOB_STATE_CANCELLED"):
        print(f"[{dataset}] already submitted: {existing_job['job_name']} "
              f"({existing_job.get('state')}). Use 'status' or 'fetch'.")
        return

    print(f"[{dataset}] collecting samples...")
    samples, tasks_per_rowid = collect_samples(dataset)
    n_requests = sum(len(v) for v in tasks_per_rowid.values())
    print(f"[{dataset}] {len(samples)} unique audios, "
          f"{n_requests} (audio,task) inference requests, "
          f"tasks={DATASETS[dataset]}")

    print(f"[{dataset}] uploading audio (workers={workers})...")
    audio_recs = upload_audio(client, dataset, samples, workers=workers)

    n_ok = sum(1 for r in audio_recs.values() if r.get("uri"))
    n_failed = len(audio_recs) - n_ok
    print(f"[{dataset}] uploads: {n_ok} ok, {n_failed} failed")
    if n_ok == 0:
        print(f"[{dataset}] nothing to submit (no audio uploaded)")
        return

    print(f"[{dataset}] building requests.jsonl ...")
    req_path, rows_meta = build_requests_jsonl(dataset, samples, tasks_per_rowid, audio_recs)
    (state_dir(dataset) / "rows_meta.json").write_text(json.dumps(rows_meta))
    print(f"[{dataset}] wrote {len(rows_meta)} requests to {req_path}")

    print(f"[{dataset}] uploading JSONL ...")
    from google.genai import types
    src = client.files.upload(
        file=str(req_path),
        config=types.UploadFileConfig(display_name=f"{dataset}-batch-input",
                                      mime_type="jsonl"),
    )
    print(f"[{dataset}] JSONL uploaded as {src.name}")

    print(f"[{dataset}] creating batch job ...")
    job = client.batches.create(
        model=MODEL,
        src=src.name,
        config=types.CreateBatchJobConfig(display_name=f"speechdx-{dataset}"),
    )
    info = {
        "dataset": dataset,
        "job_name": job.name,
        "model": MODEL,
        "input_file_name": src.name,
        "dest_file_name": getattr(getattr(job, "dest", None), "file_name", None),
        "state": job.state.name if job.state else None,
        "submitted_at": time.time(),
        "n_requests": len(rows_meta),
        "n_audios": n_ok,
        "n_audios_failed": n_failed,
    }
    save_batch_job(dataset, info)
    print(f"[{dataset}] submitted: {job.name}  state={info['state']}")


# ============================================================
# Status
# ============================================================

def status_dataset(client, dataset: str) -> dict | None:
    info = load_batch_job(dataset)
    if not info:
        return None
    try:
        job = client.batches.get(name=info["job_name"])
    except Exception as e:
        print(f"[{dataset}] status fetch failed: {type(e).__name__}: {e}")
        return info
    state = job.state.name if job.state else "UNKNOWN"
    dest = getattr(getattr(job, "dest", None), "file_name", None)
    info["state"] = state
    info["dest_file_name"] = dest
    save_batch_job(dataset, info)
    return info


def cmd_status(client, args) -> None:
    datasets = [args.dataset] if args.dataset else list(DATASETS.keys())
    rows = []
    for d in datasets:
        info = status_dataset(client, d) if load_batch_job(d) else None
        rows.append({
            "dataset": d,
            "submitted": "yes" if info else "no",
            "state": info["state"] if info else "-",
            "n_requests": info["n_requests"] if info else "-",
            "job": info["job_name"].split("/")[-1] if info else "-",
        })
    print(pd.DataFrame(rows).to_string(index=False))


# ============================================================
# Fetch + parse
# ============================================================

def _parse_response(task: str, text: str):
    cfg = TASKS[task]
    kind = cfg["parser"]
    if kind == "yes_no":
        return parse_yes_no(text), None
    if kind == "label":
        return parse_label(text, cfg["classes"]), None
    if kind == "integer":
        lo, hi = cfg.get("range", (-1e9, 1e9))
        return parse_integer(text, lo, hi), None
    if kind == "multilabel":
        return None, parse_multilabel(text, cfg["classes"])
    return None, None


def _extract_text(response_obj) -> str:
    """Best-effort: pull text from a batch result's response dict."""
    if not isinstance(response_obj, dict):
        return ""
    cands = response_obj.get("candidates", [])
    if not cands:
        return ""
    parts = cands[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts)


def fetch_dataset(client, dataset: str) -> None:
    info = load_batch_job(dataset)
    if not info:
        print(f"[{dataset}] no batch job recorded")
        return
    job = client.batches.get(name=info["job_name"])
    state = job.state.name if job.state else "UNKNOWN"
    if state not in ("JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"):
        print(f"[{dataset}] not ready: state={state}")
        info["state"] = state
        save_batch_job(dataset, info)
        return

    dest_name = getattr(getattr(job, "dest", None), "file_name", None)
    if not dest_name:
        print(f"[{dataset}] succeeded but no dest file name; raw job: {job}")
        return
    print(f"[{dataset}] downloading results from {dest_name} ...")
    blob = client.files.download(file=dest_name)
    results_path = state_dir(dataset) / "results.jsonl"
    results_path.write_bytes(blob if isinstance(blob, bytes) else blob.encode())

    # Reconstruct (key -> meta) map for label + duration lookup
    meta_path = state_dir(dataset) / "rows_meta.json"
    meta_list = json.loads(meta_path.read_text())
    meta_by_key = {m["key"]: m for m in meta_list}

    # Per-task sample map (so multilabel/binary tasks share rowids but have
    # the right ground-truth column for THIS task)
    per_task_samples: dict[str, dict[str, dict]] = {}
    for task in DATASETS[dataset]:
        samples_t, _ = load_samples(task)
        per_task_samples[task] = {str(s["_rowid"]): s for s in samples_t}

    # Group output rows per task
    per_task_rows: dict[str, list[dict]] = {t: [] for t in DATASETS[dataset]}

    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        key = rec.get("key", "")
        meta = meta_by_key.get(key)
        if meta is None:
            continue
        task = meta["task"]
        cfg = TASKS[task]
        sample = per_task_samples.get(task, {}).get(meta["rowid"], {})
        try:
            true_val = normalize_true(task, sample.get(cfg["label_col"]))
        except Exception as e:
            print(f"  [warn] {task}/{meta['rowid']}: normalize_true failed "
                  f"({type(e).__name__}); true_val=None", flush=True)
            true_val = None

        if "error" in rec and rec["error"]:
            resp_text = f"<batch-error: {json.dumps(rec['error'])[:200]}>"
            pred, pred_vec = None, None
        else:
            resp_text = _extract_text(rec.get("response", {}))
            pred, pred_vec = _parse_response(task, resp_text)

        per_task_rows[task].append({
            "rowid": meta["rowid"],
            "fold": meta.get("fold"),
            "duration_sec": meta.get("duration_sec"),
            "true": true_val if cfg["metric"] != "multilabel" else None,
            "pred": pred,
            "true_vec": true_val if cfg["metric"] == "multilabel" else None,
            "pred_vec": pred_vec,
            "response": resp_text,
        })

    # Also record any rowids that had no upload (so predictions.csv covers
    # the same sample set as the manifest).
    audio_recs = load_audio_uris(dataset)
    failed_rids = [r for r, rec in audio_recs.items() if not rec.get("uri")]
    for rid in failed_rids:
        for task in DATASETS[dataset]:
            cfg = TASKS[task]
            sample = per_task_samples.get(task, {}).get(rid, {"_rowid": rid})
            try:
                true_val = normalize_true(task, sample.get(cfg["label_col"]))
            except Exception:
                true_val = None
            per_task_rows[task].append({
                "rowid": rid, "fold": sample.get("_fold"),
                "duration_sec": None,
                "true": true_val if cfg["metric"] != "multilabel" else None,
                "pred": None,
                "true_vec": true_val if cfg["metric"] == "multilabel" else None,
                "pred_vec": None,
                "response": f"<upload-failed: {audio_recs[rid].get('error')}>",
            })

    for task, rows in per_task_rows.items():
        task_dir = PRED_ROOT / task
        task_dir.mkdir(parents=True, exist_ok=True)
        preds_path = task_dir / "predictions.csv"
        existing_rids = set()
        if preds_path.exists():
            existing_rids = set(pd.read_csv(preds_path, usecols=["rowid"])
                                ["rowid"].astype(str).tolist())
        new_rows = [r for r in rows if str(r["rowid"]) not in existing_rids]
        if new_rows:
            df = pd.DataFrame(new_rows)
            header = not preds_path.exists()
            df.to_csv(preds_path, mode="a", header=header, index=False)
        print(f"[{dataset}/{task}] {len(new_rows)} new rows  "
              f"(existing {len(existing_rids)} preserved)")
        _compute_and_save_metrics(task, preds_path)

    info["state"] = state
    save_batch_job(dataset, info)


def _compute_and_save_metrics(task: str, preds_path: Path) -> None:
    if not preds_path.exists():
        return
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
        print(f"[{task}] metrics -> {out}")
        print(json.dumps(metrics["overall"], indent=2, default=str))
    except Exception as e:
        print(f"[{task}] metrics calc failed: {type(e).__name__}: {e}", flush=True)


# ============================================================
# Delta retry (failed-upload + failed-inference)
# ============================================================

def _find_failed_predictions(dataset: str) -> dict[str, list[str]]:
    """rowid -> [task,...] for any row whose prediction is missing or errored."""
    failed: dict[str, list[str]] = {}
    for task in DATASETS[dataset]:
        f = PRED_ROOT / task / "predictions.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, dtype={"rowid": str})
        resp = df["response"].astype(str).fillna("")
        bad_mask = resp.str.startswith("<batch-error") | resp.str.startswith("<upload-failed")
        for rid in df.loc[bad_mask, "rowid"]:
            failed.setdefault(str(rid), []).append(task)
    return failed


def _load_retry_uris(dataset: str) -> dict[str, dict]:
    p = state_dir(dataset) / "audio_uris_retry.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("uri"):
            out[rec["rowid"]] = rec
    return out


def submit_delta(client, dataset: str, workers: int = 4) -> None:
    """Build + submit a retry batch for any predictions that came back as
    <batch-error> or <upload-failed>. Reuses existing file URIs where they're
    still ACTIVE; re-uploads where the audio was cleaned."""
    failed = _find_failed_predictions(dataset)
    if not failed:
        print(f"[{dataset}] no failed predictions")
        return

    n_calls = sum(len(v) for v in failed.values())
    print(f"[{dataset}] {len(failed)} rowids, {n_calls} (rowid,task) calls to retry")

    # Build a per-task sample map (so true labels match the right task)
    per_task_samples: dict[str, dict[str, dict]] = {}
    for task in DATASETS[dataset]:
        samples_t, _ = load_samples(task)
        per_task_samples[task] = {str(s["_rowid"]): s for s in samples_t}

    # Resolve audio URI for each failed rowid:
    #   1. existing retry URI (resumable)
    #   2. probe main audio_uris.jsonl — keep if still ACTIVE
    #   3. else re-upload
    main_recs = load_audio_uris(dataset)
    retry_recs = _load_retry_uris(dataset)
    uri_for: dict[str, dict] = dict(retry_recs)

    to_upload: list[tuple[str, str]] = []
    for rid in failed:
        if rid in uri_for:
            continue
        rec = main_recs.get(rid)
        if rec and rec.get("uri") and rec.get("name"):
            # Probe — main batch may have cleaned this up
            try:
                f = client.files.get(name=rec["name"])
                if f.state and f.state.name == "ACTIVE":
                    uri_for[rid] = rec
                    continue
            except Exception:
                pass
        # Need re-upload. Find audio path from any task's sample.
        local_path = None
        for task in DATASETS[dataset]:
            s = per_task_samples.get(task, {}).get(rid)
            if s:
                p = rewrite_path(s["path"])
                if Path(p).exists():
                    local_path = p
                    break
        if local_path is None:
            print(f"  [warn] {dataset}/{rid}: no audio on disk; skipping")
            continue
        to_upload.append((rid, local_path))

    if to_upload:
        print(f"[{dataset}] re-uploading {len(to_upload)} audios (workers={workers})...")
        retry_path = state_dir(dataset) / "audio_uris_retry.jsonl"
        done_n = 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_upload_one, client, p, r): (r, p)
                       for r, p in to_upload}
            for fut in as_completed(futures):
                rid, path = futures[fut]
                try:
                    rec = fut.result()
                    uri_for[rid] = rec
                    with retry_path.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                except Exception as e:
                    print(f"  [warn] {dataset}/{rid} re-upload failed: "
                          f"{type(e).__name__}: {str(e)[:120]}", flush=True)
                done_n += 1
                if done_n % 50 == 0 or done_n == len(to_upload):
                    rate = (time.time() - t0) / done_n
                    eta = rate * (len(to_upload) - done_n)
                    print(f"  [{dataset}] re-uploaded {done_n}/{len(to_upload)} "
                          f"({rate:.1f}s/file, eta {eta:.0f}s)", flush=True)

    # Build requests JSONL
    rows_meta = []
    req_path = state_dir(dataset) / "requests_retry.jsonl"
    with req_path.open("w") as f:
        for rid, tasks in failed.items():
            rec = uri_for.get(rid)
            if not rec or not rec.get("uri"):
                continue
            local = next((rewrite_path(per_task_samples[t][rid]["path"])
                          for t in tasks if rid in per_task_samples.get(t, {})), None)
            duration = audio_duration_sec(local) if local else None
            for task in tasks:
                key = f"{rid}__{task}__retry"
                req = {
                    "key": key,
                    "request": {"contents": [{"parts": [
                        {"file_data": {"mime_type": rec["mime_type"],
                                       "file_uri": rec["uri"]}},
                        {"text": PROMPTS[task]},
                    ]}]},
                }
                f.write(json.dumps(req) + "\n")
                # Take fold from whichever task's sample first matches
                fold = None
                for t in tasks:
                    s = per_task_samples.get(t, {}).get(rid)
                    if s and s.get("_fold") is not None:
                        fold = s["_fold"]; break
                rows_meta.append({"key": key, "rowid": rid, "task": task,
                                  "duration_sec": duration, "fold": fold})

    if not rows_meta:
        print(f"[{dataset}] nothing to submit (all retries failed)")
        return
    (state_dir(dataset) / "rows_meta_retry.json").write_text(json.dumps(rows_meta))
    print(f"[{dataset}] wrote {len(rows_meta)} retry requests to {req_path}")

    print(f"[{dataset}] uploading retry JSONL ...")
    from google.genai import types
    src = client.files.upload(
        file=str(req_path),
        config=types.UploadFileConfig(display_name=f"{dataset}-retry-input",
                                      mime_type="jsonl"),
    )

    print(f"[{dataset}] creating retry batch ...")
    job = client.batches.create(
        model=MODEL, src=src.name,
        config=types.CreateBatchJobConfig(display_name=f"speechdx-{dataset}-retry"),
    )
    info = {
        "dataset": dataset, "job_name": job.name, "model": MODEL,
        "input_file_name": src.name,
        "dest_file_name": getattr(getattr(job, "dest", None), "file_name", None),
        "state": job.state.name if job.state else None,
        "submitted_at": time.time(),
        "n_requests": len(rows_meta), "is_retry": True,
    }
    (state_dir(dataset) / "batch_job_retry.json").write_text(json.dumps(info, indent=2))
    print(f"[{dataset}] retry batch submitted: {job.name}")


def fetch_delta(client, dataset: str) -> None:
    """Download retry results; overwrite failed rows in predictions.csv."""
    p = state_dir(dataset) / "batch_job_retry.json"
    if not p.exists():
        print(f"[{dataset}] no retry batch")
        return
    info = json.loads(p.read_text())
    job = client.batches.get(name=info["job_name"])
    state = job.state.name if job.state else "?"
    if state not in ("JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"):
        print(f"[{dataset}] retry not ready: {state}")
        info["state"] = state
        p.write_text(json.dumps(info, indent=2))
        return

    dest_name = getattr(getattr(job, "dest", None), "file_name", None)
    if not dest_name:
        print(f"[{dataset}] succeeded but no dest")
        return
    print(f"[{dataset}] downloading retry results from {dest_name} ...")
    blob = client.files.download(file=dest_name)
    results_path = state_dir(dataset) / "results_retry.jsonl"
    results_path.write_bytes(blob if isinstance(blob, bytes) else blob.encode())

    meta = json.loads((state_dir(dataset) / "rows_meta_retry.json").read_text())
    meta_by_key = {m["key"]: m for m in meta}

    per_task_samples: dict[str, dict[str, dict]] = {}
    for task in DATASETS[dataset]:
        samples_t, _ = load_samples(task)
        per_task_samples[task] = {str(s["_rowid"]): s for s in samples_t}

    per_task_updates: dict[str, dict[str, dict]] = {t: {} for t in DATASETS[dataset]}
    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        meta_row = meta_by_key.get(rec.get("key", ""))
        if not meta_row:
            continue
        task = meta_row["task"]
        rid = meta_row["rowid"]
        cfg = TASKS[task]
        sample = per_task_samples.get(task, {}).get(rid, {})
        try:
            true_val = normalize_true(task, sample.get(cfg["label_col"]))
        except Exception:
            true_val = None

        if "error" in rec and rec["error"]:
            resp_text = f"<batch-error-retry: {json.dumps(rec['error'])[:200]}>"
            pred, pred_vec = None, None
        else:
            resp_text = _extract_text(rec.get("response", {}))
            pred, pred_vec = _parse_response(task, resp_text)

        per_task_updates[task][rid] = {
            "rowid": rid, "fold": meta_row.get("fold"),
            "duration_sec": meta_row.get("duration_sec"),
            "true": true_val if cfg["metric"] != "multilabel" else None,
            "pred": pred,
            "true_vec": true_val if cfg["metric"] == "multilabel" else None,
            "pred_vec": pred_vec,
            "response": resp_text,
        }

    for task, updates in per_task_updates.items():
        if not updates:
            continue
        preds_path = PRED_ROOT / task / "predictions.csv"
        df = pd.read_csv(preds_path, dtype={"rowid": str})
        # Coerce list-bearing columns to object dtype; pandas' str dtype rejects
        # list assignment at .at with "iterable length" broadcast error.
        for col in ("true_vec", "pred_vec"):
            if col in df.columns:
                df[col] = df[col].astype(object)
        for rid, new in updates.items():
            idxs = df.index[df["rowid"] == rid].tolist()
            if idxs:
                for idx in idxs:
                    for col, val in new.items():
                        df.at[idx, col] = val
            else:
                df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
        df.to_csv(preds_path, index=False)
        print(f"[{dataset}/{task}] applied {len(updates)} retry updates")
        _compute_and_save_metrics(task, preds_path)

    info["state"] = state
    info["fetched_at"] = time.time()
    p.write_text(json.dumps(info, indent=2))


# ============================================================
# Cleanup
# ============================================================

def cleanup_dataset(client, dataset: str) -> None:
    recs = load_audio_uris(dataset)
    n = 0
    for rid, rec in recs.items():
        name = rec.get("name")
        if not name:
            continue
        try:
            client.files.delete(name=name)
            n += 1
        except Exception as e:
            print(f"  [warn] delete {name}: {type(e).__name__}: {e}", flush=True)
    print(f"[{dataset}] deleted {n} uploaded files")


# ============================================================
# CLI
# ============================================================

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit", help="upload audio + create batch job")
    s.add_argument("--dataset", choices=list(DATASETS.keys()))
    s.add_argument("--all", action="store_true",
                   help="Submit every dataset (sequential, smallest first)")
    s.add_argument("--workers", type=int, default=32,
                   help="Parallel audio uploads")

    sub.add_parser("status", help="show batch state").add_argument(
        "--dataset", choices=list(DATASETS.keys()), default=None)

    f = sub.add_parser("fetch", help="download results + write predictions")
    f.add_argument("--dataset", choices=list(DATASETS.keys()))
    f.add_argument("--all", action="store_true")

    c = sub.add_parser("cleanup", help="delete uploaded audio files")
    c.add_argument("--dataset", choices=list(DATASETS.keys()))
    c.add_argument("--all", action="store_true")

    sd = sub.add_parser("submit-delta",
                        help="retry failed-upload + failed-inference rows in a new batch")
    sd.add_argument("--dataset", choices=list(DATASETS.keys()))
    sd.add_argument("--all", action="store_true")
    sd.add_argument("--workers", type=int, default=4)

    fd = sub.add_parser("fetch-delta", help="fetch retry batch + merge into predictions")
    fd.add_argument("--dataset", choices=list(DATASETS.keys()))
    fd.add_argument("--all", action="store_true")

    args = p.parse_args()
    client = make_client()

    if args.cmd == "submit":
        if args.all:
            # Order by smallest-first (fast feedback on pipeline correctness)
            order = sorted(DATASETS.keys(),
                           key=lambda d: sum(1 for _ in load_samples(DATASETS[d][0])[0]))
            for d in order:
                if d == "edaic":
                    print("[edaic] skipping (already evaluated interactively)")
                    continue
                submit_dataset(client, d, workers=args.workers)
        else:
            submit_dataset(client, args.dataset, workers=args.workers)

    elif args.cmd == "status":
        cmd_status(client, args)

    elif args.cmd == "fetch":
        datasets = list(DATASETS.keys()) if args.all else [args.dataset]
        for d in datasets:
            fetch_dataset(client, d)

    elif args.cmd == "cleanup":
        datasets = list(DATASETS.keys()) if args.all else [args.dataset]
        for d in datasets:
            cleanup_dataset(client, d)

    elif args.cmd == "submit-delta":
        datasets = list(DATASETS.keys()) if args.all else [args.dataset]
        for d in datasets:
            submit_delta(client, d, workers=args.workers)

    elif args.cmd == "fetch-delta":
        datasets = list(DATASETS.keys()) if args.all else [args.dataset]
        for d in datasets:
            fetch_delta(client, d)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
