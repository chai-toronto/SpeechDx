"""Zero-shot Gemini-3 evaluation for SpeechDx tasks.

Optimization: tasks are organized into *groups* that share audio. For each
sample, the audio is uploaded once via the Files API and reused across every
task in the group — saving upload bandwidth and (when --cache is on) audio
tokens via explicit context caching.

E-DAIC (T1 depression yes/no + T2 PHQ-8 score) is the first group.

Usage:
    python scripts/run_gemini_eval.py --group edaic
    python scripts/run_gemini_eval.py --group edaic --limit 2  # smoke test
    python scripts/run_gemini_eval.py --group edaic --cache    # explicit ctx cache
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import soundfile as sf

# Reuse registry, loaders, parsers, metrics from the Qwen reference script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_qwen3omni_all import (  # noqa: E402
    TASKS, PROMPTS, load_samples, normalize_true,
    parse_yes_no, parse_label, parse_integer, parse_multilabel,
    compute_metrics, cv_aggregate,
    _preds_path, _metrics_path, _done_rowids, _append_rows,
)

# Correct stale registry paths
TASKS["T7"]["source"] = {"manifest": "exps/single_task/dbank_adC/manifest", "split": "test"}
TASKS["T8"]["source"] = {"manifest": "exps/single_task/dbank_mmseR/manifest", "split": "test"}

MODEL = "gemini-3.1-pro-preview"

# Task groups: lists of task IDs that share audio. Uploading + (optionally)
# caching once per sample saves bandwidth and tokens.
TASK_GROUPS: dict[str, list[str]] = {
    "edaic": ["T1", "T2"],
    # future: "mvdr": ["T13", "T14", "T15", "T16"], "iemocap": ["T5", "T6"], ...
}

# Path rewrites for manifests written against renamed dirs
_PATH_REWRITES = {
    "/data/dbank/": "/data/dementiabank/",
    "/data/c9s/": "/data/c19sounds/",
}
_TORGO_PREFIX_RE = re.compile(r"^s\d+_")


def _rewrite_path(p: str) -> str:
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


def _parse_response(cfg, resp):
    kind = cfg["parser"]
    if kind == "yes_no":
        return parse_yes_no(resp), None
    if kind == "label":
        return parse_label(resp, cfg["classes"]), None
    if kind == "integer":
        lo, hi = cfg.get("range", (-1e9, 1e9))
        return parse_integer(resp, lo, hi), None
    if kind == "multilabel":
        return None, parse_multilabel(resp, cfg["classes"])
    return None, None


# ============================================================
# Gemini backend
# ============================================================

class GeminiBackend:
    def __init__(self, model: str, flex: bool, max_retries: int = 10,
                 retry_cap_s: float = 180.0):
        from google import genai
        from google.genai import types
        self._types = types
        self.client = genai.Client(api_key=get_api_key())
        self.model = model
        self.service_tier = types.ServiceTier.FLEX if flex else types.ServiceTier.STANDARD
        self.max_retries = max_retries
        self.retry_cap_s = retry_cap_s

    def _retry(self, label: str, fn):
        last = None
        for attempt in range(self.max_retries):
            try:
                return fn()
            except Exception as e:
                last = e
                # Full-jitter binary exponential backoff, capped at retry_cap_s.
                # Jitter breaks thundering-herd from concurrent workers on flex.
                base = min(self.retry_cap_s, 2 ** attempt)
                wait = random.uniform(0.5 * base, base)
                print(f"  [{label} retry {attempt}] {type(e).__name__}: "
                      f"{str(e)[:160]}; sleeping {wait:.1f}s", flush=True)
                time.sleep(wait)
        raise RuntimeError(f"{label}: max retries exceeded ({last!r})")

    def upload_audio(self, path: str):
        f = self._retry("upload", lambda: self.client.files.upload(file=path))
        # Wait for ACTIVE state (long audio can take 30+ seconds to process)
        deadline = time.time() + 600
        while f.state and f.state.name == "PROCESSING":
            if time.time() > deadline:
                raise RuntimeError(f"file processing timed out for {path}")
            time.sleep(2)
            f = self.client.files.get(name=f.name)
        if f.state and f.state.name == "FAILED":
            raise RuntimeError(f"file processing FAILED for {path}")
        return f

    def create_cache(self, file_obj, ttl_seconds: int = 600):
        t = self._types
        config = t.CreateCachedContentConfig(
            contents=[t.Content(role="user", parts=[
                t.Part.from_uri(file_uri=file_obj.uri, mime_type=file_obj.mime_type),
            ])],
            ttl=f"{ttl_seconds}s",
        )
        return self._retry("cache-create",
                           lambda: self.client.caches.create(model=self.model, config=config))

    def generate_with_file(self, file_obj, prompt: str) -> str:
        t = self._types
        contents = [
            t.Part.from_uri(file_uri=file_obj.uri, mime_type=file_obj.mime_type),
            prompt,
        ]
        config = t.GenerateContentConfig(service_tier=self.service_tier)
        return self._retry("gen", lambda: (
            self.client.models.generate_content(
                model=self.model, contents=contents, config=config).text or ""
        ))

    def generate_with_cache(self, cache_name: str, prompt: str) -> str:
        t = self._types
        config = t.GenerateContentConfig(
            cached_content=cache_name, service_tier=self.service_tier)
        return self._retry("gen", lambda: (
            self.client.models.generate_content(
                model=self.model, contents=prompt, config=config).text or ""
        ))

    def delete_file(self, name: str) -> None:
        try:
            self.client.files.delete(name=name)
        except Exception:
            pass

    def delete_cache(self, name: str) -> None:
        try:
            self.client.caches.delete(name=name)
        except Exception:
            pass


# ============================================================
# Per-sample worker
# ============================================================

def _row_for(task: str, sample, duration, resp, pred, pred_vec):
    cfg = TASKS[task]
    true_val = normalize_true(task, sample.get(cfg["label_col"]))
    return {
        "rowid": str(sample["_rowid"]),
        "fold": sample.get("_fold"),
        "duration_sec": duration,
        "true": true_val if cfg["metric"] != "multilabel" else None,
        "pred": pred,
        "true_vec": true_val if cfg["metric"] == "multilabel" else None,
        "pred_vec": pred_vec,
        "response": resp,
    }


def process_sample(backend: GeminiBackend, sample: dict, task_ids: list[str],
                   pending_tasks: list[str], out_dir: Path, use_cache: bool) -> None:
    rowid = str(sample["_rowid"])
    raw = sample["path"]
    path = _rewrite_path(raw)

    if not Path(path).exists():
        for task in pending_tasks:
            _append_rows(_preds_path(out_dir, task),
                         [_row_for(task, sample, None, "<load-failed>", None, None)])
        return

    duration = audio_duration_sec(path)

    try:
        file_obj = backend.upload_audio(path)
    except Exception as e:
        msg = f"<upload-failed: {type(e).__name__}: {str(e)[:160]}>"
        for task in pending_tasks:
            _append_rows(_preds_path(out_dir, task),
                         [_row_for(task, sample, duration, msg, None, None)])
        return

    cache_obj = None
    try:
        if use_cache and len(pending_tasks) >= 2:
            try:
                cache_obj = backend.create_cache(file_obj, ttl_seconds=900)
            except Exception as e:
                print(f"  [warn] cache create failed for {rowid} "
                      f"({type(e).__name__}); falling back to file-only", flush=True)
                cache_obj = None

        for task in pending_tasks:
            cfg = TASKS[task]
            try:
                if cache_obj is not None:
                    resp = backend.generate_with_cache(cache_obj.name, PROMPTS[task])
                else:
                    resp = backend.generate_with_file(file_obj, PROMPTS[task])
            except Exception as e:
                resp = f"<generate-failed: {type(e).__name__}: {str(e)[:160]}>"
            pred, pred_vec = _parse_response(cfg, resp)
            _append_rows(_preds_path(out_dir, task),
                         [_row_for(task, sample, duration, resp, pred, pred_vec)])
            tail = resp.splitlines()[-1] if resp else ""
            print(f"  [{rowid}/{task}] pred={pred if pred is not None else pred_vec}  "
                  f"({tail[:80]})", flush=True)
    finally:
        if cache_obj is not None:
            backend.delete_cache(cache_obj.name)
        backend.delete_file(file_obj.name)


# ============================================================
# Group runner
# ============================================================

def run_group(group_name: str, task_ids: list[str], args) -> None:
    backend = GeminiBackend(model=args.model, flex=args.flex)
    out_dir = Path(args.out_dir).resolve()
    for t in task_ids:
        (out_dir / t).mkdir(parents=True, exist_ok=True)

    samples, _ = load_samples(task_ids[0])
    seen: dict[str, dict] = {}
    for s in samples:
        seen[str(s["_rowid"])] = s
    samples = list(seen.values())
    if args.limit:
        samples = samples[: args.limit]

    done = {t: _done_rowids(_preds_path(out_dir, t)) for t in task_ids}
    work = []
    for s in samples:
        rid = str(s["_rowid"])
        pending = [t for t in task_ids if rid not in done[t]]
        if pending:
            work.append((s, pending))
    print(f"[{group_name}] tasks={task_ids}  pending={len(work)}/{len(samples)}  "
          f"model={args.model}  tier={'flex' if args.flex else 'standard'}  "
          f"cache={'on' if args.cache else 'off'}", flush=True)

    if not work:
        print(f"[{group_name}] nothing to do.", flush=True)
    else:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = {
                ex.submit(process_sample, backend, s, task_ids, pending, out_dir, args.cache): s
                for s, pending in work
            }
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    fut.result()
                except Exception as e:
                    s = futures[fut]
                    print(f"  [error] sample {s.get('_rowid')}: "
                          f"{type(e).__name__}: {e}", flush=True)
                if i % 5 == 0 or i == len(futures):
                    elapsed = time.time() - t0
                    rate = elapsed / i
                    eta = rate * (len(futures) - i)
                    print(f"  progress: {i}/{len(futures)}  elapsed={elapsed:.0f}s  "
                          f"eta={eta:.0f}s", flush=True)

    # Metrics per task
    for task in task_ids:
        _compute_and_save_metrics(task, _preds_path(out_dir, task),
                                  _metrics_path(out_dir, task))


def _compute_and_save_metrics(task: str, preds_path: Path, metrics_path: Path) -> None:
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
        metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
        print(f"\n[{task}] metrics -> {metrics_path}")
        print(json.dumps(metrics["overall"], indent=2, default=str))
    except Exception as e:
        print(f"[{task}] metrics calc failed: {type(e).__name__}: {e}", flush=True)


# ============================================================
# Entry
# ============================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", required=True, choices=list(TASK_GROUPS.keys()),
                   help="Task group to evaluate")
    p.add_argument("--out-dir", default="exps/gemini_3_1_pro")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--concurrency", type=int, default=8,
                   help="Samples processed in parallel")
    p.add_argument("--limit", type=int, default=None,
                   help="Per-group sample cap (debug / smoke test)")
    p.add_argument("--no-flex", dest="flex", action="store_false", default=True,
                   help="Use standard tier instead of flex (faster, ~2x cost)")
    p.add_argument("--cache", action="store_true", default=False,
                   help="Use explicit context caching (worth it for groups with 3+ tasks)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if get_api_key() is None:
        print("ERROR: no Gemini API key in env or macOS keychain "
              "(service 'gemini-api-key').", file=sys.stderr)
        return 1
    run_group(args.group, TASK_GROUPS[args.group], args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
