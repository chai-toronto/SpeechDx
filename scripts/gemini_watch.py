"""Watcher loop for SpeechDx Gemini batches.

Every CYCLE_SEC seconds:
  1. retry submit for any dataset that doesn't yet have batch_job.json
  2. poll batch state for every submitted dataset
  3. on state transition, print a single line; on SUCCEEDED, auto-fetch
  4. on RESOURCE_EXHAUSTED, back off (longer sleep)

Stops when every dataset (excluding 'edaic') is in a terminal state and
has been fetched.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gemini_batch import (  # noqa: E402
    DATASETS, make_client, submit_dataset, status_dataset,
    fetch_dataset, fetch_delta, state_dir, load_batch_job,
)

# Datasets to manage (skip edaic — already done interactively)
DATASETS_TO_WATCH = [d for d in DATASETS if d != "edaic"]

TERMINAL = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
            "JOB_STATE_PARTIALLY_SUCCEEDED"}

CYCLE_SEC = 300         # 5 min normal cadence
BACKOFF_SEC = 3600      # 1 h after a 429 from submit
LAST_STATE_PATH = Path("exps/gemini_batches/_last_states.json")
RETRY_SUBMIT = False    # poll-only mode; flip to True to retry submits


def now() -> str:
    return time.strftime("%H:%M:%S")


def load_last_states() -> dict[str, str]:
    if LAST_STATE_PATH.exists():
        return json.loads(LAST_STATE_PATH.read_text())
    return {}


def save_last_states(states: dict[str, str]) -> None:
    LAST_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAST_STATE_PATH.write_text(json.dumps(states, indent=2))


def already_fetched(dataset: str) -> bool:
    info = load_batch_job(dataset)
    return bool(info and info.get("fetched_at"))


def mark_fetched(dataset: str) -> None:
    info = load_batch_job(dataset) or {}
    info["fetched_at"] = time.time()
    (state_dir(dataset) / "batch_job.json").write_text(json.dumps(info, indent=2))


def cycle(client) -> tuple[int, int, bool]:
    """Return (n_terminal, n_total, hit_resource_exhausted)."""
    last_states = load_last_states()
    new_states: dict[str, str] = dict(last_states)
    n_terminal = 0
    hit_429 = False

    # 1. retry-submit any unsubmitted (only if RETRY_SUBMIT is on; else poll-only)
    if RETRY_SUBMIT:
        for d in DATASETS_TO_WATCH:
            if load_batch_job(d) is None:
                print(f"[{now()}] {d}: retrying submit ...", flush=True)
                try:
                    submit_dataset(client, d, workers=1)
                except Exception as e:
                    msg = str(e)
                    if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                        print(f"[{now()}] {d}: 429 again, backing off", flush=True)
                        hit_429 = True
                    else:
                        print(f"[{now()}] {d}: submit error: {type(e).__name__}: "
                              f"{msg[:200]}", flush=True)
                # only one retry attempt per cycle to avoid quota burn
                break

    # 2. poll state of every submitted batch
    for d in DATASETS_TO_WATCH:
        if not load_batch_job(d):
            continue
        try:
            info = status_dataset(client, d)
        except Exception as e:
            print(f"[{now()}] {d}: status error: {type(e).__name__}: "
                  f"{str(e)[:160]}", flush=True)
            continue
        if not info:
            continue
        state = info["state"]
        prev = last_states.get(d)
        if state != prev:
            print(f"[{now()}] {d}: {prev or '(new)'} -> {state}", flush=True)
            new_states[d] = state
        if state in TERMINAL:
            n_terminal += 1
            # Auto-fetch on first successful terminal observation
            if state in ("JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED") \
                    and not already_fetched(d):
                print(f"[{now()}] {d}: fetching results ...", flush=True)
                try:
                    fetch_dataset(client, d)
                    mark_fetched(d)
                    print(f"[{now()}] {d}: fetched.", flush=True)
                except Exception as e:
                    print(f"[{now()}] {d}: fetch error: {type(e).__name__}: "
                          f"{str(e)[:200]}", flush=True)

    # 3. Track retry batches (batch_job_retry.json). Auto-fetch on SUCCEEDED.
    import json as _json
    for d in DATASETS_TO_WATCH:
        retry_path = state_dir(d) / "batch_job_retry.json"
        if not retry_path.exists():
            continue
        info = _json.loads(retry_path.read_text())
        if info.get("fetched_at"):
            continue
        try:
            job = client.batches.get(name=info["job_name"])
        except Exception as e:
            print(f"[{now()}] {d}/retry: status err {type(e).__name__}", flush=True)
            continue
        state = job.state.name if job.state else "?"
        prev = last_states.get(f"{d}__retry")
        if state != prev:
            print(f"[{now()}] {d}/retry: {prev or '(new)'} -> {state}", flush=True)
            new_states[f"{d}__retry"] = state
        info["state"] = state
        retry_path.write_text(_json.dumps(info, indent=2))
        if state in ("JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"):
            print(f"[{now()}] {d}/retry: fetching ...", flush=True)
            try:
                fetch_delta(client, d)
                info["fetched_at"] = time.time()
                retry_path.write_text(_json.dumps(info, indent=2))
                print(f"[{now()}] {d}/retry: fetched.", flush=True)
            except Exception as e:
                print(f"[{now()}] {d}/retry: fetch err {type(e).__name__}: "
                      f"{str(e)[:160]}", flush=True)

    save_last_states(new_states)
    return n_terminal, len(DATASETS_TO_WATCH), hit_429


def main() -> int:
    client = make_client()
    print(f"[{now()}] watcher started; cycle={CYCLE_SEC}s, "
          f"backoff={BACKOFF_SEC}s, datasets={DATASETS_TO_WATCH}", flush=True)
    while True:
        try:
            n_terminal, n_total, hit_429 = cycle(client)
        except Exception as e:
            print(f"[{now()}] cycle error: {type(e).__name__}: "
                  f"{str(e)[:200]}", flush=True)
            n_terminal, n_total, hit_429 = 0, len(DATASETS_TO_WATCH), True

        # Done when both main + retry batches are fetched for everything
        all_main_fetched = all(
            already_fetched(d) for d in DATASETS_TO_WATCH
            if load_batch_job(d) and load_batch_job(d).get("state")
            in ("JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED")
        )
        retry_outstanding = False
        for d in DATASETS_TO_WATCH:
            rp = state_dir(d) / "batch_job_retry.json"
            if rp.exists():
                ri = json.loads(rp.read_text())
                if not ri.get("fetched_at"):
                    retry_outstanding = True
                    break
        if n_terminal == n_total and all_main_fetched and not retry_outstanding:
            print(f"[{now()}] all done. exiting watcher.", flush=True)
            return 0

        sleep_s = BACKOFF_SEC if hit_429 else CYCLE_SEC
        print(f"[{now()}] {n_terminal}/{n_total} terminal; sleeping {sleep_s}s",
              flush=True)
        time.sleep(sleep_s)


if __name__ == "__main__":
    raise SystemExit(main())
