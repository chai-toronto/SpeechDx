#!/usr/bin/env python3
"""Track queued, running, and finished SLURM jobs by job name.

The tracker polls ``squeue`` and keeps one tracked run id per job name,
recording the current status and time left for each name. Queued jobs are
included alongside running jobs. Jobs that disappear from ``squeue`` after
being tracked are marked with their final ``sacct`` status when available, or
``DONE`` otherwise. By default it refreshes every 10 minutes and writes the
latest snapshot to JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_OUTPUT = Path("exps/slurm_logs/job_tracker.json")
DEFAULT_INTERVAL_SECONDS = 600


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track current SLURM jobs by name.",
    )
    parser.add_argument(
        "--user",
        default=os.environ.get("USER"),
        help="SLURM user to query. Defaults to the current shell user.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=DEFAULT_INTERVAL_SECONDS,
        help="Refresh interval in seconds. Default: 600 (10 minutes).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to the JSON snapshot file. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single refresh instead of looping.",
    )
    return parser


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def query_slurm_jobs(user: str | None) -> list[dict[str, str]]:
    command = [
        "squeue",
        "--noheader",
        "--format=%i|%j|%T|%L|%M|%S",
    ]
    if user:
        command.extend(["--user", user])

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "squeue is not available on this machine. Run the tracker on a SLURM login or compute node."
        ) from exc
    if result.returncode != 0:
        stderr = result.stderr.strip() or "unknown squeue error"
        raise RuntimeError(f"squeue failed: {stderr}")

    jobs = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) != 6:
            continue
        job_id, name, status, time_left, elapsed, start_time = parts
        jobs.append(
            {
                "job_id": job_id.strip(),
                "name": name.strip(),
                "status": status.strip(),
                "time_left": time_left.strip(),
                "elapsed": elapsed.strip(),
                "start_time": start_time.strip(),
            }
        )
    return jobs


def query_final_job_status(job_id: str) -> str | None:
    command = [
        "sacct",
        "--noheader",
        "--parsable2",
        f"--jobs={job_id}",
        "--format=JobIDRaw,State",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None

    if result.returncode != 0:
        return None

    base_job_id = job_id.split("_", maxsplit=1)[0]
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        sacct_job_id, state = parts[0].strip(), parts[1].strip()
        if sacct_job_id == job_id or sacct_job_id == base_job_id:
            return state.split(maxsplit=1)[0]
    return None


def choose_active_job(jobs_for_name: list[dict[str, str]]) -> tuple[dict[str, str], list[str]]:
    def sort_key(job: dict[str, str]) -> tuple[int, int]:
        status_priority = 0 if job["status"] == "RUNNING" else 1
        try:
            job_id = int(job["job_id"].split("_", maxsplit=1)[0])
        except ValueError:
            job_id = -1
        return (status_priority, -job_id)

    ordered = sorted(jobs_for_name, key=sort_key)
    chosen = ordered[0]
    duplicates = [job["job_id"] for job in ordered[1:]]
    return chosen, duplicates


def read_previous_snapshot(output_path: Path) -> dict[str, object]:
    if not output_path.exists():
        return {}
    try:
        loaded = json.loads(output_path.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(loaded, dict):
        return {}
    return loaded


def build_snapshot(user: str | None, previous_snapshot: dict[str, object] | None = None) -> dict[str, object]:
    jobs = query_slurm_jobs(user)
    jobs_by_name: dict[str, list[dict[str, str]]] = {}
    for job in jobs:
        jobs_by_name.setdefault(job["name"], []).append(job)

    tracked_jobs: dict[str, dict[str, object]] = {}
    duplicate_names: list[str] = []
    now = utc_now_iso()
    for name in sorted(jobs_by_name):
        chosen, duplicates = choose_active_job(jobs_by_name[name])
        tracked_jobs[name] = {
            "run_id": chosen["job_id"],
            "status": chosen["status"],
            "time_left": chosen["time_left"],
            "elapsed": chosen["elapsed"],
            "start_time": chosen["start_time"],
            "last_seen_utc": now,
        }
        if duplicates:
            duplicate_names.append(name)
            tracked_jobs[name]["other_active_run_ids"] = duplicates

    if previous_snapshot:
        previous_jobs = previous_snapshot.get("jobs_by_name", {})
        if isinstance(previous_jobs, dict):
            for name, record in previous_jobs.items():
                if name in tracked_jobs or not isinstance(record, dict):
                    continue
                done_record = dict(record)
                run_id = str(done_record.get("run_id", ""))
                final_status = query_final_job_status(run_id) if run_id else None
                done_record["status"] = final_status or "DONE"
                done_record["time_left"] = "0:00"
                done_record["done_at_utc"] = done_record.get("done_at_utc", now)
                done_record.pop("other_active_run_ids", None)
                tracked_jobs[str(name)] = done_record

    return {
        "updated_at_utc": now,
        "user": user,
        "job_count": len(jobs),
        "tracked_name_count": len(tracked_jobs),
        "done_name_count": sum(
            1
            for record in tracked_jobs.values()
            if isinstance(record, dict)
            and record.get("status") not in {"PENDING", "RUNNING", "SUSPENDED", "CONFIGURING"}
        ),
        "duplicate_name_count": len(duplicate_names),
        "duplicate_names": duplicate_names,
        "jobs_by_name": dict(sorted(tracked_jobs.items())),
    }


def write_snapshot(snapshot: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")


def format_table(snapshot: dict[str, object]) -> str:
    jobs_by_name = snapshot["jobs_by_name"]
    assert isinstance(jobs_by_name, dict)

    if not jobs_by_name:
        return "No tracked SLURM jobs found."

    headers = ("job_name", "run_id", "status", "time_left")
    rows = []
    for name, record in jobs_by_name.items():
        assert isinstance(record, dict)
        rows.append(
            (
                name,
                str(record["run_id"]),
                str(record["status"]),
                str(record["time_left"]),
            )
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    lines = [
        " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)),
        "-+-".join("-" * width for width in widths),
    ]
    for row in rows:
        lines.append(
            " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))
        )

    duplicate_names = snapshot.get("duplicate_names", [])
    if duplicate_names:
        lines.append("")
        lines.append(
            "Warning: multiple active jobs share these names; the newest active run id is tracked."
        )
        lines.append(", ".join(str(name) for name in duplicate_names))

    return "\n".join(lines)


def refresh(user: str | None, output_path: Path) -> None:
    previous_snapshot = read_previous_snapshot(output_path)
    snapshot = build_snapshot(user, previous_snapshot)
    write_snapshot(snapshot, output_path)
    print(f"[{snapshot['updated_at_utc']}] wrote {output_path}")
    print(format_table(snapshot))
    sys.stdout.flush()


def main() -> int:
    args = build_parser().parse_args()
    if args.interval_seconds <= 0:
        raise SystemExit("--interval-seconds must be a positive integer.")

    try:
        while True:
            refresh(args.user, args.output)
            if args.once:
                return 0
            time.sleep(args.interval_seconds)
    except KeyboardInterrupt:
        print("\nTracker stopped.")
        return 0
    except Exception as exc:
        print(f"Tracker failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
