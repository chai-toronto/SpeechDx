#!/usr/bin/env python3
"""Track active SLURM jobs by job name.

The tracker polls ``squeue`` and keeps one active run id per job name,
recording the current status and time left for each name. By default it
refreshes every 10 minutes and writes the latest snapshot to JSON.
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


def build_snapshot(user: str | None) -> dict[str, object]:
    jobs = query_slurm_jobs(user)
    jobs_by_name: dict[str, list[dict[str, str]]] = {}
    for job in jobs:
        jobs_by_name.setdefault(job["name"], []).append(job)

    tracked_jobs: dict[str, dict[str, object]] = {}
    duplicate_names: list[str] = []
    for name in sorted(jobs_by_name):
        chosen, duplicates = choose_active_job(jobs_by_name[name])
        tracked_jobs[name] = {
            "run_id": chosen["job_id"],
            "status": chosen["status"],
            "time_left": chosen["time_left"],
            "elapsed": chosen["elapsed"],
            "start_time": chosen["start_time"],
        }
        if duplicates:
            duplicate_names.append(name)
            tracked_jobs[name]["other_active_run_ids"] = duplicates

    return {
        "updated_at_utc": utc_now_iso(),
        "user": user,
        "job_count": len(jobs),
        "tracked_name_count": len(tracked_jobs),
        "duplicate_name_count": len(duplicate_names),
        "duplicate_names": duplicate_names,
        "jobs_by_name": tracked_jobs,
    }


def write_snapshot(snapshot: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")


def format_table(snapshot: dict[str, object]) -> str:
    jobs_by_name = snapshot["jobs_by_name"]
    assert isinstance(jobs_by_name, dict)

    if not jobs_by_name:
        return "No active SLURM jobs found."

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
    snapshot = build_snapshot(user)
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
