"""Thread-safe logging primitives shared by the orchestrator subcommands.

Leaf utilities only — no orchestrator business logic. Atomic terminal writes,
progress counters, per-job dashboard rows, file-tail reads, and small helpers
for capturing one job's stdout/stderr into a log file.
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Thread-local stdout so concurrent _run_logged_job workers each write to
# their own log file without stomping on each other's sys.stdout redirect.
# ---------------------------------------------------------------------------
_tls = threading.local()


class _ThreadLocalStdout(io.TextIOBase):
    """Proxy that routes each thread's writes to its own per-job log file."""

    def write(self, s: str) -> int:
        f = getattr(_tls, "out", sys.__stdout__)
        try:
            return f.write(s)
        except ValueError:
            return sys.__stdout__.write(s)

    def flush(self) -> None:
        f = getattr(_tls, "out", sys.__stdout__)
        try:
            f.flush()
        except ValueError:
            pass

    @property
    def encoding(self):
        return getattr(getattr(_tls, "out", sys.__stdout__), "encoding", "utf-8")


_tl_stdout = _ThreadLocalStdout()


@contextlib.contextmanager
def _redirect_tls(f):
    """Set this thread's log-file target; install proxy on first call."""
    prev = getattr(_tls, "out", None)
    _tls.out = f
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = _tl_stdout
    sys.stderr = _tl_stdout
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        if prev is None:
            del _tls.out
        else:
            _tls.out = prev

# Fixed-width role label so concurrent rows align.
ROLE_W = 7

# Single process-wide lock so concurrent workers' multi-line writes never
# interleave on the terminal.
_terminal_lock = threading.Lock()


def _emit(*lines: str) -> None:
    """Atomic multi-line print so concurrent workers don't interleave."""
    with _terminal_lock:
        for ln in lines:
            print(ln, flush=True)


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "x"


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _new_log_dir(logs_root: Path) -> Path:
    """Create and return a timestamped log directory under ``logs_root``."""
    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_log_dir = logs_root / run_stamp
    run_log_dir.mkdir(parents=True, exist_ok=True)
    return run_log_dir


def _tail(path: Path, n: int = 20) -> list[str]:
    try:
        text = path.read_text(errors="replace")
    except FileNotFoundError:
        return ["(log not written)"]
    except Exception as e:
        return [f"(log unreadable: {e})"]
    lines = text.splitlines()
    return lines[-n:] if lines else ["(log empty)"]


class _Progress:
    """Thread-safe counters surfaced as snapshot strings in driver output."""

    def __init__(self, total: int):
        self.total = total
        self.active = 0
        self.done = 0
        self.failed = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            self.active += 1

    def finish(self, ok: bool) -> None:
        with self._lock:
            self.active = max(0, self.active - 1)
            if ok:
                self.done += 1
            else:
                self.failed += 1

    def snap(self) -> str:
        with self._lock:
            queued = max(0, self.total - self.active - self.done - self.failed)
            return (f"act={self.active} queued={queued} "
                    f"done={self.done} fail={self.failed}/{self.total}")


def _run_logged_job(label: str, log_path: Path, fn: Callable[[], None], *,
                    detail_lines: Iterable[str] = ()) -> tuple[bool, float]:
    """Run ``fn`` while redirecting stdout/stderr to ``log_path``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    success = True
    with log_path.open("w", buffering=1) as f:
        f.write(f"{'='*60}\n  Running: {label}\n")
        for line in detail_lines:
            f.write(f"  {line}\n")
        f.write(f"  Started: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"{'='*60}\n")
        f.flush()
        try:
            with _redirect_tls(f):
                fn()
        except Exception:
            success = False
            with _redirect_tls(f):
                traceback.print_exc()
    elapsed = time.time() - start
    with log_path.open("a", buffering=1) as f:
        marker = "✓ OK" if success else "✗ FAILED"
        f.write(f"\n{'='*60}\n  {marker}: {label} in {elapsed/60:.1f} min\n")
        f.write(f"  Finished: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"{'='*60}\n")
    return success, elapsed


class _JobDashboard:
    """Shared terminal dashboard for batched warm/train/run-style drivers."""

    def __init__(self, *, total_jobs: int, logs_root: Path, header_lines: Iterable[str]):
        self.total_jobs = total_jobs
        self.run_log_dir = _new_log_dir(logs_root)
        self.progress = _Progress(total_jobs)
        self.completed = 0
        self.failed: list[str] = []
        self.failed_log_paths: dict[str, Path] = {}
        self._lock = threading.Lock()
        _emit("", *header_lines, f"           logs    : {self.run_log_dir}/  (one file per job)", "")

    def start_job(self, idx: int, role: str, label: str, log_path: Path) -> None:
        self.progress.start()
        _emit(
            f"[{_now()}] START [{idx+1:>3}/{self.total_jobs}] {role:<{ROLE_W}} {label}",
            f"           log : {log_path}",
            f"           prog: {self.progress.snap()}",
        )

    def finish_job(self, idx: int, label: str, *, ok: bool,
                   elapsed: float, log_path: Path) -> None:
        self.progress.finish(ok)
        with self._lock:
            if ok:
                self.completed += 1
            else:
                self.failed.append(label)
                self.failed_log_paths[label] = log_path

        status = "✓ OK  " if ok else "✗ FAIL"
        head = (f"[{_now()}] END   [{idx+1:>3}/{self.total_jobs}] {status:<{ROLE_W}} {label}    "
                f"elapsed={elapsed/60:.1f} min   prog: {self.progress.snap()}")
        if not ok:
            extra = [f"           tail of {log_path}:"]
            extra += [f"             | {ln}" for ln in _tail(log_path, 20)]
            _emit(head, *extra)
        else:
            _emit(head)

    def print_summary(self, *, done_label: str, skipped: int = 0) -> None:
        print(f"\n{'='*60}")
        print(f"  Done — {self.completed} {done_label}, {skipped} skipped, {len(self.failed)} failed")
        print(f"  Logs : {self.run_log_dir}/")
        if self.failed:
            print("  Failed runs:")
            for name in self.failed:
                lp = self.failed_log_paths.get(name)
                print(f"    - {name}" + (f"   →  {lp}" if lp else ""))
        print(f"{'='*60}")
