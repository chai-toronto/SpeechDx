"""Thread-safe logging primitives used by the orchestrators.

These are leaf utilities — no orchestrator state, no I/O policy, just
helpers for atomic terminal writes, progress counters, and tail reads.
"""

from __future__ import annotations

import re
import threading
from datetime import datetime
from pathlib import Path

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
