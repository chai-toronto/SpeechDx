"""Helpers for reading and aggregating result files.

Currently exposes the mapping from ``task_type`` to the confidence-interval
field names that ``training/brain.py`` emits, used by the orchestrators'
``has_ci_results`` / completion-checking logic.
"""

from __future__ import annotations


def expected_ci_keys(task_type: str) -> tuple[str, str] | None:
    """CI field names brain.py emits per task_type. ``None`` ⇒ no CI produced."""
    if task_type == "R":
        return ("MAE_CI_low", "MAE_CI_high")
    if task_type in ("B", "C", "L"):
        return ("AUROC_CI_low", "AUROC_CI_high")
    return None
