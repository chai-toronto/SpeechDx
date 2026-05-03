"""``ahb run-cross-category`` — same machinery as ``ahb run-cross`` but
restricted to ``category_*`` tasks under a separate output root.

Salvaged from ``run_all_cross_category.py`` (28 lines). The category yamls
have a different schema (plural ``train_datasets`` / ``test_datasets`` and
``setting_N`` entries) but the orchestrator dispatch is identical.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ahb.run_cross import cmd_run_cross

CATEGORY_EXPS_ROOT = Path("./cross_cat_exps")
CATEGORY_LOGS_ROOT = Path("logs/run_all_cross_category")


def cmd_run_cross_category(args: argparse.Namespace) -> None:
    cmd_run_cross(
        args,
        include_categories=True,
        exps_root=CATEGORY_EXPS_ROOT,
        logs_root=CATEGORY_LOGS_ROOT,
    )
