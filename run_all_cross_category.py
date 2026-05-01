#!/usr/bin/env python3
"""Run category-cross training with the shared cross runner."""

from pathlib import Path

import run_all_cross as cross_runner


def configure_category_cross() -> None:
    """Pin run_all_cross to category-only tasks and config."""
    cross_runner.BASE_CONFIG = Path("training/config/main_cross_category.yaml")
    cross_runner.TASK = sorted(
        p.stem for p in cross_runner.TASKS_DIR.glob("category_*.yaml")
    )
    # Defensive reset in case this module is imported/reused in-process.
    cross_runner._main_yaml_cache = None
    cross_runner._task_yaml_cache.clear()
    cross_runner._task_ids_cache.clear()


def main() -> None:
    configure_category_cross()
    cross_runner.main()


if __name__ == "__main__":
    main()
