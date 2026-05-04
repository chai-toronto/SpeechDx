"""``ahb summary-cross`` and ``ahb summary-cross-category``.

Mirrors ``ahb summary`` but operates on cross-task results under
``./exps/cross/`` (non-category) or ``./exps/cross_cat/`` (category).
Output CSVs land under ``<root>/_summary_<tag>/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ahb.orchestrator_cross import (
    EXPS_ROOT,
    _load_task_yaml,
    discover_all_encoders,
    discover_all_tasks,
    discover_tasks,
    get_output_folder,
    get_results_file,
    get_task_info,
    is_complete,
)
from ahb.registry import encoders as registry_encoders
from ahb.run_cross_category import CATEGORY_EXPS_ROOT
from ahb.summary import (
    _collect_matrices,
    _write_completion_csv,
    _write_metric_csvs,
)


def _cmd(args: argparse.Namespace, *, include_categories: bool,
         exps_root: Path) -> None:
    tasks = discover_tasks(args.dataset, args.task,
                           include_categories=include_categories)
    encoders = list(registry_encoders().keys())

    def folder_for(task_stem: str, model_name: str) -> Path:
        dataset, task = get_task_info(task_stem)
        return get_output_folder(dataset, task, model_name, args.tag,
                                 exps_root=exps_root)

    matrices, regression_tasks, classification_tasks, found, missing = _collect_matrices(
        tasks=tasks,
        encoders=encoders,
        folder_for=folder_for,
        results_file_for=get_results_file,
        task_type_for=lambda ts: _load_task_yaml(ts).get("task_type"),
    )

    out_dir = Path(args.out_dir or exps_root / f"_summary_{args.tag}")
    written = _write_metric_csvs(
        matrices, encoders, regression_tasks, classification_tasks, out_dir,
    )

    # Completion grid: every cross-task yaml × every encoder yaml under the
    # right scope (category vs non-category).
    if include_categories:
        all_tasks_list = sorted(
            s for s in discover_all_tasks() if s.startswith("category_")
        )
    else:
        all_tasks_list = sorted(
            s for s in discover_all_tasks() if not s.startswith("category_")
        )
    all_encoders = discover_all_encoders()
    comp_path = _write_completion_csv(
        all_tasks=all_tasks_list,
        all_encoders=all_encoders,
        folder_for=folder_for,
        is_complete_for=is_complete,
        out_dir=out_dir,
    )
    written.append(comp_path)

    from ahb.summary import _print_metric_tables
    _print_metric_tables(matrices, encoders, regression_tasks, classification_tasks)
    print(f"\nParsed {found} result files, {missing} missing")
    print(f"Wrote {len(written)} CSV(s) to {out_dir}/")


def cmd_summary_cross(args: argparse.Namespace) -> None:
    _cmd(args, include_categories=False, exps_root=EXPS_ROOT)


def cmd_summary_cross_category(args: argparse.Namespace) -> None:
    _cmd(args, include_categories=True, exps_root=CATEGORY_EXPS_ROOT)
