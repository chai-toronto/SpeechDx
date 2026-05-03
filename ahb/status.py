"""``ahb status`` — completion table over ./exps/single_task/.

Salvaged from ``run_all.py:cmd_status`` (lines 505-561). Output format
unchanged so existing tooling that greps the table keeps working.
"""

from __future__ import annotations

import argparse

from ahb.orchestrator import (
    _load_task_yaml,
    discover_tasks,
    get_output_folder,
    get_results_file,
    get_task_info,
    has_ci_results,
    is_complete,
)
from ahb.registry import encoders as registry_encoders
from ahb.results import expected_ci_keys


def cmd_status(args: argparse.Namespace) -> None:
    tasks = discover_tasks()
    test_only = getattr(args, "test_only", False)
    complete = incomplete = absent = 0

    encoders = list(registry_encoders().keys())
    rows = []
    for task_stem in tasks:
        dataset, task = get_task_info(task_stem)
        row = [task_stem]
        ci_applicable = (
            expected_ci_keys(_load_task_yaml(task_stem).get("task_type", "")) is not None
        ) if test_only else True
        for model_name in encoders:
            folder = get_output_folder(dataset, task, model_name)
            if test_only:
                if not ci_applicable or not (folder / get_results_file(task_stem)).exists():
                    row.append(" ")
                    absent += 1
                elif has_ci_results(folder, task_stem):
                    row.append("☑")
                    complete += 1
                else:
                    row.append("☐")
                    incomplete += 1
            elif is_complete(folder, task_stem):
                row.append("☑")
                complete += 1
            else:
                row.append("☐")
                incomplete += 1
        rows.append(row)

    headers = ["task", *encoders]
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def format_row(row):
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))

    if test_only:
        total = complete + incomplete + absent
        print(f"\nLegend: ☑ CI present, ☐ tested without CI, blank = no test result")
        print(f"Summary: {complete}/{total} with CI, {incomplete} pending CI, {absent} no test result")
    else:
        total = complete + incomplete
        print(f"\nLegend: ☑ complete, ☐ incomplete")
        print(f"Summary: {complete}/{total} complete, {incomplete} remaining")
