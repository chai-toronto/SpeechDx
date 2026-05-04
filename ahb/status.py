"""``ahb status`` — completion table over ./exps/single_task/."""

from __future__ import annotations

import argparse

from ahb.orchestrator import (
    discover_tasks,
    get_output_folder,
    get_task_info,
    is_complete,
)
from ahb.registry import encoders as registry_encoders


def cmd_status(args: argparse.Namespace) -> None:
    tasks = discover_tasks()
    tag = getattr(args, "tag", "run1")
    complete = incomplete = 0

    encoders = list(registry_encoders().keys())
    rows = []
    for task_stem in tasks:
        dataset, task = get_task_info(task_stem)
        row = [task_stem]
        for model_name in encoders:
            folder = get_output_folder(dataset, task, model_name, tag)
            if is_complete(folder, task_stem):
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

    total = complete + incomplete
    print(f"\nLegend: ☑ complete, ☐ incomplete")
    print(f"Summary: {complete}/{total} complete, {incomplete} remaining")
