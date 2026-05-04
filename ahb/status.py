"""``ahb <mode> status`` — completion grids per mode.

Single / cross / cross-cat: per-task × per-encoder ☑ / ☐ grid.
Data-eff: same with an extra ``level`` column.
All: stacks the four mode grids.
"""

from __future__ import annotations

import argparse

from pathlib import Path

from ahb.registry import encoders as registry_encoders


def _is_complete_any(folder: Path) -> bool:
    """Mode-agnostic completeness — either kind of result file marks done."""
    return ((folder / "test_results.txt").exists()
            or (folder / "test_results.yaml").exists())


def _format_grid(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt(row):
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))


def _resolve_encoders(arg_encoder: list[str] | None) -> list[str]:
    encs = list(registry_encoders().keys())
    if arg_encoder:
        wanted = set(arg_encoder)
        encs = [e for e in encs if e in wanted]
    return encs


def _scan_single(args: argparse.Namespace) -> tuple[int, int, int]:
    """Returns (complete, incomplete, total)."""
    from ahb.orchestrator import (
        discover_tasks, get_output_folder, get_task_info,
    )
    tag = getattr(args, "tag", "run1")
    tasks = discover_tasks(getattr(args, "dataset", None),
                           getattr(args, "task", None))
    encoders = _resolve_encoders(getattr(args, "encoder", None))
    complete = incomplete = 0
    rows: list[list[str]] = []
    for ts in tasks:
        ds, t = get_task_info(ts)
        row = [ts]
        for enc in encoders:
            folder = get_output_folder(ds, t, enc, tag)
            if _is_complete_any(folder):
                row.append("☑"); complete += 1
            else:
                row.append("☐"); incomplete += 1
        rows.append(row)
    _format_grid(["task", *encoders], rows)
    return complete, incomplete, complete + incomplete


def _scan_cross(args: argparse.Namespace, *, include_categories: bool,
                exps_root=None) -> tuple[int, int, int]:
    from ahb.orchestrator_cross import (
        discover_tasks as cross_disc,
        get_output_folder as cross_folder,
        get_task_info as cross_info,
    )
    tag = getattr(args, "tag", "run1")
    tasks = cross_disc(getattr(args, "dataset", None),
                       getattr(args, "task", None),
                       include_categories=include_categories)
    encoders = _resolve_encoders(getattr(args, "encoder", None))
    complete = incomplete = 0
    rows: list[list[str]] = []
    for ts in tasks:
        ds, t = cross_info(ts)
        row = [ts]
        for enc in encoders:
            folder = cross_folder(ds, t, enc, tag, exps_root=exps_root)
            if _is_complete_any(folder):
                row.append("☑"); complete += 1
            else:
                row.append("☐"); incomplete += 1
        rows.append(row)
    _format_grid(["task", *encoders], rows)
    return complete, incomplete, complete + incomplete


def _scan_data_eff(args: argparse.Namespace) -> tuple[int, int, int]:
    from ahb.orchestrator import get_task_info
    from ahb.orchestrator_data_eff import (
        discover_tasks as de_disc,
        get_output_folder as de_folder,
    )
    from ahb.registry import data_eff_levels
    tag = getattr(args, "tag", "run1")
    tasks = de_disc(getattr(args, "dataset", None),
                    getattr(args, "task", None))
    encoders = _resolve_encoders(getattr(args, "encoder", None))
    level_filter = set(getattr(args, "level", None) or [])
    levels = [name for name, _ in data_eff_levels()]
    if level_filter:
        levels = [lv for lv in levels if lv in level_filter]
    complete = incomplete = 0
    rows: list[list[str]] = []
    for level in levels:
        for ts in tasks:
            ds, t = get_task_info(ts)
            row = [level, ts]
            for enc in encoders:
                folder = de_folder(ds, t, enc, level, tag)
                if _is_complete_any(folder):
                    row.append("☑"); complete += 1
                else:
                    row.append("☐"); incomplete += 1
            rows.append(row)
    _format_grid(["level", "task", *encoders], rows)
    return complete, incomplete, complete + incomplete


def _print_summary(label: str, complete: int, incomplete: int, total: int) -> None:
    print(f"\nLegend: ☑ complete, ☐ incomplete")
    print(f"Summary ({label}): {complete}/{total} complete, {incomplete} remaining")


def cmd_status(args: argparse.Namespace) -> None:
    """``ahb single status``."""
    c, i, t = _scan_single(args)
    _print_summary("single", c, i, t)


def cmd_status_cross(args: argparse.Namespace) -> None:
    c, i, t = _scan_cross(args, include_categories=False)
    _print_summary("cross", c, i, t)


def cmd_status_crosscat(args: argparse.Namespace) -> None:
    from ahb.run_cross_category import CATEGORY_EXPS_ROOT
    c, i, t = _scan_cross(args, include_categories=True,
                          exps_root=CATEGORY_EXPS_ROOT)
    _print_summary("cross-cat", c, i, t)


def cmd_status_dataeff(args: argparse.Namespace) -> None:
    c, i, t = _scan_data_eff(args)
    _print_summary("data-eff", c, i, t)


def cmd_status_all(args: argparse.Namespace) -> None:
    """``ahb all status`` — print all four mode grids stacked."""
    print("=== single ===")
    cs, is_, ts = _scan_single(args)
    _print_summary("single", cs, is_, ts)
    print("\n=== cross ===")
    cc, ic, tc = _scan_cross(args, include_categories=False)
    _print_summary("cross", cc, ic, tc)
    print("\n=== cross-cat ===")
    from ahb.run_cross_category import CATEGORY_EXPS_ROOT
    ck, ik, tk = _scan_cross(args, include_categories=True,
                             exps_root=CATEGORY_EXPS_ROOT)
    _print_summary("cross-cat", ck, ik, tk)
    print("\n=== data-eff ===")
    cd, id_, td = _scan_data_eff(args)
    _print_summary("data-eff", cd, id_, td)
    print(f"\nGrand total: {cs+cc+ck+cd}/{ts+tc+tk+td} complete.")
