"""``ahb summary-data-eff`` — per-level + combined progression CSVs.

Salvaged from legacy ``run_all_data_eff.py:cmd_summary`` (lines 490-680).

Layout under ``./exps/data_eff/_summary_<tag>/``:
  - ``<level_dir>/`` — one CSV per metric + ``completion.csv`` for that level
    (rows = task, cols = encoder).
  - ``<metric>.csv`` at the root — combined progression with two-row header
    (encoders × levels), where the ``100`` column is pulled from the matching
    single-task run under ``./exps/single_task/...``.
  - ``completion.csv`` at the root — same wide layout for done/not-done flags.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from typing import Callable

from ahb.orchestrator import (
    _load_task_yaml,
    get_output_folder as get_single_output_folder,
    task_label,
)
from ahb.orchestrator_data_eff import (
    EXP_ROOT_BASE,
    LEVELS,
    discover_tasks,
    encoders_default,
    get_output_folder,
    get_results_file,
    is_complete,
)
from ahb.summary import (
    CSV_LAYOUT,
    REGRESSION_METRICS,
    _collect_matrices,
    _load_metrics,
    _write_completion_csv,
    _write_metric_csvs,
)

FULL_LEVEL = "100"


def _format_value(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, str):
        return v
    return ""


def _write_progression_csv(path: Path, *, encoders: list[str],
                           levels: list[str], tasks: list[str],
                           value_for,
                           label_for: Callable[[str], str] = lambda s: s) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        header_models = [""]
        header_levels = ["task"]
        for enc in encoders:
            for i, lvl in enumerate(levels):
                header_models.append(enc if i == 0 else "")
                header_levels.append(lvl)
        writer.writerow(header_models)
        writer.writerow(header_levels)
        for task_stem in tasks:
            row = [label_for(task_stem)]
            for enc in encoders:
                for lvl in levels:
                    row.append(value_for(task_stem, enc, lvl))
            writer.writerow(row)


def cmd_summary_data_eff(args: argparse.Namespace) -> None:
    tasks = discover_tasks(args.dataset, args.task)
    enc_filter = set(getattr(args, "encoder", None) or [])
    encoders = [e for e in encoders_default().keys()
                if not enc_filter or e in enc_filter]

    if getattr(args, "level", None):
        active_levels = [(d, v) for d, v in LEVELS if d in set(args.level)]
    else:
        active_levels = list(LEVELS)

    base_out = Path(args.out_dir or f"{EXP_ROOT_BASE}/_summary_{args.tag}")
    base_out.mkdir(parents=True, exist_ok=True)

    regression_tasks = {ts for ts in tasks
                        if _load_task_yaml(ts).get("task_type") == "R"}
    classification_tasks = set(tasks) - regression_tasks

    # progression[metric][task][encoder][level] = value
    progression: dict[str, dict[str, dict[str, dict[str, float | str]]]] = {
        k: defaultdict(lambda: defaultdict(dict)) for k in CSV_LAYOUT.values()
    }
    progression_completion: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    combined_levels = [d for d, _ in LEVELS] + [FULL_LEVEL]

    # Pull 100% column from single-task results, regardless of --level filter.
    for task_stem in tasks:
        for model_name in encoders:
            folder = get_single_output_folder(task_stem, model_name, args.tag)
            metrics = _load_metrics(folder, get_results_file(task_stem))
            if metrics is not None:
                for metric_key in CSV_LAYOUT.values():
                    if metric_key in metrics:
                        progression[metric_key][task_stem][model_name][FULL_LEVEL] = metrics[metric_key]
            progression_completion[task_stem][model_name][FULL_LEVEL] = (
                1 if is_complete(folder, task_stem) else 0
            )

    grand_found = grand_missing = 0
    written: list[Path] = []

    for level_dir, _ in active_levels:
        def folder_for(task_stem: str, model_name: str,
                       level_dir=level_dir) -> Path:
            return get_output_folder(task_stem, model_name, level_dir,
                                     args.tag)

        matrices, _, _, found, missing = _collect_matrices(
            tasks=tasks,
            encoders=encoders,
            folder_for=folder_for,
            results_file_for=get_results_file,
            task_type_for=lambda ts: _load_task_yaml(ts).get("task_type"),
        )
        grand_found += found
        grand_missing += missing

        # Per-level CSVs.
        out_dir = base_out / level_dir
        per_level_written = _write_metric_csvs(
            matrices, encoders, regression_tasks, classification_tasks, out_dir,
            label_for=task_label,
        )
        written.extend(per_level_written)

        # Per-level completion: only over the paper task set (not all yaml tasks).
        comp_path = out_dir / "completion.csv"
        totals = {enc: 0 for enc in encoders}
        with comp_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["task", *encoders])
            for task_stem in tasks:
                row = [task_label(task_stem)]
                for enc in encoders:
                    done = is_complete(folder_for(task_stem, enc), task_stem)
                    row.append("1" if done else "0")
                    if done:
                        totals[enc] += 1
                    progression_completion[task_stem][enc][level_dir] = 1 if done else 0
                w.writerow(row)
            w.writerow(["TOTAL", *(str(totals[e]) for e in encoders)])
        written.append(comp_path)

        # Feed level data into progression matrices.
        for metric_key, by_task in matrices.items():
            for ts, by_enc in by_task.items():
                for enc, val in by_enc.items():
                    progression[metric_key][ts][enc][level_dir] = val

        print(f"  level {level_dir}: parsed {found}, missing {missing}")

    # Combined progression CSVs at out_root: rows = task, cols = (encoder × level).
    for csv_name, metric_key in CSV_LAYOUT.items():
        pool = regression_tasks if metric_key in REGRESSION_METRICS else classification_tasks
        task_rows = sorted(pool)
        if not task_rows:
            continue
        data = progression[metric_key]

        def value_for(ts, enc, lvl, _data=data):
            return _format_value(_data[ts].get(enc, {}).get(lvl, ""))

        path = base_out / f"{csv_name}.csv"
        _write_progression_csv(
            path, encoders=encoders, levels=combined_levels,
            tasks=task_rows, value_for=value_for, label_for=task_label,
        )
        written.append(path)

    # Combined completion across data-eff levels + 100.
    comp_path = base_out / "completion.csv"
    totals = {(enc, lvl): 0 for enc in encoders for lvl in combined_levels}

    def comp_value(ts, enc, lvl):
        v = progression_completion[ts].get(enc, {}).get(lvl, 0)
        if v:
            totals[(enc, lvl)] += 1
        return "1" if v else "0"

    _write_progression_csv(
        comp_path, encoders=encoders, levels=combined_levels,
        tasks=list(tasks), value_for=comp_value, label_for=task_label,
    )
    # Append TOTAL row.
    with comp_path.open("a", newline="") as f:
        w = csv.writer(f)
        total_row = ["TOTAL"]
        for enc in encoders:
            for lvl in combined_levels:
                total_row.append(str(totals[(enc, lvl)]))
        w.writerow(total_row)
    written.append(comp_path)

    print(f"\nParsed {grand_found} result files, {grand_missing} missing")
    print(f"Wrote {len(written)} CSV(s) under {base_out}/")
