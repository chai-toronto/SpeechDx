"""``ahb summary`` — collect results into per-metric CSVs.

Salvaged from ``run_all.py:cmd_summary`` (lines 417-502). Output CSVs
(AUC.csv, F1.csv, MAE.csv, ..., completion.csv) are byte-identical to
the legacy path's output for the same ``--out-dir`` / ``--tag``.

The lower helpers (``_collect_matrices``, ``_write_metric_csvs``,
``_write_completion_csv``) are reused by ``ahb.summary_cross`` and
``ahb.summary_data_eff``.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Callable

from ahb.orchestrator import (
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
from ahb.results import parse_results_txt, parse_results_yaml

CLASSIFICATION_METRICS = ["AUROC", "AUC_CI", "F1", "accuracy"]
REGRESSION_METRICS = ["MAE", "MAE_CI", "MSE", "PearsonR", "R2"]
CSV_LAYOUT = {
    "AUC":      "AUROC",
    "AUC_CI":   "AUC_CI",
    "F1":       "F1",
    "Acc":      "accuracy",
    "MAE":      "MAE",
    "MAE_CI":   "MAE_CI",
    "MSE":      "MSE",
    "PearsonR": "PearsonR",
    "R2":       "R2",
}


def _load_metrics(output_folder: Path, results_file: str) -> dict | None:
    path = output_folder / results_file
    if not path.exists():
        return None
    if path.suffix == ".yaml":
        return parse_results_yaml(path)
    return parse_results_txt(path)


def _collect_matrices(
    tasks: list[str],
    encoders: list[str],
    folder_for: Callable[[str, str], Path],
    results_file_for: Callable[[str], str],
    task_type_for: Callable[[str], str],
) -> tuple[dict, set[str], set[str], int, int]:
    """Collect per-metric (task → encoder → value) matrices."""
    matrices: dict = {k: defaultdict(dict) for k in CSV_LAYOUT.values()}
    regression_tasks: set[str] = set()
    classification_tasks: set[str] = set()
    for task_stem in tasks:
        if task_type_for(task_stem) == "R":
            regression_tasks.add(task_stem)
        else:
            classification_tasks.add(task_stem)

    found = missing = 0
    for task_stem in tasks:
        for model_name in encoders:
            metrics = _load_metrics(
                folder_for(task_stem, model_name),
                results_file_for(task_stem),
            )
            if metrics is None:
                missing += 1
                continue
            found += 1
            for metric_key in CSV_LAYOUT.values():
                if metric_key in metrics:
                    matrices[metric_key][task_stem][model_name] = metrics[metric_key]
    return matrices, regression_tasks, classification_tasks, found, missing


def _write_metric_csvs(
    matrices: dict,
    encoders: list[str],
    regression_tasks: set[str],
    classification_tasks: set[str],
    out_dir: Path,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for csv_name, metric_key in CSV_LAYOUT.items():
        data = matrices[metric_key]
        pool = regression_tasks if metric_key in REGRESSION_METRICS else classification_tasks
        task_rows = sorted(pool)
        if not task_rows:
            continue

        csv_path = out_dir / f"{csv_name}.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task", *encoders])
            for task_stem in task_rows:
                row = [task_stem]
                for enc in encoders:
                    val = data[task_stem].get(enc, "")
                    if isinstance(val, float):
                        row.append(f"{val:.4f}")
                    elif isinstance(val, str):
                        row.append(val)
                    else:
                        row.append("")
                writer.writerow(row)
        written.append(csv_path)
    return written


def _write_completion_csv(
    all_tasks: list[str],
    all_encoders: dict[str, str],
    folder_for: Callable[[str, str], Path],
    is_complete_for: Callable[[Path, str], bool],
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    comp_path = out_dir / "completion.csv"
    totals = {stem: 0 for stem in all_encoders}
    with comp_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", *all_encoders.keys()])
        for task_stem in all_tasks:
            row = [task_stem]
            for stem, folder_name in all_encoders.items():
                done = is_complete_for(folder_for(task_stem, folder_name), task_stem)
                row.append("1" if done else "0")
                if done:
                    totals[stem] += 1
            writer.writerow(row)
        writer.writerow(["TOTAL", *(str(totals[s]) for s in all_encoders)])
    return comp_path


def cmd_summary(args: argparse.Namespace) -> None:
    tasks = discover_tasks()
    encoders = list(registry_encoders().keys())

    def folder_for(task_stem: str, model_name: str) -> Path:
        dataset, task = get_task_info(task_stem)
        return get_output_folder(dataset, task, model_name, args.tag)

    matrices, regression_tasks, classification_tasks, found, missing = _collect_matrices(
        tasks=tasks,
        encoders=encoders,
        folder_for=folder_for,
        results_file_for=get_results_file,
        task_type_for=lambda ts: _load_task_yaml(ts).get("task_type"),
    )

    out_dir = Path(args.out_dir or f"exps/single_task/_summary_{args.tag}")
    written = _write_metric_csvs(
        matrices, encoders, regression_tasks, classification_tasks, out_dir,
    )

    all_tasks = discover_all_tasks()
    all_encoders = discover_all_encoders()
    comp_path = _write_completion_csv(
        all_tasks=all_tasks,
        all_encoders=all_encoders,
        folder_for=folder_for,
        is_complete_for=is_complete,
        out_dir=out_dir,
    )
    written.append(comp_path)

    print(f"\nParsed {found} result files, {missing} missing")
    print(f"Wrote {len(written)} CSV(s) to {out_dir}/")
    for p in written:
        print(f"  {p}")
