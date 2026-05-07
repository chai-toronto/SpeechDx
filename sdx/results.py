"""Helpers for reading and aggregating result files.

``parse_results_txt`` / ``parse_results_yaml``: parse a finished run's
``test_results.{txt,yaml}`` into a metric dict for the summary CSV writer.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def parse_results_txt(path: Path) -> dict[str, float | str]:
    """Parse 'key: value' lines from a test_results.txt file."""
    metrics: dict[str, float | str] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        try:
            metrics[key.strip()] = float(val.strip())
        except ValueError:
            pass
    lo, hi = metrics.get("AUROC_CI_low"), metrics.get("AUROC_CI_high")
    if isinstance(lo, float) and isinstance(hi, float):
        metrics["AUC_CI"] = f"({lo:.4f}, {hi:.4f})"
    lo, hi = metrics.get("MAE_CI_low"), metrics.get("MAE_CI_high")
    if isinstance(lo, float) and isinstance(hi, float):
        metrics["MAE_CI"] = f"({lo:.4f}, {hi:.4f})"
    return metrics


def parse_results_yaml(path: Path) -> dict[str, float | str]:
    """Parse test_results.yaml (per-fold CV) — take summary means.

    For mdvr (CV) AUROC and MAE, synthesize AUC_CI / MAE_CI as
    (mean - std, mean + std) since per-fold CIs aren't aggregated; cross-fold
    std is the meaningful uncertainty.
    """
    data = yaml.safe_load(path.read_text())
    summary = data.get("summary", {}) if isinstance(data, dict) else {}
    metrics: dict[str, float | str] = {
        k: float(v["mean"]) for k, v in summary.items()
        if isinstance(v, dict) and "mean" in v
    }
    auroc = summary.get("AUROC")
    if isinstance(auroc, dict) and "mean" in auroc and "std" in auroc:
        m, s = float(auroc["mean"]), float(auroc["std"])
        metrics["AUC_CI"] = f"({m - s:.4f}, {m + s:.4f})"
    mae = summary.get("MAE")
    if isinstance(mae, dict) and "mean" in mae and "std" in mae:
        m, s = float(mae["mean"]), float(mae["std"])
        metrics["MAE_CI"] = f"({m - s:.4f}, {m + s:.4f})"
    return metrics
