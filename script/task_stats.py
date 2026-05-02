"""Per-task split stats (samples, subjects, label/score distribution)
from generated metadata/<dataset>/<task>.csv files and task yaml configs.
"""
import os
import glob
from pathlib import Path

import pandas as pd
import yaml


class _Tolerant(yaml.SafeLoader):
    pass


def _ignore(loader, tag_suffix, node):
    return None


_Tolerant.add_multi_constructor("!", _ignore)

ROOT = Path("/Users/lkieu/PycharmProjects/Audio-Health-Benchmark")
TASK_DIR = ROOT / "training/config/tasks"
META_DIR = ROOT / "metadata"


def fmt_count(df):
    return f"{len(df):,} ({df['Participant_ID'].nunique():,})"


def class_ratio(series, top=15):
    vc = series.value_counts(dropna=False)
    total = vc.sum()
    head = vc.head(top)
    parts = [f"{k}:{v} ({v/total:.0%})" for k, v in head.items()]
    if len(vc) > top:
        parts.append(f"...(+{len(vc)-top} more, total_uniq={len(vc)})")
    return ", ".join(parts)


def score_range(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "n/a"
    return f"[{s.min():g}, {s.max():g}] (mean={s.mean():.2f}, n_uniq={s.nunique()})"


def main():
    yamls = sorted(TASK_DIR.glob("*.yaml"))
    rows = []
    for y in yamls:
        cfg = yaml.load(y.read_text(), Loader=_Tolerant)
        dataset = cfg.get("dataset")
        task = cfg.get("task") or y.stem.split("_", 1)[-1]
        task_type = cfg.get("task_type", "")
        meta_path = META_DIR / dataset / f"{task}.csv"
        if not meta_path.exists():
            rows.append((dataset, task, task_type, "MISSING METADATA", "", "", "", ""))
            continue
        df = pd.read_csv(meta_path, low_memory=False)
        if "split" not in df.columns or "label" not in df.columns:
            rows.append((dataset, task, task_type, f"missing cols: {df.columns.tolist()[:5]}", "", "", "", ""))
            continue
        tr = df[df["split"] == 0]
        va = df[df["split"] == 1]
        te = df[df["split"] == 2]
        is_reg = task_type.upper() == "R" or task.endswith("R")
        if is_reg:
            info = score_range(df["label"])
        else:
            info = class_ratio(df["label"])
        rows.append((
            dataset, task, task_type,
            fmt_count(tr), fmt_count(va), fmt_count(te),
            info,
            ""
        ))

    # Print TSV (paste directly into Google Sheets)
    hdr = ["Dataset", "Task", "Type", "Train", "Validation", "Test", "Label/Score"]
    print("\t".join(hdr))
    for r in rows:
        print("\t".join(str(x) for x in r[:-1]))


if __name__ == "__main__":
    main()
