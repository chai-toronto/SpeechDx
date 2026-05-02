"""Verify split integrity of a dataset CSV.

Checks:
1. Data leakage — no participant appears in more than one split.
2. Label distribution — whether label proportions are similar across splits
   (chi-squared test, p < 0.05 flags a significant difference).

Usage:
    python script/verify_splits.py path/to/dataset.csv
    python script/verify_splits.py path/to/*.csv          # multiple files
"""

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency


def check_leakage(df: pd.DataFrame) -> list[str]:
    """Return participant IDs that appear in more than one split."""
    pid_splits = df.groupby("Participant_ID")["split"].apply(set)
    leaked = pid_splits[pid_splits.apply(len) > 1]
    return leaked.index.tolist()


def check_label_distribution(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """Return crosstab, chi2 statistic, and p-value."""
    ct = pd.crosstab(df["split"], df["label"])
    chi2, p, _, _ = chi2_contingency(ct)
    return ct, chi2, p


def verify(csv_path: str):
    path = Path(csv_path)
    print(f"\n{'='*60}")
    print(f"  {path.name}")
    print(f"{'='*60}")

    df = pd.read_csv(path)

    required = {"Participant_ID", "split", "label"}
    missing = required - set(df.columns)
    if missing:
        print(f"  SKIP — missing columns: {', '.join(sorted(missing))}")
        return

    print(f"  Rows: {len(df)}  |  Participants: {df['Participant_ID'].nunique()}"
          f"  |  Labels: {df['label'].nunique()}  |  Splits: {sorted(df['split'].unique())}")

    # --- Leakage ---
    leaked = check_leakage(df)
    if leaked:
        print(f"\n  !! LEAKAGE: {len(leaked)} participant(s) in multiple splits:")
        for pid in leaked[:20]:
            splits = sorted(df.loc[df["Participant_ID"] == pid, "split"].unique())
            print(f"     {pid} -> splits {splits}")
        if len(leaked) > 20:
            print(f"     ... and {len(leaked) - 20} more")
    else:
        print("\n  OK — No data leakage detected.")

    # --- Label distribution ---
    ct, chi2, p = check_label_distribution(df)

    # Show proportions per split
    props = ct.div(ct.sum(axis=1), axis=0).round(3)
    print(f"\n  Label proportions per split:")
    print(props.to_string().replace("\n", "\n  "))

    print(f"\n  Chi-squared = {chi2:.2f},  p = {p:.4f}", end="")
    if p < 0.05:
        print("  !! Distribution differs significantly across splits.")
    else:
        print("  OK — distribution is similar across splits.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python script/verify_splits.py <csv_path> [csv_path ...]")
        sys.exit(1)

    for arg in sys.argv[1:]:
        for p in sorted(Path(".").glob(arg)) if "*" in arg else [Path(arg)]:
            verify(str(p))

    print()


if __name__ == "__main__":
    main()
