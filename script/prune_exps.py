#!/usr/bin/env python3
"""Prune redundant artifacts from completed experiments under exps/single_task/.

Targets the bloat documented in /Users/lkieu/.claude/plans/we-need-a-strategy-snuggly-hopper.md:
- Per-trial save/CKPT*/model.ckpt files (650 MB - 2.4 GB each from full-model checkpoints)
- ray_results/ (CV) or results/ (non-CV) Ray Tune storage

Only acts on dirs matching the experiment shape
`exps/single_task/<task>/<encoder>-<probe>-<tag>/` that already have a final
results file (test_results.yaml or test_results.txt with CI).

Usage:
    python -m script.prune_exps --dry-run                # report what would be deleted
    python -m script.prune_exps                          # execute
    python -m script.prune_exps --root exps/cross        # prune a different mode dir
"""
import argparse
import shutil
import sys
from pathlib import Path

# Reuse completion-detection logic from the orchestrator so the pruner stays
# consistent with the harness's own skip-gating.
from ahb.orchestrator import has_ci_results

# script/extract_layer_weights.py reads model.ckpt only from
# exps/single_task/*/*CLTP-2*/brain-logs/save/CKPT*/ — a different naming pattern that the
# structural guard below does not match (no `CLTP-2` substring in any current
# encoder-probe-tag combo). Those CKPTs are never touched.


def is_experiment_dir(p: Path) -> bool:
    """Structural guard: True iff `p` looks like exps/single_task/<task>/<encoder>-<probe>-<tag>/.

    Requires:
      - p is a directory
      - leaf name has >= 3 dash-separated parts
      - leaf name does not start with '_'
      - parent (task) name does not start with '_' and is not 'manifest'
    """
    if not p.is_dir():
        return False
    name = p.name
    if name.startswith("_") or name in ("manifest",):
        return False
    if len(name.split("-")) < 3:
        return False
    parent = p.parent.name
    if parent.startswith("_") or parent in ("manifest",):
        return False
    return True


def deletable_paths(exp_dir: Path) -> list[Path]:
    """Return the set of paths inside `exp_dir` that are safe to rmtree.

    Covers both layouts:
      CV:     fold_*/<trial_id>/save/  +  ray_results/
      non-CV: <trial_id>/save/         +  results/
    """
    targets: list[Path] = []
    # Trial save/ dirs at any depth under exp_dir (CV: depth 3, non-CV: depth 2).
    # Use rglob so we catch both. Skip any save/ directly under exp_dir (the
    # legacy top-level `save/` from non-CV runs that's typically empty but
    # belongs to the experiment — leave it alone for safety; it's bytes, not GB).
    for save_dir in exp_dir.rglob("save"):
        if save_dir.parent == exp_dir:
            continue
        if save_dir.is_dir():
            targets.append(save_dir)
    # Ray Tune storage at the experiment root.
    for name in ("ray_results", "results"):
        candidate = exp_dir / name
        if candidate.is_dir():
            targets.append(candidate)
    return targets


def dir_size(p: Path) -> int:
    """Total bytes under p (best-effort; ignores broken symlinks)."""
    total = 0
    for f in p.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def prune_root(root: Path, dry_run: bool) -> tuple[int, int]:
    """Walk `root`, prune completed experiments. Returns (paths_deleted, bytes_freed)."""
    if not root.is_dir():
        print(f"[skip] {root} does not exist")
        return 0, 0

    total_paths = 0
    total_bytes = 0
    skipped_incomplete = 0
    skipped_shape = 0

    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("_"):
            continue
        for exp_dir in sorted(task_dir.iterdir()):
            if not is_experiment_dir(exp_dir):
                skipped_shape += 1
                continue
            task_stem = task_dir.name
            if not has_ci_results(exp_dir, task_stem):
                skipped_incomplete += 1
                continue
            targets = deletable_paths(exp_dir)
            if not targets:
                continue
            for target in targets:
                size = dir_size(target)
                rel = target.relative_to(root)
                action = "[dry-run]" if dry_run else "[delete] "
                print(f"{action} {fmt_bytes(size):>10}  {root.name}/{rel}")
                total_paths += 1
                total_bytes += size
                if not dry_run:
                    shutil.rmtree(target, ignore_errors=True)

    label = "would free" if dry_run else "freed"
    print(
        f"\n{root}: {total_paths} paths, {label} {fmt_bytes(total_bytes)} "
        f"(skipped {skipped_incomplete} incomplete, {skipped_shape} non-experiment)"
    )
    return total_paths, total_bytes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="exps/single_task",
                    help="Experiments root to walk (default: exps/single_task)")
    ap.add_argument("--dry-run", action="store_true", help="Report what would be deleted, do not delete")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    paths, freed = prune_root(root, dry_run=args.dry_run)

    if not args.dry_run and paths > 0:
        print(f"\nDone. Re-run `du -sh {args.root}/` to confirm.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
