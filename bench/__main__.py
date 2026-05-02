"""Unified entry point for the four benchmark orchestrators.

Usage:
    python -m bench <mode> [args...]

Where ``<mode>`` is one of:
    single          — full single-dataset benchmark (was: run_all.py)
    data-eff        — data-efficiency sweep (was: run_all_data_eff.py)
    cross           — zero-shot cross-task (was: run_all_cross.py)
    cross-category  — category-cross variant (was: run_all_cross_category.py)

All flags after the mode are forwarded to the underlying script. For example:

    python -m bench single status
    python -m bench single run --encoder wavlm --task c9s_t1 -j 4
    python -m bench data-eff status --level 25
    python -m bench cross summary

The legacy ``run_all*.py`` scripts continue to work and remain the entry
point for SLURM submission templates.
"""

from __future__ import annotations

import sys

_MODES = {
    "single":         ("run_all",                "run_all.py"),
    "data-eff":       ("run_all_data_eff",       "run_all_data_eff.py"),
    "cross":          ("run_all_cross",          "run_all_cross.py"),
    "cross-category": ("run_all_cross_category", "run_all_cross_category.py"),
}


def _print_usage() -> None:
    print("Usage: python -m bench <mode> [args...]\n")
    print("Modes:")
    for mode, (_, script) in _MODES.items():
        print(f"  {mode:<15} forwards to {script}")
    print("\nRun  python -m bench <mode> --help  to see mode-specific flags.")


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help", "help"}:
        _print_usage()
        return 0
    mode, *forwarded = argv
    if mode not in _MODES:
        print(f"bench: unknown mode {mode!r}\n", file=sys.stderr)
        _print_usage()
        return 2

    module_name, script_name = _MODES[mode]
    module = __import__(module_name)
    # Underlying scripts call ``argparse`` against ``sys.argv``, so rewrite it.
    sys.argv = [script_name, *forwarded]
    module.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
