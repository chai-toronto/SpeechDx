"""Split a frozen-Alliance pip-freeze into wheelhouse vs PyPI requirement files.

Usage: python split_requirements.py <full_requirements.txt> <out_dir>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def split(full: Path, out_dir: Path) -> None:
    cc_lines: list[str] = []
    pypi_lines: list[str] = []

    for raw in full.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if "+computecanada" in line:
            cleaned = line.replace("+computecanada", "")
            cc_lines.append(cleaned)
            continue

        m = re.match(r"^\s*([A-Za-z0-9_.\-]+)\s*@\s*file:///", line)
        if m:
            cc_lines.append(m.group(1))
            continue

        pypi_lines.append(line)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "requirements_cc.txt").write_text("\n".join(cc_lines) + "\n")
    (out_dir / "requirements_pypi.txt").write_text("\n".join(pypi_lines) + "\n")
    print(f"wheelhouse pkgs : {len(cc_lines)}")
    print(f"pypi pkgs      : {len(pypi_lines)}")


if __name__ == "__main__":
    full = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    split(full, out_dir)
