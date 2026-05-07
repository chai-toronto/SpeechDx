"""``python -m sdx`` entry point — delegates to ``sdx.cli.main``."""

from __future__ import annotations

import sys

from sdx.cli import main

if __name__ == "__main__":
    sys.exit(main())
