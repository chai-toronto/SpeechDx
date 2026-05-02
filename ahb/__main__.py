"""``python -m ahb`` entry point — delegates to ``ahb.cli.main``."""

from __future__ import annotations

import sys

from ahb.cli import main

if __name__ == "__main__":
    sys.exit(main())
