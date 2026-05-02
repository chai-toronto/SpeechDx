#!/usr/bin/env python3
"""Top-level convenience shim — equivalent to ``python -m ahb``."""

from __future__ import annotations

import sys

from ahb.cli import main

if __name__ == "__main__":
    sys.exit(main())
