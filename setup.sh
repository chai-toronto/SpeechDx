#!/usr/bin/env bash
set -euo pipefail

# Python 3.11 — see .python-version (gitignored)
python -m pip install --upgrade pip
pip install -r requirements.txt
