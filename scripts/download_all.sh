#!/usr/bin/env bash
# Fetch every openly-downloadable dataset into data/<name>/raw/.
# Runs each per-dataset downloader in sequence; safe to re-run (downloaders
# skip already-fetched archives and extracted trees).
#
# Covers: ravdess (Zenodo), mdvr / MDVR-KCL (Zenodo), coswara (GitHub clone +
# upstream extract_data.py — heavy, tens of GB once extracted).
#
# License-gated datasets (aphasia, avfad, c19sounds, dementiabank, edaic, iemocap, ksof,
# nemours, torgo, uaspeech) need a DTA / EULA / email request — see the
# Datasets table in the top-level README.
#
# Usage: bash script/download_all.sh [--skip-coswara]

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SKIP_COSWARA=0

for arg in "$@"; do
  case "$arg" in
    --skip-coswara) SKIP_COSWARA=1 ;;
    *) echo "Unknown flag: $arg" >&2; exit 2 ;;
  esac
done

echo "==> ravdess"
bash "$HERE/download_ravdess.sh"

echo "==> mdvr (MDVR-KCL)"
bash "$HERE/download_mdvr.sh"

if [[ "$SKIP_COSWARA" == "1" ]]; then
  echo "==> coswara: skipped (--skip-coswara)"
else
  echo "==> coswara (heavy, ~tens of GB extracted)"
  bash "$HERE/download_coswara.sh"
fi

echo
echo "All open downloads done. License-gated datasets must be requested manually — see README."
