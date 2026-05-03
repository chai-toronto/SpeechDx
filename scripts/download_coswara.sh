#!/usr/bin/env bash
# Fetch the Coswara COVID-19 sounds dataset into data/coswara/raw/.
# Source: https://github.com/iiscleap/Coswara-Data (the upstream repo bundles
# split tar archives per YYYYMMDD/ folder; their own extract_data.py joins and
# unpacks them).
#
# Usage: bash script/download_coswara.sh [DEST]
# Default DEST = data/coswara/raw
#
# After extraction, audio lives under:
#   $DEST/Extracted_data/<YYYYMMDD>/<participant_id>/{counting-normal,counting-fast,...}.wav
#
# NOTE: metadata_script/create_coswara_metadata.py currently expects audio at
#   data/coswara/<YYYYMMDD>/<pid>/...  (no `raw/Extracted_data/` prefix).
# Adjust DATA_ROOT in that script — or move/symlink Extracted_data/ — before
# building the metadata CSV.

set -euo pipefail

DEST="${1:-data/coswara/raw}"
REPO="https://github.com/iiscleap/Coswara-Data.git"

if [[ ! -d "$DEST/.git" ]]; then
  mkdir -p "$(dirname "$DEST")"
  echo "Cloning $REPO into $DEST ..."
  git clone --depth 1 "$REPO" "$DEST"
else
  echo "Repo already cloned at $DEST; pulling latest ..."
  git -C "$DEST" pull --ff-only
fi

if [[ ! -d "$DEST/Extracted_data" ]]; then
  echo "Running upstream extract_data.py (joins split tarballs, ~tens of GB) ..."
  ( cd "$DEST" && python3 extract_data.py )
fi

echo
echo "Done. Audio under $DEST/Extracted_data/<YYYYMMDD>/<pid>/*.wav"
