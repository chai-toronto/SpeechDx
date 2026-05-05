#!/usr/bin/env bash
# Fetch the TORGO Database of Dysarthric Articulation into data/torgo/raw/.
#
# Source: https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html
# Four archives (one per speaker group), ~9.5 GB total compressed:
#   F.tar.bz2   (~1.1 GB) — female dysarthric
#   FC.tar.bz2  (~2.5 GB) — female controls
#   M.tar.bz2   (~2.5 GB) — male dysarthric
#   MC.tar.bz2  (~3.4 GB) — male controls
#
# Each archive extracts a flat list of <speaker>/Session*/wav_arrayMic/*.wav
# at the destination root. We re-extract into a per-group subdirectory so the
# layout matches what metadata_script/create_torgo_metadata.py expects:
#   data/torgo/raw/<group>/<speaker>/Session*/wav_arrayMic/*.wav
#
# Usage: bash scripts/download_torgo.sh [DEST]
# Default DEST = data/torgo/raw

set -euo pipefail

DEST="${1:-data/torgo/raw}"
BASE_URL="https://www.cs.toronto.edu/~complingweb/data/TORGO"
ARCHIVES=(F.tar.bz2 FC.tar.bz2 M.tar.bz2 MC.tar.bz2)

mkdir -p "$DEST"
cd "$DEST"

for arc in "${ARCHIVES[@]}"; do
  group="${arc%.tar.bz2}"
  if [[ -d "$group" ]]; then
    echo "==> $group/ already extracted, skipping"
    continue
  fi
  if [[ ! -f "$arc" ]]; then
    echo "==> Downloading $arc ..."
    curl -L --fail -o "$arc" "$BASE_URL/$arc"
  fi
  echo "==> Extracting $arc into $group/ ..."
  mkdir -p "$group"
  tar -xjf "$arc" -C "$group"
done

echo
echo "Done. Raw audio under $DEST/{F,FC,M,MC}/<speaker>/Session*/wav_arrayMic/*.wav"
echo "Next: python metadata_script/create_torgo_metadata.py"
