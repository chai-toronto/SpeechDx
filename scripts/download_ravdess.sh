#!/usr/bin/env bash
# Fetch RAVDESS speech audio (CC BY-NC-SA 4.0) into data/ravdess/raw/.
# Source: https://zenodo.org/records/1188976
#
# Usage: bash script/download_ravdess.sh [DEST]
# Default DEST = data/ravdess/raw

set -euo pipefail

DEST="${1:-data/ravdess/raw}"
URL="https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
ZIP="Audio_Speech_Actors_01-24.zip"

mkdir -p "$DEST"
cd "$DEST"

if [[ ! -f "$ZIP" ]]; then
  echo "Downloading $ZIP (~208 MB) into $DEST ..."
  curl -L --fail -o "$ZIP" "$URL"
fi

if [[ ! -d "Actor_01" ]]; then
  echo "Extracting $ZIP ..."
  unzip -q "$ZIP"
fi

echo
echo "Done. Audio under $DEST/Actor_*/*.wav"
echo "Next: copy the Actor_*/ tree into data/ravdess/processed/audio/, then run"
echo "      python metadata_script/create_ravdess_metadata.py"
