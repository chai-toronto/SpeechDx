#!/usr/bin/env bash
# Fetch MDVR-KCL (Mobile Device Voice Recordings at King's College London)
# Parkinson's disease + healthy-control speech, into data/mdvr/raw/.
#
# Source: https://zenodo.org/records/2867216
# Citation: Jaeger, Trivedi, Stadtschnitzer (2019). Funded by EU i-PROGNOSIS.
# License: CC BY 4.0
# Filename format: ID{nn}_{hc|pd}_{H&Y}_{UPDRS_II-5}_{UPDRS_III-18}.wav
#
# Usage: bash script/download_mdvr.sh [DEST]
# Default DEST = data/mdvr/raw

set -euo pipefail

DEST="${1:-data/mdvr/raw}"
URL="https://zenodo.org/records/2867216/files/26_29_09_2017_KCL.zip?download=1"
ZIP="26_29_09_2017_KCL.zip"

mkdir -p "$DEST"
cd "$DEST"

if [[ ! -f "$ZIP" ]]; then
  echo "Downloading $ZIP (~606 MB) into $DEST ..."
  curl -L --fail -o "$ZIP" "$URL"
fi

# Extract once. The archive top-level dir is "26-29_09_2017_KCL".
if [[ ! -d "26-29_09_2017_KCL" ]] && [[ ! -d "ReadText" ]]; then
  echo "Extracting $ZIP ..."
  unzip -q "$ZIP"
fi

echo
echo "Done. Run: python metadata_script/create_mdvr_metadata.py"
