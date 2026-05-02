#!/usr/bin/env bash
# Invalidate cache.hdf5 entries for the (encoder, dataset) pairs below.
# Dry-run by default. Pass --apply as the only arg to actually delete.
#
# Edit ENCODERS and DATASETS to choose what to invalidate.
set -euo pipefail

ENCODERS=(ast audiomae)
DATASETS=(avfad coswara edaic ksof ravdess)

cd "$(dirname "$0")"
APPLY="${1:-}"

for ENC in "${ENCODERS[@]}"; do
    for DS in "${DATASETS[@]}"; do
        echo "=============================================="
        echo ">>> encoder=$ENC dataset=$DS"
        python -m training.dataio.invalidate_cache "$ENC" "$DS" $APPLY
    done
done