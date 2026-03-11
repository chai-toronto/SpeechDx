#!/usr/bin/env bash
set -e

BASE_CONFIG=training/config/main.yaml
TMP_CONFIG=training/config/_tmp_run.yaml

# Generate a temp config from main.yaml with the given encoder/probe/dataset combo
# Usage: make_config <model_name> <probe_name> <encoder_yaml> <probe_yaml> <dataset_yaml>
make_config() {
  local model_name="$1" probe_name="$2" encoder_yaml="$3" probe_yaml="$4" dataset_yaml="$5"
  sed \
    -e "s|^model_name:.*|model_name: ${model_name}|" \
    -e "s|^probe_name:.*|probe_name: ${probe_name}|" \
    -e "s|^encoder_params: !include:.*|encoder_params: !include:encoders/${encoder_yaml}|" \
    -e "s|^probe_params: !include:.*|probe_params: !include:probes/${probe_yaml}|" \
    -e "s|^data_params: !include:.*|data_params: !include:tasks/${dataset_yaml}|" \
    "$BASE_CONFIG" > "$TMP_CONFIG"
}

cleanup() { rm -f "$TMP_CONFIG"; }
trap cleanup EXIT


# WavJepa
#for DATASET in torgo uaspeech; do
#  make_config wavjepa TProbe wavjepa.yaml TProbe.yaml "${DATASET}.yaml"
#  python -m training.train "$TMP_CONFIG"
#done

# Qwen3Voice
for DATASET in torgo ravdess uaspeech mvdr; do
  make_config qwen3voice CLTP-2 qwen3_voice.yaml LTProbe.yaml "${DATASET}.yaml"
  python -m training.train "$TMP_CONFIG" --device=="$1"
done

# WavLM TP
#for DATASET in torgo ravdess uaspeech mvdr; do
#  make_config wavlm-basep-all TProbe wavlm.yaml TProbe.yaml "${DATASET}.yaml"
#  python -m training.train "$TMP_CONFIG"
#done

# XTTS
#for DATASET in uaspeech; do
#  make_config xtts_spk Probe xtts.yaml Probe.yaml "${DATASET}.yaml"
#  python -m training.train "$TMP_CONFIG"
#done




