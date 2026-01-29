"""
Dataset-specific preparation script and dataio pipelines.

This file contains functions that are unique to the 'CS-Res-L' dataset,
including manifest generation and label encoding logic.

To adapt to a new dataset, copy this file and modify:
1. prepare_data (to generate the correct train/valid/test JSON manifest files).
2. label_pipeline (to handle the specific labels and label-to-index mapping).
3. The 'output_keys' in dataio_prep (if the label key changes).
"""
import json
import random
import warnings
from pathlib import Path

import torchaudio
import torchaudio.functional as F
import speechbrain as sb
from speechbrain.utils.data_pipeline import CachedDynamicItem
import torch
import pandas as pd

from training.dataio.cache_dynamic_item import CachedHDF5DynamicItem, CachedPersistDynamicItem
from training.dataio.stratified_group_k_fold import stratified_group_kfold_df
from training.dataio.utils import ensure_dir, PathEncoder, locate_bad


def prepare_data(
        wav_folder,
        audio_archive_path,
        metadata_path,
        manifest_train_path,
        manifest_val_path,
        manifest_test_path,
        ratio,
        random_seed,
        raw_label_key,
        new_test,
        num_fold,
        max_length=1e10,
):
    """
    This function is dataset-specific.
    It takes raw data info (like metadata_path and wav_folder) and
    creates the SpeechBrain manifest JSON files (train, valid, test).

    Replace the body of this function with the logic needed for your
    specific dataset (e.g., reading a CSV/TSV file and writing JSONs).

    It first takes the test split out of metadata. Then it splits the
    rest deterministically with seed. Each train manifest now contains
    k subdict for each fold. Each is like the original format.
    """
    manifest_train_path = Path(manifest_train_path)
    manifest_val_path = Path(manifest_val_path)
    manifest_test_path = Path(manifest_test_path)

    df = pd.read_csv(metadata_path)

    # Resolve path to be absolute
    df["path"] = Path(wav_folder).resolve() / df["path"]
    df["max_length"] = int(max_length)

    # split into test and non-test
    if new_test:
        raise NotImplementedError
    else:
        df_test = df[df['split'] == 2]
        df_nontest = df[df['split'] != 2]
        df_train_og = df[df['split'] == 0]
        df_val_og = df[df['split'] == 1]

    # split train data into folds, each goes by manifest id
    folds = stratified_group_kfold_df(df_nontest,
                                      'uid',
                                      raw_label_key,
                                      "Participant_ID",
                                      random_seed=random_seed,
                                      n_splits=num_fold)

    train_dicts = []
    valid_dicts = [] # list of folds
    for train_df, val_df in folds:
        train_dicts.append(train_df.set_index('uid').to_dict(orient='index'))
        valid_dicts.append(val_df.set_index('uid').to_dict(orient='index'))

    test_val = df_test.set_index("uid").to_dict(orient='index')
    test_train = df_nontest.set_index("uid").to_dict(orient='index')
    test_data = {
        "train": test_train,
        "val": test_val
    }

    train_dicts.append(df_train_og.set_index('uid').to_dict(orient='index'))
    valid_dicts.append(df_val_og.set_index('uid').to_dict(orient='index'))
    print("Train og size:", len(df_train_og))
    print("Val og size:", len(df_val_og))
    print("Test size:", len(test_val))

    import json
    ensure_dir(manifest_train_path)
    with open(manifest_train_path, 'w') as f:
        json.dump(train_dicts, f, indent=5, cls=PathEncoder)
    ensure_dir(manifest_val_path)
    with open(manifest_val_path, 'w') as f:
        json.dump(valid_dicts, f, indent=5, cls=PathEncoder)
    ensure_dir(manifest_test_path)
    with open(manifest_test_path, 'w') as f:
        json.dump(test_data, f, indent=4, cls=PathEncoder)
    print("Manifests created.")
    print("--- prepare_data finished ---")


def dataio_prep(hparams):
    """
    This function is dataset-specific.
    It defines the data processing pipelines and creates the DynamicItemDatasets.

    For a new task, modify the label_pipeline and the output_keys.
    """

    dynamic_items = []
    output_keys = ["id", "path"]

    # Initialization of the label encoder.
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    max_length = hparams.get("max_length", 160000)  # default to 10 seconds at 16kHz
    sample_rate = hparams.get("sample_rate", 16000)
    min_length = hparams.get("min_length", 16000)  # default to 1 second at 16kHz

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("path")
    @sb.utils.data_pipeline.provides("raw_signal", "raw_duration")
    def audio_pipeline(file_path):
        """Load the signal, resample, and pass it and its length."""

        raw_signal, sr_og = torchaudio.load(file_path)
        # handle multi-channel
        if raw_signal.shape[0] > 1:
            raw_signal = raw_signal.mean(dim=0, keepdim=False)

        if sr_og != sample_rate:
            raw_signal = F.resample(raw_signal, sr_og, new_freq=sample_rate,
                                lowpass_filter_width=64,
                                rolloff=0.9475937167399596,
                                resampling_method="sinc_interp_kaiser",
                                beta=14.769656459379492
                                )

        raw_signal = raw_signal.squeeze()
        raw_duration = len(raw_signal)
        return raw_signal, raw_duration

    dynamic_items.append(audio_pipeline)
    output_keys.extend(["raw_signal", "raw_duration"])

    # Define label pipeline
    @sb.utils.data_pipeline.takes("label")
    @sb.utils.data_pipeline.provides("label_encoded")
    def label_pipeline(label):
        """Defines the pipeline to process the input label ('non'/'symptomatic')."""
        # The key produced here ('label_encoded') must match
        # the 'label_key' used in train.py and the YAML.
        label_encoded = label
        yield label_encoded
    dynamic_items.append(label_pipeline)
    output_keys.append("label_encoded")

    if hparams["cache_encoder"]:
        speech_encoder = hparams["encoder"]
        # Do this to take advantage of auto padding
        num_layers = hparams["num_layers"]
        num_outputs = num_layers if speech_encoder.output_hidden_states else 1
        raw_output_vars = [f"raw_emb_{i}" for i in range(num_outputs)]

        @CachedHDF5DynamicItem.cache(hparams["cache_dir"], 'a')
        @sb.utils.data_pipeline.takes("id", "signal")
        @sb.utils.data_pipeline.provides(*raw_output_vars)
        def cache_emb(id, signal):
            # signal is 1D tensor
            with torch.no_grad():
                emb = speech_encoder(signal.unsqueeze(0))
            if speech_encoder.output_hidden_states:
                emb = tuple(x.squeeze(0) for x in emb)
            else:
                emb = emb.squeeze(0)
            return emb

        dynamic_items.append(cache_emb)
        output_keys += raw_output_vars

        output_vars = [f"emb_{i}" for i in range(num_outputs)]

        # Handling too short or too long data
        @sb.utils.data_pipeline.takes(*raw_output_vars, "raw_duration")
        @sb.utils.data_pipeline.provides(*output_vars, "duration")
        def process_emb(*args):
            raw_embs = args[:-1]
            duration = args[-1]
            rel_min_length = duration / min_length
            if rel_min_length < 1.0:
                # pad
                output_embs = []
                for raw_emb in raw_embs:
                    T, D = raw_emb.shape
                    n_repeats = int(1.0 / rel_min_length) + 1
                    padded_emb = raw_emb.repeat(n_repeats, 1)[:min_length]
                    output_embs.append(padded_emb)
                return (*output_embs, min_length)

            rel_max_length = duration / max_length
            if rel_max_length > 1.0:
                # randomly crop
                output_embs = []
                for raw_emb in raw_embs:
                    T, D = raw_emb.shape
                    new_length = int(T / rel_max_length)
                    start = random.randint(0, T - new_length)
                    cropped_emb = raw_emb[start:start + new_length]
                    output_embs.append(cropped_emb)
                return (*output_embs, new_length)

            return (*raw_embs, duration)
        dynamic_items.append(process_emb)
        output_keys += output_vars + ["duration"]

    else:
        # Handling too short or too long data.
        @sb.utils.data_pipeline.takes("raw_signal", "raw_duration")
        @sb.utils.data_pipeline.provides("signal", "duration")
        def process_signal(raw_signal, raw_duration):
            if raw_duration > max_length: # randomly crop if too long
                start = random.randint(0, raw_duration - max_length)
                signal = raw_signal[start:start + max_length]
                duration = max_length

            if raw_duration < min_length:  # Concat to itself if too short
                n_repeats = int(min_length / raw_duration) + 1
                signal = raw_signal.repeat(n_repeats)[:min_length]
                duration = len(signal)
            return signal, duration


    # Retrieve the data
    with open(hparams["train_annotation"], "r") as f:
        train_folds = json.load(f)

    with open(hparams["val_annotation"], "r") as f:
        val_folds = json.load(f)

    with open(hparams['test_annotation'], "r") as f:
        test_data = json.load(f)

    data_dict = {}
    for i in range(hparams['num_fold']):
        data_dict[f'train_{i}'] = train_folds[i]
        data_dict[f'val_{i}'] = val_folds[i]

    data_dict['test_train'] = test_data['train']
    data_dict['test_val'] = test_data['val']

    data_dict['train_og'] = train_folds[-1]
    data_dict['val_og'] = val_folds[-1]

    # Define datasets.
    datasets = {}
    for dataset in data_dict:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset(
            data=data_dict[dataset],
            dynamic_items=dynamic_items,
            output_keys=output_keys,
        )

    if hparams["cache_encoder"]:
        warmup_ds = [datasets['test_train'], datasets['test_val']]
        for i, ds in enumerate(warmup_ds):
            print(f"Iterating dataset {i} to warm the cache.")
            ds.iterate_once()
        cache_emb.change_file_mode('r')  # change to read mode

    return datasets


