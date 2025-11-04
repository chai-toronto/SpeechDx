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
import torch
import pandas as pd

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
    if not all(
            (
                manifest_train_path.exists(),
                manifest_val_path.exists(),
                manifest_test_path.exists(),
            )
    ):
        df = pd.read_csv(metadata_path)

        # Resolve path to be absolute
        df["path"] = Path(wav_folder) / df["path"].astype(str)
        df["max_length"] = max_length

        # split into test and non-test
        if new_test:
            raise NotImplementedError
        else:
            df_test = df[df['split'] == 2]
            df_nontest = df[df['split'] != 2]

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

        import json
        ensure_dir(manifest_train_path)
        with open(manifest_train_path, 'w') as f:
            print("First non-JSONable:", locate_bad(train_dicts))
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

    # Initialization of the label encoder.
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("path", "max_length")
    @sb.utils.data_pipeline.provides("signal", "duration")
    def audio_pipeline(file_path, max_length):
        """Load the signal, resample, and pass it and its length."""

        signal, sr_og = torchaudio.load(file_path)
        # handle multi-channel
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)

        if sr_og != 16000:
            signal = F.resample(signal, sr_og, new_freq=16000,
                                lowpass_filter_width=64,
                                rolloff=0.9475937167399596,
                                resampling_method="sinc_interp_kaiser",
                                beta=14.769656459379492
                                )

        signal = signal.squeeze()
        duration = len(signal)
        if duration > max_length: # randomly crop if too long
            start = random.randint(0, duration - max_length)
            signal = signal[start:start + max_length]
            duration = max_length

        if duration == 0:  # handle empty audio
            signal = torch.zeros(16000)
            duration = 16000
            warnings.warn("Empty audio file found: {}".format(file_path))
        return signal, duration

    # Define label pipeline
    @sb.utils.data_pipeline.takes("label")
    @sb.utils.data_pipeline.provides("label_encoded")
    def label_pipeline(label):
        """Defines the pipeline to process the input label ('non'/'symptomatic')."""
        # The key produced here ('label_encoded') must match
        # the 'label_key' used in train.py and the YAML.
        label_encoded = label
        yield label_encoded

    @sb.utils.data_pipeline.takes("id")
    @sb.utils.data_pipeline.provides("emb")
    def cache_emb(id):
        """Fe"""
        return None

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

    # Define datasets.
    datasets = {}
    for dataset in data_dict:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset(
            data=data_dict[dataset],
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "signal", "duration", "path", "label_encoded"],
        )
    return datasets


