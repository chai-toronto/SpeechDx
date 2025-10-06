"""
Dataset-specific preparation script and dataio pipelines.

This file contains functions that are unique to the 'CS-Res-L' dataset,
including manifest generation and label encoding logic.

To adapt to a new dataset, copy this file and modify:
1. prepare_data (to generate the correct train/valid/test JSON manifest files).
2. label_pipeline (to handle the specific labels and label-to-index mapping).
3. The 'output_keys' in dataio_prep (if the label key changes).
"""

import torchaudio
import torchaudio.functional as F
import speechbrain as sb
import torch
import pandas as pd

from training.dataio.stratified_group_k_fold import stratified_group_kfold_df


def prepare_data(
        wav_folder,
        audio_archive_path,
        metadata_path,
        manifest_train_path,
        manifest_fold_path,
        manifest_test_path,
        ratio,
        random_seed,
        label_key,
        new_test
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

    if not all(
            (
                    sb.utils.checkpoints.is_pytorch_object_in_folder(manifest_train_path),
                    sb.utils.checkpoints.is_pytorch_object_in_folder(manifest_fold_path),
                    sb.utils.checkpoints.is_pytorch_object_in_folder(manifest_test_path)
            )
    ):

        print(f"Creating mock manifest files for demonstration in: {manifest_train_path}")

        df = pd.read_csv(metadata_path)

        # split into test and non-test
        if new_test:
            raise NotImplementedError
        else:
            df_test = df[df['split'] == 2]
            df_nontest = df[df['split'] != 2]

        # split train data into folds, each goes by manifest id
        folds = stratified_group_kfold_df(df_nontest,
                                          'uid',
                                          label_key,
                                          "Participant_ID",
                                          random_seed=random_seed)

        folds_dict = {}
        for i, (train_ids, val_ids) in enumerate(folds):
            folds_dict[f'{i}'] = {
                'train': train_ids,
                'val': val_ids
            }

        train_data = df_nontest.set_index("uid").to_dict(orient='index')
        test_data = df_test.set_index("uid").to_dict(orient='index')

        import json
        with open(manifest_train_path, 'w') as f:
            json.dump(train_data, f, indent=4)
        with open(manifest_fold_path, 'w') as f:
            json.dump(folds_dict, f, indent=4)
        with open(manifest_test_path, 'w') as f:
            json.dump(test_data, f, indent=4)
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
    @sb.utils.data_pipeline.takes("path")
    @sb.utils.data_pipeline.provides("signal", "duration")
    def audio_pipeline(file_path):
        """Load the signal, resample, and pass it and its length."""

        signal, sr_og = torchaudio.load(file_path)
        # handle multi-channel
        if signal.shape[0] > 1:
            signal = torch.mean(signal, axis=0)

        if sr_og != 16000:
            signal = F.resample(signal, sr_og, new_freq=16000,
                                lowpass_filter_width=64,
                                rolloff=0.9475937167399596,
                                resampling_method="sinc_interp_kaiser",
                                beta=14.769656459379492
                                )
        signal = signal.squeeze()
        duration = len(signal)
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

    # Define datasets.
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "test": hparams["test_annotation"]
    }

    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["uid", "signal", "duration", "path", "label_encoded"],
        )

    return datasets


