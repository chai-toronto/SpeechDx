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
import os
import random
import warnings
from pathlib import Path

import soundfile as sf
import librosa
import speechbrain as sb
from speechbrain.augment.time_domain import AddReverb, AddNoise, SpeedPerturb
from speechbrain.utils.data_pipeline import CachedDynamicItem
import torch
import pandas as pd

from training.dataio.cache_dynamic_item import CachedHDF5DynamicItem
from training.dataio.stratified_group_k_fold import stratified_group_kfold_df
from training.dataio.utils import ensure_dir, PathEncoder, locate_bad, proc_length_vec


def prepare_data(
        wav_folder,
        metadata_path,
        manifest_train_path,
        manifest_val_path,
        manifest_test_path,
        ratio,
        random_seed,
        raw_label_key,
        new_test,
        num_fold,
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

    # split into test and non-test
    if new_test:
        raise NotImplementedError
    else:
        df_test = df[df['split'] == 2]
        df_nontest = df[df['split'] != 2]

        # Official split
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

    train_dicts.append(df_train_og.set_index('uid').to_dict(orient='index'))
    valid_dicts.append(df_val_og.set_index('uid').to_dict(orient='index'))

    final_test = df_test.set_index("uid").to_dict(orient='index')
    final_train = df_nontest.set_index("uid").to_dict(orient='index')
    test_data = {
        "train": final_train,
        "val": final_test
    }

    print("Train og size:", len(df_train_og))
    print("Val og size:", len(df_val_og))
    print("Test size:", len(final_test))

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

    train_dynamic_items, val_dynamic_items = [], []
    output_keys = ["id", "path"]

    # Initialization of the label encoder.
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    sample_rate = hparams.get("sample_rate", 16000)

    max_samples = hparams.get("max_length", 10e5) * sample_rate  # default to longest
    min_samples = hparams.get("min_length", 3) * sample_rate  # default to torgo's avg lengths

    noise_folder = hparams.get("noise_folder", None)
    noise_folder = os.path.abspath(noise_folder)

    if noise_folder is None:
        raise ValueError("Noise folder must be specified in hparams for this task.")

    noisifier = AddNoise(os.path.join(noise_folder, 'noises.csv'),
                         replacements={'noise_folder': os.path.join(noise_folder, 'audio')},
                         snr_low= hparams["data_params"]["snr_low"],
                         snr_high= hparams["data_params"]["snr_high"],
                         noise_sample_rate=sample_rate,
                         clean_sample_rate=sample_rate)

    rir_folder = hparams.get("rir_folder", None)
    if rir_folder is None:
        raise ValueError("RIR folder must be specified in hparams for this task.")

    reverb = AddReverb(os.path.join(rir_folder, 'rirs.csv'),
                       replacements={'rir_folder': os.path.join(rir_folder, 'audio')},
                       reverb_sample_rate=sample_rate,
                       clean_sample_rate=sample_rate,
                       )


    # 90% to 109% speed perturbation
    perturbator = SpeedPerturb(orig_freq=sample_rate,
                               speeds=hparams["data_params"]["speed"])

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("path")
    @sb.utils.data_pipeline.provides("raw_signal", "duration")
    def audio_pipeline(file_path):
        """Load the signal, resample, and pass it and its length."""

        data, sr_og = sf.read(file_path, dtype='float32')
        # sf.read returns (samples,) or (samples, channels)
        if data.ndim > 1:
            data = data.mean(axis=1)

        if len(data) == 0:
            raise ValueError(f"Zero-length audio file: {file_path}")

        if sr_og != sample_rate:
            data = librosa.resample(data, orig_sr=sr_og, target_sr=sample_rate)

        raw_signal = torch.from_numpy(data)
        duration = len(raw_signal)

        return raw_signal, duration

    train_dynamic_items.append(audio_pipeline)
    val_dynamic_items.append(audio_pipeline)
    # output_keys.extend(["raw_signal", "duration"])

    @sb.utils.data_pipeline.takes("raw_signal")
    @sb.utils.data_pipeline.provides("raw_signal", "duration")
    def augment(raw_signal):
        raw_signal = raw_signal.unsqueeze(0)  # add batch dimension for augmentations
        raw_signal = perturbator(raw_signal)
        raw_signal = noisifier(raw_signal, torch.ones(1))
        raw_signal = reverb(raw_signal)
        raw_signal = raw_signal.squeeze(0)
        duration = raw_signal.shape[0]
        return raw_signal, duration

    # Notice we only augment the training data, not validation or test.
    train_dynamic_items.append(augment)

    # Handling too short or too long data.
    @sb.utils.data_pipeline.takes("raw_signal", "duration")
    @sb.utils.data_pipeline.provides("signal", "duration")
    def process_signal(signal, duration):
        if duration > max_samples:  # randomly crop if too long
            start = random.randint(0, duration - max_samples)
            signal = signal[start:start + max_samples]

        if duration < min_samples:  # Center pad with silence if too short
            pad_total = min_samples - duration
            pad_left = int(pad_total // 2)
            pad_right = int(pad_total - pad_left)
            signal = torch.nn.functional.pad(signal, (pad_left, pad_right), value=0.0)
        duration = len(signal)
        return signal, duration

    train_dynamic_items.append(process_signal)
    val_dynamic_items.append(process_signal)
    output_keys += ["signal"]

    # Define label pipeline
    @sb.utils.data_pipeline.takes("label")
    @sb.utils.data_pipeline.provides("label_encoded")
    def label_pipeline(label):
        """Defines the pipeline to process the input label ('non'/'symptomatic')."""
        # The key produced here ('label_encoded') must match
        # the 'label_key' used in train.py and the YAML.
        label_encoded = label
        yield label_encoded

    train_dynamic_items.append(label_pipeline)
    val_dynamic_items.append(label_pipeline)
    output_keys.append("label_encoded")

    if hparams["cache_encoder"]:
        speech_encoder = hparams["encoder"]

        # Do this to take advantage of auto padding
        num_layers = hparams["num_layers"]
        num_outputs = num_layers if speech_encoder.output_hidden_states else 1
        output_vars = [f"emb_{i}" for i in range(num_outputs)]

        warm_cache = hparams.get("warm_cache", False)
        file_mode = 'a' if warm_cache else 'r'

        train_cache_dir = hparams.get("train_cache_dir")
        val_cache_dir = hparams.get("val_cache_dir")

        cache_mode = f"multi_L{num_layers}" if speech_encoder.output_hidden_states else "single"
        train_cache_dir = os.path.join(train_cache_dir, cache_mode)
        val_cache_dir = os.path.join(val_cache_dir, cache_mode)

        num_versions = hparams.get("num_aug_ver", 1)

        def make_cache_emb(cache_dir, num_ver=1):
            if warm_cache:
                @CachedHDF5DynamicItem.cache(cache_dir, file_mode, num_ver)
                @sb.utils.data_pipeline.takes("id", "signal")
                @sb.utils.data_pipeline.provides(*output_vars)
                def cache_emb(id, raw_signal):
                    device = next(speech_encoder.parameters()).device
                    with torch.no_grad():
                        raw_signal = raw_signal.unsqueeze(0).to(device)
                        emb = speech_encoder(raw_signal)
                    if speech_encoder.output_hidden_states:
                        emb = tuple(x.squeeze(0).cpu() for x in emb)
                    else:
                        emb = emb.squeeze(0).cpu()
                    return emb
                return cache_emb

            # Relieve dependency of signal onto resolving other dynamic items
            @CachedHDF5DynamicItem.cache(cache_dir, file_mode, num_ver)
            @sb.utils.data_pipeline.takes("id")
            @sb.utils.data_pipeline.provides(*output_vars)
            def read_cache(id):
                warnings.warn("Cache doesn't exist for one or more data points.")
                pass # never called, expect cache hit

            return read_cache

        train_cache_emb = make_cache_emb(train_cache_dir, num_versions)
        val_cache_emb = make_cache_emb(val_cache_dir)

        train_dynamic_items.append(train_cache_emb)
        val_dynamic_items.append(val_cache_emb)

        output_keys += output_vars
        output_keys.remove("signal")  # we don't need the raw signal after caching


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

    data_dict['final_train'] = test_data['train']
    data_dict['final_test'] = test_data['val']

    data_dict['train_og'] = train_folds[-1]
    data_dict['val_og'] = val_folds[-1]

    # Define datasets.
    datasets = {}
    for dataset in data_dict:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset(
            data=data_dict[dataset],
            dynamic_items=train_dynamic_items if "train" in dataset else val_dynamic_items,
            output_keys=output_keys,
        )

    if hparams["cache_encoder"] and hparams.get("warm_cache", True):
        # for all to be augmented
        warmup_ds = ['final_train'] * num_versions

        # For all not to be augmented
        for i in range(hparams['num_fold']):
            warmup_ds.append(f"val_{i}")
        warmup_ds.append(f"final_test")

        for ds_name in warmup_ds:
            print(f"Iterating dataset {ds_name} to warm the cache.")
            ds = datasets[ds_name]
            ds.iterate_once()

        train_cache_emb.change_file_mode('r')  # change to read mode
        val_cache_emb.change_file_mode('r')

        # Adapt read only cache Dynamic Items.
        for dataset in data_dict:
            datasets[dataset] = sb.dataio.dataset.DynamicItemDataset(
                data=data_dict[dataset],
                dynamic_items=train_dynamic_items if "train" in dataset else val_dynamic_items,
                output_keys=output_keys,
            )


    return datasets




