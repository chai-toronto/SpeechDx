import json
import os
import warnings
from typing import Any

import soundfile as sf
import librosa
import speechbrain as sb
from speechbrain.augment.time_domain import AddReverb, AddNoise, SpeedPerturb
from speechbrain.utils.data_pipeline import CachedDynamicItem
import torch


from training.dataio.cache_dynamic_item import CachedHDF5DynamicItem


def master_dataio_prep(data_dict: dict[str, Any], hparams) -> dict[Any, Any]:
    train_dynamic_items, val_dynamic_items = [], []
    output_keys = ["id", "path"]

    sample_rate = hparams.get("sample_rate", 16000)

    max_samples = hparams.get("max_length", 10e5) * sample_rate  # default to longest
    min_samples = hparams.get("min_length", 3) * sample_rate  # default to torgo's avg lengths

    noise_folder = hparams.get("noise_folder", None)
    noise_folder = os.path.abspath(noise_folder)

    if noise_folder is None:
        raise ValueError("Noise folder must be specified in hparams for this task.")

    noisifier = AddNoise(os.path.join(noise_folder, 'noises.csv'),
                         replacements={'noise_folder': os.path.join(noise_folder, 'audio')},
                         snr_low=hparams["data_params"]["snr_low"],
                         snr_high=hparams["data_params"]["snr_high"],
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
        """Defines the pipeline to process the input label."""
        if isinstance(label, list):
            label_encoded = torch.tensor(label, dtype=torch.float)
        else:
            label_encoded = label
        yield label_encoded

    train_dynamic_items.append(label_pipeline)
    val_dynamic_items.append(label_pipeline)
    output_keys.append("label_encoded")

    if hparams["cache_encoder"]:
        speech_encoder = hparams["encoder"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        speech_encoder = speech_encoder.to(device)

        # Do this to take advantage of auto padding
        num_layers = hparams["num_layers"]
        num_outputs = num_layers if speech_encoder.output_hidden_states else 1
        output_vars = [f"emb_{i}" for i in range(num_outputs)]

        warm_cache = hparams.get("warm_cache", False)

        train_cache_dir = hparams.get("train_cache_dir")
        val_cache_dir = hparams.get("val_cache_dir")

        cache_mode = f"multi_L{num_layers}" if speech_encoder.output_hidden_states else "single"
        train_cache_dir = os.path.join(train_cache_dir, cache_mode)
        val_cache_dir = os.path.join(val_cache_dir, cache_mode)

        num_versions = hparams["data_params"].get("num_aug_ver", 1)

        def make_cache_emb(cache_dir, warm, num_ver=1):
            file_mode = 'a' if warm else 'r'
            if warm:
                @CachedHDF5DynamicItem.cache(cache_dir, file_mode, num_ver)
                @sb.utils.data_pipeline.takes("id", "signal")
                @sb.utils.data_pipeline.provides(*output_vars)
                def cache_emb(id, raw_signal):
                    device = next(speech_encoder.parameters()).device
                    with torch.no_grad():
                        if len(raw_signal) > max_samples:
                            chunks = list(raw_signal.split(int(max_samples)))
                            if len(chunks) > 1 and len(chunks[-1]) < min_samples:
                                chunks = chunks[:-1]
                            embs = []
                            for chunk in chunks:
                                embs.append(speech_encoder(chunk.unsqueeze(0).to(device)))
                            if speech_encoder.output_hidden_states:
                                # T dim is always -2
                                emb = tuple(
                                    torch.cat([e[i].squeeze(0) for e in embs], dim=-2).cpu()
                                    for i in range(len(embs[0]))
                                )
                            else:
                                # T dim is always -2
                                emb = torch.cat([e.squeeze(0) for e in embs], dim=-2).cpu()
                        else:
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
                pass  # never called, expect cache hit

            return read_cache

        if warm_cache:
            train_cache_emb = make_cache_emb(train_cache_dir, warm_cache, num_versions)
            val_cache_emb = make_cache_emb(val_cache_dir, warm_cache)

            dataset_all_aug = sb.dataio.dataset.DynamicItemDataset(
                data=data_dict["all"],
                dynamic_items=train_dynamic_items + [train_cache_emb],
                output_keys=output_keys + output_vars,
            )

            dataset_all_no_aug = sb.dataio.dataset.DynamicItemDataset(
                data=data_dict["all"],
                dynamic_items=val_dynamic_items + [val_cache_emb],
                output_keys=output_keys + output_vars,
            )

            warmup_ds = [dataset_all_aug] * num_versions
            warmup_ds.append(dataset_all_no_aug)

            for i, ds in enumerate(warmup_ds):
                print(f"Iterating dataset {i} to warm the cache.")
                ds.iterate_once()

            train_cache_emb.close()
            val_cache_emb.close()

        train_cache_emb = make_cache_emb(train_cache_dir, False, num_versions)
        val_cache_emb = make_cache_emb(val_cache_dir, False)

        train_dynamic_items.append(train_cache_emb)
        val_dynamic_items.append(val_cache_emb)

        output_keys += output_vars
        output_keys.remove("signal")

    # Define datasets.
    datasets = {}
    for dataset in data_dict:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset(
            data=data_dict[dataset],
            dynamic_items=train_dynamic_items if "train" in dataset else val_dynamic_items,
            output_keys=output_keys,
        )
    return datasets
