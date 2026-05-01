import json
import os
import warnings
from typing import Any

# # Disable HDF5 file locking to allow multiple processes to access cache
# os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import soundfile as sf
import librosa
import speechbrain as sb
from speechbrain.augment.time_domain import AddReverb, AddNoise, SpeedPerturb
from speechbrain.utils.data_pipeline import CachedDynamicItem
import torch


from training.dataio.cache_dynamic_item import CachedHDF5DynamicItem


def master_dataio_prep(data_dict: dict[str, Any], hparams) -> dict[Any, Any]:
    train_dynamic_items, val_dynamic_items = [], []
    output_keys = ["id", "path", "pid"]

    sample_rate = hparams.get("sample_rate", 16000)

    max_samples = hparams.get("max_length", 10e5) * sample_rate  # default to longest
    min_samples = hparams.get("min_length", 3) * sample_rate  # default to torgo's avg lengths

    # Reader-mode jobs (cache_encoder=True, warm_cache=False) never invoke the
    # audio chain — output_keys excludes "signal" so audio_pipeline/augment are
    # dead. Skip building the augment objects: SpeedPerturb's per-speed Resample
    # kernels alone can balloon to 25 GB+ for fine-grained speed grids.
    cache_encoder = hparams["cache_encoder"]
    warm_cache = hparams.get("warm_cache", False)
    build_augment = warm_cache or not cache_encoder

    noisifier = reverb = perturbator = None
    if build_augment:
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

        perturbator = SpeedPerturb(orig_freq=sample_rate,
                                   speeds=hparams["data_params"]["speed"])

    @sb.utils.data_pipeline.takes("Participant_ID")
    @sb.utils.data_pipeline.provides("pid")
    def get_pid(pid: str):
        return pid

    train_dynamic_items.append(get_pid)
    val_dynamic_items.append(get_pid)

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("path")
    @sb.utils.data_pipeline.provides("signal", "raw_duration")
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

    @sb.utils.data_pipeline.takes("signal")
    @sb.utils.data_pipeline.provides("signal", "duration")
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

    # Val/test don't go through augment, so provide a duration that mirrors raw_duration.
    @sb.utils.data_pipeline.takes("signal")
    @sb.utils.data_pipeline.provides("signal", "duration")
    def passthrough_duration(signal):
        return signal, signal.shape[0]

    val_dynamic_items.append(passthrough_duration)

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

    # Handling too short data.
    @sb.utils.data_pipeline.takes("signal", "duration")
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

    if hparams["cache_encoder"]:
        # Audio can be split based on given boundary first. Boundary is a list of splits in seconds from metadata.
        split_by_boundary = hparams["data_params"].get("split_by_boundary", False)
        if split_by_boundary:
            # No need to handle too short audio anymore.
            train_dynamic_items.pop(-1)
            val_dynamic_items.pop(-1)

        split_fn_takes = ["signal"] if not split_by_boundary else ["signal", "boundaries", "raw_duration", "duration"]

        def split_by_boundaries(signal, boundaries, raw_duration, duration):
            # Boundaries are in seconds relative to the raw (pre-perturb) audio.
            # Map to sample indices in the (possibly speed-perturbed) signal by
            # scaling with the actual/raw length ratio.
            scale = duration / raw_duration
            chunks, prev = [], 0
            for b in boundaries:
                end = int(round(b * sample_rate * scale))
                chunks.append(signal[prev:end])
                prev = end
            return chunks

        def chunk_signal(signal):
            if len(signal) > max_samples:
                chunks = list(signal.split(int(max_samples)))
                if len(chunks) > 1 and len(chunks[-1]) < min_samples:
                    chunks = chunks[:-1]
            else:
                chunks = [signal]

            padded = []
            for c in chunks:
                if len(c) < min_samples:
                    pad_total = min_samples - len(c)
                    pad_left = int(pad_total // 2)
                    pad_right = int(pad_total - pad_left)
                    c = torch.nn.functional.pad(c, (pad_left, pad_right), value=0.0)
                padded.append(c)
            return padded

        @sb.utils.data_pipeline.takes(*split_fn_takes)
        @sb.utils.data_pipeline.provides("signals")
        def split_signal(signal, boundaries=None, raw_duration=None, duration=None):
            """
            Cut by boundaries (if any), then chunk any piece longer than max_samples
            and center-pad any piece shorter than min_samples.
            """
            if boundaries is not None:
                pieces = split_by_boundaries(signal, boundaries, raw_duration, duration)
            else:
                pieces = [signal]
            signals = []
            for p in pieces:
                signals.extend(chunk_signal(p))
            return signals

        train_dynamic_items.append(split_signal)
        val_dynamic_items.append(split_signal)

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

        cache_pool = hparams.get("cache_pool", "none")
        if speech_encoder.output_hidden_states:
            cache_mode = f"multi_L{num_layers}"
        elif cache_pool == "mean":
            cache_mode = "single_avg"
        else:
            cache_mode = "single"
        train_cache_dir = os.path.join(train_cache_dir, cache_mode)
        val_cache_dir = os.path.join(val_cache_dir, cache_mode)

        num_versions = hparams["data_params"].get("num_aug_ver", 1)

        def make_cache_emb(cache_dir, warm, num_ver=1):
            file_mode = 'a' if warm else 'r'
            # if not warm:
            #     cache_file = os.path.join(cache_dir, "cache.hdf5")
            #     if not os.path.exists(cache_file):
            #         raise RuntimeError(
            #             f"Cache not found at {cache_file}. "
            #             f"Set warm_cache: true in your config to create it."
            #         )
            if warm:
                @CachedHDF5DynamicItem.cache(cache_dir, file_mode, num_ver)
                @sb.utils.data_pipeline.takes("id", "signals")
                @sb.utils.data_pipeline.provides(*output_vars)
                def cache_emb(id, raw_signals):
                    device = next(speech_encoder.parameters()).device
                    with torch.no_grad():
                        embs = []
                        for chunk in raw_signals:
                            embs.append(speech_encoder(chunk.unsqueeze(0).to(device)))
                        if speech_encoder.output_hidden_states:
                            # T dim is always -2
                            n_layers = len(embs[0])
                            emb = []
                            for i in range(n_layers):
                                layer_chunks = [e[i].squeeze(0) for e in embs]
                                layer_emb = torch.cat(layer_chunks, dim=-2).cpu()
                                emb.append(layer_emb)
                            emb = tuple(emb)
                        else:
                            # T dim is always -2
                            emb = torch.cat([e.squeeze(0) for e in embs], dim=-2).cpu()
                            if cache_pool == "mean":
                                emb = emb.mean(dim=-2, keepdim=True)
                    return emb

                return cache_emb

            # Relieve dependency of signal onto resolving other dynamic items
            @CachedHDF5DynamicItem.cache(cache_dir, file_mode, num_ver)
            @sb.utils.data_pipeline.takes("id")
            @sb.utils.data_pipeline.provides(*output_vars)
            def read_cache(id):
                # Reached only when _is_cached(id) is False — i.e. the v{num_ver-1}
                # slot is missing. Typically means num_aug_ver in the current fork
                # exceeds the cache's actual fill depth (e.g. cache warmed under an
                # older num_aug_ver, or a stale fork after the task yaml changed).
                # The parent CachedDynamicItem would otherwise call _cache(None, id)
                # → h5py.create_dataset(key, data=None) → inscrutable TypeError.
                raise RuntimeError(
                    f"Cache miss for id={id!r} at {cache_dir} with num_ver={num_ver}. "
                    f"Expected v{num_ver - 1} to exist. Re-warm the cache "
                    f"(set warm_cache: True) or check that the trial fork's "
                    f"num_aug_ver matches what was used to build the cache."
                )

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

            all_ids = list(data_dict["all"].keys())

            # One warmup pass per version slot to write. Train writes v0..v{N-1};
            # val has a single version (v0).
            warmup_ds = [
                (dataset_all_aug, train_cache_emb, "train", v) for v in range(num_versions)
            ]
            warmup_ds.append((dataset_all_no_aug, val_cache_emb, "val", 0))

            try:
                for i, (ds, cache, kind, version) in enumerate(warmup_ds):
                    # Refine the id list every iteration so we only warm what's
                    # still missing for this version slot — this lets us resume
                    # a partially-warmed cache without redoing work.
                    uncached = cache.uncached_ids(all_ids, version)
                    if not uncached:
                        print(
                            f"Iteration {i} ({kind} v{version}): "
                            f"cache already fully warmed, skipping."
                        )
                        continue
                    print(
                        f"Iterating dataset {i} ({kind} v{version}) to warm the cache "
                        f"({len(uncached)}/{len(all_ids)} uncached)."
                    )
                    subset = sb.dataio.dataset.FilteredSortedDynamicItemDataset(ds, uncached)
                    subset.iterate_once()

                train_dynamic_items.append(train_cache_emb)
                val_dynamic_items.append(val_cache_emb)
            finally:
                # Always close so HDF5 flushes its object header to disk,
                # even on SIGTERM/exception. Prevents corrupt-cache resume bugs.
                train_cache_emb.close()
                val_cache_emb.close()

            # Update output_keys even when warming - datasets need correct structure
            output_keys += output_vars
            output_keys.remove("signal")
        else:
            train_cache_emb = make_cache_emb(train_cache_dir, False, num_versions)
            val_cache_emb = make_cache_emb(val_cache_dir, False)

            train_dynamic_items.append(train_cache_emb)
            val_dynamic_items.append(val_cache_emb)

            output_keys += output_vars
            output_keys.remove("signal")

    # TODO: if not use cache, Random Crop
    # Define datasets.
    datasets = {}
    for dataset in data_dict:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset(
            data=data_dict[dataset],
            dynamic_items=train_dynamic_items if "train" in dataset else val_dynamic_items,
            output_keys=output_keys,
        )
    return datasets

def master_dataio_prep_cross(data_dict: dict[str, Any], hparams) -> dict[Any, Any]:
    train_dynamic_items, val_dynamic_items, test_dynamic_items = [], [], []
    output_keys = ["id", "path", "pid"]

    sample_rate = hparams.get("sample_rate", 16000)
    max_samples = hparams.get("max_length", 10e5) * sample_rate
    min_samples = hparams.get("min_length", 3) * sample_rate

    # Reader-mode jobs skip the augment chain — see master_dataio_prep for why
    # SpeedPerturb's per-speed Resample kernels can otherwise eat tens of GB.
    cache_encoder = hparams["cache_encoder"]
    warm_cache = hparams.get("warm_cache", False)
    build_augment = warm_cache or not cache_encoder

    noisifier = reverb = perturbator = None
    if build_augment:
        noise_folder = hparams.get("noise_folder", None)
        noise_folder = os.path.abspath(noise_folder)
        if noise_folder is None:
            raise ValueError("Noise folder must be specified in hparams for this task.")
        noisifier = AddNoise(
            os.path.join(noise_folder, "noises.csv"),
            replacements={"noise_folder": os.path.join(noise_folder, "audio")},
            snr_low=hparams["data_params"]["snr_low"],
            snr_high=hparams["data_params"]["snr_high"],
            noise_sample_rate=sample_rate,
            clean_sample_rate=sample_rate,
        )

        rir_folder = hparams.get("rir_folder", None)
        if rir_folder is None:
            raise ValueError("RIR folder must be specified in hparams for this task.")

        reverb = AddReverb(
            os.path.join(rir_folder, "rirs.csv"),
            replacements={"rir_folder": os.path.join(rir_folder, "audio")},
            reverb_sample_rate=sample_rate,
            clean_sample_rate=sample_rate,
        )

        perturbator = SpeedPerturb(
            orig_freq=sample_rate,
            speeds=hparams["data_params"]["speed"],
        )

    @sb.utils.data_pipeline.takes("Participant_ID")
    @sb.utils.data_pipeline.provides("pid")
    def get_pid(pid: str):
        return pid

    train_dynamic_items.append(get_pid)
    val_dynamic_items.append(get_pid)
    test_dynamic_items.append(get_pid)

    @sb.utils.data_pipeline.takes("path")
    @sb.utils.data_pipeline.provides("signal", "raw_duration")
    def audio_pipeline(file_path):
        data, sr_og = sf.read(file_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if len(data) == 0:
            raise ValueError(f"Zero-length audio file: {file_path}")
        if sr_og != sample_rate:
            data = librosa.resample(data, orig_sr=sr_og, target_sr=sample_rate)
        signal = torch.from_numpy(data)
        raw_duration = len(signal)
        return signal, raw_duration

    train_dynamic_items.append(audio_pipeline)
    val_dynamic_items.append(audio_pipeline)
    test_dynamic_items.append(audio_pipeline)

    @sb.utils.data_pipeline.takes("signal")
    @sb.utils.data_pipeline.provides("signal", "duration")
    def augment(signal):
        signal = signal.unsqueeze(0)
        signal = perturbator(signal)
        signal = noisifier(signal, torch.ones(1))
        signal = reverb(signal)
        signal = signal.squeeze(0)
        duration = signal.shape[0]
        return signal, duration

    train_dynamic_items.append(augment)

    # Val/test don't go through augment, so provide a duration mirroring raw_duration.
    @sb.utils.data_pipeline.takes("signal")
    @sb.utils.data_pipeline.provides("signal", "duration")
    def passthrough_duration(signal):
        return signal, signal.shape[0]

    val_dynamic_items.append(passthrough_duration)
    test_dynamic_items.append(passthrough_duration)

    @sb.utils.data_pipeline.takes("signal", "duration")
    @sb.utils.data_pipeline.provides("signal", "duration")
    def process_signal(signal, duration):
        if duration < min_samples:
            pad_total = min_samples - duration
            pad_left = int(pad_total // 2)
            pad_right = int(pad_total - pad_left)
            signal = torch.nn.functional.pad(signal, (pad_left, pad_right), value=0.0)
        duration = len(signal)
        return signal, duration

    train_dynamic_items.append(process_signal)
    val_dynamic_items.append(process_signal)
    test_dynamic_items.append(process_signal)
    output_keys += ["signal"]

    @sb.utils.data_pipeline.takes("label")
    @sb.utils.data_pipeline.provides("label_encoded")
    def label_pipeline(label):
        if isinstance(label, list):
            label_encoded = torch.tensor(label, dtype=torch.float)
        else:
            label_encoded = label
        yield label_encoded

    train_dynamic_items.append(label_pipeline)
    val_dynamic_items.append(label_pipeline)
    test_dynamic_items.append(label_pipeline)
    output_keys.append("label_encoded")

    if hparams["cache_encoder"]:
        split_by_boundary = hparams["data_params"].get("split_by_boundary", False)
        if split_by_boundary:
            # No need for pre-padding here; chunking/padding is handled after boundary split.
            train_dynamic_items.pop(-1)
            val_dynamic_items.pop(-1)
            test_dynamic_items.pop(-1)

        split_fn_takes = ["signal"] if not split_by_boundary else ["signal", "boundaries", "raw_duration", "duration"]

        def split_by_boundaries(signal, boundaries, raw_duration, duration):
            scale = duration / raw_duration
            chunks, prev = [], 0
            for b in boundaries:
                end = int(round(b * sample_rate * scale))
                chunks.append(signal[prev:end])
                prev = end
            return chunks

        def chunk_signal(signal):
            if len(signal) > max_samples:
                chunks = list(signal.split(int(max_samples)))
                if len(chunks) > 1 and len(chunks[-1]) < min_samples:
                    chunks = chunks[:-1]
            else:
                chunks = [signal]

            padded = []
            for c in chunks:
                if len(c) < min_samples:
                    pad_total = min_samples - len(c)
                    pad_left = int(pad_total // 2)
                    pad_right = int(pad_total - pad_left)
                    c = torch.nn.functional.pad(c, (pad_left, pad_right), value=0.0)
                padded.append(c)
            return padded

        @sb.utils.data_pipeline.takes(*split_fn_takes)
        @sb.utils.data_pipeline.provides("signals")
        def split_signal(signal, boundaries=None, raw_duration=None, duration=None):
            if boundaries is not None:
                pieces = split_by_boundaries(signal, boundaries, raw_duration, duration)
            else:
                pieces = [signal]
            signals = []
            for p in pieces:
                signals.extend(chunk_signal(p))
            return signals

        train_dynamic_items.append(split_signal)
        val_dynamic_items.append(split_signal)
        test_dynamic_items.append(split_signal)

        speech_encoder = hparams["encoder"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        speech_encoder = speech_encoder.to(device)

        num_layers = hparams["num_layers"]
        num_outputs = num_layers if speech_encoder.output_hidden_states else 1
        output_vars = [f"emb_{i}" for i in range(num_outputs)]

        warm_cache = hparams.get("warm_cache", False)

        train_cache_dir_1 = hparams.get("train_cache_dir_1")
        val_cache_dir_1 = hparams.get("val_cache_dir_1")
        val_cache_dir_2 = hparams.get("val_cache_dir_2")

        cache_pool = hparams.get("cache_pool", "none")
        if speech_encoder.output_hidden_states:
            cache_mode = f"multi_L{num_layers}"
        elif cache_pool == "mean":
            cache_mode = "single_avg"
        else:
            cache_mode = "single"
               
        train_cache_dir_1 = os.path.join(train_cache_dir_1, cache_mode)
        val_cache_dir_1 = os.path.join(val_cache_dir_1, cache_mode)
        val_cache_dir_2 = os.path.join(val_cache_dir_2, cache_mode)

        num_versions = hparams["data_params"].get("num_aug_ver", 1)

        def make_cache_emb(cache_dir, warm, num_ver=1):
            file_mode = "a" if warm else "r"

            if warm:
                @CachedHDF5DynamicItem.cache(cache_dir, file_mode, num_ver)
                @sb.utils.data_pipeline.takes("id", "signals")
                @sb.utils.data_pipeline.provides(*output_vars)
                def cache_emb(id, raw_signals):
                    device = next(speech_encoder.parameters()).device
                    with torch.no_grad():
                        embs = []
                        for chunk in raw_signals:
                            embs.append(speech_encoder(chunk.unsqueeze(0).to(device)))
                        if speech_encoder.output_hidden_states:
                            emb = tuple(
                                torch.cat([e[i].squeeze(0) for e in embs], dim=-2).cpu()
                                for i in range(len(embs[0]))
                            )
                        else:
                            emb = torch.cat([e.squeeze(0) for e in embs], dim=-2).cpu()
                            if cache_pool == "mean":
                                emb = emb.mean(dim=-2, keepdim=True)
                    return emb

                return cache_emb

            @CachedHDF5DynamicItem.cache(cache_dir, file_mode, num_ver)
            @sb.utils.data_pipeline.takes("id")
            @sb.utils.data_pipeline.provides(*output_vars)
            def read_cache(id):
                raise RuntimeError(
                    f"Cache miss for id={id!r} at {cache_dir} with num_ver={num_ver}. "
                    f"Expected v{num_ver - 1} to exist. Re-warm the cache "
                    f"(set warm_cache: True) or check num_aug_ver."
                )

            return read_cache

        if warm_cache:
            train_cache_emb = make_cache_emb(train_cache_dir_1, True, num_versions)
            val_cache_emb = make_cache_emb(val_cache_dir_1, True, 1)
            test_cache_emb = make_cache_emb(val_cache_dir_2, True, 1)

            dataset_all_train_aug = sb.dataio.dataset.DynamicItemDataset(
                data=data_dict["all_train"],
                dynamic_items=train_dynamic_items + [train_cache_emb],
                output_keys=output_keys + output_vars,
            )

            dataset_all_train_no_aug = sb.dataio.dataset.DynamicItemDataset(
                data=data_dict["all_train"],
                dynamic_items=val_dynamic_items + [val_cache_emb],
                output_keys=output_keys + output_vars,
            )

            dataset_all_test_no_aug = sb.dataio.dataset.DynamicItemDataset(
                data=data_dict["all_test"],
                dynamic_items=test_dynamic_items + [test_cache_emb],
                output_keys=output_keys + output_vars,
            )

            warmup_ds = [
                (dataset_all_train_aug, train_cache_emb, list(data_dict["all_train"].keys()), "train", v)
                for v in range(num_versions)
            ]
            warmup_ds.append(
                (dataset_all_train_no_aug, val_cache_emb, list(data_dict["all_train"].keys()), "val", 0)
            )
            warmup_ds.append(
                (dataset_all_test_no_aug, test_cache_emb, list(data_dict["all_test"].keys()), "test", 0)
            )

            try:
                for i, (ds, cache, ids, kind, version) in enumerate(warmup_ds):
                    uncached = cache.uncached_ids(ids, version)
                    if not uncached:
                        print(f"Iteration {i} ({kind} v{version}): cache already fully warmed, skipping.")
                        continue
                    print(
                        f"Iterating dataset {i} ({kind} v{version}) to warm the cache "
                        f"({len(uncached)}/{len(ids)} uncached)."
                    )
                    subset = sb.dataio.dataset.FilteredSortedDynamicItemDataset(ds, uncached)
                    subset.iterate_once()

                train_dynamic_items.append(train_cache_emb)
                val_dynamic_items.append(val_cache_emb)
                test_dynamic_items.append(test_cache_emb)
            finally:
                train_cache_emb.close()
                val_cache_emb.close()
                test_cache_emb.close()

            output_keys += output_vars
            output_keys.remove("signal")

        else:
            train_cache_emb = make_cache_emb(train_cache_dir_1, False, num_versions)
            val_cache_emb = make_cache_emb(val_cache_dir_1, False, 1)
            test_cache_emb = make_cache_emb(val_cache_dir_2, False, 1)

            train_dynamic_items.append(train_cache_emb)
            val_dynamic_items.append(val_cache_emb)
            test_dynamic_items.append(test_cache_emb)

            output_keys += output_vars
            output_keys.remove("signal")

    datasets = {
        "train": sb.dataio.dataset.DynamicItemDataset(
            data=data_dict["train"],
            dynamic_items=train_dynamic_items,
            output_keys=output_keys,
        ),
        "val": sb.dataio.dataset.DynamicItemDataset(
            data=data_dict["val"],
            dynamic_items=val_dynamic_items,
            output_keys=output_keys,
        ),
        "test": sb.dataio.dataset.DynamicItemDataset(
            data=data_dict["test"],
            dynamic_items=test_dynamic_items,
            output_keys=output_keys,
        ),
    }

    return datasets


def master_dataio_prep_cross_category(data_dict: dict[str, Any], hparams) -> dict[Any, Any]:
    """Category-cross dataio with per-dataset precomputed cache reads.

    Expects each manifest item to include:
      - id (unique manifest key)
      - cache_uid (key used inside per-dataset cache)
      - source_dataset
      - num_aug_ver (train only; per-dataset augmentation versions)
      - label
      - path

    Cache roots are read from:
      - train_cache_dir_1 (train split root)
      - val_cache_dir_1 (val/test split root)
    with final layout:
      <root>/<dataset>/<model_name>/<split>/<cache_mode>/cache.hdf5
    """
    output_keys = ["id", "path"]

    @sb.utils.data_pipeline.takes("label")
    @sb.utils.data_pipeline.provides("label_encoded")
    def label_pipeline(label):
        if isinstance(label, list):
            label_encoded = torch.tensor(label, dtype=torch.float)
        else:
            label_encoded = label
        yield label_encoded

    output_keys.append("label_encoded")

    speech_encoder = hparams["encoder"]
    num_layers = hparams["num_layers"]
    num_outputs = num_layers if speech_encoder.output_hidden_states else 1
    output_vars = [f"emb_{i}" for i in range(num_outputs)]
    output_keys += output_vars

    cache_pool = hparams.get("cache_pool", "none")
    if speech_encoder.output_hidden_states:
        cache_mode = f"multi_L{num_layers}"
    elif cache_pool == "mean":
        cache_mode = "single_avg"
    else:
        cache_mode = "single"

    model_name = hparams["model_name"]
    train_root = hparams.get("train_cache_dir_1")
    eval_root = hparams.get("val_cache_dir_1")
    if train_root is None or eval_root is None:
        raise ValueError(
            "Category-cross requires train_cache_dir_1 and val_cache_dir_1 roots in hparams."
        )

    def make_read_cache(cache_dir, num_ver):
        @CachedHDF5DynamicItem.cache(cache_dir, "r", num_ver)
        @sb.utils.data_pipeline.takes("id")
        @sb.utils.data_pipeline.provides(*output_vars)
        def read_cache(id):
            raise RuntimeError(
                f"Cache miss for id={id!r} at {cache_dir} with num_ver={num_ver}. "
                f"Expected v{num_ver - 1} to exist."
            )

        return read_cache

    # Build train readers keyed by (dataset, num_versions) so category mixes can
    # keep per-dataset augmentation settings while sharing the same cache roots.
    train_versions_by_dataset = {}
    for item in data_dict["train"].values():
        ds = item.get("source_dataset")
        if not ds:
            continue
        num_ver = int(item.get("num_aug_ver", 1))
        train_versions_by_dataset.setdefault(ds, set()).add(max(1, num_ver))
    if not train_versions_by_dataset:
        raise ValueError("No source_dataset values found in category train manifest.")

    eval_datasets = {
        item.get("source_dataset")
        for item in data_dict["val"].values()
    } | {
        item.get("source_dataset")
        for item in data_dict["test"].values()
    }
    eval_datasets = {d for d in eval_datasets if d}
    if not eval_datasets:
        raise ValueError("No source_dataset values found in category val/test manifests.")

    train_readers = {}
    eval_readers = {}
    for ds in sorted(eval_datasets):
        eval_dir = os.path.join(eval_root, ds, model_name, "val", cache_mode)
        eval_readers[ds] = make_read_cache(eval_dir, 1)

    for ds in sorted(train_versions_by_dataset):
        train_dir = os.path.join(train_root, ds, model_name, "train", cache_mode)
        versions = sorted(train_versions_by_dataset[ds])
        for num_ver in versions:
            train_readers[(ds, num_ver)] = make_read_cache(train_dir, num_ver)

    @sb.utils.data_pipeline.takes("cache_uid", "source_dataset", "num_aug_ver")
    @sb.utils.data_pipeline.provides(*output_vars)
    def train_emb_pipeline(cache_uid, source_dataset, num_aug_ver):
        num_ver = max(1, int(num_aug_ver))
        reader_key = (source_dataset, num_ver)
        if reader_key not in train_readers:
            raise KeyError(
                "No train cache reader configured for "
                f"source_dataset={source_dataset!r}, num_aug_ver={num_ver}."
            )
        return train_readers[reader_key](cache_uid)

    @sb.utils.data_pipeline.takes("cache_uid", "source_dataset")
    @sb.utils.data_pipeline.provides(*output_vars)
    def eval_emb_pipeline(cache_uid, source_dataset):
        if source_dataset not in eval_readers:
            raise KeyError(
                f"No eval cache reader configured for source_dataset={source_dataset!r}"
            )
        return eval_readers[source_dataset](cache_uid)

    datasets = {
        "train": sb.dataio.dataset.DynamicItemDataset(
            data=data_dict["train"],
            dynamic_items=[label_pipeline, train_emb_pipeline],
            output_keys=output_keys,
        ),
        "val": sb.dataio.dataset.DynamicItemDataset(
            data=data_dict["val"],
            dynamic_items=[label_pipeline, eval_emb_pipeline],
            output_keys=output_keys,
        ),
        "test": sb.dataio.dataset.DynamicItemDataset(
            data=data_dict["test"],
            dynamic_items=[label_pipeline, eval_emb_pipeline],
            output_keys=output_keys,
        ),
    }
    return datasets