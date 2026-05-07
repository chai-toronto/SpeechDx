"""Audio-pipeline factories used by both the warmer and the read path.

Each ``make_*`` builds a SpeechBrain ``DynamicItem``. Logic is salvaged
from ``training/dataio/preprocessing.py:62-205`` (originally inline closures
over ``master_dataio_prep`` parameters), exposed here as standalone factory
functions so warm and read code paths can share them without dragging the
encoder into the read path.

This module imports nothing from ``model.*``.

Pipeline-key registry
---------------------
The names below are the single source of truth for keys speechbrain
manages on a data row — either auto-injected (``ID``) or produced by one
of the ``make_*`` factories (``SIGNAL`` / ``RAW_DURATION`` / ``DURATION`` /
``SIGNALS`` / ``LABEL_ENCODED``). Static manifest rows must NOT carry any
of these keys, since speechbrain's ``compute_outputs`` reads
``data[key]`` *before* dynamic-provider outputs and a stray static value
(e.g. ``duration: NaN`` from a pandas concat) silently shadows the
provider. Callers strip these via ``PIPELINE_KEYS`` (see
``sdx.warm_cross_cat._per_dataset_subset``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import librosa
import soundfile as sf
import speechbrain as sb
import torch
from speechbrain.augment.time_domain import AddNoise, AddReverb, SpeedPerturb


# Speechbrain reserves ``id`` for the outer dict key.
ID = "id"
# Provided by make_audio_pipeline / make_augment / make_passthrough_duration.
SIGNAL = "signal"
RAW_DURATION = "raw_duration"
DURATION = "duration"
# Provided by make_split_signal.
SIGNALS = "signals"
# Provided by make_label_pipeline.
LABEL_ENCODED = "label_encoded"

PIPELINE_KEYS: frozenset[str] = frozenset({
    ID, SIGNAL, RAW_DURATION, DURATION, SIGNALS, LABEL_ENCODED,
})


def make_audio_pipeline(sample_rate: int):
    @sb.utils.data_pipeline.takes("path")
    @sb.utils.data_pipeline.provides(SIGNAL, RAW_DURATION)
    def audio_pipeline(file_path):
        data, sr_og = sf.read(file_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if len(data) == 0:
            raise ValueError(f"Zero-length audio file: {file_path}")
        if sr_og != sample_rate:
            data = librosa.resample(data, orig_sr=sr_og, target_sr=sample_rate)
        signal = torch.from_numpy(data)
        return signal, len(signal)

    return audio_pipeline


def build_augmenters(noise_folder: str, rir_folder: str, sample_rate: int,
                     snr_low: float, snr_high: float,
                     speeds: list[int]) -> tuple:
    """Construct the ``(noisifier, reverb, perturbator)`` tuple used by ``make_augment``."""
    noisifier = AddNoise(
        str(Path(noise_folder) / "noises.csv"),
        replacements={"noise_folder": str(Path(noise_folder) / "audio")},
        snr_low=snr_low,
        snr_high=snr_high,
        noise_sample_rate=sample_rate,
        clean_sample_rate=sample_rate,
    )
    reverb = AddReverb(
        str(Path(rir_folder) / "rirs.csv"),
        replacements={"rir_folder": str(Path(rir_folder) / "audio")},
        reverb_sample_rate=sample_rate,
        clean_sample_rate=sample_rate,
    )
    perturbator = SpeedPerturb(orig_freq=sample_rate, speeds=speeds)
    return noisifier, reverb, perturbator


def make_augment(noisifier, reverb, perturbator):
    @sb.utils.data_pipeline.takes(SIGNAL)
    @sb.utils.data_pipeline.provides(SIGNAL, DURATION)
    def augment(signal):
        signal = signal.unsqueeze(0)
        signal = perturbator(signal)
        signal = noisifier(signal, torch.ones(1))
        signal = reverb(signal)
        signal = signal.squeeze(0)
        return signal, signal.shape[0]

    return augment


def make_passthrough_duration():
    @sb.utils.data_pipeline.takes(SIGNAL)
    @sb.utils.data_pipeline.provides(SIGNAL, DURATION)
    def passthrough_duration(signal):
        return signal, signal.shape[0]

    return passthrough_duration


def make_label_pipeline():
    @sb.utils.data_pipeline.takes("label")
    @sb.utils.data_pipeline.provides(LABEL_ENCODED)
    def label_pipeline(label):
        if isinstance(label, list):
            label_encoded = torch.tensor(label, dtype=torch.float)
        else:
            label_encoded = label
        yield label_encoded

    return label_pipeline


def make_process_signal(min_samples: int):
    @sb.utils.data_pipeline.takes(SIGNAL, DURATION)
    @sb.utils.data_pipeline.provides(SIGNAL, DURATION)
    def process_signal(signal, duration):
        if duration < min_samples:
            pad_total = min_samples - duration
            pad_left = int(pad_total // 2)
            pad_right = int(pad_total - pad_left)
            signal = torch.nn.functional.pad(signal, (pad_left, pad_right), value=0.0)
        return signal, len(signal)

    return process_signal


def _split_by_boundaries(signal, boundaries: Iterable[float],
                         raw_duration: int, duration: int, sample_rate: int):
    """Split the (possibly speed-perturbed) signal at boundary timestamps.

    Boundaries are in seconds relative to the raw (pre-perturb) audio. Map
    to sample indices in the perturbed signal by scaling with the actual /
    raw length ratio.
    """
    scale = duration / raw_duration
    chunks, prev = [], 0
    for b in boundaries:
        end = int(round(b * sample_rate * scale))
        chunks.append(signal[prev:end])
        prev = end
    return chunks


def _chunk_signal(signal, max_samples: int, min_samples: int):
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


def make_split_signal(max_samples: int, min_samples: int, sample_rate: int,
                      *, split_by_boundary: bool):
    """Build the ``split_signal`` DynamicItem.

    When ``split_by_boundary`` is True the function consumes both the audio
    signal and the per-utterance boundary list; otherwise it consumes only
    the signal and chunks at ``max_samples`` length.
    """
    if split_by_boundary:
        @sb.utils.data_pipeline.takes(SIGNAL, "boundaries", RAW_DURATION, DURATION)
        @sb.utils.data_pipeline.provides(SIGNALS)
        def split_signal(signal, boundaries, raw_duration, duration):
            # Cross tasks set split_by_boundary=true at the task level for the
            # boundary-bearing source dataset; the partner dataset's manifest
            # rows carry boundaries=None. Fall back to plain length-chunking
            # for those rows so warm-cross doesn't TypeError on test uids.
            if boundaries is None:
                return _chunk_signal(signal, max_samples, min_samples)
            pieces = _split_by_boundaries(signal, boundaries, raw_duration, duration, sample_rate)
            signals = []
            for p in pieces:
                signals.extend(_chunk_signal(p, max_samples, min_samples))
            return signals

    else:
        @sb.utils.data_pipeline.takes(SIGNAL)
        @sb.utils.data_pipeline.provides(SIGNALS)
        def split_signal(signal):
            return _chunk_signal(signal, max_samples, min_samples)

    return split_signal
