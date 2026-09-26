"""Batched cache warm must write what the serial warm writes (sdx/warm.py).

``warm_batch_size > 1`` pads a batch of chunks into one ``forward`` call. It
is a throughput knob, so the cached embeddings must match the one-chunk-at-a-
time path. A small random WavLM stands in for the padding encoders and a
fake encoder drives the OOM halving. No data or checkpoints.

Before setting ``warm_batch_size`` for a new encoder, call
``assert_batched_matches_serial`` with the real wrapper and a few recordings
of real audio whose chunks differ in length.
"""

import h5py
import pytest
import torch
from torch import nn

from sdx.warm import (
    _combine_chunk_embs,
    _compute_emb_from_signals,
    _encode_chunk_batch_oom_safe,
    _read_chunk,
)


def assert_batched_matches_serial(encoder, recordings, tmp_path, *,
                                  cache_pool="none", atol=1e-5):
    """Warm ``recordings`` (one list of 1-D chunks per uid) both ways and compare.

    All chunks go through a single padded batch, as ``_drive_warm_batched``
    does; each uid is then reassembled and compared with the serial payload.
    """
    ohs = encoder.output_hidden_states
    buffer = [(f"u{u}/c{c}", sig)
              for u, chunks in enumerate(recordings)
              for c, sig in enumerate(chunks)]
    with h5py.File(tmp_path / "chunks.hdf5", "w") as tmp:
        _encode_chunk_batch_oom_safe(buffer, encoder, tmp, ohs,
                                     torch.device("cpu"))
        for u, chunks in enumerate(recordings):
            serial = _compute_emb_from_signals(chunks, encoder, cache_pool)
            batched = _combine_chunk_embs(
                [_read_chunk(tmp, f"u{u}/c{c}", ohs) for c in range(len(chunks))],
                ohs, cache_pool)
            pairs = zip(batched, serial) if ohs else [(batched, serial)]
            for b, s in pairs:
                assert b.shape == s.shape
                torch.testing.assert_close(b, s, atol=atol, rtol=0)


# -- a real transformers WavLM behind model/wavlm.py ---------------------------

def _tiny_wavlm_encoder(monkeypatch, output_hidden_states):
    from transformers import Wav2Vec2FeatureExtractor, WavLMConfig, WavLMModel

    import model.wavlm as mw

    # Same conv / norm layout as WavLM-Large (feat_extract_norm="layer").
    cfg = WavLMConfig(
        hidden_size=32, num_hidden_layers=4, num_attention_heads=4,
        intermediate_size=64, do_stable_layer_norm=True,
        conv_dim=(16, 16), conv_stride=(5, 2), conv_kernel=(10, 3),
        num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=4,
        feat_extract_norm="layer", num_buckets=32, max_bucket_distance=80)
    torch.manual_seed(0)
    model = WavLMModel(cfg).eval()
    # WavLM-Large's preprocessor_config.json.
    processor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0,
        do_normalize=True, return_attention_mask=True)
    monkeypatch.setattr(mw.WavLMModel, "from_pretrained",
                        staticmethod(lambda *_a, **_k: model))
    monkeypatch.setattr(mw.AutoFeatureExtractor, "from_pretrained",
                        staticmethod(lambda *_a, **_k: processor))
    return mw.WavLM("tiny", True, output_hidden_states, 16000).eval()


def _recordings():
    torch.manual_seed(1)
    # Ragged chunk lengths, so the batch really is padded.
    return [[torch.randn(n) for n in (4000, 2600)],
            [torch.randn(3300)],
            [torch.randn(n) for n in (4000, 1700, 3900)]]


@pytest.mark.parametrize("output_hidden_states", [False, True])
@pytest.mark.parametrize("cache_pool", ["mean", "none"])
def test_wavlm_batched_warm_matches_serial(monkeypatch, tmp_path,
                                           output_hidden_states, cache_pool):
    from model.wavlm_sdpa_patch import (
        disable_wavlm_sdpa,
        enable_wavlm_sdpa,
        wavlm_sdpa_enabled,
    )

    was = wavlm_sdpa_enabled()
    try:
        enc = _tiny_wavlm_encoder(monkeypatch, output_hidden_states)
        assert_batched_matches_serial(enc, _recordings(), tmp_path,
                                      cache_pool=cache_pool)
    finally:
        (enable_wavlm_sdpa if was else disable_wavlm_sdpa)()


# -- OOM halving ---------------------------------------------------------------

class _OOMAbove(nn.Module):
    """Identity encoder, one frame per sample, that 'OOMs' above ``max_batch``."""

    output_hidden_states = False

    def __init__(self, max_batch, message="CUDA out of memory. Tried to allocate 2 GiB"):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))  # gives the warmer a device
        self.max_batch = max_batch
        self.message = message
        self.batch_sizes = []

    def forward(self, x, lengths=None):
        self.batch_sizes.append(x.shape[0])
        if x.shape[0] > self.max_batch:
            raise RuntimeError(self.message)
        return x.unsqueeze(-1)

    def feature_lengths(self, sample_lengths):
        return torch.as_tensor(sample_lengths).long()


def _chunks(n):
    return [(f"u{i}/c0", torch.arange(1.0, 3.0 + i)) for i in range(n)]


def test_oom_halves_the_batch_until_it_fits(tmp_path):
    enc, buffer = _OOMAbove(max_batch=2), _chunks(5)
    with h5py.File(tmp_path / "chunks.hdf5", "w") as tmp:
        _encode_chunk_batch_oom_safe(buffer, enc, tmp, False, torch.device("cpu"))
        for key, sig in buffer:
            assert torch.equal(_read_chunk(tmp, key, False), sig.unsqueeze(-1))
    # 5 -> OOM -> 2 + 3; 3 -> OOM -> 1 + 2.
    assert enc.batch_sizes == [5, 2, 3, 1, 2]


def test_other_errors_are_not_retried(tmp_path):
    enc = _OOMAbove(max_batch=2, message="shape mismatch")
    with h5py.File(tmp_path / "chunks.hdf5", "w") as tmp, \
            pytest.raises(RuntimeError, match="shape mismatch"):
        _encode_chunk_batch_oom_safe(_chunks(5), enc, tmp, False,
                                     torch.device("cpu"))
    assert enc.batch_sizes == [5]


def test_a_single_chunk_oom_propagates(tmp_path):
    enc = _OOMAbove(max_batch=0)
    with h5py.File(tmp_path / "chunks.hdf5", "w") as tmp, \
            pytest.raises(RuntimeError, match="out of memory"):
        _encode_chunk_batch_oom_safe(_chunks(2), enc, tmp, False,
                                     torch.device("cpu"))
    assert enc.batch_sizes == [2, 1]
