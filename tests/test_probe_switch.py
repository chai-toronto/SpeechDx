"""``--probe`` switching: default readout unchanged, ASP selects the T x D cache.

Composes real configs through the stub encoder, so no weights are loaded.

    python -m pytest tests/test_probe_switch.py
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from sdx.config import compose_config, probe_cache_pool, resolve_probe
from sdx.dataio.cache import CachedHDF5DynamicItem
from sdx.dataio.read import _cache_mode_for, _read_max_frames
from sdx.orchestrator import get_output_folder


@pytest.fixture(autouse=True)
def _repo_cwd(monkeypatch):
    # main.yaml paths are relative to the repo root.
    from pathlib import Path
    monkeypatch.chdir(Path(__file__).resolve().parent.parent)


def test_default_probe_is_mean_pool_linear():
    assert resolve_probe(None) == ("AvgTProbe", "Probe.yaml")
    assert resolve_probe("AvgTProbe") == ("AvgTProbe", "Probe.yaml")
    hp = compose_config("T13", "whisper", mode="read")
    assert hp["cache_pool"] == "mean"
    assert _cache_mode_for(hp) == "single_avg"
    assert type(hp["probe_params"]["probe"]).__name__ == "LinearProbe"
    assert _read_max_frames(hp) == -1
    assert "whisper-AvgTProbe-run1" in hp["output_folder"]


def test_asp_selects_temporal_cache_and_its_own_folder():
    assert probe_cache_pool("ASP.yaml") == "none"
    assert probe_cache_pool("Probe.yaml") is None
    hp = compose_config("T13", "whisper", probe="ASP", mode="read")
    assert hp["cache_pool"] == "none"
    assert _cache_mode_for(hp) == "single"
    assert type(hp["probe_params"]["probe"]).__name__ == "TemporalProbe"
    assert "whisper-ASP-run1" in hp["output_folder"]
    assert str(get_output_folder("T13", "whisper", "run1", probe="ASP")).endswith(
        "whisper-ASP-run1")
    assert get_output_folder("T13", "whisper", "run1") == get_output_folder(
        "T13", "whisper", "run1", probe="AvgTProbe")


def test_unknown_probe_fails_loudly():
    with pytest.raises(FileNotFoundError):
        resolve_probe("NoSuchProbe")


def test_crop_override_reaches_the_reader():
    hp = compose_config("T13", "whisper", probe="ASP", mode="read",
                        overrides={"cache_max_frames": 30000})
    assert _read_max_frames(hp) == 30000


def test_read_time_crop_only_touches_time_axis(tmp_path):
    with h5py.File(tmp_path / "cache.hdf5", "w") as f:
        f["long/v0"] = np.random.rand(50, 4).astype("float32")
        f["short/v0"] = np.random.rand(10, 4).astype("float32")
        f["pooled/v0"] = np.random.rand(4).astype("float32")

    def shapes(max_frames):
        r = CachedHDF5DynamicItem(tmp_path, "r", 1, max_frames, takes=["id"],
                                  func=lambda i: i, provides=["emb_0"])
        try:
            return [r._load(u).shape for u in ("long", "short", "pooled")]
        finally:
            r.close()

    assert shapes(0) == [(50, 4), (10, 4), (4,)]
    assert shapes(-1) == [(50, 4), (10, 4), (4,)]
    assert shapes(20) == [(20, 4), (10, 4), (4,)]


def test_asp_probe_handles_padded_batch():
    from model.pool import ASP
    from model.probe import TemporalProbe

    probe = TemporalProbe(64, 3, ASP(64))
    x = torch.randn(4, 37, 64)
    lengths = torch.tensor([1.0, 0.7, 0.4, 0.1])
    assert probe(x, lengths).shape == (4, 3)
