"""Boundary-cache tests — layout, identity, chunk round trip."""

import numpy as np
import pytest
import torch

from sdx.lora.cache import BoundaryCache, cache_mode, identity_hash


def _meta(**over):
    m = {
        "encoder": "wavlm", "source": "microsoft/wavlm-large",
        "revision": "abc123", "cut_index": 21, "num_blocks": 24, "width": 1024,
        "dataset": "coswara", "sample_rate": 16000, "max_length": 300,
        "min_length": 1, "split_by_boundary": False, "num_aug_ver": 3,
        "snr_low": 0, "snr_high": 15, "speed": "range(90,110)",
        "dtype": "float16", "rank": 8, "budget": 100000,
        "target_paths": ["a.q_proj", "a.v_proj"],
    }
    m.update(over)
    return m


def test_cache_mode_carries_the_cut():
    """Two different cut layers must never share a directory."""
    assert cache_mode(21) == "boundary_L21"
    assert cache_mode(8) != cache_mode(21)
    # And it must not collide with the ASP campaign's final-layer caches.
    assert cache_mode(21) not in ("single", "single_avg", "multi_L24")


def test_chunk_round_trip_preserves_boundaries(tmp_path):
    chunks = [torch.randn(5, 16), torch.randn(3, 16), torch.randn(7, 16)]
    p = tmp_path / "cache.hdf5"
    with BoundaryCache(p, "a", _meta(width=16)) as c:
        c.write("uid1", 0, chunks)
    with BoundaryCache(p) as c:
        got = c.read("uid1", 0)
    assert [g.shape for g in got] == [(5, 16), (3, 16), (7, 16)]
    for a, b in zip(chunks, got):
        assert torch.allclose(a.half().float(), b.float(), atol=1e-3)


def test_single_chunk_recording(tmp_path):
    p = tmp_path / "c.hdf5"
    with BoundaryCache(p, "a", _meta(width=8)) as c:
        c.write("u", 0, [torch.randn(11, 8)])
    with BoundaryCache(p) as c:
        assert len(c.read("u", 0)) == 1


def test_versions_are_independent(tmp_path):
    p = tmp_path / "c.hdf5"
    a, b = torch.zeros(4, 8), torch.ones(4, 8)
    with BoundaryCache(p, "a", _meta(width=8)) as c:
        c.write("u", 0, [a])
        c.write("u", 1, [b])
    with BoundaryCache(p) as c:
        assert c.read("u", 0)[0].sum() == 0
        assert c.read("u", 1)[0].sum() == 32
        assert c.num_versions("u") == 2


def test_uncached_reports_only_missing(tmp_path):
    p = tmp_path / "c.hdf5"
    with BoundaryCache(p, "a", _meta(width=8)) as c:
        c.write("a", 0, [torch.randn(2, 8)])
        assert c.uncached(["a", "b", "c"], 0) == ["b", "c"]
        assert c.uncached(["a"], 1) == ["a"]  # version 1 not written


def test_second_read_is_identical(tmp_path):
    """Only the frozen prefix is cached; the tail never re-runs it."""
    p = tmp_path / "c.hdf5"
    with BoundaryCache(p, "a", _meta(width=8)) as c:
        c.write("u", 0, [torch.randn(6, 8), torch.randn(2, 8)])
    with BoundaryCache(p) as c:
        assert all(torch.equal(x, y) for x, y in zip(c.read("u", 0), c.read("u", 0)))


def test_identity_hash_tracks_value_determining_fields():
    base = _meta()
    assert identity_hash(base) == identity_hash(_meta())
    for field, value in [("cut_index", 20), ("revision", "zzz"),
                         ("min_length", 30),
                         ("snr_high", 20), ("dtype", "float32"),
                         ("source", "other/model")]:
        assert identity_hash(_meta(**{field: value})) != identity_hash(base), field


def test_identity_hash_ignores_bookkeeping():
    """Rank/budget do not change the stored activations, so they must not
    fragment the cache — the boundary depends on the CUT, not the adapter."""
    assert identity_hash(_meta(rank=16)) == identity_hash(_meta())
    assert identity_hash(_meta(target_paths=["x"])) == identity_hash(_meta())


def test_identity_hash_ignores_num_aug_ver():
    """num_aug_ver is a COUNT of versions written, not a property of any
    stored value. The reader draws from its own task's num_version
    (dataio/cache.py:249), so a 1-version task always reads v0 and a
    3-version cache is a valid superset for it. c19sounds relies on this:
    T19/T21 write 3 versions, T20/T22/T23 need 1, one shared cache.

    Treating it as value-determining wrongly forced a second ~2 TiB cache."""
    assert identity_hash(_meta(num_aug_ver=1)) == identity_hash(_meta(num_aug_ver=3))
    # ...but the augmentation PARAMETERS still fragment it, since they change
    # what a given version contains.
    assert identity_hash(_meta(snr_low=5)) != identity_hash(_meta())
    assert identity_hash(_meta(speed="range(95,105)")) != identity_hash(_meta())


def test_covers_is_at_least_not_equality(tmp_path):
    """A cache may hold more versions than a reader needs, and different
    counts for different uids."""
    p = tmp_path / "c.hdf5"
    with BoundaryCache(p, "a", _meta(width=8)) as c:
        for v in range(3):
            c.write("rich", v, [torch.randn(2, 8)])
        c.write("lean", 0, [torch.randn(2, 8)])
    with BoundaryCache(p) as c:
        assert c.covers(["rich", "lean"], 1) == (True, 0)   # both have v0
        ok, missing = c.covers(["rich", "lean"], 3)
        assert not ok and missing == 2                      # lean lacks v1,v2
        assert c.covers(["rich"], 3) == (True, 0)


def test_reopening_with_a_different_identity_is_refused(tmp_path):
    p = tmp_path / "c.hdf5"
    with BoundaryCache(p, "a", _meta(width=8)) as c:
        c.write("u", 0, [torch.randn(2, 8)])
    with pytest.raises(ValueError, match="different identity"):
        with BoundaryCache(p, "a", _meta(width=8, cut_index=20)) as c:
            c.write("u2", 0, [torch.randn(2, 8)])


def test_corrupt_chunk_lengths_are_detected(tmp_path):
    import h5py
    p = tmp_path / "c.hdf5"
    with BoundaryCache(p, "a", _meta(width=8)) as c:
        c.write("u", 0, [torch.randn(4, 8), torch.randn(4, 8)])
    with h5py.File(p, "a") as f:
        f["u/v0"].attrs["chunk_lengths"] = np.array([4, 99], dtype=np.int32)
    with BoundaryCache(p) as c:
        with pytest.raises(ValueError, match="corrupt"):
            c.read("u", 0)


def test_missing_uid_raises_rather_than_returning_empty(tmp_path):
    """A silently-skipped uid would train on a smaller set than reported."""
    p = tmp_path / "c.hdf5"
    with BoundaryCache(p, "a", _meta(width=8)) as c:
        c.write("u", 0, [torch.randn(2, 8)])
    with BoundaryCache(p) as c:
        with pytest.raises(KeyError):
            c.read("absent", 0)


def test_writing_no_chunks_raises(tmp_path):
    with BoundaryCache(tmp_path / "c.hdf5", "a", _meta(width=8)) as c:
        with pytest.raises(ValueError):
            c.write("u", 0, [])


def test_dataset_draws_a_version_per_read(tmp_path):
    from sdx.lora.train import BoundaryDataset

    p = tmp_path / "c.hdf5"
    with BoundaryCache(p, "a", _meta(width=4)) as c:
        for v in range(3):
            c.write("u", v, [torch.full((2, 4), float(v))])
    train = BoundaryDataset(p, ["u"], {"u": 1.0}, num_versions=3, train=True)
    seen = {train[0][1][0][0, 0].item() for _ in range(40)}
    assert len(seen) > 1, "train should sample across augmentation versions"

    val = BoundaryDataset(p, ["u"], {"u": 1.0}, num_versions=3, train=False)
    assert {val[0][1][0][0, 0].item() for _ in range(5)} == {0.0}, \
        "val must be deterministic on v0"
