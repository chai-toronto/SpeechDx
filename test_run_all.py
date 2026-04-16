"""Lightweight scheduler tests — no training processes spawned.

Exercises the writer/reader classification, warm_cache override plumbing, and
lock behavior by stubbing out ``run_all.run_one``. Run with ``./spa/bin/python
-m unittest test_run_all``.
"""

import argparse
import threading
import time
import types
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import run_all


def _fake_args(**overrides) -> argparse.Namespace:
    base = dict(device=None, max_workers=1, encoder=["fake"], dataset=["c9s"])
    base.update(overrides)
    return argparse.Namespace(**base)


class MetadataTests(unittest.TestCase):
    """Sanity-check the memoized task-metadata helpers against the real configs."""

    def test_c9s_weights_and_versions(self):
        self.assertEqual(run_all.task_num_ver("c9s_L_t1"), 1)
        self.assertEqual(run_all.task_num_ver("c9s_t1"), 3)
        self.assertGreater(len(run_all.task_ids("c9s_L_t1")), 50000)
        # L_t1 beats t1 on weight despite t1's higher num_ver
        self.assertGreater(run_all.task_weight("c9s_L_t1"),
                           run_all.task_weight("c9s_t1"))


class MakeConfigTests(unittest.TestCase):
    """make_config should rewrite ``warm_cache:`` only when asked."""

    def _read_warm_cache(self, cfg_path: Path) -> str:
        for line in cfg_path.read_text().splitlines():
            if line.startswith("warm_cache:"):
                return line.split(":", 1)[1].strip()
        return ""

    def test_override_false(self):
        cfg = run_all.make_config("fake", "hubert.yaml", "c9s_ageR.yaml",
                                  config_id="_test_false",
                                  warm_cache_override=False)
        try:
            self.assertEqual(self._read_warm_cache(cfg), "false")
        finally:
            cfg.unlink(missing_ok=True)

    def test_override_true(self):
        cfg = run_all.make_config("fake", "hubert.yaml", "c9s_ageR.yaml",
                                  config_id="_test_true",
                                  warm_cache_override=True)
        try:
            self.assertEqual(self._read_warm_cache(cfg), "true")
        finally:
            cfg.unlink(missing_ok=True)

    def test_no_override_keeps_default(self):
        cfg = run_all.make_config("fake", "hubert.yaml", "c9s_ageR.yaml",
                                  config_id="_test_default")
        try:
            # main.yaml default is `warm_cache: true`; unchanged when no override
            self.assertEqual(self._read_warm_cache(cfg).lower(), "true")
        finally:
            cfg.unlink(missing_ok=True)


def _stub_run_one_factory(calls, sleep_s=0.0, fail=()):
    """Return a stub replacing run_all.run_one that records each call."""
    lock = threading.Lock()

    def stub(task_stem, model_name, encoder_yaml, device=None, config_id="",
            warm_cache_override=None):
        started = time.monotonic()
        time.sleep(sleep_s)
        ended = time.monotonic()
        label = f"{task_stem} × {model_name}"
        success = task_stem not in fail
        with lock:
            calls.append({
                "task": task_stem, "model": model_name,
                "warm_cache": warm_cache_override,
                "start": started, "end": ended, "success": success,
            })
        return label, success, ended - started
    return stub


class SchedulerOrderingTests(unittest.TestCase):
    """With max_workers=1, verify dispatch order + warm_cache overrides."""

    def setUp(self):
        self.encoders_patch = patch.object(run_all, "ENCODERS", {"fake": "hubert.yaml"})
        self.encoders_patch.start()
        # Pretend nothing is complete — every job is pending.
        self.complete_patch = patch.object(run_all, "is_complete", return_value=False)
        self.complete_patch.start()

    def tearDown(self):
        self.encoders_patch.stop()
        self.complete_patch.stop()

    def test_c9s_single_worker_ordering(self):
        calls = []
        with patch.object(run_all, "run_one", _stub_run_one_factory(calls)):
            run_all.cmd_run(_fake_args(max_workers=1))

        c9s = [c for c in calls if c["task"].startswith("c9s_")]
        order = [c["task"] for c in c9s]
        # First job must be the heaviest writer for c9s.
        self.assertEqual(order[0], "c9s_L_t1")
        # c9s_L_t1 is a writer: warm_cache override must be None (yaml default).
        self.assertIsNone(c9s[0]["warm_cache"])

        # After L_t1 completes, every task whose needed_keys ⊆ L_t1.keys and
        # whose num_aug_ver ≤ 1 must be classified as reader (warm_cache=False).
        reader_expected = {"c9s_L_t2", "c9s_ageR", "c9s_sexC",
                           "c9s_smokerC", "c9s_sympL"}
        for call in c9s[1:]:
            if call["task"] in reader_expected:
                self.assertIs(call["warm_cache"], False,
                              msg=f"{call['task']} should be a reader")
            if call["task"] in {"c9s_t1", "c9s_t2"}:
                # num_aug_ver=3 — still a writer
                self.assertIsNone(call["warm_cache"],
                                  msg=f"{call['task']} should be writer")


class SchedulerFailureTests(unittest.TestCase):
    """If a writer fails, subsequent subset tasks must stay writers (safe)."""

    def setUp(self):
        self.encoders_patch = patch.object(run_all, "ENCODERS", {"fake": "hubert.yaml"})
        self.encoders_patch.start()
        self.complete_patch = patch.object(run_all, "is_complete", return_value=False)
        self.complete_patch.start()

    def tearDown(self):
        self.encoders_patch.stop()
        self.complete_patch.stop()

    def test_failed_writer_does_not_unlock_immediate_reader(self):
        """If L_t1 fails, the *next* dispatched task must still be a writer —
        nothing has been written yet so no subset can be satisfied."""
        calls = []
        stub = _stub_run_one_factory(calls, fail={"c9s_L_t1"})
        with patch.object(run_all, "run_one", stub):
            run_all.cmd_run(_fake_args(max_workers=1))

        c9s = [c for c in calls if c["task"].startswith("c9s_")]
        self.assertEqual(c9s[0]["task"], "c9s_L_t1")
        self.assertFalse(c9s[0]["success"])
        # The task dispatched right after the failure cannot have its keys
        # covered (written set is empty), so it must run as writer.
        self.assertIsNone(
            c9s[1]["warm_cache"],
            msg=f"{c9s[1]['task']} became reader after failed writer — unsafe")


class SchedulerConcurrencyTests(unittest.TestCase):
    """Writers on the same (dataset, encoder) must serialize via the lock."""

    def setUp(self):
        self.encoders_patch = patch.object(run_all, "ENCODERS", {"fake": "hubert.yaml"})
        self.encoders_patch.start()
        self.complete_patch = patch.object(run_all, "is_complete", return_value=False)
        self.complete_patch.start()

    def tearDown(self):
        self.encoders_patch.stop()
        self.complete_patch.stop()

    def test_c9s_writers_serialize(self):
        calls = []
        # Use a stub that sleeps so any overlap would be visible.
        stub = _stub_run_one_factory(calls, sleep_s=0.15)
        with patch.object(run_all, "run_one", stub):
            run_all.cmd_run(_fake_args(max_workers=4))

        c9s = [c for c in calls if c["task"].startswith("c9s_")]
        writers = [c for c in c9s if c["warm_cache"] is None]
        # c9s only has one (ds,enc) bucket here — writers must not overlap.
        writers.sort(key=lambda c: c["start"])
        for prev, curr in zip(writers, writers[1:]):
            self.assertLessEqual(
                prev["end"], curr["start"] + 1e-3,
                msg=f"writer overlap: {prev['task']} ended {prev['end']} "
                    f"after {curr['task']} started {curr['start']}")


if __name__ == "__main__":
    unittest.main()
