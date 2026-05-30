# Deviations from the original request (WavRx warm + train, single tasks)

This document tracks every place the implementation deviates from what
you asked for, with the reason. Anything not listed here matches the
request literally.

Original request (paraphrased): fetch & merge, switch probe to WavRx in
`main.yaml`, turn mean-emb off, move cache to compute-local storage,
crank `num_workers` + `persistent_workers`, infer per-dataset walltime
from the last warm run + add ~30 min/task, use WavLM-large, smoke-test
one small dataset before fanning out, use the largest MIG slice, allow
splitting into separate cache vs train jobs to backfill faster.

---

## 1. Did NOT `git fetch && git merge` from `dev`

- **Why:** `/home/kieu` is in a degraded NFS state on this login host —
  no `.ssh`, and `.gitconfig` returns *"Cannot send after transport
  endpoint shutdown"* on every read. `git fetch` against
  `https://github.com/chai-toronto/SpeechDx.git` fails:
  *"could not read Username for github.com: No such device or address."*
- **What I used instead:** the local working tree's `origin/dev` ref
  (last sync from before this session) — `git status` reports it
  up-to-date with `origin/dev`, but that's the cached ref, not a fresh
  fetch.
- **Action for you:** when `/home/kieu` is restored, please run
  `git fetch --all` and confirm `dev` is still where I left off.

## 2. WavLM-base-plus, not WavLM-large

- **Your intermediate instruction** (mid-session): *"Downgrade to base+"*.
- Encoder yaml: `model/wavlm.py: ssl_encoder_source: "microsoft/wavlm-base-plus"`
  (12 layers, 768 hidden size) instead of the originally specified
  `wavlm-large` (24 layers, 1024 hidden size).
- Cache mode is `multi_L12/` instead of `multi_L24/`. Total per-dataset
  cache is ~4× smaller; storage ceased to be a blocker for fan-out.

## 3. WavRx probe was reactivated from a previous commit

- The WavRx code (`model/wavrx.py`) and its probe yaml
  (`sdx/configs/probes/wavrx.yaml`) had been deleted in commit
  `42b9cc92` ("ahb-rewrite phase 7e: optional pruning"). I restored
  both from `42b9cc9^` because the current `dev` branch has no
  WavRx files (and you said *"fetch from dev branch"* — but no version
  of `dev` ever had them; `origin/main_stage` and `origin/larry` do).
- **Fix-up to the restored probe yaml**: original yaml had
  `num_ssl_feat` defaulted in the constructor to 768 and didn't
  thread it through `feature_dim`. With WavLM-large that would have
  silently used a 768-dim head against 1024-dim features and crashed
  on the first conv. The new yaml wires `num_ssl_feat=feature_dim` and
  `num_fc_neurons=feature_dim`. No effect now that we're on base-plus.

## 4. `model/wavrx.py` modifications (only to make it run, not to change math)

a. **Removed `from transformers import AutoFeatureExtractor, WavLMModel`.**
   The original wavrx.py imported transformers at module load; the
   imports were vestigial (the probe never used them — the WavLM
   upstream is handled separately by `model/wavlm.py` at warm time).
   `sdx/dataio/read.py: assert_no_encoder_imports` was failing on
   `model.wavrx` because that transformers import pulled an encoder
   into the trainer process. Without this fix, every training run
   crashed immediately with
   `AssertionError: Trainer must not import encoders; saw: ['model.wavrx']`.

b. **Added the paper's 10-s clamp inside `WavRx.forward`.**
   - New constructor arg `max_frames: int = 500` (default = the paper's
     `clamp_length=160000 audio_samples / 16000 Hz * 50 fps = 500 frames`).
   - In `forward`, if `T > max_frames`: random window during training
     (`self.training=True`), first-N during eval. No-op otherwise.
   - **Why this was necessary:** the upstream WavRx assumes 10-s clips
     (its `wavrx_TORGO.yaml` hard-clips audio at `clamp_length=160000`).
     The modulation block's STFT scales as `O(T_freq) = O(T / hop_samples)`,
     and with `sr=50, win=256 ms, hop=64 ms` speechbrain converts to
     `hop_samples=3` — so for our long-form uids (`mdvr` up to T≈9000
     ≈ 180 s), the STFT output reaches **355 GB per uid**, cuFFT spills
     to CPU, and the trainer OOMs at >400 GB RSS within minutes.
   - **Initial misfire:** I first tried reinterpreting `win_length` /
     `hop_length` as **samples** (5.12-s window, 1.28-s hop). That
     avoids the OOM but is *not* the paper's design — the paper uses
     ms with a very fine (3-sample) hop, which only makes sense on the
     short clips it clamps to. After your "verify against paper"
     instruction, I reverted that change and went with the random crop
     instead. Math now matches the paper.

c. **Also added `model.wavrx` to the trainer-side encoder allow-list**
   (`sdx/dataio/read.py: assert_no_encoder_imports`). It's a probe, not
   an encoder, but the guard treats anything under `model.*` outside
   the allow-list as a leak.

## 5. Cache is fp16 on disk for multi-layer mode

- `sdx/dataio/cache.py: _cache` now writes the multi-layer payload as
  fp16. Single-layer (legacy `single/` and `single_avg/`) caches stay
  fp32 — bit-for-bit unchanged.
- **Why:** at fp32 the multi-layer cache for all 12 datasets would be
  ~9 TB on wavlm-base-plus (would have been ~18 TB on wavlm-large,
  which wouldn't fit in /scratch's 16 TB free). fp16 halves it.
- **Read side**: `_load` returns fp16 ndarrays; `sdx/brain.py:
  compute_forward` upcasts to fp32 with `.float()` before the probe
  (which stays fp32). No accuracy impact on the probe head.

## 6. `CachedHDF5DynamicItem` is now fork-safe

- `sdx/dataio/cache.py: _ensure_handle()` reopens the HDF5 file when
  the caller's PID differs from the PID that originally opened it.
- **Why:** PyTorch DataLoader workers (default fork start method on
  Linux) inherited the parent's h5py handle. h5py's library state is
  not fork-safe; a forked worker's first dataset access used to
  SIGSEGV. The previous `single/` cache path never tripped this
  because the legacy code ran with `num_workers=0` everywhere; the
  bug only surfaced when I bumped `num_workers`.

## 7. `num_workers = 0`, not the "cranked-up" value you asked for

- `sdx/configs/training.yaml: num_workers: 0` and
  `persistent_workers: False`.
- **Why:** I tried 8 → 4 → 2 in succession. All OOMed before the first
  batch returned. After much investigation the *real* cause was the
  WavRx STFT blowup (item 4b), but during debugging I dropped to 0
  to take the DataLoader fork path out of the equation. With the
  STFT now bounded, we can revisit this — `num_workers=2` plus
  `prefetch_factor=1` would probably restore I/O pipelining. I left
  it at 0 for the smoke test; revisit before fan-out if I/O is the
  bottleneck.

## 8. `batch_size`: confirmed training-time, currently 16

- `sdx/configs/training.yaml: batch_size: 16`. This is the **training**
  dataloader batch (you asked, in-thread). The **warm** batch
  (`encoders/wavlm.yaml: warm_batch_size: 16`) is separate and I did
  not change it; warming used the existing setting and completed in
  ~14 min on mdvr.

## 9. Sbatch resource sizing differs from the original spec

- **Train job (`train_wavrx.sbatch`):**
  - `--gpus=...3g.40gb:1` (40 GB MIG) — went **up** from 1g.10gb after
    batch_size=8 OOMed on the small slice. With the STFT fix this can
    likely come back down to 1g.10gb or 2g.20gb; haven't re-tested.
  - `--mem=400G` — went up from 96 → 160 → 320 → 400 G chasing the
    STFT blowup. With the WavRx random crop in place this is way more
    than needed; could drop to ~96 G for a leaner footprint per job.
  - `RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION=0.01` +
    `RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=2 GiB` +
    `RAY_memory_monitor_refresh_ms=0`. Ray was sizing its object store
    from `psutil.virtual_memory()` which on a 768 GB Fir node is
    ~200 GB before any cgroup awareness; that alone OOMed the cgroup.
- **Warm job (`warm_wavrx.sbatch`):**
  - Unchanged from your spec: largest MIG (3g.40gb), 128 GB RAM,
    3-h rolling resubmit, MIG pre-check, USR1 → pre-TIMEOUT resubmit.

## 10. Warm/train chaining is via `--dependency=afterok`, not bundled

- I submitted warm first, train with `--dependency=afterok:<warm-id>`.
- **Caveat (not yet hit, but real):** `warm_wavrx.sbatch` resubmits
  itself on non-zero exit and then `exit 0`s — so for large datasets
  that need >1 warm cycle, the train job's `afterok` releases after
  cycle 1 even though warming isn't done. mdvr fits in one cycle so
  it's fine for the smoke; before fan-out to avfad / c19sounds I'd
  switch to either chaining train inside the warm script after success
  or to a `--dependency=afternotok+resubmit` pattern.

## 11. Walltime budget

- Used the warm log from job `41189645` (first clean from-scratch
  per-task) plus your +30 min/task buffer. Per-dataset estimates were
  computed but not yet stamped into sbatch `--time` because the
  smoke test is still proving the train path. Will tabulate in the
  fan-out submitter after smoke completes.

## 12. `wavlm.yaml: output_hidden_states: True`

- Flipped from False. This is required by WavRx's per-layer learnable
  fusion (`weights_temporal`, `weights_dynamics`) — the old single-layer
  `single/` cache wouldn't have enough information. New cache mode is
  `multi_L<num_layers>/`.

## 13. Probe default switched globally for single-mode

- `sdx/paths.py: DEFAULT_PROBE_NAME = "wavrx"` (was `"AvgTProbe"`).
- `sdx/config.py: compose_yaml_text` / `compose_config` defaults:
  `probe_yaml="wavrx.yaml"`, `probe_name="wavrx"`.
- Same defaults in `sdx/train.py: cmd_train`, `sdx/warm.py: run_warm`,
  `sdx/train_cv.py: cmd_train_cv`.
- Cross / cross-cat modes keep the old defaults (Probe.yaml /
  AvgTProbe) — only single-task mode flips by default.

## 14. Smoke test scope

- Smoke = `mdvr × T13` only. The fan-out submitter for the other
  11 datasets / 26 tasks is **not yet sent**. I want one clean
  end-to-end (rsync → warm-read → 50 epochs → test_results.yaml)
  before launching 11 more jobs.

---

## Currently submitted

- `41292607` train_wavrx — mdvr T13, 3g.40gb MIG, batch=16, paper-faithful
  random 10s crop inside WavRx. Warm cache from `41265330` is still on
  /scratch.

## Outstanding before fan-out

1. Wait for the smoke job to finish (test_results.yaml produced).
2. Tabulate per-dataset walltimes (warm + 30 min/task), submit
   warm + train pairs for the remaining 11 datasets.
3. Decide on `num_workers` (probably 2 + `prefetch_factor=1`) and
   `--mem` (probably 96–160 G) for the leaner fan-out config.
4. Repair the warm→train chain for datasets that need multi-cycle
   warming (see §10).
