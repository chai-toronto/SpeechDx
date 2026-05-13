# Gemini 3.1 Pro evaluation handoff

Zero-shot benchmark of `gemini-3.1-pro-preview` across all 27 SpeechDx tasks,
to be compared against the Qwen3-Omni-Thinking baseline at
`exps/qwen3omni_all/`.

## Status

**All 27 tasks have predictions** at `exps/gemini_3_1_pro/<task>/predictions.csv`.
Most tasks are fully clean. Outstanding work:

- **c19sounds retry batch is PENDING**: `batches/9bg8e7crmj8s5hlxj46pt0k15iao98a95sox`
  - 891 retry calls for previously-failed predictions in T19–T23
  - Watcher (`scripts/gemini_watch.py`) is running and will auto-`fetch-delta` on SUCCESS
  - Once fetched, re-run `python scripts/regen_metrics.py` to refresh metrics

## How to resume after compact

1. Check c19sounds retry batch state:
   ```
   python scripts/gemini_batch.py status --dataset c19sounds
   ```
   (state lives in `exps/gemini_batches/c19sounds/batch_job_retry.json`)
2. If watcher is dead (`ps aux | grep gemini_watch`), restart:
   ```
   export GEMINI_API_KEY="$(security find-generic-password -s gemini-api-key -w)"
   python scripts/gemini_watch.py >> /tmp/gemini_watch.log 2>&1 &
   ```
3. After retry batch succeeds + fetches, finalize:
   ```
   python scripts/regen_metrics.py
   ```

## Auth (API key)

- Key lives in **macOS keychain** under service name `gemini-api-key`
- Every script auto-loads via `security find-generic-password -s gemini-api-key -w`
- Never paste the key into shell history or files

## Files written this session

### Scripts (`scripts/`)

| File | Purpose |
|---|---|
| `count_gemini_tokens.py` | Counts Gemini tokens per task (32 tok/sec audio + prompt). Writes `gemini_token_counts.csv` |
| `run_gemini_eval.py` | Interactive eval (group-based, used for edaic). Files API + retry/backoff + flex/standard tier |
| `gemini_batch.py` | Batch API runner. Subcommands: `submit`, `status`, `fetch`, `cleanup`, `submit-delta`, `fetch-delta` |
| `gemini_watch.py` | Long-running watcher: polls main + retry batches, auto-fetches on SUCCEEDED, 5-min cycle |
| `gemini_retry_interactive.py` | Interactive retry for failed (rowid,task) pairs in predictions.csv |
| `regen_metrics.py` | Filters predictions to canonical manifest sample sets + recomputes metrics |

### State (`exps/gemini_batches/<dataset>/`)

- `audio_uris.jsonl` — append-only log of (rowid → file URI) for main uploads
- `audio_uris_retry.jsonl` — same for delta retries
- `batch_job.json` — main batch metadata (job_name, state, fetched_at)
- `batch_job_retry.json` — delta batch metadata
- `requests.jsonl` / `requests_retry.jsonl` — batch input JSONL (audit trail)
- `results.jsonl` / `results_retry.jsonl` — downloaded raw batch output
- `rows_meta.json` / `rows_meta_retry.json` — key → {rowid, task, fold, duration} map

### Outputs (`exps/gemini_3_1_pro/<task>/`)

- `predictions.csv` — columns: rowid, fold, duration_sec, true, pred, true_vec, pred_vec, response (matches Qwen layout, drop-in comparable)
- `predictions.metrics.json` — overall + per_fold + fold_aggregate

## Datasets and task groups

```
edaic       T1, T2          (done interactively, not batch)
ravdess     T3, T4
iemocap     T5, T6
dbank       T7, T8          (cleaned up on Files API)
aphasia     T9
torgo       T10, T11
uaspeech    T12
mvdr        T13, T14, T15, T16
ksof        T17, T18
c19sounds   T19, T20, T21, T22, T23    (cleaned up; retry batch PENDING)
coswara     T24, T25, T26
avfad       T27
```

## Headline results vs Qwen3-Omni-Thinking

Significant Gemini wins:
- **T8 MMSE** (Pearson 0.53 vs Qwen -0.03 — Qwen MMSE essentially random)
- **T9 aphasia** (acc +28pp, AUC +0.35)
- **T11 dysarthria severity** (MAE 0.44 vs 1.86)
- **T12 dysarthria binary** (acc +10pp, AUC +0.12)
- **T13–T15 PD tasks** (T14 UPDRS-II ⑤ speech: MAE 0.23 vs 0.55, Pearson 0.74 vs 0.56)
- **T5/T6 iemocap emotion** (acc +4pp, AUC +0.03)

Qwen wins (slight):
- T1/T2 edaic (Qwen acc +3pp, F1 +0.05)
- T4 ravdess binary
- T16 Hoehn-Yahr (Pearson 0.53 vs 0.43)
- T24–T26 coswara (both poor; Qwen marginally better)
- T27 avfad voice pathology

Both at chance/random:
- T19–T23 covid tasks (acoustic signal weak)

## Critical gotchas learned during run

1. **Files API rate limit is brutal**: hit `429 RESOURCE_EXHAUSTED` at ~67K uploads in 38 min. Daily-ish quota. Use `--workers 4` for slow trickle. Recovered overnight.
2. **Batch API "shedding"**: flex tier returns lots of 503/UNAVAILABLE during peak hours. Use standard tier for time-sensitive runs.
3. **`<batch-error>` in results**: even on `JOB_STATE_SUCCEEDED`, individual requests inside the batch can have errors (Deadline expired / Cancelled). These need retry. ~880 such failures on c19sounds.
4. **Path rewrites needed for legacy manifests**:
   - `/data/dbank/` → `/data/dementiabank/`
   - `/data/c9s/` → `/data/c19sounds/`
   - Torgo files in manifest have `sN_` prefix that's not on disk (strip it)
5. **T7/T8 source paths**: inspiration `test_qwen3omni_all.py` has stale paths; corrected to `exps/single_task/dbank_adC/manifest` in scripts.
6. **multilabel/binary task overlap (c19sounds)**: T19–T22 binary, T23 multilabel; same rowid can appear in both. Fetch step must look up `true` value per-task, not from a merged sample map.
7. **submit guard**: `submit_dataset` skips datasets with a SUCCEEDED batch. To do a delta retry, use `submit-delta` (separate code path, saves to `batch_job_retry.json`).

## File-API cleanup

- `dbank`: cleaned ✅
- `c19sounds`: cleaned ✅ (then re-uploaded ~826 files for delta retry — still on API)
- All other datasets: still uploaded; will expire ~48 h after upload (Google auto-deletes)

User said no need to actively cleanup; let Google's TTL handle the rest.

## Bugs fixed during the run

- `_parse_response(cfg, ...)` should be `_parse_response(task, ...)` — TASKS[cfg] crashed with `unhashable type: 'dict'`. Fixed.
- `fetch_dataset` was using a merged `sample_by_rowid` map → wrong `true` value when same rowid spans multiple tasks. Patched to use `per_task_samples` lookup.
- `fetch_dataset` added `<upload-failed>` rows to *every* task in a dataset, even if the rowid wasn't a member of that task. Caused 24 spurious rows across T11 + T19–T23. `regen_metrics.py` cleans these up.
- Cleanup function has no progress logging (slow on c19sounds 17K files at ~2.7 deletes/sec). Known, not critical.

## Suggested next steps after compact

1. Wait for c19sounds retry batch (PENDING). Watcher auto-fetches.
2. Run `python scripts/regen_metrics.py` after fetch.
3. Build a leaderboard / summary table (Gemini vs Qwen across all 27 tasks). Existing `scripts/leaderboard.py` may already do this — sanity check it picks up the Gemini results from `exps/gemini_3_1_pro/`.
4. Optionally: write a markdown report of headline findings for the paper/writeup.
5. The `sdx/prep/ravdess.py` file was opened in the IDE during the session but no edits were planned there — probably user reviewing prep code; nothing actionable from us.

## Costs (estimate)

- Interactive E-DAIC (standard tier, 2 tasks × 56 samples): ~$5
- Batch API (10 main batches + 1 retry, ~85K inference calls, 50% off standard): ~$25–40 estimated
- Files API uploads: free
- Total session: probably **~$30–50**

## Key model and parameters

- `MODEL = "gemini-3.1-pro-preview"`
- Default generation params (thinking_level=high implicit per docs)
- Audio token rate: 32 tok/sec (documented)
- For batch: 50% discount, target 24h turnaround (often < 5h)
