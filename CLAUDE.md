# Audio-Health-Benchmark — agent notes

## Benchmark integrity: change HOW FAST it runs, never WHAT it runs

The benchmark protocol is the science. You may optimize **throughput**
(speed, memory, concurrency, scheduling) freely, but you must NOT change
**what is computed** — the values that affect a task's result.

- **Free to change (performance only):** orchestration concurrency
  (`--workers`), Ray plumbing (`SDX_RAY_NUM_CPUS`, temp dir, dashboard),
  thread counts (OMP/MKL), node/MIG/mem sizing, rolling-resubmit cadence,
  and bit-identical compute rewrites (e.g. `WAVRX_MOD_CHUNK` chunks the
  modulation STFT — verified max|diff|=0, same numbers).
- **Methodological — do NOT change to fit hardware:** `batch_size`,
  `precision`, `number_of_epochs`, the HP search space (`num_samples`,
  `lr_s`/`l2` ranges), `grace_period`/`limit_warmup`, the probe/loss, the
  crop length, augmentation. These define *what* the benchmark measures.
  Changing any of them makes results non-comparable across tasks.
- **Absolute last resort only:** if a methodological value genuinely
  cannot run on the available hardware (e.g. CPUs have no native fp16, so
  CPU tasks run fp32 while GPU tasks keep the canonical fp16), it is a
  deviation, not an optimization. Before doing it: (1) exhaust the
  performance levers above, (2) get the user's sign-off, (3) record it in
  `DEVIATIONS_wavrx_smoke.md` with the reason and the cross-task caveat.
- **Test/debug configs are ALWAYS a temporary override or a forked test
  file — never an edit to the canonical config.** A smoke test runs with
  `--overrides "number_of_epochs: 3"` (reverted after) or a
  `*_test.sbatch`, never by editing `sdx/configs/training.yaml`. The
  committed configs must always hold the real protocol values so a plain
  run reproduces the benchmark. (Lesson learned the hard way: an
  `ep=3`/`batch=4` value that leaks into a real run silently corrupts the
  benchmark — wipe and rerun.)

## `data/` is a SquashFS image — wrap anything that touches it

`data/` is **not a regular directory**. The ~610k loose audio files it used to
hold were packed into a single SquashFS image, `data.sqfs` (1 inode instead of
~610k, to stay under the `/scratch` inode quota). On disk `data/` is now an
**empty mountpoint** — it holds the audio files only while `data.sqfs` is
mounted onto it.

**Rule: any script, job, or command that reads (or writes paths under) `data/`
MUST run through `./run_with_data.sh`.** It mounts `data.sqfs` onto `data/`
inside an unprivileged user namespace (`unshare -rm`), runs the command, and
tears the mount down automatically on exit.

```bash
./run_with_data.sh python -u scripts/test_qwen3omni_all.py --all ...
./run_with_data.sh bash        # interactive shell with data/ mounted
```

### Writing a new sbatch script / launcher

Wrap whatever touches `data/` — in practice the `python …` invocation:

```bash
# instead of:
python -u scripts/my_eval.py ...
# do:
./run_with_data.sh python -u scripts/my_eval.py ...
```

`qwen3omni_all.sbatch` and `qwen3omni_loop.sbatch` already do this — copy the
pattern. A script that does **not** read `data/` (e.g. `submit_meanpool.sh`,
which only uses `tmp/`) does not need the wrapper. If unsure, wrap it: the
wrapper is harmless when `data/` goes unused.

### Why it works this way

- The manifests under `metadata/` reference audio by **absolute path**
  (`/scratch/kieu/Audio-Health-Benchmark/data/…`), so the image must appear at
  exactly that path — pointing code at a copy elsewhere will not work.
- This cluster has no `fusermount3` (the `fuse3` package is not installed), so
  `squashfuse` cannot mount the normal way. Inside `unshare -rm` the caller is
  root in the namespace and libfuse mounts directly, no setuid helper needed.
- The mount is private to each process tree, so concurrent jobs never collide.

Do **not** "fix" `data/` by extracting it back to loose files — that re-creates
the ~610k-inode problem. If the dataset contents ever need to change, rebuild
the image from a populated tree with `build_data_sqfs.sbatch`.

## `/loop` on a Slurm job: Monitor on state change, not time

When `/loop` is babysitting a Slurm job (queue → run → finish, possibly
self-resubmitting), **never** use `ScheduleWakeup` as the primary cadence.
Wake on the events that actually change what you'd do next:

- **Start** — `PENDING → RUNNING` (the job got a node, the log file appears).
- **Stop** — `RUNNING →` one of `COMPLETED / FAILED / TIMEOUT / CANCELLED /
  NODE_FAIL / PREEMPTED`.
- **Requeue** — the rolling-loop pattern in this repo (`qwen3omni_loop.sbatch`,
  `warm_wavlm_loop.sbatch`) prints `Submitted batch job <NEW_ID>` from the
  parent's `[sbatch]` lines just before it exits. Grep for that and re-target
  the monitor on the new ID.
- **~10% progress** — for the warm path, that's the per-task `Iteration N` /
  `X chunks encoded` lines. Throttle aggressively (≈1 event per 10% of the
  task's chunks, not 1 per log line) so the conversation isn't flooded.

Arm one `Monitor` with `persistent: true` that covers all of these in a
single stdout-event stream. `ScheduleWakeup` is only a fallback heartbeat
(≥30 min) in case the monitor itself hangs — never the primary signal.

Sketch:

```bash
JOB=$1
LOG=logs/<jobname>_${JOB}.out
prev=""
chunks_emitted=0
follow_requeues=true
while sleep 30; do
  s=$(sacct -j "$JOB" -P -n -o State 2>/dev/null | head -1 | awk '{print $1}')
  [[ -z "$s" ]] && s=GONE
  if [[ "$s" != "$prev" ]]; then
    echo "$(date -Iseconds) STATE ${prev:-?} → $s  job=$JOB"
    prev=$s
  fi
  if [[ -f "$LOG" ]]; then
    # Emit per-task starts + completions + every Nth "chunks encoded" line +
    # any traceback/OOM marker. Use tail -n +<last> to avoid re-emitting.
    tail -n +$((last_line+1)) "$LOG" 2>/dev/null \
      | grep -E "Iteration |fully warm|already warmed|warn|OOM|Traceback|Submitted batch job" \
      | head -40
    last_line=$(wc -l < "$LOG")
  fi
  if $follow_requeues && [[ -f "$LOG" ]]; then
    new=$(grep -oE 'Submitted batch job [0-9]+' "$LOG" | tail -1 | awk '{print $4}')
    if [[ -n "$new" && "$new" != "$JOB" ]]; then
      echo "$(date -Iseconds) REQUEUE $JOB → $new"; JOB=$new; LOG=logs/<jobname>_${JOB}.out
      prev=""; last_line=0
    fi
  fi
  case "$s" in COMPLETED|FAILED|TIMEOUT|CANCELLED|NODE_FAIL|PREEMPTED)
    $follow_requeues || break
    # If requeue isn't visible yet, give it 60s, then exit.
    sleep 60; new=$(grep -oE 'Submitted batch job [0-9]+' "$LOG" | tail -1 | awk '{print $4}')
    [[ -n "$new" && "$new" != "$JOB" ]] || break ;;
  esac
done
```

Reach for a `Bash run_in_background` "until the file appears, then exit"
script only when there's truly a single one-shot event to await — never to
*replace* the state-change Monitor. The right pattern for a long-running
Slurm job is one persistent Monitor for its whole lifetime.
