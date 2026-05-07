#!/bin/bash -l
# Verify the T1..T27 task-id rename (single), the T<train>_T<test> rename
# (cross), the c<train>_c<test> rename (cross-cat), the matching exp-folder
# layouts, and the new "<id> (<descriptive>)" row labels — end-to-end against
# real warmed embeddings, in --no-writer mode (cheap, no encoder weight load).
#
# Coverage by default:
#   single   : T1=edaic_depC(C) T7=dementiabank_adC(C) T8=dementiabank_mmseR(R) T9=aphasia_pwaC(C)
#              → both task types, 3 datasets, exercises the brain.py CUDA-pids fix
#              via T1 (edaic numeric Participant_ID).
#   cross    : T9_T7 (aphasia → dementiabank), T1_T6 (edaic → iemocap)
#              → exercises _proxy_single_tasks_for_cross + the new exps/cross/T9_T7/ path.
#   cross-cat: c2_c3 (aphasia,dementiabank → torgo,uaspeech,mdvr,ksof)
#              → 6-dataset category cross at exps/cross_cat/c2_c3/.
#
# Re-runnable: each mode's `run` skips pairs whose results already exist. Pass
# OVERWRITE=1 to redo them.
#
# Submit:
#   sbatch verify_rename_slurm.sh                # one-shot full sweep
#   sbatch --export=ALL,OVERWRITE=1 verify_rename_slurm.sh   # force redo of all 3 modes
#   sbatch --export=ALL,SKIP_MODES="cross cross-cat" verify_rename_slurm.sh   # single only
# Tail logs:
#   tail -f exps/slurm_logs/verify_rename_*_out.txt
# After it succeeds, revert the temp slurm_tmpdir edits:
#   git checkout sdx/configs/main.yaml sdx/configs/main_cross.yaml sdx/configs/main_cross_category.yaml
#
#SBATCH --job-name=verify_rename
#SBATCH --chdir=./
#SBATCH --output=exps/slurm_logs/%x_%j_out.txt
#SBATCH --error=exps/slurm_logs/%x_%j_err.txt
#SBATCH --time=3:00:00
#SBATCH --signal=TERM@120
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
# --no-writer trains the probe only — barely any GPU memory needed; smallest MIG
# slice is plenty. Override with --gpus-per-node=h100:1 if your account can't
# get a MIG slice.
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1

set -euo pipefail

# ---------------------------------------------------------------------------
# Tasks under test. Edit / extend per mode; defaults exercise the rename of
# all 3 modes plus the brain.py CUDA-pids fix (T1).
# ---------------------------------------------------------------------------
SINGLE_TASKS=(T1 T7 T8 T9)
CROSS_TASKS=(T9_T7 T1_T6)
CROSSCAT_TASKS=(c2_c3)

ENCODER="${ENCODER:-wavlm}"
TAG="${TAG:-run1}"
SKIP_MODES="${SKIP_MODES:-}"   # space-separated subset of: single cross cross-cat
OVERWRITE_FLAG=()
[[ "${OVERWRITE:-0}" == "1" ]] && OVERWRITE_FLAG+=(--overwrite)

# ---------------------------------------------------------------------------
# Cluster env
# ---------------------------------------------------------------------------
VENV="/scratch/aina10/SpeechDx/.venv"

export HF_HOME="${HF_HOME:-$SCRATCH/hf}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$SLURM_TMPDIR/matplotlib-$USER}"
export FONTCONFIG_PATH="${FONTCONFIG_PATH:-$SLURM_TMPDIR/fontconfig-$USER}"
mkdir -p "$MPLCONFIGDIR" "$FONTCONFIG_PATH"

# Disk-backed Ray tmp (under the 107-byte AF_UNIX socket-path limit on nibi)
export TMPDIR="$SLURM_TMPDIR/t"
export RAY_TMPDIR="$SLURM_TMPDIR"
mkdir -p "$TMPDIR"
export RAY_object_store_memory="${RAY_object_store_memory:-2000000000}"

module --force purge
module load StdEnv/2023 gcc python/3.11 arrow ffmpeg rust
source "$VENV/bin/activate"

mkdir -p exps/slurm_logs

echo "============================================================"
echo "  verify-rename slurm job"
echo "============================================================"
echo "node          : $(hostname)"
echo "job id        : ${SLURM_JOB_ID:-?}"
echo "python        : $(which python)  ($(python -V))"
echo "single tasks  : ${SINGLE_TASKS[*]:-<none>}"
echo "cross tasks   : ${CROSS_TASKS[*]:-<none>}"
echo "crosscat tasks: ${CROSSCAT_TASKS[*]:-<none>}"
echo "encoder       : $ENCODER"
echo "tag           : $TAG"
echo "overwrite     : ${OVERWRITE:-0}"
echo "skip_modes    : ${SKIP_MODES:-<none>}"
echo

# ---------------------------------------------------------------------------
# 0a. Stage per-dataset CSVs into the layout `sdx prep` expects. Idempotent.
#     CSVs live at /scratch/aina10/data/<ds>/<ds>.csv; the harness wants them
#     at data/<ds>/processed/<ds>.csv. Audio paths in the CSVs are relative
#     and never dereferenced in --no-writer mode (the brain reads from the
#     warm cache.hdf5), so we don't need to stage raw audio — just satisfy
#     ensure_raw()'s "raw/ exists and is non-empty" check with a marker file.
#
#     Datasets touched are the union across single/cross/cross-cat tasks
#     (cross uses train_dataset+test_dataset; cross-cat uses the plural
#     train_datasets+test_datasets lists).
# ---------------------------------------------------------------------------
echo "=== [0a] stage per-dataset CSVs (symlinks) ==="
SCRATCH_DATA="${SCRATCH_DATA_ROOT:-/scratch/aina10/data}"
DATASETS_NEEDED=$(SINGLE_TASKS="${SINGLE_TASKS[*]}" \
                  CROSS_TASKS="${CROSS_TASKS[*]}" \
                  CROSSCAT_TASKS="${CROSSCAT_TASKS[*]}" python - <<'PY'
import os, re, yaml
from pathlib import Path
from sdx.orchestrator import get_task_info, TASKS_DIR
from sdx.orchestrator_cross import TASKS_DIR as CROSS_DIR
from sdx.yaml_io import TolerantLoader

def _load(p):
    return yaml.load(p.read_text(), Loader=TolerantLoader) or {}

datasets, seen = [], set()
def add(ds):
    if ds and ds not in seen:
        seen.add(ds); datasets.append(ds)

# Single: dataset comes from task yaml's `dataset:` field.
for t in os.environ.get("SINGLE_TASKS", "").split():
    try:
        add(get_task_info(t)[0])
    except Exception as e:
        print(f"# single {t}: {e}", flush=True)

# Cross / cross-cat: read from the cross_tasks/ yaml.
for t in os.environ.get("CROSS_TASKS", "").split():
    try:
        cfg = _load(CROSS_DIR / f"{t}.yaml")
        add(cfg.get("train_dataset")); add(cfg.get("test_dataset"))
    except Exception as e:
        print(f"# cross {t}: {e}", flush=True)
for t in os.environ.get("CROSSCAT_TASKS", "").split():
    try:
        cfg = _load(CROSS_DIR / f"{t}.yaml")
        for ds in (cfg.get("train_datasets") or []): add(ds)
        for ds in (cfg.get("test_datasets")  or []): add(ds)
    except Exception as e:
        print(f"# cross-cat {t}: {e}", flush=True)

print(" ".join(datasets))
PY
)
for DS in $DATASETS_NEEDED; do
    mkdir -p "data/$DS/raw" "data/$DS/processed"
    [[ -f "data/$DS/raw/.staged" ]] || touch "data/$DS/raw/.staged"
    SRC="$SCRATCH_DATA/$DS/$DS.csv"
    DEST="data/$DS/processed/$DS.csv"
    if [[ -e "$DEST" ]]; then
        echo "  ✓ $DS  $DEST already present"
    elif [[ -f "$SRC" ]]; then
        ln -sf "$SRC" "$DEST"
        echo "  ✓ $DS  symlinked $SRC → $DEST"
    else
        echo "  ⚠ $DS  no CSV at $SRC and none at $DEST — task(s) on this dataset will fail prep"
    fi
done
echo

# ---------------------------------------------------------------------------
# Helpers — used by each mode-specific section below.
#
# Each mode reads its OWN main*.yaml (single/main_cross/main_cross_category),
# each with its own slurm_tmpdir setting. So the pre-flight resolves the
# cache root *per task* via compose_config — ensuring we validate the same
# path the runtime will actually open. (A single global cache root would
# silently miss a stale slurm_tmpdir in main_cross.yaml.)
# ---------------------------------------------------------------------------

# Skip a mode if listed in SKIP_MODES (space-separated).
mode_skipped() {
    local m="$1"
    [[ " $SKIP_MODES " == *" $m "* ]]
}

# Cache + CSV pre-flight for one task. Echoes "OK" or a semicolon-joined
# list of missing items. Mode-aware: invokes compose_config so the
# resolved slurm_tmpdir comes from the same main*.yaml the runtime uses
# (single/main.yaml, cross/main_cross.yaml, cross-cat/main_cross_category.yaml).
preflight_one() {
    local mode="$1" stem="$2"
    MODE="$mode" STEM="$stem" ENC="$ENCODER" python - <<'PY'
import os, yaml
from pathlib import Path
from sdx.config import compose_config
from sdx.orchestrator import TASKS_DIR as SINGLE
from sdx.orchestrator_cross import TASKS_DIR as CROSS
from sdx.yaml_io import TolerantLoader

mode, stem, enc = os.environ["MODE"], os.environ["STEM"], os.environ["ENC"]
cfg = compose_config(stem, enc, mode="read")

# Cache root = the per-mode slurm_tmpdir from main*.yaml. The runtime always
# opens <root>/<dataset>/<encoder>/{train,val}/single_avg/cache.hdf5, so
# checking that on disk validates exactly what compose_config will give the
# trainer at run time.
root = Path(cfg["slurm_tmpdir"])

# Datasets touched by this task (for CSV staging + per-dataset cache check).
tdir = SINGLE if mode == "single" else CROSS
tcfg = yaml.load((tdir / f"{stem}.yaml").read_text(), Loader=TolerantLoader) or {}
datasets = []
if mode == "single":
    datasets.append(tcfg["dataset"])
elif mode == "cross":
    for k in ("train_dataset", "test_dataset"):
        v = tcfg.get(k)
        if v: datasets.append(v)
elif mode == "cross-cat":
    for k in ("train_datasets", "test_datasets"):
        for v in (tcfg.get(k) or []):
            datasets.append(v)
datasets = list(dict.fromkeys(datasets))

reasons = []
for ds in datasets:
    tr = root / ds / enc / "train" / "single_avg" / "cache.hdf5"
    vl = root / ds / enc / "val"   / "single_avg" / "cache.hdf5"
    if not tr.exists(): reasons.append(f"missing {tr}")
    if not vl.exists(): reasons.append(f"missing {vl}")
    csv = Path(f"data/{ds}/processed/{ds}.csv")
    if not csv.exists(): reasons.append(f"missing {csv}")

print("OK" if not reasons else "; ".join(reasons))
PY
}

# Per-mode end-to-end test: pre-flight → status before → run --no-writer →
# status after → summary → label spot-check → folder check. Args:
#   $1 = mode label   (single | cross | cross-cat)
#   $2 = sdx subcmd   (single | cross | cross-cat)
#   $3 = section idx  (e.g. "1", "2", "3" — used as the [N.x] prefix)
#   shift 3 + remaining args = task stems for this mode
run_mode_verify() {
    local mode="$1" subcmd="$2" idx="$3"; shift 3
    local tasks=("$@")

    if mode_skipped "$mode"; then
        echo "=== [$idx] $mode mode SKIPPED (SKIP_MODES=\"$SKIP_MODES\") ==="
        echo
        return 0
    fi
    if (( ${#tasks[@]} == 0 )); then
        echo "=== [$idx] $mode mode SKIPPED (no tasks configured) ==="
        echo
        return 0
    fi

    echo "=== [$idx] $mode mode: pre-flight ==="
    local runnable=()
    for t in "${tasks[@]}"; do
        local why; why=$(preflight_one "$mode" "$t")
        if [[ "$why" == "OK" ]]; then
            echo "  ✓ $t  cache + CSV OK"
            runnable+=("$t")
        else
            echo "  ⚠ $t  SKIP — $why"
        fi
    done
    echo
    if (( ${#runnable[@]} == 0 )); then
        echo "  no runnable $mode tasks; skipping the rest of [$idx]"
        echo
        return 0
    fi

    local task_filters=()
    for t in "${runnable[@]}"; do task_filters+=(-t "$t"); done

    echo "=== [$idx.1] $mode status BEFORE ==="
    python -m sdx "$subcmd" status "${task_filters[@]}" -e "$ENCODER" --tag "$TAG" || true
    echo

    echo "=== [$idx.2] $mode run --no-writer ==="
    python -m sdx "$subcmd" run "${task_filters[@]}" -e "$ENCODER" --tag "$TAG" \
           --no-writer -j 1 --yes "${OVERWRITE_FLAG[@]}"
    echo

    echo "=== [$idx.3] $mode status AFTER ==="
    python -m sdx "$subcmd" status "${task_filters[@]}" -e "$ENCODER" --tag "$TAG" || true
    echo

    echo "=== [$idx.4] $mode summary ==="
    python -m sdx "$subcmd" summary "${task_filters[@]}" -e "$ENCODER" --tag "$TAG" || true
    echo

    # Summary CSVs land at exps/<scope>/_summary_<tag>/.
    local sum_dir
    case "$subcmd" in
        single)    sum_dir="exps/single_task/_summary_${TAG}" ;;
        cross)     sum_dir="exps/cross/_summary_${TAG}" ;;
        cross-cat) sum_dir="exps/cross_cat/_summary_${TAG}" ;;
    esac
    echo "=== [$idx.5] $mode CSV row spot-check (labels should read '<id> (<descriptive>)') ==="
    for f in AUC.csv MAE.csv F1.csv completion.csv; do
        if [[ -f "$sum_dir/$f" ]]; then
            echo "--- $sum_dir/$f ---"
            head -1 "$sum_dir/$f"
            for t in "${runnable[@]}"; do grep -E "^${t}( |,)" "$sum_dir/$f" || true; done
            echo
        fi
    done

    echo "=== [$idx.6] $mode result files at the new <id>/ exp folders ==="
    for t in "${runnable[@]}"; do
        local mode_for_python="$mode"   # re-export under expected env var name
        eval "$(MODE="$mode_for_python" STEM="$t" ENC="$ENCODER" TAG="$TAG" python - <<'PY'
import os
mode, stem, enc, tag = os.environ["MODE"], os.environ["STEM"], os.environ["ENC"], os.environ["TAG"]
if mode == "single":
    from sdx.orchestrator import get_output_folder, get_results_file
    fold = get_output_folder(stem, enc, tag)
    res = fold / get_results_file(stem)
elif mode == "cross":
    from sdx.orchestrator_cross import get_output_folder, get_results_file
    fold = get_output_folder(stem, enc, tag)
    res = fold / get_results_file(stem)
elif mode == "cross-cat":
    from sdx.orchestrator_cross import get_output_folder, get_results_file
    from sdx.run_cross_category import CATEGORY_EXPS_ROOT
    fold = get_output_folder(stem, enc, tag, exps_root=CATEGORY_EXPS_ROOT)
    res = fold / get_results_file(stem)
print(f"FOLD='{fold}'")
print(f"RES='{res}'")
PY
)"
        if [[ -f "$RES" ]]; then
            echo "  ✓ $t  $RES"
        else
            echo "  ✗ $t  MISSING $RES"
        fi
    done
    echo
}

# ---------------------------------------------------------------------------
# 1. Single mode end-to-end.
# ---------------------------------------------------------------------------
run_mode_verify single   single    1 "${SINGLE_TASKS[@]}"

# ---------------------------------------------------------------------------
# 2. Cross mode end-to-end. Cross uses the same per-(dataset, encoder)
#    warm cache that single populates, so --no-writer here just exercises
#    the new exps/cross/T<x>_T<y>/ layout + label rendering.
# ---------------------------------------------------------------------------
run_mode_verify cross    cross     2 "${CROSS_TASKS[@]}"

# ---------------------------------------------------------------------------
# 3. Cross-category mode end-to-end. Reads the same per-(dataset, encoder)
#    warm cache (every dataset in train_datasets + test_datasets must have
#    one), and exercises the new exps/cross_cat/c<x>_c<y>/ layout.
# ---------------------------------------------------------------------------
run_mode_verify cross-cat cross-cat 3 "${CROSSCAT_TASKS[@]}"

echo "=== finished. status: ✓"
echo "Revert the temp slurm_tmpdir edits when you're done verifying:"
echo "  git checkout sdx/configs/main.yaml sdx/configs/main_cross.yaml sdx/configs/main_cross_category.yaml"
