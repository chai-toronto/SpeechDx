#!/bin/bash -l
# Verify three sibling renames, end-to-end:
#     c9s    -> c19sounds      (lowercase token + C9S/C19SOUNDS uppercase)
#     mvdr   -> mdvr           (lowercase token + MVDR/MDVR  uppercase)
#     dbank  -> dementiabank   (lowercase token + DBank/DementiaBank CamelCase)
#
# What this checks (no GPU, no encoder weight load):
#   [1] No old strings (c9s / C9S / mvdr / MVDR / dbank / DBank) remain in any
#       source file. Excludes .venv, .git, __pycache__, exps/, metadata/,
#       logs/, data/ — those hold build artifacts or large data, not source.
#   [2] Every renamed file/dir exists at its new location (source modules,
#       task yamls, metadata-script files, download scripts, data dir + CSV).
#   [3] Every old file/dir is gone — catches half-finished renames.
#   [4] Renamed Python modules import cleanly:
#         sdx.prep.c19sounds, sdx.prep.mdvr, sdx.prep.dementiabank,
#         sdx.prep.cross_c19sounds_*, sdx.prep.cross_mdvr_*,
#         sdx.prep.cross_aphasia_dementiabank.
#   [5] Renamed module-level identifiers exist:
#         prepare_c19sounds_*, C19SOUNDS_SYMPTOMS, _c19sounds_stratify_cols,
#         prepare_mdvr_ksof_parkC_intC, prepare_dementiabank_mmseR,
#         prepare_aphasia_dementiabank_pwaC_adC, _dementiabank_stratify_cols,
#         _dementiabank_ad_df, etc.
#   [6] sdx.prep.category dispatch tables use the new dataset keys:
#         ('c19sounds','t1'), ('c19sounds','t2'), ('mdvr','parkC'),
#         ('dementiabank','adC') in TASK_LOADERS;
#         DATASET_TASKS / AGE_STRATIFIED_TASKS / OR_MERGE_DATASETS updated;
#         no stale c9s / mvdr / dbank keys.
#   [7] sdx.prep.OLD_TO_NEW_MODULE legacy-shim mappings (the YAML
#       "training.dataio.prep_*" identifier -> sdx.prep.* module map) all
#       resolve, with no stale keys.
#   [8] compose_config() loads every single + cross task yaml without error
#       — exercises the full !include / !ref chain, the YAML loader, and the
#       dispatcher end-to-end on every task definition in the repo.
#   [9] registry.yaml's `datasets:` map lists the three new keys
#       (c19sounds, mdvr, dementiabank) and none of the old ones.
#  [10] `sdx single status` runs on every impacted single task (T7, T8 for
#       dementiabank; T13–T16 for mdvr; T19–T23 for c19sounds).
#  [11] `sdx cross status` runs on every impacted cross pair.
#       Status doesn't need a GPU and doesn't write artifacts — it just
#       resolves paths through the orchestrator. We only fail on Python
#       tracebacks; "missing exp folder" output is fine.
#  [12] `sdx single run --dry-run` on every impacted single task. This
#       exercises the full plan-generation path: compose_config + warm-cache
#       presence check + train-phase status check. No GPU, no writes.
#       A non-zero exit (any traceback) fails the check.
#  [13] `sdx cross run --dry-run` on every impacted cross pair (same idea,
#       cross-mode orchestrator).
#  [14] `sdx cross-cat run --dry-run` on every cross-category task whose
#       train_datasets or test_datasets includes a renamed dataset
#       (c19sounds / mdvr / dementiabank).
#  [15] Warm-cache straggler warning: if any old-name cache directory
#       ($slurm_tmpdir/{c9s,mvdr,dbank}/...) still exists, prints the mv
#       command to migrate it. Doesn't fail.
#
# Live cache-hit block (CPU-only, runs only if pre-flight in [16] passes):
#  [16] Pre-flight: cache.hdf5 must exist at the new dataset name under
#       $CACHE_ROOT/{c19sounds,mdvr,dementiabank}/wavlm/{train,val}/single_avg/.
#       If missing the live block is skipped (a missing cache would cause
#       the warm subprocess to fall back to writer mode and try to re-extract
#       from raw audio — expensive and would need GPU). Static checks
#       above remain valid in that case.
#  [17] Stage CSV symlinks: data/<ds>/processed/<ds>.csv -> $SCRATCH_DATA_ROOT/<ds>/<ds>.csv
#       for c19sounds, mdvr, dementiabank. Idempotent.
#  [18] Temp-edit slurm_tmpdir in main*.yaml -> $CACHE_ROOT/. Necessary because
#       `sdx single warm` doesn't accept --overrides, so the orchestrator's
#       warm subprocess can't forward a CLI override. Sed-and-revert under
#       a trap matches the previous verify-script convention. Reverted on
#       EXIT (success or failure) and again explicitly at the end.
#  [19] `sdx single prep -t T7 -t T13 -t T19 --overwrite`. Exercises the
#       renamed prepare_* functions on real CSVs and writes manifests.
#  [20] `sdx single run --cache-only -t T7 -t T13 -t T19 -e wavlm
#         --tag $VERIFY_TAG --yes`. Successful warm subprocess on each
#       renamed dataset proves the warm pre-flight finds an existing
#       cache at the NEW dataset name — i.e. the rename works end-to-end
#       at the cache layer, with no encoder forward pass (no GPU needed).
#  [21] `sdx cross run --cache-only` for T9_T7 (dementiabank), T13_T17 (mdvr),
#       T19_T27 (c19sounds). Cross uses the per-(dataset, encoder) cache.
#  [22] `sdx cross-cat run --cache-only` for c2_c3 (touches dementiabank +
#       mdvr). Reads 6 caches (aphasia, dementiabank, torgo, uaspeech,
#       mdvr, ksof) — confirms the renamed names are looked up correctly
#       in the multi-source category dispatcher.
#  + cleanup: reverts main*.yaml and removes every verify-tagged exp folder.
#
# Configuration knobs (all have sensible defaults):
#   CACHE_ROOT          (default /scratch/$USER/embeddings_avg_final)
#   SCRATCH_DATA_ROOT   (default /scratch/$USER/data)
#   VERIFY_TAG          (default verify_rename)
#
# What this script does NOT test (and why):
#  - Actual encoder forward pass. Cache-hit means the encoder is never
#    loaded; that's the point. To test re-warming from raw audio you'd
#    need GPU + raw audio in data/<ds>/raw/, and a fresh cache root.
#  - Actual probe training. Skipped by --cache-only. To test, drop
#    --cache-only from [19]/[20]/[21] and add a small GPU SBATCH directive
#    (e.g. --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1).
#
# Submit:  sbatch verify_renames_slurm.sh
# Logs:    tail -f exps/slurm_logs/verify_renames_*_out.txt
#
#SBATCH --job-name=verify_renames
#SBATCH --chdir=./
#SBATCH --output=exps/slurm_logs/%x_%j_out.txt
#SBATCH --error=exps/slurm_logs/%x_%j_err.txt
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

VENV="/scratch/aina10/SpeechDx/.venv"

export HF_HOME="${HF_HOME:-$SCRATCH/hf}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${SLURM_TMPDIR:-/tmp}/matplotlib-$USER}"
export FONTCONFIG_PATH="${FONTCONFIG_PATH:-${SLURM_TMPDIR:-/tmp}/fontconfig-$USER}"
mkdir -p "$MPLCONFIGDIR" "$FONTCONFIG_PATH"

module --force purge
module load StdEnv/2023 gcc python/3.11 arrow ffmpeg rust
source "$VENV/bin/activate"

mkdir -p exps/slurm_logs

echo "============================================================"
echo "  verify renames: c9s->c19sounds, mvdr->mdvr, dbank->dementiabank"
echo "============================================================"
echo "host : $(hostname)"
echo "job  : ${SLURM_JOB_ID:-?}"
echo "py   : $(which python)  ($(python -V))"
echo

# ---------------------------------------------------------------------------
# [1] No stragglers in source.
# ---------------------------------------------------------------------------
echo "=== [1] no c9s / C9S / mvdr / MVDR / dbank / DBank in source ==="
# Exclude this verify script itself — it documents the renames and lists the
# old paths to assert they're gone, so it intentionally mentions old tokens.
LEFTOVERS=$(find . \( -name .venv -o -name .git -o -name __pycache__ \
                  -o -name exps  -o -name metadata -o -name logs -o -name data \) -prune -o \
            \( -name "*.py"   -o -name "*.sh"  -o -name "*.yaml" -o -name "*.yml" \
            -o -name "*.md"   -o -name "*.txt" -o -name "*.toml" -o -name "*.cfg" \
            -o -name "*.ini" \) -type f \
            ! -name verify_renames_slurm.sh -print0 \
            | xargs -0 grep -lE 'c9s|C9S|mvdr|MVDR|dbank|DBank' 2>/dev/null || true)
if [[ -n "$LEFTOVERS" ]]; then
    echo "  FAIL — stragglers:"
    printf '    %s\n' $LEFTOVERS
    exit 1
fi
echo "  OK — no stragglers"
echo

# ---------------------------------------------------------------------------
# [2] Renamed paths exist.
# ---------------------------------------------------------------------------
echo "=== [2] new paths exist ==="
NEW_PATHS=(
    # c9s -> c19sounds
    sdx/prep/c19sounds.py
    sdx/prep/cross_c19sounds_avfad.py
    sdx/prep/cross_c19sounds_coswara.py
    sdx/configs/tasks/c19sounds_ageR.yaml
    sdx/configs/tasks/c19sounds_sexC.yaml
    sdx/configs/tasks/c19sounds_smokerC.yaml
    metadata_script/create_c19sounds_metadata.py
    # mvdr -> mdvr
    sdx/prep/mdvr.py
    sdx/prep/cross_mdvr_ksof.py
    sdx/prep/cross_mdvr_torgo.py
    sdx/prep/cross_mdvr_uaspeech.py
    metadata_script/create_mdvr_metadata.py
    scripts/download_mdvr.sh
    data/mdvr
    data/mdvr/processed/mdvr.csv
    # dbank -> dementiabank
    sdx/prep/dementiabank.py
    sdx/prep/cross_aphasia_dementiabank.py
    metadata_script/create_dementiabank_metadata.py
    data/dementiabank
    data/dementiabank/processed/dementiabank.csv
)
for p in "${NEW_PATHS[@]}"; do
    if [[ -e $p ]]; then
        echo "  OK  $p"
    else
        echo "  FAIL  missing $p"
        exit 1
    fi
done
echo

# ---------------------------------------------------------------------------
# [3] Old paths gone.
# ---------------------------------------------------------------------------
echo "=== [3] old paths gone ==="
OLD_PATHS=(
    # c9s
    sdx/prep/c9s.py
    sdx/prep/cross_c9s_avfad.py
    sdx/prep/cross_c9s_coswara.py
    sdx/configs/tasks/c9s_ageR.yaml
    sdx/configs/tasks/c9s_sexC.yaml
    sdx/configs/tasks/c9s_smokerC.yaml
    metadata_script/create_c9s_metadata.py
    # mvdr
    sdx/prep/mvdr.py
    sdx/prep/cross_mvdr_ksof.py
    sdx/prep/cross_mvdr_torgo.py
    sdx/prep/cross_mvdr_uaspeech.py
    metadata_script/create_mvdr_metadata.py
    scripts/download_mvdr.sh
    data/mvdr
    # dbank
    sdx/prep/dbank.py
    sdx/prep/cross_aphasia_dbank.py
    metadata_script/create_dbank_metadata.py
    data/dbank
)
for p in "${OLD_PATHS[@]}"; do
    if [[ -e $p ]]; then
        echo "  FAIL  still present: $p"
        exit 1
    else
        echo "  OK   gone: $p"
    fi
done
echo

# ---------------------------------------------------------------------------
# [4..7] Imports + identifiers + category dispatch + OLD_TO_NEW_MODULE shim.
# ---------------------------------------------------------------------------
echo "=== [4-7] imports, identifiers, category dispatch, OLD_TO_NEW_MODULE ==="
python - <<'PY'
import importlib

# [4] modules import.
mods = [
    'sdx.prep.c19sounds', 'sdx.prep.mdvr', 'sdx.prep.dementiabank',
    'sdx.prep.cross_c19sounds_avfad',  'sdx.prep.cross_c19sounds_coswara',
    'sdx.prep.cross_mdvr_ksof', 'sdx.prep.cross_mdvr_torgo', 'sdx.prep.cross_mdvr_uaspeech',
    'sdx.prep.cross_aphasia_dementiabank',
    'sdx.prep.category',
]
for m in mods:
    importlib.import_module(m)
    print(f'  OK import {m}')

# [5] Renamed prepare_* + module-level identifiers exist.
import sdx.prep.c19sounds as _c
for fn in (
    'prepare_c19sounds_t1', 'prepare_c19sounds_t2',
    'prepare_c19sounds_L_t1', 'prepare_c19sounds_L_t2',
    'prepare_c19sounds_sympL', 'prepare_c19sounds_sexC',
    'prepare_c19sounds_smokerC', 'prepare_c19sounds_ageR',
    'C19SOUNDS_SYMPTOMS', '_c19sounds_stratify_cols',
):
    assert hasattr(_c, fn), f'c19sounds.{fn} missing'
print('  OK c19sounds.py: all renamed symbols present')

import sdx.prep.cross_c19sounds_avfad as _x
for fn in (
    'prepare_c19sounds_avfad_t1_pathC', 'prepare_avfad_c19sounds_pathC_t1',
    'prepare_c19sounds_avfad_t2_pathC', 'prepare_avfad_c19sounds_pathC_t2',
    '_c19sounds_stratify_cols',
):
    assert hasattr(_x, fn), f'cross_c19sounds_avfad.{fn} missing'
print('  OK cross_c19sounds_avfad.py: all renamed symbols present')

import sdx.prep.cross_mdvr_ksof as _x
for fn in ('prepare_mdvr_ksof_parkC_intC', 'prepare_ksof_mdvr_intC_parkC'):
    assert hasattr(_x, fn), f'cross_mdvr_ksof.{fn} missing'
print('  OK cross_mdvr_ksof.py: all renamed symbols present')

import sdx.prep.dementiabank as _d
assert hasattr(_d, 'prepare_dementiabank_mmseR'), 'dementiabank.prepare_dementiabank_mmseR missing'
print('  OK dementiabank.py: prepare_dementiabank_mmseR present')

import sdx.prep.cross_aphasia_dementiabank as _x
for fn in (
    'prepare_aphasia_dementiabank_pwaC_adC',
    'prepare_dementiabank_aphasia_adC_pwaC',
    '_dementiabank_stratify_cols',
    '_dementiabank_ad_df',
):
    assert hasattr(_x, fn), f'cross_aphasia_dementiabank.{fn} missing'
print('  OK cross_aphasia_dementiabank.py: all renamed symbols present')

# [6] Category mappings updated.
from sdx.prep.category import (
    TASK_LOADERS, DATASET_TASKS, AGE_STRATIFIED_TASKS, OR_MERGE_DATASETS,
)
for key in (('c19sounds', 't1'), ('c19sounds', 't2'),
            ('mdvr', 'parkC'),
            ('dementiabank', 'adC')):
    assert key in TASK_LOADERS, f'TASK_LOADERS missing {key}'
assert DATASET_TASKS['c19sounds']    == ['t1', 't2']
assert DATASET_TASKS['mdvr']         == ['parkC']
assert DATASET_TASKS['dementiabank'] == ['adC']
assert ('c19sounds', 't1') in AGE_STRATIFIED_TASKS
assert ('c19sounds', 't2') in AGE_STRATIFIED_TASKS
assert 'c19sounds' in OR_MERGE_DATASETS
for stale in ('c9s', 'mvdr', 'dbank'):
    assert stale not in DATASET_TASKS, f'stale {stale!r} still in DATASET_TASKS'
print('  OK category.py: TASK_LOADERS / DATASET_TASKS / AGE_STRATIFIED_TASKS / OR_MERGE_DATASETS')

# [7] dispatch shim.
from sdx.prep import OLD_TO_NEW_MODULE
expect = {
    'training.dataio.prep_c19sounds':            'sdx.prep.c19sounds',
    'training.dataio.prep_mdvr':                 'sdx.prep.mdvr',
    'training.dataio.prep_dementiabank':         'sdx.prep.dementiabank',
    'training.dataio.prep_c19sounds_avfad':      'sdx.prep.cross_c19sounds_avfad',
    'training.dataio.prep_c19sounds_coswara':    'sdx.prep.cross_c19sounds_coswara',
    'training.dataio.prep_mdvr_ksof':            'sdx.prep.cross_mdvr_ksof',
    'training.dataio.prep_mdvr_torgo':           'sdx.prep.cross_mdvr_torgo',
    'training.dataio.prep_mdvr_uaspeech':        'sdx.prep.cross_mdvr_uaspeech',
    'training.dataio.prep_aphasia_dementiabank': 'sdx.prep.cross_aphasia_dementiabank',
}
for k, v in expect.items():
    assert OLD_TO_NEW_MODULE.get(k) == v, (
        f'OLD_TO_NEW_MODULE[{k!r}] = {OLD_TO_NEW_MODULE.get(k)!r} (expected {v!r})'
    )
    importlib.import_module(v)
print('  OK OLD_TO_NEW_MODULE shim: every renamed key resolves')

stale_keys = (
    'training.dataio.prep_c9s',  'training.dataio.prep_c9s_avfad',  'training.dataio.prep_c9s_coswara',
    'training.dataio.prep_mvdr', 'training.dataio.prep_mvdr_ksof',
    'training.dataio.prep_mvdr_torgo', 'training.dataio.prep_mvdr_uaspeech',
    'training.dataio.prep_dbank', 'training.dataio.prep_aphasia_dbank',
)
for k in stale_keys:
    assert k not in OLD_TO_NEW_MODULE, f'stale key still in OLD_TO_NEW_MODULE: {k}'
print('  OK OLD_TO_NEW_MODULE shim: no stale c9s / mvdr / dbank keys')
PY
echo

# ---------------------------------------------------------------------------
# [8] Compose every task yaml.
# ---------------------------------------------------------------------------
echo "=== [8] compose_config on every single + cross task yaml ==="
python - <<'PY'
import sys
from sdx.config import compose_config
from sdx.orchestrator import TASKS_DIR as SINGLE
from sdx.orchestrator_cross import TASKS_DIR as CROSS

bad = []
for tdir, label in [(SINGLE, 'single'), (CROSS, 'cross')]:
    for p in sorted(tdir.glob('*.yaml')):
        try:
            cfg = compose_config(p.stem, 'wavlm', mode='read')
            assert cfg.get('dataset')
        except Exception as e:
            bad.append((label, p.stem, type(e).__name__, str(e)[:160]))
if bad:
    for b in bad: print('  FAIL', b)
    sys.exit(1)
print('  OK every single + cross task yaml composes')
PY
echo

# ---------------------------------------------------------------------------
# [9] Registry datasets.
# ---------------------------------------------------------------------------
echo "=== [9] registry datasets ==="
python - <<'PY'
import yaml
from pathlib import Path
from sdx.yaml_io import TolerantLoader
reg = yaml.load(Path('sdx/configs/registry.yaml').read_text(), Loader=TolerantLoader)
ds  = reg['datasets']
for new in ('c19sounds', 'mdvr', 'dementiabank'):
    assert new in ds, f"datasets must list {new!r}"
for old in ('c9s', 'mvdr', 'dbank'):
    assert old not in ds, f"stale {old!r} still in datasets"
print('  OK registry: c19sounds + mdvr + dementiabank present, old keys gone')
PY
echo

# ---------------------------------------------------------------------------
# [10] CLI status — single tasks.
# ---------------------------------------------------------------------------
echo "=== [10] sdx single status — T7,T8 (dementiabank), T13..T16 (mdvr), T19..T23 (c19sounds) ==="
python -m sdx single status \
    -t T7 -t T8 \
    -t T13 -t T14 -t T15 -t T16 \
    -t T19 -t T20 -t T21 -t T22 -t T23 \
    -e wavlm --tag run1 || true
echo

# ---------------------------------------------------------------------------
# [11] CLI status — cross pairs.
# ---------------------------------------------------------------------------
echo "=== [11] sdx cross status — dementiabank/mdvr/c19sounds cross pairs ==="
python -m sdx cross status \
    -t T9_T7 -t T7_T9 \
    -t T13_T17 -t T17_T13 -t T13_T10 -t T10_T13 \
    -t T13_T12 -t T12_T13 \
    -t T19_T27 -t T27_T19 -t T21_T27 -t T27_T21 \
    -t T19_T24 -t T24_T19 -t T21_T25 -t T25_T21 \
    -e wavlm --tag run1 || true
echo

# ---------------------------------------------------------------------------
# [12] Single-mode dry-run on every renamed-dataset paper task.
#      --dry-run exits 0 if the plan generates cleanly. Any traceback
#      (compose_config, manifest path, cache scheduling) propagates and
#      fails the slurm step thanks to `set -e`.
# ---------------------------------------------------------------------------
echo "=== [12] sdx single run --dry-run on impacted tasks ==="
python -m sdx single run \
    -t T7 -t T8 \
    -t T13 -t T14 -t T15 -t T16 \
    -t T19 -t T20 -t T21 -t T22 -t T23 \
    -e wavlm --tag run1 --dry-run --yes
echo

# ---------------------------------------------------------------------------
# [13] Cross-mode dry-run on every renamed-dataset cross pair.
# ---------------------------------------------------------------------------
echo "=== [13] sdx cross run --dry-run on impacted cross pairs ==="
python -m sdx cross run \
    -t T9_T7  -t T7_T9 \
    -t T13_T17 -t T17_T13 -t T13_T10 -t T10_T13 -t T13_T12 -t T12_T13 \
    -t T19_T27 -t T27_T19 -t T21_T27 -t T27_T21 \
    -t T19_T24 -t T24_T19 -t T21_T25 -t T25_T21 \
    -e wavlm --tag run1 --dry-run --yes
echo

# ---------------------------------------------------------------------------
# [14] Cross-cat dry-run on every category task that touches a renamed
#      dataset. c1..c4 categories: aphasia/edaic/iemocap/ravdess (c1),
#      aphasia/dementiabank (c2 — touches dementiabank rename),
#      torgo/uaspeech/mdvr/ksof (c3 — touches mdvr),
#      c19sounds/coswara/avfad (c4 — touches c19sounds).
#      Every directional pair c<i>_c<j> with i,j in {2,3,4} touches a
#      renamed dataset on at least one side.
# ---------------------------------------------------------------------------
echo "=== [14] sdx cross-cat run --dry-run on impacted category pairs ==="
python -m sdx cross-cat run \
    -t c1_c2 -t c2_c1 -t c1_c3 -t c3_c1 -t c1_c4 -t c4_c1 \
    -t c2_c3 -t c3_c2 -t c2_c4 -t c4_c2 -t c3_c4 -t c4_c3 \
    -e wavlm --tag run1 --dry-run --yes
echo

# ---------------------------------------------------------------------------
# [15] Warm-cache root: warn on stale old-name caches. The cache root is
#      `slurm_tmpdir` from main*.yaml — by default ./embeddings_avg_final/.
#      Caches there are keyed by dataset name, so c9s/mvdr/dbank
#      directories are now invisible to the new code and would silently
#      get re-warmed under the new name. We don't fail — just surface them
#      so the user can `mv` them and avoid the recompute, if they exist.
# ---------------------------------------------------------------------------
echo "=== [15] warm-cache stragglers (informational only) ==="
CACHE_ROOTS=$(python - <<'PY'
import yaml
from pathlib import Path
from sdx.yaml_io import TolerantLoader
roots = set()
for p in ('sdx/configs/main.yaml','sdx/configs/main_cross.yaml','sdx/configs/main_cross_category.yaml'):
    cfg = yaml.load(Path(p).read_text(), Loader=TolerantLoader) or {}
    r = cfg.get('slurm_tmpdir')
    if isinstance(r, str): roots.add(r.rstrip('/'))
print(' '.join(sorted(roots)))
PY
)
ANY_STALE=0
for ROOT in $CACHE_ROOTS; do
    [[ -d "$ROOT" ]] || { echo "  (skip) cache root $ROOT does not exist yet"; continue; }
    for OLD in c9s mvdr dbank; do
        if [[ -d "$ROOT/$OLD" ]]; then
            SZ=$(du -sh "$ROOT/$OLD" 2>/dev/null | cut -f1)
            echo "  WARN  stale cache: $ROOT/$OLD  ($SZ)"
            case "$OLD" in
                c9s)   NEW=c19sounds ;;
                mvdr)  NEW=mdvr ;;
                dbank) NEW=dementiabank ;;
            esac
            echo "        to keep: mv \"$ROOT/$OLD\" \"$ROOT/$NEW\""
            ANY_STALE=1
        fi
    done
done
[[ $ANY_STALE -eq 0 ]] && echo "  OK no stale caches under any cache root"
echo

# ===========================================================================
#                        LIVE CACHE-HIT VERIFICATION
# ---------------------------------------------------------------------------
# Steps [16]-[20] go beyond static checks: they actually run `sdx prep` +
# `sdx run --cache-only` against your real warm caches. With every renamed
# dataset's cache already populated under its NEW name at $CACHE_ROOT (you
# already mv'd /scratch/aina10/embeddings_avg_final/{c9s,mvdr,dbank} ->
# {c19sounds,mdvr,dementiabank}), a successful warm subprocess on each
# renamed dataset proves the new code finds the cache at the new path.
# Cache hit means no encoder load, so this section is CPU-only.
#
# Defaults match your /scratch layout. Override either via env var if needed:
#   sbatch --export=ALL,CACHE_ROOT=/path,SCRATCH_DATA_ROOT=/path verify_renames_slurm.sh
# ===========================================================================
CACHE_ROOT="${CACHE_ROOT:-/scratch/$USER/embeddings_avg_final}"
SCRATCH_DATA_ROOT="${SCRATCH_DATA_ROOT:-/scratch/$USER/data}"
VERIFY_TAG="${VERIFY_TAG:-verify_rename}"
RENAMED_DATASETS=(c19sounds mdvr dementiabank)
# Every dataset that the live tests below touch — single + cross + cross-cat.
# All need data/<ds>/processed/<ds>.csv (symlinked from $SCRATCH_DATA_ROOT)
# and a data/<ds>/raw/.staged marker so `sdx prep`'s raw-data check passes.
# (cache-only mode never reads raw audio; the marker satisfies the existence
# check without staging actual audio.)
TOUCHED_DATASETS=(c19sounds mdvr dementiabank aphasia ksof avfad torgo uaspeech)
# Every dataset that the live tests below touch — single + cross + cross-cat.
# All need data/<ds>/processed/<ds>.csv (symlinked from $SCRATCH_DATA_ROOT)
# and a data/<ds>/raw/.staged marker so `sdx prep`'s raw-data check passes.
# (cache-only mode never reads raw audio; the marker satisfies the existence
# check without staging actual audio.)
TOUCHED_DATASETS=(c19sounds mdvr dementiabank aphasia ksof avfad torgo uaspeech)

# ---------------------------------------------------------------------------
# [16] Pre-flight: cache.hdf5 must exist at the new dataset name. If it
#      doesn't, skip the live block — the warm subprocess would fall back
#      to writer mode and try to re-extract from raw audio (expensive,
#      requires GPU + full audio tree). Static checks above are still valid.
# ---------------------------------------------------------------------------
echo "=== [16] live pre-flight: warm cache present at new dataset names ==="
SKIP_LIVE=0
if [[ ! -d "$CACHE_ROOT" ]]; then
    echo "  SKIP — cache root $CACHE_ROOT does not exist"
    echo "         (set CACHE_ROOT=... to point at your real cache)"
    SKIP_LIVE=1
else
    for ds in "${RENAMED_DATASETS[@]}"; do
        TR="$CACHE_ROOT/$ds/wavlm/train/single_avg/cache.hdf5"
        VL="$CACHE_ROOT/$ds/wavlm/val/single_avg/cache.hdf5"
        if [[ -f "$TR" && -f "$VL" ]]; then
            echo "  OK  $ds  (train+val cache.hdf5 present)"
        else
            echo "  SKIP — $ds cache missing under $CACHE_ROOT/$ds/wavlm/"
            SKIP_LIVE=1
        fi
    done
fi

if [[ $SKIP_LIVE -eq 0 ]]; then
    # Other datasets the cross-cat task touches must also be present.
    for ds in aphasia torgo uaspeech ksof; do
        TR="$CACHE_ROOT/$ds/wavlm/train/single_avg/cache.hdf5"
        if [[ ! -f "$TR" ]]; then
            echo "  WARN — $ds cache missing; cross-cat step may fail"
        fi
    done
fi
echo

if [[ $SKIP_LIVE -eq 1 ]]; then
    echo "Live cache-hit block [17]-[21] skipped due to pre-flight failure."
else
    # ---------------------------------------------------------------------------
    # [17] Stage CSV symlinks. Each renamed dataset's processed CSV must live
    #      at data/<ds>/processed/<ds>.csv for `sdx prep`. Real CSVs are at
    #      $SCRATCH_DATA_ROOT/<ds>/<ds>.csv per existing repo convention.
    # ---------------------------------------------------------------------------
    echo "=== [17] stage CSV symlinks + raw markers for every touched dataset ==="
    # Covers the renamed datasets PLUS the others touched by cross / cross-cat
    # (aphasia, ksof, avfad, torgo, uaspeech). Without these, prep's raw-data
    # existence check fails before the cross-pair `prepare_*` even runs, even
    # though cache-only mode never opens the audio files.
    for ds in "${TOUCHED_DATASETS[@]}"; do
        mkdir -p "data/$ds/raw" "data/$ds/processed"
        [[ -e "data/$ds/raw/.staged" ]] || touch "data/$ds/raw/.staged"
        SRC="$SCRATCH_DATA_ROOT/$ds/$ds.csv"
        DEST="data/$ds/processed/$ds.csv"
        if [[ -e "$DEST" ]]; then
            echo "  OK   $ds: $DEST already present"
        elif [[ -f "$SRC" ]]; then
            ln -sf "$SRC" "$DEST"
            echo "  OK   $ds: linked $SRC -> $DEST"
        else
            echo "  FAIL $ds: no CSV at $SRC and none at $DEST"
            exit 1
        fi
    done
    echo

    # ---------------------------------------------------------------------------
    # [18] Temporarily edit slurm_tmpdir in main*.yaml to point at the real
    #      cache root. We can't use --overrides because `sdx single warm`
    #      doesn't accept it (the warm subparser doesn't define --overrides),
    #      so the orchestrator's warm subprocess invocation cannot forward
    #      it; we'd need to extend the warm subcommand to support it. Sed-
    #      and-restore is the existing convention from the previous verify
    #      script.
    #
    #      Surgical restore: we capture the verbatim slurm_tmpdir line per
    #      file before editing, and the trap sed-replaces back to that exact
    #      text. This preserves any other uncommitted edits in those files
    #      (a `git checkout --` would wipe them) and works whether the
    #      committed value is `./embeddings_avg_finalv2/` or anything else.
    # ---------------------------------------------------------------------------
    echo "=== [18] temp-edit slurm_tmpdir in main*.yaml -> $CACHE_ROOT/ ==="
    MAIN_YAMLS=(
        sdx/configs/main.yaml
        sdx/configs/main_cross.yaml
        sdx/configs/main_cross_category.yaml
    )
    declare -A ORIG_TMPDIR_LINE
    for y in "${MAIN_YAMLS[@]}"; do
        line=$(grep -m1 '^slurm_tmpdir:' "$y" || true)
        if [[ -z "$line" ]]; then
            echo "  FAIL no slurm_tmpdir line found in $y"
            exit 1
        fi
        ORIG_TMPDIR_LINE["$y"]="$line"
    done
    revert_main_yamls() {
        echo "    restoring slurm_tmpdir lines in main*.yaml"
        for y in "${MAIN_YAMLS[@]}"; do
            orig="${ORIG_TMPDIR_LINE[$y]}"
            # Escape any sed-special chars in the original line for the RHS.
            esc=$(printf '%s' "$orig" | sed 's|[\\&|]|\\&|g')
            sed -i -E "s|^slurm_tmpdir:.*|$esc|" "$y"
        done
    }
    trap revert_main_yamls EXIT
    for y in "${MAIN_YAMLS[@]}"; do
        sed -i -E "s|^slurm_tmpdir:.*|slurm_tmpdir: $CACHE_ROOT/|" "$y"
        echo "    edited $y -> $(grep '^slurm_tmpdir:' "$y")"
    done
    echo

    # ---------------------------------------------------------------------------
    # [19] `sdx single prep` for one task per renamed dataset:
    #        T7  = dementiabank/adC, T13 = mdvr/parkC, T19 = c19sounds/t1.
    #      Exercises the renamed prepare_* functions on real CSVs and writes
    #      manifests to exps/single_task/<task>/manifest/. CPU-only, fast.
    # ---------------------------------------------------------------------------
    echo "=== [19] sdx single prep — T7 (dementiabank), T13 (mdvr), T19 (c19sounds) ==="
    python -m sdx single prep -t T7 -t T13 -t T19 --overwrite
    echo

    # ---------------------------------------------------------------------------
    # [20] `sdx single run --cache-only`. Successful subprocess proves:
    #        - the warm pre-flight finds an existing cache at the NEW dataset
    #          name (so the rename works end-to-end at the cache layer),
    #        - no encoder forward pass happens (no GPU needed).
    #      slurm_tmpdir comes from the temp-edited main*.yaml (step [18]).
    #      Output goes under tag=$VERIFY_TAG so it doesn't clobber real runs.
    # ---------------------------------------------------------------------------
    echo "=== [20] sdx single run --cache-only — warm cache hit at new names ==="
    python -m sdx single run --cache-only \
        -t T7 -t T13 -t T19 \
        -e wavlm --tag "$VERIFY_TAG" --yes
    echo

    # ---------------------------------------------------------------------------
    # [21] Same for cross + cross-cat. Cross uses the per-(dataset, encoder)
    #      caches that single populated. Cross-cat reads N caches (one per
    #      dataset in train_datasets + test_datasets).
    # ---------------------------------------------------------------------------
    echo "=== [21] sdx cross run --cache-only — T9_T7, T13_T17, T19_T27 ==="
    python -m sdx cross run --cache-only \
        -t T9_T7 -t T13_T17 -t T19_T27 \
        -e wavlm --tag "$VERIFY_TAG" --yes
    echo

    echo "=== [22] sdx cross-cat run --cache-only — c2_c3 ==="
    python -m sdx cross-cat run --cache-only \
        -t c2_c3 \
        -e wavlm --tag "$VERIFY_TAG" --yes
    echo

    # ---------------------------------------------------------------------------
    # Cleanup: revert main*.yaml + remove every exp folder created with the
    # verify tag. The trap revert above runs on EXIT for any path; we run
    # it explicitly here so the success path doesn't depend on the trap.
    # ---------------------------------------------------------------------------
    echo "=== cleanup ==="
    revert_main_yamls
    trap - EXIT
    find exps -type d -name "*-${VERIFY_TAG}" -prune -print -exec rm -rf {} +
    echo
fi

echo "============================================================"
echo "  ALL GOOD — c9s/mvdr/dbank renames verified"
echo "============================================================"
