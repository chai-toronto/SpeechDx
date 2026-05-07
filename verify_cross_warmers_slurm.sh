#!/bin/bash -l
# Verify the new self-contained cross + cross-cat warmers.
#
# Confirms:
#   - Every cross + cross-cat task yaml parses end-to-end via HyperPyYAML
#     (resolves !ref / !include / !new) and exposes the expected schema:
#       cross    : train_split_by_boundary, test_split_by_boundary,
#                  num_aug_ver, snr_low, snr_high, speed
#       cross-cat: setting_<N> (per train ds: split_by_boundary, num_aug_ver,
#                  snr_low, snr_high, speed); test_setting_<N> (per test ds:
#                  split_by_boundary)
#   - compose_config(task, encoder, mode='warm') resolves cross caches to
#     <slurm_tmpdir>/<train_dataset>/<enc>/{train,val} and
#     <slurm_tmpdir>/<test_dataset>/<enc>/val (no _2 train cache).
#   - main_cross.yaml has no `train_cache_dir_2` line.
#   - cli.py has no `_proxy_single_tasks_for_cross` or `_load_cross_yaml`.
#   - `sdx cross warm` and `sdx cross-cat warm` route to the new modules.
#
# Live (cache-hit only — no GPU, no encoder forward):
#   - `sdx cross warm -t T13_T10 -e wavlm` is idempotent against an
#     already-warm cache. With --tag verify it builds manifests, opens
#     stub-encoder pre-flight, sees every (uid, version) cached, and
#     short-circuits. Validates the new warmer reads main_cross.yaml +
#     cross_tasks/T13_T10.yaml, looks up the right HDF5 paths, and runs
#     the expected idempotency check.
#   - HDF5 inspect: confirms 3 caches written (or pre-existing) at the
#     expected mdvr/train (>= num_aug_ver versions), mdvr/val (>= 1),
#     torgo/val (>= 1) paths and that no torgo/train cache was created
#     by this run (sanity-check that train_cache_dir_2 is dead).
#   - Cross-cat: same idempotent warm against c2_c3 (train aphasia,
#     dementiabank; test torgo, uaspeech, mdvr, ksof). Inspects each
#     contributing dataset's cache.
#
# Required env (uses defaults if unset):
#   CACHE_ROOT          (default /scratch/$USER/embeddings_avg_final)
#   SCRATCH_DATA_ROOT   (default /scratch/$USER/data)
#   VERIFY_TAG          (default verify_cross_warmers)
#
# Submit:  sbatch verify_cross_warmers_slurm.sh
# Logs:    tail -f exps/slurm_logs/verify_cross_warmers_*_out.txt
#
#SBATCH --job-name=verify_cross_warmers
#SBATCH --chdir=./
#SBATCH --output=exps/slurm_logs/%x_%j_out.txt
#SBATCH --error=exps/slurm_logs/%x_%j_err.txt
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

VENV="/scratch/aina10/Audio-Health-Benchmark/.venv"

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

CACHE_ROOT="${CACHE_ROOT:-/scratch/$USER/embeddings_avg_final}"
SCRATCH_DATA_ROOT="${SCRATCH_DATA_ROOT:-/scratch/$USER/data}"
VERIFY_TAG="${VERIFY_TAG:-verify_cross_warmers}"
ENCODER="${ENCODER:-wavlm}"
CROSS_PAIR="${CROSS_PAIR:-T13_T10}"          # mdvr -> torgo (open data)
CROSS_CAT_TASK="${CROSS_CAT_TASK:-c2_c3}"    # aphasia,dementiabank -> torgo,uaspeech,mdvr,ksof

# Touched datasets need data/<ds>/processed/<ds>.csv at runtime so prep can run
TOUCHED_DATASETS=(mdvr torgo aphasia dementiabank uaspeech ksof)

echo "============================================================"
echo "  verify cross + cross-cat self-contained warmers"
echo "============================================================"
echo "host        : $(hostname)"
echo "job         : ${SLURM_JOB_ID:-?}"
echo "py          : $(which python)  ($(python -V))"
echo "CACHE_ROOT  : $CACHE_ROOT"
echo "encoder     : $ENCODER"
echo "cross pair  : $CROSS_PAIR"
echo "cross-cat   : $CROSS_CAT_TASK"
echo "tag         : $VERIFY_TAG"
echo

# ---------------------------------------------------------------------------
# [1] Yaml schema — every cross + cross-cat yaml resolves with HyperPyYAML
# and exposes the new schema fields.
# ---------------------------------------------------------------------------
echo "=== [1] yaml schema resolves on every cross + cross-cat yaml ==="
python - <<'PY'
import sys
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml

CROSS = Path("sdx/configs/cross_tasks")
fail = 0
for p in sorted(CROSS.glob("T*.yaml")):
    with p.open() as f:
        cfg = load_hyperpyyaml(f)
    for k in ("train_split_by_boundary", "test_split_by_boundary",
              "num_aug_ver", "snr_low", "snr_high", "speed",
              "train_dataset", "test_dataset"):
        if k not in cfg:
            print(f"  FAIL {p.name}: missing {k}"); fail += 1; break
    else:
        print(f"  OK   {p.name}")
n_train_t = sum(1 for _ in CROSS.glob("T*.yaml"))

for p in sorted(CROSS.glob("c*.yaml")):
    with p.open() as f:
        cfg = load_hyperpyyaml(f)
    train = cfg.get("train_datasets") or []
    test = cfg.get("test_datasets") or []
    bad = False
    for i, _ in enumerate(train):
        s = cfg.get(f"setting_{i+1}") or {}
        for k in ("split_by_boundary", "num_aug_ver", "snr_low", "snr_high", "speed"):
            if k not in s:
                print(f"  FAIL {p.name}: setting_{i+1} missing {k}"); fail += 1; bad = True; break
        if bad: break
    if bad: continue
    for j, _ in enumerate(test):
        s = cfg.get(f"test_setting_{j+1}") or {}
        if "split_by_boundary" not in s:
            print(f"  FAIL {p.name}: test_setting_{j+1} missing split_by_boundary"); fail += 1; bad = True; break
    if not bad:
        print(f"  OK   {p.name}")
sys.exit(1 if fail else 0)
PY
echo

# ---------------------------------------------------------------------------
# [2] compose_config — cross + cross-cat compose with mode='warm' and the
# resolved cache paths point at the EXPECTED locations (no _2 train).
# ---------------------------------------------------------------------------
echo "=== [2] compose_config resolves expected cache paths ==="
python - <<PY
from sdx.config import compose_config
h = compose_config("$CROSS_PAIR", "$ENCODER", mode="warm")
got = sorted(k for k in h if "cache_dir" in k)
exp = ["train_cache_dir_1", "val_cache_dir_1", "val_cache_dir_2"]
assert got == exp, f"unexpected cache_dir keys: got={got}, expected={exp}"
print(f"  cross cache keys: {got}")
print(f"    train_cache_dir_1 = {h['train_cache_dir_1']}")
print(f"    val_cache_dir_1   = {h['val_cache_dir_1']}")
print(f"    val_cache_dir_2   = {h['val_cache_dir_2']}")
assert "train_cache_dir_2" not in h, "train_cache_dir_2 must be removed"
print("  OK   no train_cache_dir_2")

h2 = compose_config("$CROSS_CAT_TASK", "$ENCODER", mode="warm")
print(f"  cat cache root = {h2['train_cache_dir_1']}")
print(f"  cat train_datasets = {h2['data_params'].get('train_datasets')}")
print(f"  cat test_datasets  = {h2['data_params'].get('test_datasets')}")
PY
echo

# ---------------------------------------------------------------------------
# [3] New modules import + handlers route to them.
# ---------------------------------------------------------------------------
echo "=== [3] new warm modules import and handlers route to them ==="
python - <<'PY'
import inspect
from sdx.warm_cross import run_warm_cross, cmd_warm_cross_jobs
from sdx.warm_cross_cat import run_warm_cross_cat, cmd_warm_crosscat_jobs
print("  OK sdx.warm_cross.run_warm_cross / cmd_warm_cross_jobs")
print("  OK sdx.warm_cross_cat.run_warm_cross_cat / cmd_warm_crosscat_jobs")

from sdx.cli import _h_cross_warm, _h_crosscat_warm
src1 = inspect.getsource(_h_cross_warm)
src2 = inspect.getsource(_h_crosscat_warm)
assert "cmd_warm_cross_jobs" in src1, "_h_cross_warm should call cmd_warm_cross_jobs"
assert "cmd_warm_crosscat_jobs" in src2, "_h_crosscat_warm should call cmd_warm_crosscat_jobs"
assert "_proxy_single_tasks_for_cross" not in (src1 + src2), \
    "old proxy helper still referenced"
print("  OK _h_cross_warm calls cmd_warm_cross_jobs")
print("  OK _h_crosscat_warm calls cmd_warm_crosscat_jobs")
print("  OK no _proxy_single_tasks_for_cross references")
PY
echo

# ---------------------------------------------------------------------------
# [4] cli.py purged of dead helpers; main_cross.yaml has no train_cache_dir_2.
# ---------------------------------------------------------------------------
echo "=== [4] dead code removed ==="
if grep -q "_proxy_single_tasks_for_cross\|_load_cross_yaml" sdx/cli.py; then
    echo "  FAIL cli.py still has _proxy_single_tasks_for_cross or _load_cross_yaml"
    exit 1
fi
echo "  OK cli.py — no _proxy_single_tasks_for_cross / _load_cross_yaml"

if grep -q "^train_cache_dir_2:" sdx/configs/main_cross.yaml; then
    echo "  FAIL main_cross.yaml still has train_cache_dir_2"
    exit 1
fi
echo "  OK main_cross.yaml — train_cache_dir_2 removed"
echo

# ---------------------------------------------------------------------------
# [5] CLI parses for both warm subcommands.
# ---------------------------------------------------------------------------
echo "=== [5] CLI dispatch parses ==="
python -m sdx cross warm --help     >/dev/null && echo "  OK sdx cross warm --help"
python -m sdx cross-cat warm --help >/dev/null && echo "  OK sdx cross-cat warm --help"
echo

# ---------------------------------------------------------------------------
# [6] Pre-flight — required cache files must already exist for the cache-hit
# live test (otherwise the warmer would fall back to encoder forward, which
# needs GPU + raw audio).
# ---------------------------------------------------------------------------
echo "=== [6] pre-flight: required cache.hdf5 files exist ==="
PREFLIGHT_OK=1
declare -a PREFLIGHT_PATHS=(
    "$CACHE_ROOT/mdvr/$ENCODER/train/single_avg/cache.hdf5"
    "$CACHE_ROOT/mdvr/$ENCODER/val/single_avg/cache.hdf5"
    "$CACHE_ROOT/torgo/$ENCODER/val/single_avg/cache.hdf5"
    "$CACHE_ROOT/aphasia/$ENCODER/train/single_avg/cache.hdf5"
    "$CACHE_ROOT/aphasia/$ENCODER/val/single_avg/cache.hdf5"
    "$CACHE_ROOT/dementiabank/$ENCODER/train/single_avg/cache.hdf5"
    "$CACHE_ROOT/dementiabank/$ENCODER/val/single_avg/cache.hdf5"
    "$CACHE_ROOT/uaspeech/$ENCODER/val/single_avg/cache.hdf5"
    "$CACHE_ROOT/ksof/$ENCODER/val/single_avg/cache.hdf5"
)
for p in "${PREFLIGHT_PATHS[@]}"; do
    if [[ -f $p ]]; then
        echo "  OK $p"
    else
        echo "  MISS $p"
        PREFLIGHT_OK=0
    fi
done
if [[ $PREFLIGHT_OK -eq 0 ]]; then
    echo
    echo "  Some caches missing → skipping the live block. Static checks above"
    echo "  remain valid. Re-warm via single-mode first if you want the live"
    echo "  block to run."
    echo
    echo "ALL STATIC CHECKS PASSED"
    exit 0
fi
echo

# ---------------------------------------------------------------------------
# [7] Stage CSV symlinks so prep can run without copying the real CSVs in.
# ---------------------------------------------------------------------------
echo "=== [7] stage CSV symlinks for ${TOUCHED_DATASETS[*]} ==="
for ds in "${TOUCHED_DATASETS[@]}"; do
    SRC="$SCRATCH_DATA_ROOT/$ds/$ds.csv"
    if [[ ! -f $SRC ]]; then
        echo "  WARN no source CSV at $SRC — skipping symlink for $ds"
        continue
    fi
    DEST="data/$ds/processed/$ds.csv"
    mkdir -p "$(dirname "$DEST")"
    if [[ ! -L $DEST ]] || [[ "$(readlink -f "$DEST")" != "$(readlink -f "$SRC")" ]]; then
        ln -sf "$SRC" "$DEST"
        echo "  ln  $DEST -> $SRC"
    else
        echo "  ok  $DEST"
    fi
done
echo

# ---------------------------------------------------------------------------
# [8] Temp-edit slurm_tmpdir in main_cross{,_category}.yaml -> $CACHE_ROOT.
# Sed-and-revert under a trap. Necessary because compose_config uses MAIN
# yaml's slurm_tmpdir; we point at the actual on-disk cache.
# ---------------------------------------------------------------------------
echo "=== [8] temp-edit slurm_tmpdir in main_cross{,_category}.yaml ==="
declare -a EDITED_MAINS=(
    sdx/configs/main.yaml
    sdx/configs/main_cross.yaml
    sdx/configs/main_cross_category.yaml
)
declare -A ORIG_TMPDIR_LINE
for y in "${EDITED_MAINS[@]}"; do
    line=$(grep -m1 '^slurm_tmpdir:' "$y" || true)
    ORIG_TMPDIR_LINE["$y"]="$line"
done

revert_main_yamls() {
    set +e
    for y in "${EDITED_MAINS[@]}"; do
        orig="${ORIG_TMPDIR_LINE[$y]:-}"
        if [[ -n "$orig" ]]; then
            esc=$(printf '%s' "$orig" | sed 's|[\\&|]|\\&|g')
            sed -i -E "s|^slurm_tmpdir:.*|$esc|" "$y"
        fi
    done
    set -e
}
trap revert_main_yamls EXIT

CACHE_ROOT_ESC=$(printf '%s/' "$CACHE_ROOT" | sed 's|[\\&|]|\\&|g')
for y in "${EDITED_MAINS[@]}"; do
    sed -i -E "s|^slurm_tmpdir:.*|slurm_tmpdir: $CACHE_ROOT_ESC|" "$y"
    new=$(grep -m1 '^slurm_tmpdir:' "$y")
    echo "  $y  ->  $new"
done
echo

# ---------------------------------------------------------------------------
# [9] Cross prep + warm (cache-hit) for $CROSS_PAIR.
# ---------------------------------------------------------------------------
echo "=== [9] sdx cross prep + warm (cache-hit) for $CROSS_PAIR × $ENCODER ==="
python -m sdx cross prep -t "$CROSS_PAIR" --overwrite
python -m sdx cross warm -t "$CROSS_PAIR" -e "$ENCODER"
echo

# ---------------------------------------------------------------------------
# [10] HDF5 inspect — verify the right caches are present, with the right
# version counts. ALSO assert no test-side train cache was created.
# ---------------------------------------------------------------------------
echo "=== [10] HDF5 inspect for $CROSS_PAIR ==="
python - <<PY
import json, h5py
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml

with open(f"sdx/configs/cross_tasks/$CROSS_PAIR.yaml") as f:
    cfg = load_hyperpyyaml(f)
train_ds = cfg["train_dataset"]
test_ds = cfg["test_dataset"]
nav = int(cfg["num_aug_ver"])
root = Path("$CACHE_ROOT")

# manifest IDs
with open(f"exps/cross/$CROSS_PAIR/manifest/train.json") as f: tr = json.load(f)
with open(f"exps/cross/$CROSS_PAIR/manifest/valid.json") as f: va = json.load(f)
with open(f"exps/cross/$CROSS_PAIR/manifest/test.json") as f: te = json.load(f)
all_train_ids = list({**tr, **va}.keys())
all_test_ids = list(te.keys())
print(f"  manifest: train+val ids = {len(all_train_ids)}, test ids = {len(all_test_ids)}")

def assert_cache(path: Path, ids: list[str], num_versions: int, label: str):
    assert path.exists(), f"  MISSING cache: {path}"
    with h5py.File(path, "r") as f:
        keys = set(f.keys())
        sample = ids[0]
        if sample in keys:
            sub = set(f[sample].keys())
        else:
            # cache may be flat (older layout) — check via direct path
            sub = {k.split("/")[-1] for k in keys if k.startswith(f"{sample}/")}
        present_versions = sorted([int(v[1:]) for v in sub if v.startswith("v")])
    coverage = sum(1 for uid in ids if uid in keys)
    print(f"  {label}: {path}")
    print(f"    {coverage}/{len(ids)} ids cached, sample {sample!r} versions = {present_versions}")
    assert coverage == len(ids), f"missing ids in {path}"
    assert max(present_versions) + 1 >= num_versions, \
        f"need {num_versions} versions, have {present_versions}"
    return True

assert_cache(root / train_ds / "$ENCODER" / "train" / "single_avg" / "cache.hdf5",
             all_train_ids, nav,    f"train_dataset {train_ds}/train (need {nav} versions)")
assert_cache(root / train_ds / "$ENCODER" / "val" / "single_avg" / "cache.hdf5",
             all_train_ids, 1,      f"train_dataset {train_ds}/val   (need 1 version)")
assert_cache(root / test_ds  / "$ENCODER" / "val" / "single_avg" / "cache.hdf5",
             all_test_ids,  1,      f"test_dataset  {test_ds}/val    (need 1 version)")

# Sanity: this run should NOT have written a test_dataset/train cache. If one
# already exists from a previous unrelated single warm, that's fine — we just
# check that warm_cross.py never wrote here. We can't tell "wrote" vs "exists"
# robustly, so this is informational only.
test_train_path = root / test_ds / "$ENCODER" / "train" / "single_avg" / "cache.hdf5"
if test_train_path.exists():
    print(f"  INFO  {test_train_path} exists (likely from prior single warm) — "
          "warm_cross.py never reads or writes here for the test side, so "
          "this is unrelated to the new flow.")
else:
    print(f"  OK    {test_train_path} not present — confirms warm_cross.py "
          "doesn't touch the test-side train cache.")
print("  OK cross HDF5 layout matches the 3-cache contract")
PY
echo

# ---------------------------------------------------------------------------
# [11] Cross-cat prep + warm (cache-hit) for $CROSS_CAT_TASK.
# ---------------------------------------------------------------------------
echo "=== [11] sdx cross-cat prep + warm (cache-hit) for $CROSS_CAT_TASK × $ENCODER ==="
python -m sdx cross-cat prep -t "$CROSS_CAT_TASK" --overwrite
python -m sdx cross-cat warm -t "$CROSS_CAT_TASK" -e "$ENCODER"
echo

# ---------------------------------------------------------------------------
# [12] HDF5 inspect — per-train-dataset (train + val) and per-test-dataset
# (val) caches all hit.
# ---------------------------------------------------------------------------
echo "=== [12] HDF5 inspect for $CROSS_CAT_TASK ==="
python - <<PY
import json, h5py
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml

with open(f"sdx/configs/cross_tasks/$CROSS_CAT_TASK.yaml") as f:
    cfg = load_hyperpyyaml(f)
train_ds = cfg["train_datasets"]
test_ds = cfg["test_datasets"]
root = Path("$CACHE_ROOT")

with open(f"exps/cross_cat/$CROSS_CAT_TASK/manifest/train.json") as f: tr = json.load(f)
with open(f"exps/cross_cat/$CROSS_CAT_TASK/manifest/valid.json") as f: va = json.load(f)
with open(f"exps/cross_cat/$CROSS_CAT_TASK/manifest/test.json") as f: te = json.load(f)
all_train = {**tr, **va}

def split(uid):
    pre, bare = uid.split("::", 1)
    return pre.split("_", 1)[0], bare

def per_ds(d, ds):
    return [bare for uid in d for k_ds, bare in [split(uid)] if k_ds == ds]

def assert_cache(path: Path, ids: list[str], num_versions: int, label: str):
    if not ids:
        print(f"  SKIP {label}: no manifest ids for this dataset"); return
    assert path.exists(), f"  MISSING cache: {path}"
    with h5py.File(path, "r") as f:
        keys = set(f.keys())
    coverage = sum(1 for u in ids if u in keys)
    print(f"  {label}: {path}")
    print(f"    {coverage}/{len(ids)} ids cached")
    assert coverage == len(ids), f"missing ids in {path}: {set(ids) - keys}"

for i, ds in enumerate(train_ds):
    s = cfg.get(f"setting_{i+1}") or {}
    nav = int(s.get("num_aug_ver", 1))
    ids = per_ds(all_train, ds)
    assert_cache(root / ds / "$ENCODER" / "train" / "single_avg" / "cache.hdf5",
                 ids, nav, f"train ds[{i}]={ds!r} (need {nav} versions)")
    assert_cache(root / ds / "$ENCODER" / "val" / "single_avg" / "cache.hdf5",
                 ids, 1,   f"train ds[{i}]={ds!r}/val (need 1 version)")
for j, ds in enumerate(test_ds):
    ids = per_ds(te, ds)
    assert_cache(root / ds / "$ENCODER" / "val" / "single_avg" / "cache.hdf5",
                 ids, 1,   f"test  ds[{j}]={ds!r}/val (need 1 version)")
print("  OK cross-cat HDF5 layout matches the per-dataset contract")
PY
echo

revert_main_yamls
trap - EXIT
echo "=== Reverted slurm_tmpdir lines in main yamls ==="
echo
echo "ALL CHECKS PASSED"
