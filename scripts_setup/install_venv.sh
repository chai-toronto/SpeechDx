#!/bin/bash
# Build /scratch/aina10/SpeechDx/.venv and populate from
# requirements_cc.txt (Alliance wheelhouse) + requirements_pypi.txt (PyPI).
# Run this from a LOGIN node only.

set -euo pipefail

PROJECT="/scratch/aina10/SpeechDx"
VENV="$PROJECT/.venv"
LOG="$PROJECT/setup_install.log"

cd "$PROJECT"

# Load Alliance modules. python/3.11 matches the frozen env's 3.11.5 wheels.
module --force purge
module load StdEnv/2023 gcc python/3.11 arrow ffmpeg rust

echo "==> Python: $(which python)  ($(python --version))" | tee -a "$LOG"

if [[ ! -d "$VENV" ]]; then
    echo "==> Creating venv at $VENV" | tee -a "$LOG"
    python -m venv "$VENV"
fi

# shellcheck source=/dev/null
source "$VENV/bin/activate"

echo "==> venv python = $(which python)" | tee -a "$LOG"
echo "==> venv pip    = $(which pip)" | tee -a "$LOG"

echo "==> Upgrading pip/setuptools/wheel from wheelhouse" | tee -a "$LOG"
pip install --no-index --upgrade pip setuptools wheel 2>&1 | tee -a "$LOG"

echo "==> Installing wheelhouse packages (--no-index)" | tee -a "$LOG"
pip install --no-index -r requirements_cc.txt 2>&1 | tee -a "$LOG"

echo "==> Installing PyPI-only packages (network)" | tee -a "$LOG"
pip install -r requirements_pypi.txt 2>&1 | tee -a "$LOG"

echo "==> Sanity import" | tee -a "$LOG"
python - <<'PY' 2>&1 | tee -a "$LOG"
import importlib, sys
mods = [
    "torch", "torchaudio", "torchvision", "transformers",
    "speechbrain", "hyperpyyaml", "ray", "h5py", "librosa",
]
ok, bad = [], []
for m in mods:
    try:
        importlib.import_module(m)
        ok.append(m)
    except Exception as e:
        bad.append((m, repr(e)))
print(f"OK   ({len(ok)}): {ok}")
if bad:
    print(f"FAIL ({len(bad)}):")
    for m, e in bad:
        print(f"  - {m}: {e}")
    sys.exit(1)
PY

echo "==> DONE. venv = $VENV" | tee -a "$LOG"
