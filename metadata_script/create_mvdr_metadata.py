import csv
import random
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _drop_invalid_audio import drop_invalid_audio_rows

# MDVR-KCL (Mobile Device Voice Recordings at King's College London) —
# Jaeger, Trivedi, Stadtschnitzer (2019), Zenodo 10.5281/zenodo.2867216,
# CC BY 4.0. Funded by the EU i-PROGNOSIS project.
#
# Expected raw layout (after `script/download_mvdr.sh`):
#   data/mvdr/raw/.../{ReadText,SpontaneousDialogue}/{HC,PD}/ID*_{hc,pd}_*.wav
# Staged into:
#   data/mvdr/processed/audio/{ReadText,SpontaneousDialogue}/{HC,PD}/...
RAW_ROOT = Path("data/mvdr/raw")
PROCESSED_ROOT = Path("data/mvdr/processed")
PROCESSED_AUDIO = PROCESSED_ROOT / "audio"

# (raw subdir name, value for the `task` column). ReadText is listed first so
# its rows sort to the top by path, keeping uids 0..36 stable across the
# pre-Spontaneous CSV; spontaneous rows are appended starting at uid 37.
TASKS = [
    ("ReadText", "read"),
    ("SpontaneousDialogue", "spontaneous"),
]

# Filename format: ID{nn}_{hc|pd}_{H&Y}_{UPDRS_II5}_{UPDRS_III18}.wav
# Example: ID02_pd_1_2_1.wav
# Label: HC=0, PD=1

# One known typo in the upstream zip: SpontaneousDialogue/HC/ID22hc_0_0_0.wav
# is missing the underscore between participant id and condition. Normalize it
# at staging time so downstream parsing sees the canonical filename.
TYPO_FIX_RE = re.compile(r"^(ID\d+)(hc|pd)_")

RANDOM_SEED = 42


def stage_audio_from_raw():
    """Copy {ReadText,SpontaneousDialogue}/{HC,PD}/*.wav from raw/ → processed/audio/.

    The MDVR-KCL zip extracts to raw/26-29_09_2017_KCL/<task>/{HC,PD}/...,
    but we tolerate any depth above the task dir in case a user already moved
    the contents up. The known ID22hc typo is normalized to ID22_hc on copy.
    """
    if not RAW_ROOT.is_dir():
        return
    matches = []
    for task_dir, _ in TASKS:
        matches.extend(RAW_ROOT.glob(f"**/{task_dir}/*/*.wav"))
    if not matches:
        wanted = ",".join(t for t, _ in TASKS)
        print(
            f"WARN: {RAW_ROOT} exists but contains no {{{wanted}}}/{{HC,PD}}/*.wav. "
            "Run scripts/download_mvdr.sh or extract 26_29_09_2017_KCL.zip there."
        )
        return
    PROCESSED_AUDIO.mkdir(parents=True, exist_ok=True)
    task_names = {t for t, _ in TASKS}
    copied = skipped = renamed = 0
    for src in matches:
        # Re-root the destination at processed/audio/<task>/<HC|PD>/<file>
        # — drop everything above the task dir in the source path.
        idx = next(i for i, part in enumerate(src.parts) if part in task_names)
        rel = Path(*src.parts[idx:])
        fixed_name = TYPO_FIX_RE.sub(r"\1_\2_", rel.name)
        if fixed_name != rel.name:
            print(f"WARN: normalizing typo filename {rel} -> {rel.with_name(fixed_name)}")
            rel = rel.with_name(fixed_name)
            renamed += 1
        dst = PROCESSED_AUDIO / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            skipped += 1
        else:
            shutil.copy2(src, dst)
            copied += 1
    print(f"Staged from {RAW_ROOT}: copied {copied}, skipped {skipped}, renamed {renamed}")


stage_audio_from_raw()


def make_stratified_splits(wav_files, seed):
    """Randomly assign participants to train/val (80/20),
    stratified by label so each split has proportional HC/PD."""
    random.seed(seed)

    hc_ids = sorted({w['participant_id'] for w in wav_files if w['label'] == 0})
    pd_ids = sorted({w['participant_id'] for w in wav_files if w['label'] == 1})

    random.shuffle(hc_ids)
    random.shuffle(pd_ids)

    def split_list(ids):
        n = len(ids)
        n_val = max(1, round(n * 0.2))
        val = set(ids[:n_val])
        train = set(ids[n_val:])
        return train, val

    hc_train, hc_val = split_list(hc_ids)
    pd_train, pd_val = split_list(pd_ids)

    train_set = hc_train | pd_train
    val_set = hc_val | pd_val

    return train_set, val_set


def get_split(participant_id, train_set, val_set):
    """Return split: 0 for train, 1 for val"""
    if participant_id in train_set:
        return 0
    elif participant_id in val_set:
        return 1
    else:
        return -1


def parse_filename(filename):
    """Parse MVDR filename to extract metadata.
    Format: ID{nn}_{hc|pd}_{H&Y}_{UPDRS_II5}_{UPDRS_III18}.wav
    """
    match = re.match(r'(ID\d+)_(hc|pd)_(\d+)_(\d+)_(\d+)\.wav', filename)
    if not match:
        return None
    return {
        'participant_id': match.group(1),
        'condition': match.group(2),
        'hy_rating': int(match.group(3)),
        'updrs_ii5': int(match.group(4)),
        'updrs_iii18': int(match.group(5)),
    }


# Collect all wav files from HC and PD subdirectories under each task dir.
wav_files = []
for task_dir, task_name in TASKS:
    for condition_dir in ['HC', 'PD']:
        dir_path = PROCESSED_AUDIO / task_dir / condition_dir
        if not dir_path.exists():
            continue

        for wav_file in sorted(dir_path.glob("*.wav")):
            info = parse_filename(wav_file.name)
            if info is None:
                print(f"Warning: Could not parse filename {wav_file.name}")
                continue

            # Label: HC=0, PD=1
            label = 0 if info['condition'] == 'hc' else 1
            rel_path = str(wav_file.relative_to(PROCESSED_AUDIO))

            wav_files.append({
                'participant_id': info['participant_id'],
                'condition': info['condition'].upper(),
                'label': label,
                'hy_rating': info['hy_rating'],
                'updrs_ii5': info['updrs_ii5'],
                'updrs_iii18': info['updrs_iii18'],
                'task': task_name,
                'path': rel_path,
            })

# Sort by path. Since paths are prefixed by task dir name and ReadText sorts
# before SpontaneousDialogue, this preserves uids 0..36 for ReadText rows
# (matching the previous CSV) and assigns 37+ to SpontaneousDialogue rows.
wav_files.sort(key=lambda x: x['path'])

# Create stratified random splits
train_set, val_set = make_stratified_splits(wav_files, RANDOM_SEED)

for w in wav_files:
    w['split'] = get_split(w['participant_id'], train_set, val_set)

# Write to CSV
output_file = "data/mvdr/processed/mvdr.csv"
with open(output_file, 'w', newline='') as csvfile:
    fieldnames = ['uid', 'Participant_ID', 'condition', 'label', 'hy_rating',
                  'updrs_ii5', 'updrs_iii18', 'task', 'split', 'path']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for uid, wav_info in enumerate(wav_files):
        writer.writerow({
            'uid': uid,
            'Participant_ID': wav_info['participant_id'],
            'condition': wav_info['condition'],
            'label': wav_info['label'],
            'hy_rating': wav_info['hy_rating'],
            'updrs_ii5': wav_info['updrs_ii5'],
            'updrs_iii18': wav_info['updrs_iii18'],
            'task': wav_info['task'],
            'split': wav_info['split'],
            'path': wav_info['path'],
        })

print(f"Created {output_file} with {len(wav_files)} wav files")

drop_invalid_audio_rows(output_file, PROCESSED_AUDIO)

print(f"\nSample rows:")
for i in range(min(5, len(wav_files))):
    print(wav_files[i])

# Print statistics
print(f"\nStatistics:")
print(f"Total files: {len(wav_files)}")
print(f"Train (0): {sum(1 for w in wav_files if w['split'] == 0)}")
print(f"Val (1): {sum(1 for w in wav_files if w['split'] == 1)}")
print(f"Unknown split: {sum(1 for w in wav_files if w['split'] == -1)}")
print(f"Label 0 (HC): {sum(1 for w in wav_files if w['label'] == 0)}")
print(f"Label 1 (PD): {sum(1 for w in wav_files if w['label'] == 1)}")
for _, task_name in TASKS:
    n = sum(1 for w in wav_files if w['task'] == task_name)
    print(f"Task {task_name}: {n}")
