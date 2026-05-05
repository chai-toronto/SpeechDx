import os
import sys
import csv
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _drop_invalid_audio import drop_invalid_audio_rows

# Source: data/torgo/raw/<gender>/<pid>/Session*/wav_arrayMic/*.wav (TORGO release layout)
# Staged into:  data/torgo/processed/audio/<gender>/<pid>/Session*/wav_arrayMic/*.wav
RAW_ROOT = Path("data/torgo/raw")
PROCESSED_ROOT = Path("data/torgo/processed")
PROCESSED_AUDIO = PROCESSED_ROOT / "audio"
root_dir = str(PROCESSED_AUDIO)


def stage_audio_from_raw():
    """Copy each <gender>/<pid>/Session*/wav_arrayMic/*.wav from raw/ → processed/audio/."""
    if not RAW_ROOT.is_dir():
        return
    PROCESSED_AUDIO.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0
    for src in RAW_ROOT.glob("*/*/Session*/wav_arrayMic/*.wav"):
        rel = src.relative_to(RAW_ROOT)
        dst = PROCESSED_AUDIO / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            skipped += 1
        else:
            shutil.copy2(src, dst)
            copied += 1
    print(f"Staged from {RAW_ROOT}: copied {copied}, skipped {skipped}")


stage_audio_from_raw()

# Define splits
train_set = {'FC02', 'F03', 'F01', 'MC04', 'MC03', 'M02'}
val_set = {'MC02', 'FC01', 'M03', 'M01'}
test_set = {'FC03', 'F04', 'MC01', 'M05', 'M04'}

def get_split(participant_id):
    """Return split: 0 for train, 1 for val, 2 for test"""
    if participant_id in train_set:
        return 0
    elif participant_id in val_set:
        return 1
    elif participant_id in test_set:
        return 2
    else:
        return -1  # Unknown participant

def get_label(participant_id):
    """Return label: 0 if 'C' in participant_id, 1 otherwise"""
    return 0 if 'C' in participant_id else 1

# Severity from https://link.springer.com/article/10.1007/s11042-024-20053-w/tables/2
# 1 = very low, 2 = low, 3 = medium. Controls have no severity (NaN) — kept in
# the CSV for dysC's negative class but excluded from sevR by prepare_sevR.
SEVERITY = {
    'M01': 3, 'M02': 3, 'M03': 1, 'M04': 3, 'M05': 2,
    'F01': 2, 'F03': 1, 'F04': 1,
}

def get_severity(participant_id):
    return SEVERITY.get(participant_id, '')

# Collect all wav files
wav_files = []
for gender_dir in Path(root_dir).iterdir():
    if not gender_dir.is_dir() or gender_dir.name.startswith('.'):
        continue

    gender = gender_dir.name

    for participant_dir in gender_dir.iterdir():
        if not participant_dir.is_dir():
            continue

        participant_id = participant_dir.name

        for session_dir in participant_dir.iterdir():
            if not session_dir.is_dir():
                continue

            session_number = session_dir.name

            # Look for wav_arrayMic directory
            wav_array_dir = session_dir / "wav_arrayMic"
            if not wav_array_dir.exists() or not wav_array_dir.is_dir():
                continue

            # Get all wav files
            for wav_file in wav_array_dir.glob("*.wav"):
                # Get relative path from root
                rel_path = str(wav_file.relative_to(root_dir))

                wav_files.append({
                    'gender': gender,
                    'participant_id': participant_id,
                    'session': session_number,
                    'path': rel_path,
                    'split': get_split(participant_id),
                    'label': get_label(participant_id),
                    'severity': get_severity(participant_id)
                })

# Sort by path for consistency
wav_files.sort(key=lambda x: x['path'])

# Write to CSV
output_file = str(PROCESSED_ROOT / "torgo.csv")
PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
with open(output_file, 'w', newline='') as csvfile:
    fieldnames = ['uid', 'gender', 'Participant_ID', 'split', 'label', 'severity', 'path']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for uid, wav_info in enumerate(wav_files):
        writer.writerow({
            'uid': uid,
            'gender': wav_info['gender'],
            'Participant_ID': wav_info['participant_id'],
            'split': wav_info['split'],
            'label': wav_info['label'],
            'severity': wav_info['severity'],
            'path': wav_info['path']
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
print(f"Test (2): {sum(1 for w in wav_files if w['split'] == 2)}")
print(f"Label 0 (with C): {sum(1 for w in wav_files if w['label'] == 0)}")
print(f"Label 1 (without C): {sum(1 for w in wav_files if w['label'] == 1)}")
