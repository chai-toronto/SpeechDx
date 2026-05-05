import os
import csv
import shutil
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _drop_invalid_audio import drop_invalid_audio_rows

# Source (extracted Audio_Speech_Actors_01-24.zip) -> processed/audio/
RAW_ROOT = Path("data/ravdess/raw")
PROCESSED_ROOT = Path("data/ravdess/processed")
root_dir = str(PROCESSED_ROOT / "audio")


def stage_audio_from_raw():
    """Copy Actor_*/<wav> from raw/ into processed/audio/ if not already there."""
    if not RAW_ROOT.is_dir():
        return
    dst_root = Path(root_dir)
    dst_root.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0
    for src in RAW_ROOT.glob("Actor_*/*.wav"):
        rel = src.relative_to(RAW_ROOT)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            skipped += 1
        else:
            shutil.copy2(src, dst)
            copied += 1
    print(f"Staged from {RAW_ROOT}: copied {copied}, skipped {skipped}")


stage_audio_from_raw()

def parse_filename(filename):
    """
    Parse RAVDESS filename with 7 parts:
    Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor
    """
    # Remove .wav extension
    name = filename.replace('.wav', '')
    parts = name.split('-')

    if len(parts) != 7:
        return None

    return {
        'modality': int(parts[0]),
        'vocal_channel': int(parts[1]),
        'emotion': int(parts[2]),
        'intensity': int(parts[3]),
        'statement': int(parts[4]),
        'repetition': int(parts[5]),
        'actor': int(parts[6])
    }

def get_split(actor_num):
    """Return split based on actor: 0 for 1-18, 1 for 19-20, 2 for 21-24"""
    if 1 <= actor_num <= 18:
        return 0
    elif 19 <= actor_num <= 20:
        return 1
    elif 21 <= actor_num <= 24:
        return 2
    else:
        return -1  # Unknown actor

def get_gender(actor_num):
    """Odd numbered actors are male, even numbered actors are female"""
    return 'male' if actor_num % 2 == 1 else 'female'

def get_audio_duration(file_path):
    """Get duration of audio file in seconds"""
    try:
        with wave.open(str(file_path), 'rb') as audio_file:
            frames = audio_file.getnframes()
            rate = audio_file.getframerate()
            duration = frames / float(rate)
            return round(duration, 3)
    except Exception as e:
        print(f"Warning: Could not read duration for {file_path}: {e}")
        return None

# Collect all wav files
wav_files = []
for actor_dir in Path(root_dir).iterdir():
    if not actor_dir.is_dir() or actor_dir.name.startswith('.'):
        continue

    actor_name = actor_dir.name  # e.g., "Actor_16"

    # Get all wav files in this actor directory
    for wav_file in actor_dir.glob("*.wav"):
        # Parse filename
        parsed = parse_filename(wav_file.name)

        if parsed is None:
            print(f"Warning: Could not parse {wav_file.name}")
            continue

        # Get relative path from root
        rel_path = str(wav_file.relative_to("data/ravdess/processed/audio"))

        # Get audio duration
        duration = get_audio_duration(wav_file)

        wav_files.append({
            'actor': parsed['actor'],
            'actor_name': actor_name,
            'gender': get_gender(parsed['actor']),
            'emotion': parsed['emotion'],
            'intensity': parsed['intensity'],
            'statement': parsed['statement'],
            'repetition': parsed['repetition'],
            'vocal_channel': parsed['vocal_channel'],
            'modality': parsed['modality'],
            'path': rel_path,
            'duration': duration,
            'split': get_split(parsed['actor']),
            'label': parsed['emotion'] - 1  # Emotion starting from 0
        })

# Sort by path for consistency
wav_files.sort(key=lambda x: x['path'])

# Write to CSV
output_file = "data/ravdess/processed/ravdess.csv"
with open(output_file, 'w', newline='') as csvfile:
    fieldnames = ['uid', 'Participant_ID', 'gender', 'emotion', 'intensity', 'statement',
                  'repetition', 'vocal_channel', 'modality', 'split', 'label', 'duration', 'path']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for uid, wav_info in enumerate(wav_files):
        writer.writerow({
            'uid': uid,
            'Participant_ID': wav_info['actor'],
            'gender': wav_info['gender'],
            'emotion': wav_info['emotion'],
            'intensity': wav_info['intensity'],
            'statement': wav_info['statement'],
            'repetition': wav_info['repetition'],
            'vocal_channel': wav_info['vocal_channel'],
            'modality': wav_info['modality'],
            'split': wav_info['split'],
            'label': wav_info['label'],
            'duration': wav_info['duration'],
            'path': wav_info['path']
        })

print(f"Created {output_file} with {len(wav_files)} wav files")

drop_invalid_audio_rows(output_file, root_dir)

print(f"\nSample rows:")
for i in range(min(5, len(wav_files))):
    print(wav_files[i])

# Print statistics
print(f"\nStatistics:")
print(f"Total files: {len(wav_files)}")
print(f"Train (0): {sum(1 for w in wav_files if w['split'] == 0)}")
print(f"Val (1): {sum(1 for w in wav_files if w['split'] == 1)}")
print(f"Test (2): {sum(1 for w in wav_files if w['split'] == 2)}")
print(f"\nLabel distribution:")
for label in range(8):
    count = sum(1 for w in wav_files if w['label'] == label)
    print(f"Label {label} (emotion {label+1}): {count}")
print(f"\nGender distribution:")
print(f"Male: {sum(1 for w in wav_files if w['gender'] == 'male')}")
print(f"Female: {sum(1 for w in wav_files if w['gender'] == 'female')}")