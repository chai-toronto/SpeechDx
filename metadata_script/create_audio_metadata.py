import os
import csv
import argparse
from pathlib import Path
import soundfile as sf

def get_audio_duration(audio_path):
    """Get duration of audio file in seconds"""
    try:
        info = sf.info(audio_path)
        return info.duration
    except Exception as e:
        print(f"Warning: Could not read {audio_path}: {e}")
        return None

def create_audio_metadata_csv(folder_path, output_csv=None, replacement_prefix='noise_folder'):
    """
    Create a CSV with metadata for all audio files in a folder.

    Args:
        folder_path: Path to folder containing audio files
        output_csv: Output CSV path (default: <folder_name>_metadata.csv)
        replacement_prefix: Prefix to replace the folder path in the 'wav' column
    """
    folder_path = Path(folder_path)

    if not folder_path.exists():
        print(f"Error: Folder {folder_path} does not exist")
        return

    # Determine output CSV path
    if output_csv is None:
        output_csv = f"{folder_path.name}_metadata.csv"

    # Common audio extensions
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.opus'}

    # Collect all audio files
    audio_files = []
    for file_path in sorted(folder_path.rglob('*')):
        if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
            # Get filename without extension
            file_id = file_path.stem

            # Get duration
            duration = get_audio_duration(file_path)
            if duration is None:
                continue

            # Get format (extension without dot)
            wav_format = file_path.suffix.lstrip('.')

            # Get relative path from folder
            try:
                wav_path = file_path.relative_to(folder_path)
            except ValueError:
                wav_path = file_path

            wav_path = f'${replacement_prefix}' / wav_path
            wav_path = str(wav_path)
            audio_files.append({
                'ID': file_id,
                'duration': duration,
                'wav': wav_path,
                'wav_format': wav_format,
                'wav_opts': ''
            })

    # Write to CSV
    if not audio_files:
        print(f"No audio files found in {folder_path}")
        return

    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['ID', 'duration', 'wav', 'wav_format', 'wav_opts']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for audio_info in audio_files:
            writer.writerow(audio_info)

    print(f"Created {output_csv} with {len(audio_files)} audio files")
    print(f"\nSample rows:")
    for i in range(min(5, len(audio_files))):
        print(f"  {audio_files[i]}")

    print(f"\nStatistics:")
    print(f"Total files: {len(audio_files)}")
    total_duration = sum(f['duration'] for f in audio_files)
    print(f"Total duration: {total_duration:.2f} seconds ({total_duration/3600:.2f} hours)")

    # Format statistics
    formats = {}
    for f in audio_files:
        fmt = f['wav_format']
        formats[fmt] = formats.get(fmt, 0) + 1
    print(f"Formats: {dict(formats)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create CSV metadata for audio files in a folder')
    parser.add_argument('folder', type=str, help='Path to folder containing audio files')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output CSV file path (default: <folder_name>_metadata.csv)')

    parser.add_argument('--replacement_prefix', '-p', type=str, nargs='?', default='noise_folder',)
    args = parser.parse_args()
    create_audio_metadata_csv(args.folder, args.output, args.replacement_prefix)
