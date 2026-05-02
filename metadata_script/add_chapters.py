"""
Add chapters to audio file based on vector values.
Each vector value represents a 25ms frame with 20ms stride (5ms overlap).

Chunk boundaries are guaranteed to be at least min_chunk_size frames apart,
matching the ChunkPool logic (default: 4 tokens = 80ms minimum spacing).
"""

import subprocess
from pathlib import Path
import torch



def get_audio_duration(audio_path):
    """Get audio duration in seconds using ffprobe."""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(audio_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def find_chapters(vector, threshold=0.1, frame_stride_ms=20, min_chunk_size=4):
    """
    Find chapter boundaries where vector values exceed threshold.
    Uses guaranteed minimum spacing logic matching ChunkPool.

    Args:
        vector: List of values, one per frame (25ms frame with 20ms stride)
        threshold: Values above this create chapters
        frame_stride_ms: Stride between frames in milliseconds (default: 20ms)
        min_chunk_size: Minimum spacing between boundaries in frames (default: 4 tokens = 80ms)

    Returns:
        List of (start_time_ms, end_time_ms, title) tuples
    """
    n = len(vector)
    if n == 0:
        return []

    # Pick boundaries with guaranteed spacing (matching ChunkPool logic)
    boundaries = pick_boundaries_with_spacing(vector, threshold, min_chunk_size)

    # Convert boundaries to chapters
    chapters = []
    for i in range(len(boundaries) - 1):
        start_ms = boundaries[i] * frame_stride_ms
        end_ms = (boundaries[i + 1] * frame_stride_ms) - 1
        chapters.append((start_ms, end_ms, f"Chunk {i + 1}"))

    # Last chapter goes to the end
    if boundaries:
        start_ms = boundaries[-1] * frame_stride_ms
        end_ms = (n * frame_stride_ms) - 1
        chapters.append((start_ms, end_ms, f"Chunk {len(chapters) + 1}"))

    return chapters


def pick_boundaries_with_spacing(vector, threshold, k):
    """
    Pick boundary indices with guaranteed minimum spacing.
    Matches the logic of pick_with_spacing_mask_batched from ChunkPool.

    Args:
        vector: List/array of boundary scores
        threshold: Minimum score to be considered a boundary
        k: Minimum spacing between boundaries (in frames)

    Returns:
        List of boundary indices
    """
    n = len(vector)
    if n == 0:
        return []

    step = max(int(k), 1)

    # Always pick first frame as boundary
    boundaries = [0]
    cur = 0  # Last picked index
    start = cur + step  # Next allowed starting point

    while start < n:
        # Find next candidate at or after start that exceeds threshold
        nxt = None
        for i in range(start, n):
            if vector[i] > threshold:
                nxt = i
                break

        if nxt is None:
            break

        boundaries.append(nxt)
        cur = nxt
        start = cur + step

    return boundaries

def create_audacity_project(root, audio, outroot, scores, threshold=0.1, min_chunk_size=4):
    """
    Create a folder with audio file and Audacity label file.

    Args:
        audio_path: Path to the input audio file
        output_folder: Path to output folder
        vector: torch.Tensor of values (25ms frames with 20ms stride)
        threshold: Threshold for chapter creation (default: 0.1)
        min_chunk_size: Minimum spacing between boundaries in frames (default: 4 tokens = 80ms)
    """
    chunk_sizes = []

    import shutil
    vector = scores['boundary_prob']
    tscores = scores['scores']
    # Convert tensor to list
    if isinstance(vector, torch.Tensor):
        vector = vector.flatten().cpu().tolist()

    # Find chapters with guaranteed minimum spacing
    chapters = find_chapters(vector, threshold, min_chunk_size=min_chunk_size)

    print(f"Found {len(chapters)} chapter(s)")

    # Create output folder
    audio_out = outroot / audio
    audio_out.parent.mkdir(parents=True, exist_ok=True)

    # Copy audio file
    shutil.copy2(root/audio, audio_out)
    print(f"Copied audio to: {audio_out}")

    # Create label file with same name as audio
    label = Path(audio_out).stem + "_labels.txt"
    label = audio_out.parent / label

    # Create Audacity label file
    # Format: start_time\tend_time\tlabel
    with open(label, 'w') as f:
        for i, (start_ms, end_ms, title) in enumerate(chapters):
            start_sec = start_ms / 1000.0
            end_sec = end_ms / 1000.0
            chunk_sizes.append(end_sec-start_sec)
            tscore = tscores[i] if i < len(tscores) else 0.0
            f.write(f"{start_sec:.6f}\t{end_sec:.6f}\t{title}:{tscore}\n")
            print(f"  {title}: {start_sec:.2f}s - {end_sec:.2f}s")

    print(f"\nCreated Audacity labels: {label}")

    print(f"Average chunk size: {sum(chunk_sizes)/len(chunk_sizes) if chunk_sizes else 0:.2f} seconds")
    print(f"Std of chunk size: {torch.tensor(chunk_sizes).std().item() if chunk_sizes else 0:.2f} seconds")

# Example usage:
import pandas as pd
dataset = "torgo"
root = Path(f"/Users/lkieu/PycharmProjects/Audio-Health-Benchmark/data/{dataset}/processed/audio/")
csv_path = f"/Users/lkieu/PycharmProjects/Audio-Health-Benchmark/data/{dataset}/processed/{dataset}.csv"
scores_path = Path("/Users/lkieu/PycharmProjects/Audio-Health-Benchmark/exps/torgo/wavlm-basep-all-chunkAtt_LTprobe-upsampler-t0.1-regs-chunk4-guarantee/brain-logs/final_model/test_diagnostics.pt")
outroot = scores_path.parent / "with_chapters"

# Load CSV and filter for split == 2
df = pd.read_csv(csv_path, index_col=0)
df_filtered = df[df['split'] == 2]

# Load all scores
all_scores = torch.load(scores_path)

print(f"Processing {len(df_filtered)} audio files...")

# Process each row
for idx, row in df_filtered.iterrows():
    audio_path = Path(row['path'])
    label = row['label']

    # Get scores for this audio file
    scores = all_scores[str(idx)]

    # Create output folder organized by label
    output_folder = outroot / str(label)

    print(f"\nProcessing {idx+1}/{len(df_filtered)}: {row['path']}")

    # Create Audacity project folder
    create_audacity_project(root, audio_path, output_folder, scores, threshold=0.1)