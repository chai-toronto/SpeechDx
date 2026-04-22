"""
Preparing the metadata for KSoF dataset tasks.
Uses the official speaker-independent partition stored in the `split` column.
"""
from pathlib import Path

import pandas as pd

from training.dataio.prep_utils import save_task_csv, to_sb_dict_and_save

# 8-class multilabel order:
#   0 block, 1 prolongation, 2 sound_rep, 3 word_rep,
#   4 modified, 5 interjection, 6 no_disfl, 7 garbage
STUT_CLASS_COLS = [
    ("block", "Block"),
    ("prolongation", "Prolongation"),
    ("sound_rep", "Sound Repetition"),
    ("word_rep", "Word / Phrase Repetition"),
    ("modified", "Modified/ Speech technique"),
    ("interjection", "Interjection"),
    ("no_disfl", "No dysfluencies"),
]
# "Garbage" = unintelligible / no speech / poor audio / background music
# (any one of these columns has a majority vote).
STUT_GARBAGE_COLS = [
    "Unintelligible", "No Speech", "Poor Audio Quality", "Music (Background Noise)"
]
STUT_MAJORITY = 2  # >=2 of 3 annotators


def prepare_ksof_stutL(
        wav_folder, metadata_path,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset, task,
):
    """8-class multilabel stuttering classification.

    Labels are derived per clip from the three-annotator counts in the source
    CSV via majority vote (count >= 2). The 8th class (`garbage`) fires if a
    majority of annotators marked the clip as unintelligible, no-speech, poor
    audio, or background music.

    Splits come from the official KSoF partition already encoded in the
    `split` column (0=train, 1=dev, 2=test). All 5597 segments are kept:
    multilabel naturally handles segments the challenge would have dropped as
    ambiguous in its single-label setting.
    """
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]

    # Per-class majority-vote binary targets -> 8-dim list per row.
    class_bits = []
    for _, src_col in STUT_CLASS_COLS:
        class_bits.append((df[src_col] >= STUT_MAJORITY).astype(int).values)
    garbage = pd.concat(
        [(df[c] >= STUT_MAJORITY) for c in STUT_GARBAGE_COLS], axis=1
    ).any(axis=1).astype(int).values
    class_bits.append(garbage)

    label_matrix = list(zip(*class_bits))  # list of 8-tuples
    df["label"] = [list(t) for t in label_matrix]

    n_pos = [int(col.sum()) for col in class_bits]
    names = [n for n, _ in STUT_CLASS_COLS] + ["garbage"]
    print("Per-class positives:", dict(zip(names, n_pos)))

    df_test = df[df["split"] == 2]
    df_train = df[df["split"] == 0]
    df_val = df[df["split"] == 1]

    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(
        df_train, manifest_train_path,
        df_val, manifest_val_path,
        df_test, manifest_test_path,
    )
    print("--- prepare_ksof_stutL finished ---")