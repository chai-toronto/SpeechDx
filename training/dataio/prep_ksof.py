"""
Preparing the metadata for KSoF dataset tasks.
Uses the official speaker-independent partition stored in the `split` column.
"""
from pathlib import Path

import pandas as pd

from training.dataio.prep_utils import save_task_csv, to_sb_dict_and_save

# 7-class multilabel order:
#   0 block, 1 prolongation, 2 sound_rep, 3 word_rep,
#   4 modified, 5 interjection, 6 no_disfl
STUT_CLASS_COLS = [
    ("block", "Block"),
    ("prolongation", "Prolongation"),
    ("sound_rep", "Sound Repetition"),
    ("word_rep", "Word / Phrase Repetition"),
    ("modified", "Modified/ Speech technique"),
    ("interjection", "Interjection"),
    ("no_disfl", "No dysfluencies"),
]

STUT_MAJORITY = 2  # >=2 of 3 annotators

# Six disfluency classes used for the binary stutter/no-stutter task.
STUT_DISFL_COLS = [
    "Block", "Prolongation", "Sound Repetition", "Word / Phrase Repetition",
    "Modified/ Speech technique", "Interjection",
]


def prepare_ksof_stutL(
        wav_folder, metadata_path,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset, task,
):
    """7-class multilabel stuttering classification.

    Labels are derived per clip from the three-annotator counts in the source
    CSV via majority vote (count >= 2). Clips where no class reaches majority
    (all-zero label, e.g. garbage/ambiguous) are dropped.

    Splits come from the official KSoF partition already encoded in the
    `split` column (0=train, 1=dev, 2=test).
    """
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]

    # Per-class majority-vote binary targets -> 7-dim list per row.
    class_bits = []
    for _, src_col in STUT_CLASS_COLS:
        class_bits.append((df[src_col] >= STUT_MAJORITY).astype(int).values)

    label_matrix = list(zip(*class_bits))  # list of 7-tuples
    df["label"] = [list(t) for t in label_matrix]

    n_before = len(df)
    keep = [any(t) for t in label_matrix]
    df = df[keep].reset_index(drop=True)
    class_bits = [col[keep] for col in class_bits]
    print(f"Dropped {n_before - len(df)} clips with no majority label; kept {len(df)}.")

    n_pos = [int(col.sum()) for col in class_bits]
    names = [n for n, _ in STUT_CLASS_COLS]
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


def prepare_ksof_stutC(
        wav_folder, metadata_path,
        manifest_train_path, manifest_val_path, manifest_test_path,
        ratio, random_seed, dataset, task,
):
    """Binary stutter vs. no-disfluency classification.

    label = 1 iff any of the six disfluency classes (block, prolongation,
    sound_rep, word_rep, modified, interjection) reaches majority (>=2 of 3
    annotators); label = 0 iff `No dysfluencies` reaches majority. Clips where
    neither side has a majority (ambiguous / garbage) are dropped.
    """
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]

    pos = pd.concat(
        [(df[c] >= STUT_MAJORITY) for c in STUT_DISFL_COLS], axis=1
    ).any(axis=1)
    neg = df["No dysfluencies"] >= STUT_MAJORITY

    n_before = len(df)
    keep = pos | neg
    df = df[keep].reset_index(drop=True)
    df["label"] = pos[keep].astype(int).values
    print(f"Dropped {n_before - len(df)} clips with no majority; kept {len(df)}.")
    print(f"Positives: {int(df['label'].sum())}, Negatives: {int((1 - df['label']).sum())}")

    df_test = df[df["split"] == 2]
    df_train = df[df["split"] == 0]
    df_val = df[df["split"] == 1]

    save_task_csv(df_train, df_val, df_test, dataset, task)
    to_sb_dict_and_save(
        df_train, manifest_train_path,
        df_val, manifest_val_path,
        df_test, manifest_test_path,
    )
    print("--- prepare_ksof_stutC finished ---")