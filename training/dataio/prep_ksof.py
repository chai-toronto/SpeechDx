"""
Preparing the metadata for KSoF dataset tasks.
Speaker-disjoint k-fold CV (replaces the official partition).
"""
from pathlib import Path

import pandas as pd

from training.dataio.prep_utils import (
    kfold_split, save_kfold_task_csv, to_sb_kfold_dict_and_save,
)

# Six disfluency classes used for the binary stutter/no-stutter task.
STUT_DISFL_COLS = [
    "Block", "Prolongation", "Sound Repetition", "Word / Phrase Repetition",
    "Modified/ Speech technique", "Interjection",
]


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
        manifest_train_path, manifest_val_path,
        random_seed, raw_label_key, num_fold, dataset, task,
):
    """8-class multilabel stuttering classification, speaker-disjoint k-fold.

    Per-clip labels come from majority vote (>=2 of 3 annotators) on each
    disfluency class, plus a "garbage" bit if a majority marked the clip as
    unintelligible / no-speech / poor-audio / background-music.

    Stratification key is a derived signature of the multilabel vector
    (positive class indices joined, "none" if all-zero) plus gender — falls
    back to gender / GroupKFold via ``kfold_split`` if too sparse.
    """
    df = pd.read_csv(metadata_path)
    df["path"] = Path(wav_folder).resolve() / df["path"]

    class_bits = []
    for _, src_col in STUT_CLASS_COLS:
        class_bits.append((df[src_col] >= STUT_MAJORITY).astype(int).values)
    garbage = pd.concat(
        [(df[c] >= STUT_MAJORITY) for c in STUT_GARBAGE_COLS], axis=1
    ).any(axis=1).astype(int).values
    class_bits.append(garbage)

    label_matrix = list(zip(*class_bits))  # list of 8-tuples
    df["label"] = [[int(x) for x in t] for t in label_matrix]

    df["_strat_sig"] = [
        "_".join(str(i) for i, x in enumerate(t) if x) or "none"
        for t in label_matrix
    ]

    n_pos = [int(col.sum()) for col in class_bits]
    names = [n for n, _ in STUT_CLASS_COLS] + ["garbage"]
    print("Per-class positives:", dict(zip(names, n_pos)))

    folds = kfold_split(
        df, stratify_cols=["_strat_sig", "gender"],
        group_col="Participant_ID",
        n_splits=num_fold, random_seed=random_seed,
    )
    df = df.drop(columns=["_strat_sig"])
    folds = [(tr.drop(columns=["_strat_sig"]), va.drop(columns=["_strat_sig"]))
             for tr, va in folds]

    save_kfold_task_csv(df, folds, dataset, task)
    to_sb_kfold_dict_and_save(folds, manifest_train_path, manifest_val_path)
    print(f"--- prepare_ksof_stutL ({dataset}_{task}) finished ---")


def prepare_ksof_stutC(
        wav_folder, metadata_path,
        manifest_train_path, manifest_val_path,
        random_seed, raw_label_key, num_fold, dataset, task,
):
    """Binary stutter vs. no-disfluency classification, speaker-disjoint k-fold.

    label = 1 iff any of the six disfluency classes (block, prolongation,
    sound_rep, word_rep, modified, interjection) reaches majority (>=2 of 3
    annotators); label = 0 iff `No dysfluencies` reaches majority. Clips where
    neither side has a majority (ambiguous / garbage) are dropped before the
    split.
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
    df["label"] = pos[keep].reset_index(drop=True).astype(int).values
    print(f"Dropped {n_before - len(df)} clips with no majority; kept {len(df)}.")
    print(f"Positives: {int(df['label'].sum())}, Negatives: {int((1 - df['label']).sum())}")

    folds = kfold_split(
        df, stratify_cols=[raw_label_key, "gender"],
        group_col="Participant_ID",
        n_splits=num_fold, random_seed=random_seed,
    )
    save_kfold_task_csv(df, folds, dataset, task)
    to_sb_kfold_dict_and_save(folds, manifest_train_path, manifest_val_path)
    print(f"--- prepare_ksof_stutC ({dataset}_{task}) finished ---")
