"""Build ``docs/data.js`` for the interactive SpeechDx leaderboard site.

Reads the per-task ``AUC.csv`` (classification ROC-AUC) and ``Cindex.csv``
(regression C-index) from a summary dir, restricts to the *external-baseline*
encoders (drops the from-scratch models + the LLM columns), keeps the raw
per-dataset tasks (no merges), computes the mean-reciprocal-rank headline, and
writes a single ``docs/data.js`` assigning ``window.LEADERBOARD_DATA``.

This is the data backing ``docs/index.html``. It mirrors ``leaderboard.csv`` on
main (``scripts/leaderboard.py --no-merge --mrr --drop ...``) and adds the raw
per-task scores + task / encoder metadata the static page needs for the heatmap
and the hover tooltips. Regenerate with::

    .venv/bin/python scripts/build_site_data.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO / "exps" / "_summary_jun3_full" / "single_task"
OUT = REPO / "docs" / "data.js"

# The from-scratch models (audio-pretrainer2 checkpoints) + the LLM columns.
# wavlm_large == microsoft/wavlm-large is a *published* baseline and stays in.
DROP = [
    "dmel", "wavlm_base", "wavlm_base_preln", "wavlm_base_preln_366k",
    "wavlm_parcae", "wavlm_rx", "gemini", "qwen3omni",
]

# Jun2+ summaries label tasks "T2 (edaic_T2)"; map the T-number to its canonical
# id (same table as scripts/leaderboard.py).
JUN2_TASK_MAP = {
    "T1": "edaic_depC", "T2": "edaic_phqR", "T3": "ravdess_emoC",
    "T4": "ravdess_emoBC", "T5": "iemocap_emoC", "T6": "iemocap_emoBC",
    "T7": "dbank_adC", "T8": "dbank_mmseR", "T9": "aphasia_pwaC",
    "T10": "torgo_dysC", "T11": "torgo_sevR", "T12": "uaspeech_dysC",
    "T13": "mdvr_parkC", "T14": "mdvr_updrs5R", "T15": "mdvr_updrs18R",
    "T16": "mdvr_hyR", "T17": "ksof_intC", "T18": "ksof_stutL",
    "T19": "c9s_t1", "T20": "c9s_L_t1", "T21": "c9s_t2", "T22": "c9s_L_t2",
    "T23": "c9s_sympL", "T24": "coswara_sympC", "T25": "coswara_covidC",
    "T26": "coswara_sympL", "T27": "avfad_pathC",
}

# Speech-production taxonomy categories (paper). Colours are a colourblind-safe
# placeholder palette — swap for the exact paper hexes if they differ.
CATEGORIES = [
    {"code": "c1", "label": "Affective", "color": "#E15759"},
    {"code": "c2", "label": "Cognitive", "color": "#4E79A7"},
    {"code": "c3", "label": "Motor", "color": "#59A14F"},
    {"code": "c4", "label": "Respiratory", "color": "#B07AA1"},
]

# canonical task id -> (category code, short label, long hover description)
TASK_META = {
    # c1 — Affective
    "edaic_depC":    ("c1", "Depression / healthy (E-DAIC)", "Binary depression screening from clinical-interview speech (E-DAIC-WOZ)."),
    "edaic_phqR":    ("c1", "PHQ-8 score (E-DAIC)", "Regress the PHQ-8 depression-severity score from interview speech (E-DAIC-WOZ)."),
    "ravdess_emoC":  ("c1", "Emotion class (RAVDESS)", "Multi-class emotion recognition on acted speech (RAVDESS)."),
    "ravdess_emoBC": ("c1", "Neg. emotion (RAVDESS)", "Binary negative-vs-non-negative affect on acted speech (RAVDESS)."),
    "iemocap_emoC":  ("c1", "Emotion class (IEMOCAP)", "Multi-class emotion recognition on dyadic sessions (IEMOCAP)."),
    "iemocap_emoBC": ("c1", "Neg. emotion (IEMOCAP)", "Binary negative-vs-non-negative affect (IEMOCAP)."),
    # c2 — Cognitive
    "dbank_adC":     ("c2", "Dementia / healthy (DementiaBank)", "Binary Alzheimer's-dementia detection from picture-description speech (DementiaBank / ADReSS-M)."),
    "dbank_mmseR":   ("c2", "MMSE score (DementiaBank)", "Regress the MMSE cognitive score (DementiaBank / ADReSS-M)."),
    "aphasia_pwaC":  ("c2", "Aphasia / healthy (AphasiaBank)", "Binary aphasia detection (AphasiaBank)."),
    # c3 — Motor
    "torgo_dysC":    ("c3", "Dysarthria / healthy (TORGO)", "Binary dysarthria detection (TORGO)."),
    "torgo_sevR":    ("c3", "Dysarthria severity (TORGO)", "Regress dysarthria severity (TORGO)."),
    "uaspeech_dysC": ("c3", "Dysarthria / healthy (UASpeech)", "Binary dysarthria detection (UASpeech)."),
    "mdvr_parkC":    ("c3", "Parkinson's / healthy (MDVR-KCL)", "Binary Parkinson's detection from voice (MDVR-KCL)."),
    "mdvr_updrs5R":  ("c3", "UPDRS-II.5 (MDVR-KCL)", "Regress UPDRS-II item 5 (speech disability) (MDVR-KCL)."),
    "mdvr_updrs18R": ("c3", "UPDRS-III.18 (MDVR-KCL)", "Regress UPDRS-III item 18 (speech, examiner-rated) (MDVR-KCL)."),
    "mdvr_hyR":      ("c3", "Hoehn & Yahr (MDVR-KCL)", "Regress the Hoehn & Yahr Parkinson's stage (MDVR-KCL)."),
    "ksof_intC":     ("c3", "Disfluency / healthy (KSoF)", "Binary stuttering / disfluency detection (KSoF)."),
    "ksof_stutL":    ("c3", "Disfluency type (KSoF)", "Multi-label stuttering-type classification (KSoF)."),
    # c4 — Respiratory / phonatory
    "c9s_t1":        ("c4", "Symptomatic / healthy (C19-Sounds)", "Binary symptomatic detection (COVID-19 Sounds)."),
    "c9s_L_t1":      ("c4", "Symptomatic / healthy — long (C19-Sounds)", "Binary symptomatic detection, long-form audio (COVID-19 Sounds)."),
    "c9s_t2":        ("c4", "COVID-19 / non-COVID (C19-Sounds)", "Binary COVID-19 detection (COVID-19 Sounds)."),
    "c9s_L_t2":      ("c4", "COVID-19 / non-COVID — long (C19-Sounds)", "Binary COVID-19 detection, long-form audio (COVID-19 Sounds)."),
    "c9s_sympL":     ("c4", "Resp. symptoms (C19-Sounds)", "Multi-label respiratory-symptom classification (COVID-19 Sounds)."),
    "coswara_sympC": ("c4", "Symptomatic / healthy (Coswara)", "Binary symptomatic detection (Coswara)."),
    "coswara_covidC":("c4", "COVID-19 / non-COVID (Coswara)", "Binary COVID-19 detection (Coswara)."),
    "coswara_sympL": ("c4", "Resp. symptoms (Coswara)", "Multi-label respiratory-symptom classification (Coswara)."),
    "avfad_pathC":   ("c4", "Vocal pathology / healthy (AVFAD)", "Binary voice-pathology detection (AVFAD)."),
}

# encoder column id -> (display name, HF / source checkpoint)
ENCODER_META = {
    "wavlm_large": ("WavLM (Large)", "microsoft/wavlm-large"),
    "whisper":     ("Whisper (Large-v3)", "openai/whisper-large-v3"),
    "qwen3voice":  ("Qwen3-TTS-Tokenizer (12 Hz)", "Qwen/Qwen3-TTS-Tokenizer-12Hz"),
    "ast":         ("AST (AudioSet-finetuned)", "MIT/ast-finetuned-audioset-10-10-0.4593"),
    "audiomae":    ("AudioMAE", "hance-ai/audiomae"),
    "wavjepa":     ("WavJEPA-Nat (Base)", "labhamlet/wavjepa-nat-base"),
    "mms":         ("MMS-1B", "facebook/mms-1b"),
    "hubert":      ("HuBERT (Large, ASR-FT)", "facebook/hubert-large-ls960-ft"),
    "emotion2vec": ("emotion2vec+ Large", "iic/emotion2vec_plus_large"),
    "opera_gt":    ("OPERA-GT", "evelyn0414/OPERA (encoder-operaGT.ckpt)"),
    "w2v2":        ("wav2vec 2.0 (Large, ASR-FT)", "facebook/wav2vec2-large-960h-lv60-self"),
    "clap":        ("CLAP (LAION-Larger-General)", "laion/larger_clap_general"),
}

CAT_INDEX = {c["code"]: i for i, c in enumerate(CATEGORIES)}
LEADERBOARD_CSV_URL = "https://github.com/chai-toronto/SpeechDx/blob/main/leaderboard.csv"


def canonical(label: str) -> str:
    m = re.match(r"\s*(T\d+)\b", label)
    return JUN2_TASK_MAP.get(m.group(1), label) if m else label


def load_typed(input_dir: Path) -> pd.DataFrame:
    """Tasks x encoders, each row tagged with its metric ('classification' from
    AUC.csv = ROC-AUC, higher better, clipped [0,1]; 'regression' from MAE.csv =
    mean absolute error, lower better, NOT clipped)."""
    frames = []
    for fname, kind, clip in (("AUC.csv", "classification", True),
                              ("MAE.csv", "regression", False)):
        p = input_dir / fname
        if not p.exists():
            continue
        df = pd.read_csv(p).set_index("task").rename(index=canonical)
        if clip:
            df = df.clip(0, 1)
        df["__kind__"] = kind
        frames.append(df)
    if not frames:
        raise SystemExit(f"no AUC.csv / MAE.csv in {input_dir}")
    return pd.concat(frames, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_dir", type=Path, nargs="?", default=DEFAULT_INPUT)
    ap.add_argument("-o", "--output", type=Path, default=OUT)
    args = ap.parse_args()

    typed = load_typed(args.input_dir)
    kinds = typed.pop("__kind__")
    scores = typed.drop(columns=[c for c in DROP if c in typed.columns])

    encoders = [e for e in scores.columns if e in ENCODER_META]
    unknown = set(scores.columns) - set(ENCODER_META)
    if unknown:
        raise SystemExit(f"encoders with no ENCODER_META: {sorted(unknown)}")
    scores = scores[encoders]
    # Drop encoders whose entire row is empty (no data on any task).
    scores = scores.loc[:, scores.notna().any(axis=0)]
    encoders = list(scores.columns)

    missing_meta = [t for t in scores.index if t not in TASK_META]
    if missing_meta:
        raise SystemExit(f"tasks with no TASK_META: {sorted(missing_meta)}")

    # Mean reciprocal rank: rank encoders within each task (regression = MAE is
    # lower-better → ascending), reciprocate, average across tasks.
    ranks = scores.rank(axis=1, method="min", ascending=False)
    reg_tasks = [t for t in scores.index if kinds[t] == "regression"]
    if reg_tasks:
        ranks.loc[reg_tasks] = scores.loc[reg_tasks].rank(axis=1, method="min", ascending=True)
    rr = 1.0 / ranks
    mrr = rr.mean(axis=0, skipna=True)

    # Tasks ordered by category, then classification before regression, then id.
    def task_sort_key(t: str):
        cat, _, _ = TASK_META[t]
        return (CAT_INDEX[cat], 0 if kinds[t] == "classification" else 1, t)

    task_ids = sorted(scores.index, key=task_sort_key)
    tasks = [{
        "id": t,
        "label": TASK_META[t][1],
        "desc": TASK_META[t][2],
        "category": TASK_META[t][0],
        "type": kinds[t],
        "metric": "ROC-AUC" if kinds[t] == "classification" else "MAE",
    } for t in task_ids]

    order = mrr.sort_values(ascending=False)
    models = []
    for rank, enc in enumerate(order.index, start=1):
        disp, ckpt = ENCODER_META[enc]
        vals = {t: (None if pd.isna(scores.loc[t, enc]) else round(float(scores.loc[t, enc]), 4))
                for t in task_ids}
        models.append({
            "id": enc, "display": disp, "checkpoint": ckpt,
            "rank": rank, "mrr": round(float(order[enc]), 4), "scores": vals,
        })

    data = {
        "generated": dt.date.today().isoformat(),
        "source": str(args.input_dir.relative_to(REPO)),
        "leaderboard_csv_url": LEADERBOARD_CSV_URL,
        "n_tasks": len(task_ids),
        "n_models": len(models),
        "categories": CATEGORIES,
        "tasks": tasks,
        "models": models,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, ensure_ascii=False)
    args.output.write_text(
        "// Generated by scripts/build_site_data.py — do not edit by hand.\n"
        f"window.LEADERBOARD_DATA = {payload};\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output} ({len(models)} models x {len(task_ids)} tasks)")


if __name__ == "__main__":
    main()
