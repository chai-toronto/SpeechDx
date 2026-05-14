"""Per-task prompt configs for the LP-style eval pipeline.

Single source of truth for all task prompts (LP and regression alike).
Both the Gemini and Qwen eval runners import from here.

Each entry has a `kind` discriminator:
  binary, multiclass -> structured: context, description, answer_map
  multilabel         -> structured: context, description, labels (+ optional
                        label_descriptions, label_questions for awkward keys)
  regression         -> single free-text `prompt` (LP doesn't apply -- the
                        target is a continuous integer scale that doesn't fit
                        the single-token letter scheme)

Letter assignment in answer_map matches TASKS["classes"] order: letter
index == class index. Keep that invariant when editing -- the parser
relies on it via probs_to_class_vector.
"""
from __future__ import annotations

from typing import Any

from lp_prompts import build_choice_prompt, build_multilabel_prompts


# ============================================================
# Configs
# ============================================================

LP_CONFIGS: dict[str, dict[str, Any]] = {
    # ---------- T1: E-DAIC depression yes/no (binary) ----------
    "T1": {
        "kind": "binary",
        "context": (
            "You are a clinical screening assistant. The audio is a "
            "participant from a depression-screening interview (E-DAIC)."
        ),
        "description": (
            "Decide whether this participant most likely meets criteria "
            "for clinical depression."
        ),
        "answer_map": {"no": "A", "yes": "B"},
    },

    # ---------- T2: E-DAIC PHQ-8 (regression) ----------
    "T2": {
        "kind": "regression",
        "prompt": (
            "You are a clinical screening assistant. The audio is a "
            "participant from a depression-screening interview (E-DAIC). "
            "Estimate the participant's PHQ-8 total score (integer between "
            "0 and 24, where higher means more severe depressive symptoms).\n\n"
            "Reply on the final line in exactly this format:\n"
            "ANSWER: <integer 0-24>"
        ),
    },

    # ---------- T3: RAVDESS 8-class emotion ----------
    "T3": {
        "kind": "multiclass",
        "context": (
            "You are listening to a recording of an actor speaking a short "
            "scripted statement while portraying a target emotion."
        ),
        "description": (
            "Identify the portrayed emotion."
        ),
        "answer_map": {
            "neutral": "A", "calm": "B", "happy": "C", "sad": "D",
            "angry": "E", "fearful": "F", "disgust": "G", "surprised": "H",
        },
    },

    # ---------- T4: RAVDESS negative-vs-non-negative (binary) ----------
    "T4": {
        "kind": "binary",
        "context": (
            "You are listening to an actor portraying an emotion."
        ),
        "description": (
            "Decide whether the emotion is NEGATIVE (sad, angry, fearful, "
            "disgust) or NON-NEGATIVE (neutral, calm, happy, surprised)."
        ),
        "answer_map": {"non-negative": "A", "negative": "B"},
    },

    # ---------- T5: IEMOCAP 4-class emotion ----------
    "T5": {
        "kind": "multiclass",
        "context": (
            "You are listening to a single utterance from a dyadic acted-"
            "conversation recording."
        ),
        "description": (
            "Identify the speaker's emotional state."
        ),
        "answer_map": {"neutral": "A", "angry": "B", "sad": "C", "happy": "D"},
    },

    # ---------- T6: IEMOCAP negative-vs-non-negative (binary) ----------
    "T6": {
        "kind": "binary",
        "context": "You are listening to a single utterance.",
        "description": (
            "Decide whether the speaker's emotion is NEGATIVE (angry, sad) "
            "or NON-NEGATIVE (neutral, happy)."
        ),
        "answer_map": {"non-negative": "A", "negative": "B"},
    },

    # ---------- T7: ADReSS-M dementia yes/no ----------
    "T7": {
        "kind": "binary",
        "context": (
            "You are a clinical screening assistant. The audio is from a "
            "participant describing the picture of a lion lying with a cub in the dessert while eating as part of a "
            "cognitive assessment."
        ),
        "description": (
            "Based on speech "
            "decide whether this participant most likely has dementia "
            "(probable Alzheimer's disease)."
        ),
        "answer_map": {"no": "A", "yes": "B"},
    },

    # ---------- T8: MMSE score (regression) ----------
    "T8": {
        "kind": "regression",
        "prompt": (
            "You are a clinical screening assistant. The audio is a "
            "picture description of a lion lying with a cub in the dessert while eating. Estimate the participant's "
            "MMSE (Mini-Mental State Examination) total score, an integer "
            "0-30 where higher values indicate better cognition. Severity "
            "bands: 0-9 severe, 10-18 moderate, 19-23 mild, 24-30 normal.\n\n"
            "Reply on the final line in exactly this format:\n"
            "ANSWER: <integer 0-30>"
        ),
    },

    # ---------- T9: aphasia yes/no ----------
    "T9": {
        "kind": "binary",
        "context": (
            "You are a clinical screening assistant. The audio is from a "
            "participant performing a connected-speech task (picture "
            "description, narrative, or naming)."
        ),
        "description": (
            "Decide whether this "
            "participant is likely to have aphasia."
        ),
        "answer_map": {"no": "A", "yes": "B"},
    },

    # ---------- T10: TORGO dysarthria yes/no ----------
    "T10": {
        "kind": "binary",
        "context": (
            "You are listening to a participant reading short utterances "
            "or sentences in English."
        ),
        "description": (
            "Decide whether this speaker has dysarthria."
        ),
        "answer_map": {"no": "A", "yes": "B"},
    },

    # ---------- T11: dysarthria severity (regression) ----------
    "T11": {
        "kind": "regression",
        "prompt": (
            "You are listening to a recording from a participant with "
            "dysarthria reading short utterances. Rate the severity of "
            "dysarthria.\n"
            "1 = very low (mild)\n"
            "2 = low (moderate)\n"
            "3 = medium (severe)\n\n"
            "Reply on the final line in exactly this format:\n"
            "ANSWER: <integer 1-3>"
        ),
    },

    # ---------- T12: UA-Speech dysarthria yes/no ----------
    "T12": {
        "kind": "binary",
        "context": (
            "You are listening to a single-word recording from either a "
            "healthy speaker or a speaker with cerebral-palsy dysarthria."
        ),
        "description": (
            "Decide whether this speaker has dysarthria."
        ),
        "answer_map": {"no": "A", "yes": "B"},
    },

    # ---------- T13: MDVR-KCL Parkinson's yes/no ----------
    "T13": {
        "kind": "binary",
        "context": (
            "You are a clinical screening assistant. The audio is from a "
            "participant performing voice tasks (sustained vowels and/or "
            "read speech)."
        ),
        "description": (
            "Decide whether this speaker has Parkinson's disease."
        ),
        "answer_map": {"no": "A", "yes": "B"},
    },

    # ---------- T14: UPDRS-II item 5 (regression) ----------
    "T14": {
        "kind": "regression",
        "prompt": (
            "You are a clinical screening assistant for Parkinson's "
            "disease. The audio is from a participant performing voice "
            "tasks. Estimate the UPDRS-II item 5 (speech disability over "
            "the past week):\n"
            "  0 = normal\n"
            "  1 = slight loss of expression, diction, and/or volume\n"
            "  2 = monotonous, slurred but understandable\n"
            "  3 = marked impairment, hard to understand\n"
            "  4 = unintelligible most of the time\n\n"
            "Reply on the final line in exactly this format:\n"
            "ANSWER: <integer 0-4>"
        ),
    },

    # ---------- T15: UPDRS-III item 18 (speech, examiner-rated) ----------
    "T15": {
        "kind": "regression",
        "prompt": (
            "You are a clinical screening assistant for Parkinson's "
            "disease. The audio is from a participant performing voice "
            "tasks. Estimate the UPDRS-III item 18 (speech, as scored on "
            "the motor examination):\n"
            "  0 = normal\n"
            "  1 = slight loss of expression, diction, and/or volume\n"
            "  2 = monotone, slurred but understandable\n"
            "  3 = marked impairment, hard to understand\n"
            "  4 = unintelligible\n\n"
            "Reply on the final line in exactly this format:\n"
            "ANSWER: <integer 0-4>"
        ),
    },

    # ---------- T16: Hoehn & Yahr stage (regression) ----------
    "T16": {
        "kind": "regression",
        "prompt": (
            "You are a clinical screening assistant for Parkinson's "
            "disease. From the audio, estimate the Hoehn & Yahr stage:\n"
            "  0 = no signs of disease\n"
            "  1 = unilateral symptoms only\n"
            "  2 = bilateral symptoms, no balance impairment\n"
            "  3 = mild-to-moderate bilateral with postural instability; "
            "physically independent\n"
            "  4 = severe disability, still walks/stands unassisted\n"
            "  5 = wheelchair-bound or bedridden unless assisted\n\n"
            "Reply on the final line in exactly this format:\n"
            "ANSWER: <integer 0-5>"
        ),
    },

    # ---------- T17: SEP-28k disfluency yes/no ----------
    "T17": {
        "kind": "binary",
        "context": "You are listening to a German speaker reading or speaking.",
        "description": (
            "Decide whether the speech contains ANY disfluency (blocks, "
            "prolongations, sound or word/phrase repetitions, modified "
            "words, interjections), or is fluent."
        ),
        "answer_map": {"no": "A", "yes": "B"},
    },

    # ---------- T18: SEP-28k multilabel disfluency ----------
    "T18": {
        "kind": "multilabel",
        "context": "You are listening to a German speaker.",
        "description": (
            "Several speech-disfluency labels may apply. Decide for each "
            "label independently."
        ),
        "labels": [
            "block", "prolongation", "sound_rep", "word_rep",
            "modified", "interjection", "no_disfl", "garbage",
        ],
        "label_descriptions": {
            "block": "speech block",
            "prolongation": "sound prolongation",
            "sound_rep": "sound repetition",
            "word_rep": "word or phrase repetition",
            "modified": "modified word",
        },
        "label_questions": {
            "no_disfl": "Is the speech entirely fluent (no disfluency present at all)?",
            "garbage": "Is this audio unintelligible, noisy, or non-speech?",
        },
    },

    # ---------- T19: Coswara symptomatic yes/no ----------
    "T19": {
        "kind": "binary",
        "context": (
            "You are listening to a respiratory sample (cough, breath, or "
            "short speech) submitted by a participant for COVID-19 screening."
        ),
        "description": (
            "Decide whether the participant is symptomatic (any "
            "respiratory or general illness symptom -- cough, fever, sore "
            "throat, shortness of breath, etc.)."
        ),
        "answer_map": {"no": "A", "yes": "B"},
    },

    # ---------- T20: Coswara symptomatic (subset, same prompt as T19) ----------
    "T20": None,  # filled in below

    # ---------- T21: Coswara COVID positive yes/no ----------
    "T21": {
        "kind": "binary",
        "context": (
            "You are listening to a respiratory sample (cough, breath, or "
            "short speech) submitted by a participant."
        ),
        "description": (
            "Decide whether this participant has tested positive for COVID-19."
        ),
        "answer_map": {"no": "A", "yes": "B"},
    },

    # ---------- T22: Coswara COVID positive (subset, same prompt as T21) ----------
    "T22": None,  # filled in below

    # ---------- T23: Coswara multilabel symptoms ----------
    "T23": {
        "kind": "multilabel",
        "context": (
            "You are listening to a respiratory sample (cough, breath, or "
            "short speech)."
        ),
        "description": (
            "Identify which symptoms the participant most likely has. "
            "Decide for each symptom independently."
        ),
        "labels": [
            "drycough", "wetcough", "fever", "headache", "muscleache",
            "dizziness", "sorethroat", "shortbreath", "tightness",
            "runnyblockednose", "smelltasteloss", "runny",
        ],
        "label_descriptions": {
            "drycough": "dry cough",
            "wetcough": "wet (productive) cough",
            "muscleache": "muscle ache",
            "sorethroat": "sore throat",
            "shortbreath": "shortness of breath",
            "tightness": "chest tightness",
            "runnyblockednose": "runny or blocked nose",
            "smelltasteloss": "loss of smell or taste",
            "runny": "runny nose",
        },
    },

    # ---------- T24: COVID-19 Sounds symptomatic yes/no ----------
    "T24": {
        "kind": "binary",
        "context": (
            "You are listening to a respiratory or voice sample submitted "
            "by a participant for COVID-19 screening."
        ),
        "description": (
            "Decide whether the participant is symptomatic (any "
            "respiratory or general illness symptom)."
        ),
        "answer_map": {"no": "A", "yes": "B"},
    },

    # ---------- T25: COVID-19 Sounds positive yes/no ----------
    "T25": {
        "kind": "binary",
        "context": "You are listening to a respiratory or voice sample.",
        "description": (
            "Decide whether this participant has COVID-19 (tested positive)."
        ),
        "answer_map": {"no": "A", "yes": "B"},
    },

    # ---------- T26: COVID-19 Sounds multilabel symptoms ----------
    "T26": {
        "kind": "multilabel",
        "context": "You are listening to a respiratory or voice sample.",
        "description": (
            "Identify which symptoms the participant most likely has. "
            "Decide for each symptom independently."
        ),
        "labels": [
            "cold", "cough", "fever", "diarrhoea", "loss_of_smell",
            "muscularpain", "breathing_difficulty", "fatigue",
            "sore_throat", "others_resp",
        ],
        "label_descriptions": {
            "loss_of_smell": "loss of smell",
            "muscularpain": "muscular pain",
            "breathing_difficulty": "breathing difficulty",
            "sore_throat": "sore throat",
            "others_resp": "any other respiratory symptom",
        },
    },

    # ---------- T27: SVD voice pathology yes/no ----------
    "T27": {
        "kind": "binary",
        "context": (
            "You are a voice clinician. The audio is a sustained vowel or "
            "short reading from a participant."
        ),
        "description": (
            "Decide whether the voice exhibits pathology (hoarseness, "
            "breathiness, strain, roughness, tremor, abnormal pitch, etc.) "
            "as opposed to a normal voice."
        ),
        "answer_map": {"no": "A", "yes": "B"},
    },
}

# T20 / T22 share T19 / T21 prompts (subset vs full population).
LP_CONFIGS["T20"] = LP_CONFIGS["T19"]
LP_CONFIGS["T22"] = LP_CONFIGS["T21"]


# ============================================================
# Prompt builder
# ============================================================

def build_prompt(task_id: str) -> str | list[tuple[str, str]]:
    """Return the prompt(s) for a task.

    binary / multiclass / regression -> single prompt string
    multilabel                       -> [(label, prompt), ...] for K subcalls
    """
    cfg = LP_CONFIGS[task_id]
    kind = cfg["kind"]
    if kind == "regression":
        return cfg["prompt"]
    if kind in ("binary", "multiclass"):
        return build_choice_prompt(
            context=cfg["context"],
            description=cfg["description"],
            answer_map=cfg["answer_map"],
        )
    if kind == "multilabel":
        return build_multilabel_prompts(
            context=cfg["context"],
            description=cfg["description"],
            labels=cfg["labels"],
            label_descriptions=cfg.get("label_descriptions"),
            label_questions=cfg.get("label_questions"),
        )
    raise ValueError(f"unknown kind {kind!r} for {task_id}")


def get_kind(task_id: str) -> str:
    return LP_CONFIGS[task_id]["kind"]


def get_answer_map(task_id: str) -> dict[str, str] | None:
    """For binary/multiclass. Returns None for regression/multilabel."""
    cfg = LP_CONFIGS[task_id]
    if cfg["kind"] in ("binary", "multiclass"):
        return cfg["answer_map"]
    return None


def get_labels(task_id: str) -> list[str] | None:
    """For multilabel. Returns None otherwise."""
    cfg = LP_CONFIGS[task_id]
    if cfg["kind"] == "multilabel":
        return list(cfg["labels"])
    return None


# ============================================================
# Sanity check on import
# ============================================================

def _validate() -> None:
    """Catch authoring drift between LP_CONFIGS and TASKS at import time."""
    import importlib
    tq = importlib.import_module("test_qwen3omni_all")
    tasks = tq.TASKS
    issues = []
    for tid, cfg in LP_CONFIGS.items():
        if tid not in tasks:
            issues.append(f"{tid}: in LP_CONFIGS but not TASKS")
            continue
        kind = cfg["kind"]
        metric = tasks[tid]["metric"]
        # binary/multiclass: answer_map keys must match TASKS classes order
        if kind in ("binary", "multiclass"):
            classes = tasks[tid].get("classes") or []
            am_keys = list(cfg["answer_map"].keys())
            if am_keys != classes:
                issues.append(
                    f"{tid}: answer_map keys {am_keys} != TASKS classes {classes}")
        # multilabel: labels list must match TASKS classes order
        if kind == "multilabel":
            classes = tasks[tid].get("classes") or []
            if list(cfg["labels"]) != classes:
                issues.append(
                    f"{tid}: labels {cfg['labels']} != TASKS classes {classes}")
        # kind/metric agreement
        expected_metrics = {
            "binary": "binary", "multiclass": "multiclass",
            "multilabel": "multilabel", "regression": "regression",
        }
        if expected_metrics[kind] != metric:
            issues.append(f"{tid}: kind {kind} but TASKS metric {metric}")
    missing = set(tasks) - set(LP_CONFIGS)
    if missing:
        issues.append(f"missing LP_CONFIGS for tasks: {sorted(missing)}")
    if issues:
        raise RuntimeError("LP_CONFIGS validation:\n  " + "\n  ".join(issues))


if __name__ == "__main__":
    _validate()
    print("LP_CONFIGS validated against TASKS.")
    # Print one example of each kind.
    for tid in ("T1", "T3", "T18", "T2"):
        kind = LP_CONFIGS[tid]["kind"]
        out = build_prompt(tid)
        print(f"\n{'='*70}\n{tid} ({kind})\n{'='*70}")
        if isinstance(out, list):
            for lab, p in out[:2]:
                print(f"\n--- {lab} ---")
                print(p)
            if len(out) > 2:
                print(f"\n... ({len(out)-2} more labels)")
        else:
            print(out)
