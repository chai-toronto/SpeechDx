"""Letter-choice prompt generator for log-probability evaluation.

Designed for Gemini (and any model with top-k logprobs) so we can read a
calibrated probability over classes instead of a thresholded text answer.

Shape:
  binary / multiclass -> single prompt; ask for one letter (A, B, C, ...).
                         At eval time, find the letter token in the response
                         and read its top-k logprobs, restrict to the option
                         letters, softmax-renormalize.
  multilabel          -> K independent binary prompts, one per label
                         (yes/no -> A/B). Aggregate per-label P(yes) into a
                         multi-hot vector or a real-valued probability vector.

Regression is intentionally not handled here -- keep free-text generation for
PHQ-8 / MMSE / UPDRS-III.
"""
from __future__ import annotations

from typing import Iterable


def build_choice_prompt(
    context: str,
    description: str,
    answer_map: dict[str, str],
) -> str:
    """Single-question multiple-choice prompt.

    answer_map: ordered {answer_text: letter}. Caller controls letter
    assignment, which is also the order options are listed in the prompt.
    The letter assigned to each text answer is what you'll later look up in
    the response logprobs to recover P(class).

    Example:
        build_choice_prompt(
            context="You are a clinical screening assistant. The audio is "
                    "from an E-DAIC depression-screening interview.",
            description="Decide whether this participant most likely meets "
                        "criteria for clinical depression.",
            answer_map={"yes": "A", "no": "B"},
        )
    """
    if not answer_map:
        raise ValueError("answer_map must be non-empty")
    if len(set(answer_map.values())) != len(answer_map):
        raise ValueError(f"answer_map letters must be unique: {answer_map}")

    options = "\n".join(f"  {letter}) {text}" for text, letter in answer_map.items())
    letters = ", ".join(answer_map.values())
    return (
        f"{context.strip()}\n\n"
        f"{description.strip()}\n\n"
        f"Choose one:\n{options}\n\n"
        f"Reply on the final line in exactly this format:\n"
        f"ANSWER: <one of: {letters}>"
    )


def build_multilabel_prompts(
    context: str,
    description: str,
    labels: Iterable[str],
    label_descriptions: dict[str, str] | None = None,
    label_questions: dict[str, str] | None = None,
    yes_letter: str = "A",
    no_letter: str = "B",
) -> list[tuple[str, str]]:
    """Decomposed yes/no prompts, one per label.

    Returns [(label, prompt), ...]. Each subprompt asks whether one
    specific label is present. At eval time, read P(yes_letter) per label
    -> per-label probability vector.

    description: shared task framing (what the audio is, what kind of
    labels are being judged). The per-label question is appended.
    label_descriptions: {label: short clarification}, appended in parens
    after the label name in the default question.
    label_questions: {label: full question text} overrides the default
    "Is '<label>' (clarification) present in this audio?" template.
    Useful for meta-labels like "no_disfl" or unidiomatic keys where
    the default phrasing reads awkwardly.
    """
    label_descriptions = label_descriptions or {}
    label_questions = label_questions or {}
    prompts: list[tuple[str, str]] = []
    for lab in labels:
        if lab in label_questions:
            q = label_questions[lab].strip()
        else:
            clarif = label_descriptions.get(lab, "").strip()
            clarif_part = f" ({clarif})" if clarif else ""
            q = f"Is '{lab}'{clarif_part} present in this audio?"
        sub_desc = f"{description.strip()}\n\nQuestion: {q}"
        prompts.append((
            lab,
            build_choice_prompt(
                context=context,
                description=sub_desc,
                answer_map={"yes": yes_letter, "no": no_letter},
            ),
        ))
    return prompts


if __name__ == "__main__":
    # Smoke-test: print a binary, multiclass, and multilabel example.
    print("=" * 70, "\nBINARY (T1 depression)\n", "=" * 70, sep="")
    print(build_choice_prompt(
        context=("You are a clinical screening assistant. The audio is from "
                 "an E-DAIC depression-screening interview."),
        description=("Decide whether this participant most likely meets "
                     "criteria for clinical depression."),
        answer_map={"yes": "A", "no": "B"},
    ))

    print("\n", "=" * 70, "\nMULTICLASS (T3 RAVDESS emotion)\n", "=" * 70, sep="")
    classes = ["neutral", "calm", "happy", "sad",
               "angry", "fearful", "disgust", "surprised"]
    print(build_choice_prompt(
        context=("You are listening to an actor speaking a short scripted "
                 "statement while portraying a target emotion."),
        description=("Identify the emotion based on vocal cues (pitch, "
                     "energy, tempo, prosody, voice quality)."),
        answer_map={c: chr(ord("A") + i) for i, c in enumerate(classes)},
    ))

    print("\n", "=" * 70, "\nMULTILABEL (T18 disfluency)\n", "=" * 70, sep="")
    labels = ["block", "prolongation", "sound_rep", "word_rep",
              "modified", "interjection"]
    for lab, p in build_multilabel_prompts(
        context="You are listening to a German speaker reading or speaking.",
        description=("Several speech-disfluency labels may apply. Decide "
                     "for each label independently."),
        labels=labels,
        label_descriptions={
            "sound_rep": "sound repetition",
            "word_rep": "word or phrase repetition",
            "modified": "modified word",
        },
    ):
        print(f"\n--- {lab} ---")
        print(p)