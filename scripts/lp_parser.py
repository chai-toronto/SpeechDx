"""Logprob-based parser for letter-choice responses.

Pairs with `lp_prompts.py`. Designed to run offline on already-saved data
(response text + serialized logprobs), so we can re-parse without
re-calling the API.

Pipeline:
  1. Eval script calls Gemini with response_logprobs=True, logprobs=K and
     stores `serialize_logprobs(resp.candidates[0].logprobs_result)` next
     to the response text.
  2. This module reads that JSON back and turns it into a calibrated
     probability over the option letters (and from there, over class
     names via the original answer_map).

Failure modes return None (rather than raising) so caller can decide how
to handle: skip the row, fall back to text parsing, etc.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable

# A token "looks like" the answer letter if, after stripping leading
# whitespace and surrounding punctuation, it's a single uppercase letter.
_LETTER_TOKEN_RE = re.compile(r"^[\s\W_]*([A-Z])[\s\W_]*$")


# ============================================================
# Serialization (call this on the eval-side before saving CSV)
# ============================================================

def serialize_logprobs(logprobs_result) -> dict | None:
    """Convert SDK LogprobsResult to a compact JSON-serializable dict.

    Output shape:
        {
          "tokens": [
            {"t": "<token str>", "lp": <float>,
             "top": [["<alt token>", <float>], ...]},
            ...
          ]
        }

    `top` may be None at positions where the SDK didn't return alternates.
    """
    if logprobs_result is None:
        return None
    chosen = logprobs_result.chosen_candidates or []
    top = logprobs_result.top_candidates or []
    tokens = []
    for i, ch in enumerate(chosen):
        top_list = None
        if i < len(top) and top[i] is not None and top[i].candidates:
            top_list = [[c.token, c.log_probability] for c in top[i].candidates]
        tokens.append({
            "t": ch.token,
            "lp": ch.log_probability,
            "top": top_list,
        })
    return {"tokens": tokens}


# ============================================================
# Core: find the answer-letter position and read its top-k
# ============================================================

def _normalize_letter_token(tok: str) -> str | None:
    """Return the bare uppercase letter if `tok` looks like an answer
    letter token (possibly with leading space / punctuation), else None."""
    if tok is None:
        return None
    m = _LETTER_TOKEN_RE.match(tok)
    return m.group(1) if m else None


def _find_answer_letter_position(tokens: list[dict]) -> int | None:
    """Locate the index of the chosen-candidate token that is the answer
    letter. Strategy:

      1. Concatenate all token strings; find the *last* occurrence of
         "ANSWER" (case-insensitive) -- the prompt asks the model to put
         it on the final line, so the last hit is the real answer.
      2. Walk forward from that token through any ':', whitespace,
         punctuation, or markdown wrappers, and return the first token
         that normalizes to a single A-Z letter.
    """
    if not tokens:
        return None

    # Find last token whose string contains "ANSWER" (case-insensitive).
    answer_idx = None
    for i, tok in enumerate(tokens):
        s = (tok.get("t") or "")
        if "answer" in s.lower():
            answer_idx = i
    if answer_idx is None:
        return None

    for j in range(answer_idx + 1, len(tokens)):
        s = tokens[j].get("t") or ""
        if not s.strip():
            continue
        # Skip pure-punctuation tokens like ':' or '**'.
        if not re.search(r"[A-Za-z]", s):
            continue
        if _normalize_letter_token(s) is not None:
            return j
        # Hit a non-letter alphabetic token before finding the letter --
        # bail. (e.g., model wrote "ANSWER: yes" instead of "ANSWER: A".)
        return None
    return None


def extract_letter_probs(
    logprobs: dict | str | None,
    allowed_letters: Iterable[str],
) -> dict[str, float] | None:
    """Return softmax-renormalized {letter: P} restricted to allowed_letters.

    Returns None if the answer letter token can't be located, or if none
    of the allowed letters appear in its top-k candidates.

    Note: we use the top-k *at the answer position*, not the chosen-token
    logprob alone -- this is what makes the probability calibrated across
    the option set.
    """
    if logprobs is None:
        return None
    if isinstance(logprobs, str):
        try:
            logprobs = json.loads(logprobs)
        except (json.JSONDecodeError, TypeError):
            return None

    tokens = (logprobs or {}).get("tokens") or []
    pos = _find_answer_letter_position(tokens)
    if pos is None:
        return None

    top = tokens[pos].get("top")
    if not top:
        # No alternates -- best we can do is report 1.0 on the chosen
        # letter if it's in the allowed set, else give up.
        chosen_letter = _normalize_letter_token(tokens[pos].get("t") or "")
        allowed = {L.upper() for L in allowed_letters}
        if chosen_letter and chosen_letter in allowed:
            return {chosen_letter: 1.0}
        return None

    allowed = {L.upper() for L in allowed_letters}
    # Collect the highest logprob seen for each allowed letter (a letter
    # may appear multiple times with different surface forms: "A", " A").
    best_lp: dict[str, float] = {}
    for tok_str, lp in top:
        L = _normalize_letter_token(tok_str)
        if L is None or L not in allowed:
            continue
        if L not in best_lp or lp > best_lp[L]:
            best_lp[L] = lp

    if not best_lp:
        return None

    # Softmax-renormalize over the restricted set.
    m = max(best_lp.values())
    exps = {L: math.exp(lp - m) for L, lp in best_lp.items()}
    z = sum(exps.values())
    return {L: e / z for L, e in exps.items()}


# ============================================================
# High-level: response + answer_map -> {answer_text: prob}
# ============================================================

def parse_choice_response(
    response_text: str,
    logprobs: dict | str | None,
    answer_map: dict[str, str],
) -> dict[str, float] | None:
    """For binary/multiclass. answer_map matches the prompt: {text: letter}.

    Returns {answer_text: probability} restricted to the option set, or
    None if the letter can't be extracted from logprobs.

    response_text is currently unused (kept in the signature so callers
    can later add a text-fallback path without changing the API).
    """
    del response_text  # reserved for future text-fallback parsing
    letter_probs = extract_letter_probs(logprobs, answer_map.values())
    if letter_probs is None:
        return None
    letter_to_text = {v.upper(): k for k, v in answer_map.items()}
    out = {}
    for L, p in letter_probs.items():
        if L in letter_to_text:
            out[letter_to_text[L]] = p
    return out or None


def probs_to_class_vector(
    text_probs: dict[str, float],
    classes: list[str],
) -> list[float] | None:
    """Reorder {answer_text: prob} into the canonical TASKS['classes']
    order. Missing classes get 0.0. Returns None if text_probs is None."""
    if text_probs is None:
        return None
    return [float(text_probs.get(c, 0.0)) for c in classes]


def argmax_class(
    text_probs: dict[str, float] | None,
    classes: list[str],
) -> int | None:
    """Hard prediction (class index) from a parsed prob dict."""
    if not text_probs:
        return None
    vec = probs_to_class_vector(text_probs, classes)
    if vec is None or sum(vec) == 0:
        return None
    return max(range(len(vec)), key=lambda i: vec[i])


# ============================================================
# Multilabel aggregation
# ============================================================

def aggregate_multilabel_probs(
    sub_results: dict[str, dict[str, float] | None],
    threshold: float = 0.5,
) -> tuple[list[float], list[int]] | None:
    """Aggregate per-label binary results into an ordered probability vector
    and a hard 0/1 multi-hot vector.

    sub_results: {label: parse_choice_response(...) output}
                 where each inner dict is {"yes": p_yes, "no": p_no}.
    Returns (probs, hard) in the order of sub_results.keys(), or None if
    every label failed to parse.

    Threshold-as-argument because some downstream metrics may want to
    sweep it; default 0.5 matches the standard convention.
    """
    if not sub_results:
        return None
    probs: list[float] = []
    any_ok = False
    for _lab, res in sub_results.items():
        if res is None:
            probs.append(float("nan"))
        else:
            any_ok = True
            probs.append(float(res.get("yes", 0.0)))
    if not any_ok:
        return None
    hard = [1 if (not math.isnan(p)) and p >= threshold else 0 for p in probs]
    return probs, hard


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    # Synthetic logprobs payload mimicking a "ANSWER: A" response.
    sample = {
        "tokens": [
            {"t": "ANSWER", "lp": -0.01,
             "top": [["ANSWER", -0.01], ["Answer", -4.6]]},
            {"t": ":",      "lp": -0.0,
             "top": [[":", 0.0]]},
            {"t": " A",     "lp": -0.51,
             "top": [[" A", -0.51], [" B", -1.20], [" C", -3.40]]},
        ]
    }

    # Binary case: yes -> A, no -> B.
    binary_map = {"yes": "A", "no": "B"}
    probs = parse_choice_response("ANSWER: A", sample, binary_map)
    print("binary probs:", probs)
    print("binary pred class idx (classes=['no','yes']):",
          argmax_class(probs, ["no", "yes"]))

    # Multiclass case with three options.
    mc_map = {"happy": "A", "sad": "B", "angry": "C"}
    probs = parse_choice_response("ANSWER: A", sample, mc_map)
    print("\nmulticlass probs:", probs)
    classes = ["happy", "sad", "angry"]
    print("class vector:", probs_to_class_vector(probs, classes))

    # Multilabel aggregation: 3 labels, each a yes/no subcall.
    sub = {
        "block":        parse_choice_response("ANSWER: A", sample, binary_map),
        "prolongation": parse_choice_response("ANSWER: A", {
            "tokens": [
                {"t": "ANSWER", "lp": 0.0,
                 "top": [["ANSWER", 0.0]]},
                {"t": ":", "lp": 0.0, "top": [[":", 0.0]]},
                {"t": " B", "lp": -0.05,
                 "top": [[" B", -0.05], [" A", -3.00]]},
            ]
        }, binary_map),
        "interjection": None,  # parse failed for this sub-call
    }
    agg = aggregate_multilabel_probs(sub)
    print("\nmultilabel:", agg)