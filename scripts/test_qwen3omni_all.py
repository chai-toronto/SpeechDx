"""Unified zero-shot evaluation of Qwen3-Omni-Thinking on SpeechDx tasks.

Usage:
    python scripts/test_qwen3omni_all.py --task T3

Reads samples from a manifest (preferred) or per-task metadata CSV. For CV
tasks, evaluates each unique sample once (zero-shot predictions are
fold-independent), then reports per-fold metrics with mean ± std.

Reasoning traces stay in the `response` column for prompt-engineering use.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import tempfile
import time
from collections import defaultdict
from pathlib import Path

# Avoid transformers' "is this a base mistral?" check making an HF API call —
# breaks under HF_HUB_OFFLINE and gets rate-limited at scale.
try:
    import transformers.tokenization_utils_base as _tub
    _tub.is_base_mistral = lambda *_a, **_kw: False
except Exception:
    pass

import math
import sys

import numpy as np
import pandas as pd
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

# LP eval helpers (prompts, parser, per-task configs).
from lp_configs import build_prompt, get_kind, get_answer_map, get_labels  # noqa: E402
from lp_parser import (  # noqa: E402
    parse_choice_response, probs_to_class_vector, argmax_class,
    aggregate_multilabel_probs,
)

# Top-K logprobs requested per generated token. Must comfortably exceed
# the largest class set (T3: 8 letters); bump to 30 here since vLLM has
# no API-side cap and the cost of larger K is just larger payloads.
LOGPROBS_K = 30

# ============================================================
# Task registry
# ============================================================


# Each task config:
#   source: where to read samples from. Either:
#       {"manifest": <path>, "split": "test"|"valid"}  (preferred)
#       {"csv": <path>, "split_col": <col>, "split_value": <int>}  (fallback)
#   label_col: which column in each row holds the ground-truth target
#   parser:    which output-parser to use ("yes_no", "label", "integer", "multilabel")
#   metric:    "binary", "multiclass", "regression", "multilabel"
#   classes:   list of class names for multiclass/multilabel (output order)
#   range:     (min, max) for regression — predictions clipped to this
def _t(**kwargs): return kwargs

TASKS = {
    "T1": _t(
        source={"manifest": "exps/single_task/T1/manifest", "split": "test"},
        label_col="label", parser="yes_no", metric="binary",
        classes=["no", "yes"],
    ),
    "T2": _t(
        source={"manifest": "exps/single_task/T2/manifest", "split": "test"},
        label_col="PHQ_Score", parser="integer", metric="regression",
        range=(0, 24),
    ),
    "T3": _t(
        source={"manifest": "exps/single_task/T3/manifest", "split": "valid"},
        label_col="label", parser="label", metric="multiclass",
        classes=["neutral","calm","happy","sad","angry","fearful","disgust","surprised"],
    ),
    "T4": _t(
        source={"manifest": "exps/single_task/T4/manifest", "split": "valid"},
        label_col="label", parser="label", metric="binary",
        classes=["non-negative", "negative"],
    ),
    "T5": _t(
        source={"manifest": "exps/single_task/T5/manifest", "split": "valid"},
        label_col="label", parser="label", metric="multiclass",
        classes=["neutral", "angry", "sad", "happy"],
    ),
    "T6": _t(
        source={"manifest": "exps/single_task/T6/manifest", "split": "valid"},
        label_col="label", parser="label", metric="binary",
        classes=["non-negative", "negative"],
    ),
    "T7": _t(
        source={"manifest": "exps/single_task/T7/manifest", "split": "test"},
        label_col="label", parser="yes_no", metric="binary",
        classes=["no", "yes"],
    ),
    "T8": _t(
        source={"manifest": "exps/single_task/T8/manifest", "split": "test"},
        label_col="mmse", parser="integer", metric="regression",
        range=(0, 30),
    ),
    "T9": _t(
        source={"manifest": "exps/single_task/T9/manifest", "split": "test"},
        label_col="label", parser="yes_no", metric="binary",
        classes=["no", "yes"],
    ),
    "T10": _t(
        source={"manifest": "exps/single_task/T10/manifest", "split": "valid"},
        label_col="label", parser="yes_no", metric="binary",
        classes=["no", "yes"],
    ),
    "T11": _t(
        source={"manifest": "exps/single_task/T11/manifest", "split": "valid"},
        label_col="severity", parser="integer", metric="regression",
        range=(1, 3),
    ),
    "T12": _t(
        source={"manifest": "exps/single_task/T12/manifest", "split": "valid"},
        label_col="label", parser="yes_no", metric="binary",
        classes=["no", "yes"],
    ),
    "T13": _t(
        source={"manifest": "exps/single_task/T13/manifest", "split": "valid"},
        label_col="label", parser="yes_no", metric="binary",
        classes=["no", "yes"],
    ),
    "T14": _t(
        source={"manifest": "exps/single_task/T14/manifest", "split": "valid"},
        label_col="updrs_ii5", parser="integer", metric="regression",
        range=(0, 4),
    ),
    "T15": _t(
        source={"manifest": "exps/single_task/T15/manifest", "split": "valid"},
        label_col="updrs_iii18", parser="integer", metric="regression",
        range=(0, 4),
    ),
    "T16": _t(
        source={"manifest": "exps/single_task/T16/manifest", "split": "valid"},
        label_col="hy_rating", parser="integer", metric="regression",
        range=(0, 5),
    ),
    "T17": _t(
        source={"manifest": "exps/single_task/T17/manifest", "split": "valid"},
        label_col="label", parser="yes_no", metric="binary",
        classes=["no", "yes"],
    ),
    "T18": _t(
        source={"manifest": "exps/single_task/T18/manifest", "split": "valid"},
        label_col="label", parser="multilabel", metric="multilabel",
        classes=["block","prolongation","sound_rep","word_rep","modified",
                 "interjection","no_disfl","garbage"],
    ),
    "T19": _t(
        source={"manifest": "exps/single_task/T19/manifest", "split": "test"},
        label_col="label", parser="yes_no", metric="binary",
        classes=["no", "yes"],
    ),
    "T20": _t(
        source={"manifest": "exps/single_task/T20/manifest", "split": "test"},
        label_col="label", parser="yes_no", metric="binary",
        classes=["no", "yes"],
    ),
    "T21": _t(
        source={"manifest": "exps/single_task/T21/manifest", "split": "test"},
        label_col="label", parser="yes_no", metric="binary",
        classes=["no", "yes"],
    ),
    "T22": _t(
        source={"manifest": "exps/single_task/T22/manifest", "split": "test"},
        label_col="label", parser="yes_no", metric="binary",
        classes=["no", "yes"],
    ),
    "T23": _t(
        source={"manifest": "exps/single_task/T23/manifest", "split": "test"},
        label_col="label", parser="multilabel", metric="multilabel",
        classes=["drycough","wetcough","fever","headache","muscleache","dizziness",
                 "sorethroat","shortbreath","tightness","runnyblockednose",
                 "smelltasteloss","runny"],
    ),
    "T24": _t(
        source={"manifest": "exps/single_task/T24/manifest", "split": "test"},
        label_col="label", parser="yes_no", metric="binary",
        classes=["no", "yes"],
    ),
    "T25": _t(
        source={"manifest": "exps/single_task/T25/manifest", "split": "test"},
        label_col="label", parser="yes_no", metric="binary",
        classes=["no", "yes"],
    ),
    "T26": _t(
        source={"manifest": "exps/single_task/T26/manifest", "split": "test"},
        label_col="label", parser="multilabel", metric="multilabel",
        classes=["cold","cough","fever","diarrhoea","loss_of_smell","muscularpain",
                 "breathing_difficulty","fatigue","sore_throat","others_resp"],
    ),
    "T27": _t(
        source={"manifest": "exps/single_task/T27/manifest", "split": "test"},
        label_col="label", parser="yes_no", metric="binary",
        classes=["no", "yes"],
    ),
}


# ============================================================
# Source loaders
# ============================================================

def _norm_path(p: str) -> str:
    """Make audio paths resolve under our renamed dirs (dbank → dementiabank etc)."""
    return str(Path(p).resolve())


def load_samples(task: str) -> tuple[list[dict], list[int] | None]:
    """Returns (samples, fold_assignments). fold_assignments is None for single-split."""
    cfg = TASKS[task]
    src = cfg["source"]

    if "manifest" in src:
        mdir = REPO / src["manifest"]
        fname = mdir / f"{src['split']}.json"
        with open(fname) as f:
            blob = json.load(f)

        # CV: list of K fold dicts; single: one dict.
        if isinstance(blob, list):
            samples, folds = [], []
            for fi, fold_dict in enumerate(blob):
                for rowid, row in fold_dict.items():
                    samples.append({**row, "_rowid": rowid, "_fold": fi})
                    folds.append(fi)
            return samples, folds
        else:
            samples = [{**row, "_rowid": rowid} for rowid, row in blob.items()]
            return samples, None

    # CSV fallback
    df = pd.read_csv(REPO / src["csv"])
    df = df[df[src["split_col"]] == src["split_value"]].reset_index(drop=True)
    samples = []
    for i, row in df.iterrows():
        samples.append({**row.to_dict(), "_rowid": str(row.get("uid", i))})
    return samples, None


# ============================================================
# Audio
# ============================================================

def load_and_trim_audio(path: str, max_duration: float, target_sr: int = 16000):
    import librosa
    wav, _ = librosa.load(path, sr=target_sr, mono=True)
    if max_duration and len(wav) > int(max_duration * target_sr):
        wav = wav[: int(max_duration * target_sr)]
    return wav.astype(np.float32)


# ============================================================
# Parsers
# ============================================================

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"ANSWER\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _answer_payload(text: str) -> str | None:
    cleaned = strip_thinking(text)
    m = _ANSWER_RE.search(cleaned)
    if m:
        return m.group(1).strip().rstrip(".").strip()
    return None


def parse_integer(text: str, lo: float, hi: float) -> float | None:
    p = _answer_payload(text)
    text_to_search = p if p is not None else strip_thinking(text)
    m = re.search(r"-?\d+(?:\.\d+)?", text_to_search)
    if m is None:
        return None
    val = float(m.group(0))
    return float(np.clip(val, lo, hi))


# ============================================================
# Backend
# ============================================================

class QwenOmniVLLM:
    def __init__(self, model_path: str, max_model_len: int,
                 gpu_memory_utilization: float, max_num_seqs: int):
        os.environ.setdefault("VLLM_USE_V1", "0")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

        import torch
        from vllm import LLM
        from transformers import Qwen3OmniMoeProcessor

        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        self.model = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=torch.cuda.device_count() or 1,
            limit_mm_per_prompt={"audio": 1},
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            seed=1234,
        )

    def generate_batch(self, batch_messages: list, max_tokens: int,
                       logprobs_k: int | None = None) -> list:
        """Return one (text, lp_dict) tuple per input message, or
        (None, None) for ones the processor / vLLM couldn't handle.

        When logprobs_k is set, lp_dict matches the format expected by
        lp_parser (see _vllm_lp_to_dict). When None, lp_dict is always
        None (free-text path, e.g. regression).

        Some audio inputs trigger Qwen3OmniMoeProcessor failures *inside*
        ``vllm.generate`` (not at our per-message prep) and the exception
        takes down the entire batch. To recover, on a batch failure we
        binary-split the items and recurse — bad items are isolated in
        O(log N) retries and only those end up as None.
        """
        from qwen_omni_utils import process_mm_info
        from vllm import SamplingParams

        sp_kw = dict(temperature=1e-2, max_tokens=max_tokens)
        if logprobs_k is not None:
            # LP calibration: keep the full vocab in the per-step distribution so
            # alternate class letters remain scorable. top_k=1 / top_p=0.1 would
            # collapse the post-filter distribution to the chosen token only.
            sp_kw["top_p"] = 1.0
            sp_kw["top_k"] = -1
            sp_kw["logprobs"] = logprobs_k
        else:
            sp_kw["top_p"] = 0.1
            sp_kw["top_k"] = 1
        sp = SamplingParams(**sp_kw)
        items, kept = [], []
        for j, messages in enumerate(batch_messages):
            try:
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                audios, _, _ = process_mm_info(messages, use_audio_in_video=False)
                item = {"prompt": text, "multi_modal_data": {},
                        "mm_processor_kwargs": {"use_audio_in_video": False}}
                if audios is not None:
                    item["multi_modal_data"]["audio"] = audios
                items.append(item)
                kept.append(j)
            except Exception as e:
                print(f"  [warn] processor failed on msg {j}: "
                      f"{type(e).__name__}: {str(e)[:200]}", flush=True)

        out = [(None, None)] * len(batch_messages)
        if items:
            self._generate_recursive(items, kept, sp, out, logprobs_k is not None)
        return out

    def _generate_recursive(self, items: list, kept: list, sp, out: list,
                            with_logprobs: bool) -> None:
        """Try generating `items` as a single batch; on failure, binary-split
        until the bad item(s) are isolated. Results land in `out` at indices
        from `kept` as (text, lp_dict) tuples."""
        if not items:
            return
        try:
            outputs = self.model.generate(items, sampling_params=sp)
            for k, j in enumerate(kept):
                vo = outputs[k].outputs[0]
                lp = _vllm_lp_to_dict(vo) if with_logprobs else None
                out[j] = (vo.text, lp)
            return
        except Exception as e:
            if len(items) == 1:
                print(f"  [warn] vLLM.generate bad on msg {kept[0]}: "
                      f"{type(e).__name__}: {str(e)[:200]}", flush=True)
                return
            print(f"  [warn] vLLM.generate failed on {len(items)}-item batch "
                  f"({type(e).__name__}); binary-splitting", flush=True)
            mid = len(items) // 2
            self._generate_recursive(items[:mid], kept[:mid], sp, out, with_logprobs)
            self._generate_recursive(items[mid:], kept[mid:], sp, out, with_logprobs)


def _vllm_lp_to_dict(vllm_output) -> dict | None:
    """Convert vllm CompletionOutput -> {tokens: [...]} matching lp_parser.

    vllm_output.logprobs is list[dict[token_id, Logprob]] (one dict per
    generated step). vllm_output.token_ids is the list of chosen ids.
    Each Logprob has .logprob, .decoded_token, .rank.
    """
    if vllm_output.logprobs is None:
        return None
    chosen_ids = list(vllm_output.token_ids)
    tokens = []
    for step_idx, step_dict in enumerate(vllm_output.logprobs):
        if step_dict is None or step_idx >= len(chosen_ids):
            continue
        cid = chosen_ids[step_idx]
        chosen = step_dict.get(cid)
        if chosen is None:
            continue
        # Sort by logprob desc; vLLM may already sort but defend against it.
        ranked = sorted(step_dict.values(), key=lambda l: -l.logprob)
        top_list = [[v.decoded_token, float(v.logprob)] for v in ranked]
        tokens.append({
            "t": chosen.decoded_token,
            "lp": float(chosen.logprob),
            "top": top_list,
        })
    return {"tokens": tokens}


def build_messages(audio_path: str, prompt: str) -> list:
    return [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text", "text": prompt},
        ],
    }]


# ============================================================
# Metrics
# ============================================================

def _dns_fill(task: str, df: pd.DataFrame) -> pd.DataFrame:
    """Impute unparsed LP probabilities as maximally-wrong values (DNS rule).

    binary      : pred_prob     = 1 - true
    multiclass  : pred_prob_vec = one-hot at the wrong class ((true+1) mod n)
    regression  : pred          = true + MAD  (DNS rows contribute MAD to MAE;
                  with report-card scoring 1 - MAE/(2*MAD), all-DNS -> 0.5)
    multilabel  : pred_prob_vec = 1.0 - true_vec elementwise (row-level None
                  fills the full vec; partial Nones in the list are filled
                  per-element).
    """
    cfg = TASKS[task]
    metric = cfg["metric"]
    df = df.copy()
    if metric == "binary":
        if "pred_prob" in df.columns:
            mask = df["pred_prob"].isna() & df["true"].notna()
            df.loc[mask, "pred_prob"] = 1.0 - df.loc[mask, "true"].astype(float)
    elif metric == "multiclass":
        if "pred_prob_vec" in df.columns:
            n_classes = len(cfg["classes"])
            for i in df.index[df["pred_prob_vec"].isna() & df["true"].notna()]:
                wrong = (int(df.at[i, "true"]) + 1) % n_classes
                vec = [0.0] * n_classes
                vec[wrong] = 1.0
                df.at[i, "pred_prob_vec"] = vec
    elif metric == "regression":
        true_vals = pd.to_numeric(df["true"], errors="coerce").dropna()
        mad = float((true_vals - true_vals.mean()).abs().mean()) if len(true_vals) else 0.0
        mask = df["pred"].isna() & df["true"].notna()
        df.loc[mask, "pred"] = df.loc[mask, "true"].astype(float) + mad
    elif metric == "multilabel":
        if "pred_prob_vec" in df.columns:
            for i in df.index:
                tv = df.at[i, "true_vec"]
                if not isinstance(tv, list):
                    continue
                ppv = df.at[i, "pred_prob_vec"]
                if ppv is None or (isinstance(ppv, float) and pd.isna(ppv)):
                    df.at[i, "pred_prob_vec"] = [1.0 - float(x) for x in tv]
                elif isinstance(ppv, list) and any(
                    v is None or (isinstance(v, float) and pd.isna(v)) for v in ppv
                ):
                    df.at[i, "pred_prob_vec"] = [
                        (1.0 - float(tv[j])) if (v is None or (isinstance(v, float) and pd.isna(v)))
                        else float(v)
                        for j, v in enumerate(ppv)
                    ]
    return df


def compute_metrics(task: str, df: pd.DataFrame) -> dict:
    """LP-only headline metrics: AUC for classification/multilabel, MAE for regression."""
    cfg = TASKS[task]
    metric = cfg["metric"]
    out = {"n_total": int(len(df))}
    parse_col = "pred_vec" if metric == "multilabel" else "pred"
    out["n_parsed"] = int(df[parse_col].notna().sum())
    out["n_unparsed"] = int(len(df) - out["n_parsed"])
    out["n_dns_imputed"] = out["n_unparsed"]

    filled = _dns_fill(task, df)
    from sklearn.metrics import roc_auc_score, mean_absolute_error

    if metric == "regression":
        sub = filled.dropna(subset=["pred", "true"])
        if len(sub) >= 2:
            y_true = sub["true"].astype(float).to_numpy()
            y_pred = sub["pred"].astype(float).to_numpy()
            out["mae"] = float(mean_absolute_error(y_true, y_pred))
        return out

    score_col = "pred_prob" if metric == "binary" else "pred_prob_vec"
    true_col = "true_vec" if metric == "multilabel" else "true"
    if score_col not in filled.columns:
        return out
    sub = filled.dropna(subset=[score_col, true_col])
    if len(sub) < 2:
        return out

    if metric == "binary":
        y_true = sub["true"].astype(int).to_numpy()
        if pd.Series(y_true).nunique() == 2:
            y_score = sub["pred_prob"].astype(float).to_numpy()
            out["auc"] = float(roc_auc_score(y_true, y_score))
    else:  # multiclass or multilabel — macro-AUC over per-class OvR
        if metric == "multiclass":
            from sklearn.preprocessing import label_binarize
            n_classes = len(cfg["classes"])
            y_true_bin = label_binarize(
                sub["true"].astype(int).to_numpy(), classes=list(range(n_classes))
            )
        else:
            y_true_bin = np.array(sub["true_vec"].tolist())
        y_score = np.array(sub["pred_prob_vec"].tolist())
        aucs = []
        for j in range(y_true_bin.shape[1]):
            if len(np.unique(y_true_bin[:, j])) < 2:
                continue
            aucs.append(roc_auc_score(y_true_bin[:, j], y_score[:, j]))
        out["auc"] = float(np.mean(aucs)) if aucs else None
    return out


def cv_aggregate(per_fold: list[dict]) -> dict:
    """Mean ± std across folds for shared numeric keys."""
    if not per_fold:
        return {}
    keys = set(per_fold[0].keys())
    for f in per_fold[1:]:
        keys &= set(f.keys())
    agg = {}
    for k in keys:
        vals = [f[k] for f in per_fold if f.get(k) is not None]
        if vals and all(isinstance(v, (int, float)) for v in vals):
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
    return agg


# ============================================================
# True-label normalization
# ============================================================

def normalize_true(task: str, raw_value) -> object:
    """Convert manifest/CSV ground-truth value to comparable form."""
    cfg = TASKS[task]
    metric = cfg["metric"]
    if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
        return None
    if metric == "multilabel":
        if isinstance(raw_value, str):
            return ast.literal_eval(raw_value)
        return list(raw_value)
    if metric in ("binary", "multiclass"):
        return int(raw_value)
    if metric == "regression":
        return float(raw_value)
    return raw_value


# ============================================================
# Resume / time-budget infrastructure
# ============================================================

import signal

_START_TIME = time.time()
_SHOULD_STOP = False  # set by SIGTERM handler

def _term_handler(sig, frame):
    global _SHOULD_STOP
    _SHOULD_STOP = True
    print(f"[signal] caught {sig}, will stop after current chunk", flush=True)

signal.signal(signal.SIGTERM, _term_handler)
signal.signal(signal.SIGUSR1, _term_handler)


def _seconds_left(walltime_s: float, safety_s: float) -> float:
    return walltime_s - (time.time() - _START_TIME) - safety_s


def _should_stop_now(walltime_s: float, safety_s: float) -> bool:
    return _SHOULD_STOP or _seconds_left(walltime_s, safety_s) <= 0


def _preds_path(out_dir: Path, task: str) -> Path:
    return out_dir / task / "predictions.csv"


def _metrics_path(out_dir: Path, task: str) -> Path:
    return out_dir / task / "predictions.metrics.json"


def _done_rowids(preds_path: Path) -> set:
    if not preds_path.exists():
        return set()
    try:
        df = pd.read_csv(preds_path, usecols=["rowid"])
        return set(df["rowid"].astype(str).tolist())
    except Exception:
        return set()


def _append_rows(preds_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not preds_path.exists()
    df.to_csv(preds_path, mode="a", header=header, index=False)


# ============================================================
# Per-task runner
# ============================================================

def _lp_row(task: str, sample, duration, true_val, response, logprobs,
            pred=None, pred_vec=None, pred_prob=None, pred_prob_vec=None):
    """Unified row-builder. response may be a string or a {label: text} dict."""
    cfg = TASKS[task]
    metric = cfg["metric"]
    if not isinstance(response, str):
        response = json.dumps(response, default=str)
    return {
        "rowid": sample["_rowid"],
        "fold": sample.get("_fold"),
        "duration_sec": duration,
        "true": true_val if metric != "multilabel" else None,
        "true_vec": true_val if metric == "multilabel" else None,
        "pred": pred,
        "pred_vec": pred_vec,
        "pred_prob": pred_prob,
        "pred_prob_vec": pred_prob_vec,
        "response": response,
        "logprobs_json": json.dumps(logprobs, default=str) if logprobs is not None else None,
    }


def _parse_lp_choice(task: str, text: str | None, lp: dict | None):
    """Binary/multiclass: returns (pred_idx, pred_prob_or_vec). Either may
    be None on parse failure.

    Binary: prob is float P(positive class) where positive = classes[1].
    Multiclass: prob is list[float] in TASKS["classes"] order.
    """
    if text is None:
        return None, None
    cfg = TASKS[task]
    classes = cfg["classes"]
    text_probs = parse_choice_response(text, lp, get_answer_map(task))
    if text_probs is None:
        return None, None
    pred = argmax_class(text_probs, classes)
    prob_vec = probs_to_class_vector(text_probs, classes)
    if cfg["metric"] == "binary":
        prob_pos = float(prob_vec[1]) if prob_vec else None
        return pred, prob_pos
    return pred, prob_vec


def _aggregate_multilabel(task: str, sub_parsed: dict[str, dict | None]):
    """Returns (pred_vec, pred_prob_vec) with per-label Nones for failed
    subcalls (so _dns_fill can apply max-wrong against the true label)."""
    label_order = get_labels(task)
    ordered = {lab: sub_parsed.get(lab) for lab in label_order}
    agg = aggregate_multilabel_probs(ordered)
    if agg is None:
        return None, None
    prob_vec_raw, _ = agg
    pred_vec, prob_vec = [], []
    for p in prob_vec_raw:
        if isinstance(p, float) and math.isnan(p):
            pred_vec.append(None); prob_vec.append(None)
        else:
            pred_vec.append(1 if p >= 0.5 else 0); prob_vec.append(float(p))
    return pred_vec, prob_vec


def process_task(
    backend: "QwenOmniVLLM",
    task: str,
    args: argparse.Namespace,
) -> bool:
    """Run inference on samples for `task`, resuming from any existing CSV.

    Returns True if all samples are predicted (task complete), False if we
    stopped early due to time budget / SIGTERM (re-runnable next requeue).
    """
    cfg = TASKS[task]
    kind = get_kind(task)
    prompt_or_subs = build_prompt(task)
    out_dir = Path(args.out_dir).resolve()
    (out_dir / task).mkdir(parents=True, exist_ok=True)
    preds_path = _preds_path(out_dir, task)

    samples, _ = load_samples(task)
    if args.limit:
        samples = samples[: args.limit]

    done = _done_rowids(preds_path)
    todo = [s for s in samples if str(s["_rowid"]) not in done]
    if not todo:
        print(f"[{task}] already complete ({len(samples)} samples)", flush=True)
        return True

    if kind == "multilabel":
        K = len(prompt_or_subs)
        print(f"[{task}] {len(todo)}/{len(samples)} samples (done={len(done)}) "
              f"kind=multilabel K={K} subcalls/sample", flush=True)
    else:
        print(f"[{task}] {len(todo)}/{len(samples)} samples (done={len(done)}) "
              f"kind={kind} prompt-len={len(prompt_or_subs)}", flush=True)

    logprobs_k = None if kind == "regression" else LOGPROBS_K

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"q3o_{task}_"))
    try:
        for chunk_idx, chunk in enumerate(_chunks(todo, args.chunk_size)):
            if _should_stop_now(args.walltime_s, args.safety_s):
                print(f"[{task}] stopping early (time/sig). chunk {chunk_idx}.", flush=True)
                return False

            chunk_clips, chunk_dur, chunk_true = [], [], []
            for s in chunk:
                path = _norm_path(s["path"])
                try:
                    wav = load_and_trim_audio(path, args.max_duration)
                except Exception as e:
                    print(f"  [warn] {task} couldn't load {path}: {e}", flush=True)
                    chunk_clips.append(None); chunk_dur.append(None); chunk_true.append(None)
                    continue
                clip = tmp_dir / f"{s['_rowid']}.wav"
                sf.write(str(clip), wav, 16000, subtype="PCM_16")
                chunk_clips.append(str(clip))
                chunk_dur.append(float(len(wav) / 16000))
                chunk_true.append(normalize_true(task, s.get(cfg["label_col"])))

            valid_idx = [i for i, c in enumerate(chunk_clips) if c is not None]
            if not valid_idx:
                rows = [_lp_row(task, s, chunk_dur[i], chunk_true[i],
                                "<load-failed>", None)
                        for i, s in enumerate(chunk)]
                _append_rows(preds_path, rows)
                continue

            # ---------- Build the vLLM batch ----------
            if kind == "multilabel":
                # Each valid sample becomes K items (one per label subcall).
                labels_list = [lab for lab, _ in prompt_or_subs]
                sub_prompt_list = [p for _, p in prompt_or_subs]
                batch_messages = []
                batch_map = []  # batch_idx -> (sample_idx_in_chunk, label_idx)
                for i in valid_idx:
                    for k_idx, p in enumerate(sub_prompt_list):
                        batch_messages.append(build_messages(chunk_clips[i], p))
                        batch_map.append((i, k_idx))
            else:
                batch_messages = [build_messages(chunk_clips[i], prompt_or_subs)
                                  for i in valid_idx]

            t0 = time.time()
            results = backend.generate_batch(
                batch_messages, max_tokens=args.max_new_tokens,
                logprobs_k=logprobs_k,
            )
            gen_s = time.time() - t0
            print(f"[{task}] chunk {chunk_idx}: {len(batch_messages)} prompts in "
                  f"{gen_s:.1f}s ({gen_s/len(batch_messages):.2f}s/prompt)", flush=True)

            # ---------- Build rows ----------
            rows = []
            if kind == "multilabel":
                # Group results back by sample index.
                per_sample_parsed: dict[int, dict[str, dict | None]] = {i: {} for i in valid_idx}
                per_sample_resp: dict[int, dict[str, str | None]] = {i: {} for i in valid_idx}
                per_sample_lp: dict[int, dict[str, dict | None]] = {i: {} for i in valid_idx}
                for batch_i, (sample_i, k_idx) in enumerate(batch_map):
                    text, lp = results[batch_i]
                    lab = labels_list[k_idx]
                    per_sample_resp[sample_i][lab] = text
                    per_sample_lp[sample_i][lab] = lp
                    per_sample_parsed[sample_i][lab] = (
                        parse_choice_response(text, lp, {"yes": "A", "no": "B"})
                        if text is not None else None
                    )
                for i, s in enumerate(chunk):
                    if chunk_clips[i] is None:
                        rows.append(_lp_row(task, s, chunk_dur[i], chunk_true[i],
                                            "<load-failed>", None))
                        continue
                    pred_vec, prob_vec = _aggregate_multilabel(task, per_sample_parsed[i])
                    rows.append(_lp_row(
                        task, s, chunk_dur[i], chunk_true[i],
                        per_sample_resp[i], per_sample_lp[i],
                        pred_vec=pred_vec, pred_prob_vec=prob_vec,
                    ))
            else:
                result_iter = iter(results)
                for i, s in enumerate(chunk):
                    if chunk_clips[i] is None:
                        rows.append(_lp_row(task, s, chunk_dur[i], chunk_true[i],
                                            "<load-failed>", None))
                        continue
                    text, lp = next(result_iter)
                    if text is None:
                        rows.append(_lp_row(task, s, chunk_dur[i], chunk_true[i],
                                            "<generate-failed>", None))
                        continue
                    if kind == "regression":
                        lo, hi = cfg.get("range", (-1e9, 1e9))
                        pred = parse_integer(text, lo, hi)
                        rows.append(_lp_row(task, s, chunk_dur[i], chunk_true[i],
                                            text, None, pred=pred))
                    elif kind == "binary":
                        pred, prob_pos = _parse_lp_choice(task, text, lp)
                        rows.append(_lp_row(task, s, chunk_dur[i], chunk_true[i],
                                            text, lp, pred=pred, pred_prob=prob_pos))
                    else:  # multiclass
                        pred, prob_vec = _parse_lp_choice(task, text, lp)
                        rows.append(_lp_row(task, s, chunk_dur[i], chunk_true[i],
                                            text, lp, pred=pred, pred_prob_vec=prob_vec))
            _append_rows(preds_path, rows)

            # Free chunk's clip files (predictions are saved).
            for i in valid_idx:
                Path(chunk_clips[i]).unlink(missing_ok=True)
    finally:
        try: tmp_dir.rmdir()
        except OSError: pass

    # All samples now predicted — compute & save metrics. Wrap in try/except
    # so a BLAS/sklearn glitch doesn't abort the whole loop; metrics can be
    # recomputed offline from the CSV at any time via summarize_qwen3omni.py.
    try:
        df = pd.read_csv(preds_path)
        for col in ("true_vec", "pred_vec", "pred_prob_vec"):
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x
                )
        metrics = {"overall": compute_metrics(task, df)}
        if df["fold"].notna().any():
            per_fold = []
            for fi in sorted(df["fold"].dropna().unique()):
                sub = df[df["fold"] == fi]
                m = compute_metrics(task, sub)
                m["fold"] = int(fi)
                per_fold.append(m)
            metrics["per_fold"] = per_fold
            metrics["fold_aggregate"] = cv_aggregate(per_fold)

        _metrics_path(out_dir, task).write_text(json.dumps(metrics, indent=2, default=str))
        print(f"[{task}] complete. metrics -> {_metrics_path(out_dir, task)}", flush=True)
        print(json.dumps(metrics, indent=2, default=str), flush=True)
    except Exception as e:
        print(f"[{task}] metrics computation FAILED ({type(e).__name__}: {e}); "
              f"predictions saved, recompute later.", flush=True)
        # Drop a marker so is_task_complete still considers this done (we have
        # all predictions). Metrics can be regenerated from the CSV.
        _metrics_path(out_dir, task).write_text(json.dumps(
            {"overall": {"n_total": int(len(df)) if 'df' in locals() else None,
                         "_note": f"metric calc failed: {e!r}"}}, indent=2))
    return True


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]


def is_task_complete(task: str, out_dir: Path, limit: int | None = None) -> bool:
    """Predictions CSV covers all (limited) samples AND metrics.json exists."""
    if not _metrics_path(out_dir, task).exists():
        return False
    preds = _preds_path(out_dir, task)
    if not preds.exists():
        return False
    try:
        samples, _ = load_samples(task)
        if limit: samples = samples[:limit]
        df = pd.read_csv(preds, usecols=["rowid"])
        return len(set(df["rowid"].astype(str))) >= len(samples)
    except Exception:
        return False


# ============================================================
# Entrypoints
# ============================================================

DEFAULT_TASK_ORDER = [f"T{i}" for i in range(1, 28)]


def run_all(args: argparse.Namespace) -> int:
    """Loop over tasks, model loaded once. Returns 0 if all done, 2 if more to do."""
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = args.tasks.split(",") if args.tasks else DEFAULT_TASK_ORDER
    tasks = [t.strip().upper() for t in tasks if t.strip()]

    # Skip any tasks already complete (avoid model load if nothing to do)
    pending = [t for t in tasks if not is_task_complete(t, out_dir, args.limit)]
    if not pending:
        print("[all] every task already complete, nothing to do.", flush=True)
        return 0
    print(f"[all] pending tasks ({len(pending)}/{len(tasks)}): {pending}", flush=True)

    # Single model load — reused across tasks
    print("[all] loading vLLM backend …", flush=True)
    backend = QwenOmniVLLM(
        model_path=args.model, max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util, max_num_seqs=args.max_num_seqs,
    )
    print("[all] backend ready.", flush=True)

    incomplete = []
    for task in pending:
        if _should_stop_now(args.walltime_s, args.safety_s):
            print("[all] time budget exhausted, exiting before next task.", flush=True)
            incomplete = pending[pending.index(task):]
            break
        try:
            done = process_task(backend, task, args)
            if not done:
                incomplete.append(task)
        except Exception as e:
            print(f"[all] {task} FAILED with {type(e).__name__}: {e}", flush=True)
            incomplete.append(task)

    if incomplete:
        print(f"[all] incomplete: {incomplete}", flush=True)
        return 2
    print("[all] every task complete.", flush=True)
    return 0


def run_one(args: argparse.Namespace) -> int:
    """Single-task entrypoint (legacy --task flag)."""
    args.tasks = args.task
    return run_all(args)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # Task selection
    g = p.add_mutually_exclusive_group()
    g.add_argument("--task", help="Single task ID like T3, T18, ...")
    g.add_argument("--tasks", help="Comma-separated task IDs, or omit/--all to sweep all")
    p.add_argument("--all", action="store_true", help="Sweep all 27 tasks (default)")
    # I/O
    p.add_argument("--out-dir", default="exps/qwen3omni_all",
                   help="Outputs go to <out-dir>/<task>/")
    # Model
    p.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Thinking")
    p.add_argument("--max-duration", type=float, default=600.0)
    p.add_argument("--limit", type=int, default=None,
                   help="Per-task sample cap (debug)")
    p.add_argument("--max-num-seqs", type=int, default=16)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-mem-util", type=float, default=0.92)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    # Resume + budget
    p.add_argument("--chunk-size", type=int, default=64,
                   help="Samples per vLLM batch (checkpoint granularity)")
    p.add_argument("--walltime-s", type=float,
                   default=float(os.environ.get("WALLTIME_S", "6900")),
                   help="Total job wallclock budget in seconds (resume on overrun)")
    p.add_argument("--safety-s", type=float, default=240.0,
                   help="Exit cleanly when this many seconds left in budget")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if args.task and not args.tasks:
        rc = run_one(args)
    else:
        rc = run_all(args)
    raise SystemExit(rc)
