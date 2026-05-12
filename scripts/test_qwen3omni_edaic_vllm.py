"""Batched zero-shot evaluation of Qwen3-Omni-Thinking on EDAIC via vLLM.

Same two tasks as ``test_qwen3omni_edaic.py``:
  1. edaic_depC : binary depression / healthy classification (target: PHQ_Binary)
  2. edaic_phqR : PHQ-8 score regression in [0, 24]            (target: PHQ_Score)

Key difference from the transformers version: all (audio, prompt) pairs are
built upfront and submitted to ``LLM.generate`` in a single call so vLLM's
continuous batching keeps the GPU saturated.

Reasoning traces (``<think>...</think>``) remain in ``dep_response`` /
``phq_response`` columns for prompt-engineering inspection.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


# -------- I/O ----------------------------------------------------------------

def load_test_rows(csv_path: Path, audio_root: Path, split: str, limit: int | None):
    df = pd.read_csv(csv_path)
    split_map = {"train": 0, "val": 1, "test": 2}
    df = df[df["split"] == split_map[split]].reset_index(drop=True)
    if limit:
        df = df.head(limit).reset_index(drop=True)
    df["abs_path"] = df["path"].map(lambda p: str((audio_root / p).resolve()))
    return df


def load_and_trim_audio(path: str, max_duration: float, target_sr: int = 16000):
    import librosa
    wav, _ = librosa.load(path, sr=target_sr, mono=True)
    if max_duration and len(wav) > int(max_duration * target_sr):
        wav = wav[: int(max_duration * target_sr)]
    return wav.astype(np.float32)


# -------- prompts ------------------------------------------------------------

DEP_PROMPT = (
    "You are a clinical screening assistant. The audio is a participant from "
    "a depression-screening interview (E-DAIC). Based on vocal and prosodic "
    "cues (tempo, pitch, energy, fluency, affect), decide whether this "
    "participant most likely meets criteria for clinical depression.\n\n"
    "Reply on the final line in exactly this format:\n"
    "ANSWER: yes  or  ANSWER: no"
)

PHQ_PROMPT = (
    "You are a clinical screening assistant. The audio is a participant from "
    "a depression-screening interview (E-DAIC). Estimate the participant's "
    "PHQ-8 total score (integer between 0 and 24, where higher means more "
    "severe depressive symptoms).\n\n"
    "Reply on the final line in exactly this format:\n"
    "ANSWER: <integer 0-24>"
)


# -------- parsing ------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def parse_yes_no(text: str) -> int | None:
    cleaned = strip_thinking(text)
    m = re.search(r"ANSWER\s*:\s*(yes|no)", cleaned, re.IGNORECASE)
    if m:
        return 1 if m.group(1).lower() == "yes" else 0
    tokens = re.findall(r"\b(yes|no)\b", cleaned, re.IGNORECASE)
    if tokens:
        return 1 if tokens[-1].lower() == "yes" else 0
    return None


def parse_phq_score(text: str) -> float | None:
    cleaned = strip_thinking(text)
    m = re.search(r"ANSWER\s*:\s*(-?\d+(?:\.\d+)?)", cleaned)
    if m is None:
        nums = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
        if not nums:
            return None
        val = float(nums[-1])
    else:
        val = float(m.group(1))
    return float(np.clip(val, 0.0, 24.0))


# -------- backend ------------------------------------------------------------

class QwenOmniVLLM:
    def __init__(self, model_path: str, max_model_len: int, gpu_memory_utilization: float,
                 max_num_seqs: int):
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

    def generate_batch(self, batch_messages: list, max_tokens: int) -> list:
        from qwen_omni_utils import process_mm_info
        from vllm import SamplingParams

        sp = SamplingParams(temperature=1e-2, top_p=0.1, top_k=1, max_tokens=max_tokens)
        items = []
        for messages in batch_messages:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            audios, _, _ = process_mm_info(messages, use_audio_in_video=False)
            item = {"prompt": text, "multi_modal_data": {},
                    "mm_processor_kwargs": {"use_audio_in_video": False}}
            if audios is not None:
                item["multi_modal_data"]["audio"] = audios
            items.append(item)
        outputs = self.model.generate(items, sampling_params=sp)
        return [o.outputs[0].text for o in outputs]


# -------- run loop -----------------------------------------------------------

def build_messages(audio_path: str, prompt: str) -> list:
    return [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text", "text": prompt},
        ],
    }]


def run(args: argparse.Namespace) -> None:
    csv_path = Path(args.csv).resolve()
    audio_root = Path(args.audio_root).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_test_rows(csv_path, audio_root, args.split, args.limit)
    print(f"[edaic] split={args.split}  n={len(rows)}  model={args.model}", flush=True)

    # Trim + write all clips up front so vLLM can batch them in one call.
    tmp_dir = Path(tempfile.mkdtemp(prefix="edaic_qwen_vllm_"))
    clip_paths, durations = [], []
    for _, row in rows.iterrows():
        wav = load_and_trim_audio(row["abs_path"], args.max_duration)
        clip = tmp_dir / f"{row['uid']}.wav"
        sf.write(str(clip), wav, 16000, subtype="PCM_16")
        clip_paths.append(str(clip))
        durations.append(float(len(wav) / 16000))
    print(f"[edaic] staged {len(clip_paths)} clips to {tmp_dir}", flush=True)

    tasks = [("dep", DEP_PROMPT), ("phq", PHQ_PROMPT)]
    batch_messages = []
    index = []  # (row_idx, task_name)
    for i, clip in enumerate(clip_paths):
        for task, prompt in tasks:
            batch_messages.append(build_messages(clip, prompt))
            index.append((i, task))
    print(f"[edaic] submitting {len(batch_messages)} prompts to vLLM "
          f"(rows={len(rows)} × tasks={len(tasks)})", flush=True)

    backend = QwenOmniVLLM(
        model_path=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        max_num_seqs=args.max_num_seqs,
    )

    t0 = time.time()
    responses = backend.generate_batch(batch_messages, max_tokens=args.max_new_tokens)
    print(f"[edaic] vLLM generate done: {len(responses)} responses "
          f"in {time.time() - t0:.1f}s", flush=True)

    # Reassemble per-row records.
    by_idx_task = {(i, t): r for (i, t), r in zip(index, responses)}
    records = []
    for i, row in rows.iterrows():
        dep_resp = by_idx_task[(i, "dep")]
        phq_resp = by_idx_task[(i, "phq")]
        records.append({
            "uid": row["uid"],
            "Participant_ID": row["Participant_ID"],
            "duration_sec": durations[i],
            "PHQ_Binary_true": int(row["PHQ_Binary"]),
            "PHQ_Score_true": float(row["PHQ_Score"]),
            "dep_pred": parse_yes_no(dep_resp),
            "dep_response": dep_resp,
            "phq_pred": parse_phq_score(phq_resp),
            "phq_response": phq_resp,
        })

    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)
    print(f"[edaic] wrote {len(df)} rows -> {out_path}", flush=True)

    metrics = compute_metrics(df)
    metrics_path = out_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[edaic] metrics -> {metrics_path}", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)

    # Clean up clip dir.
    for clip in clip_paths:
        Path(clip).unlink(missing_ok=True)
    tmp_dir.rmdir()


def compute_metrics(df: pd.DataFrame) -> dict:
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score,
        mean_absolute_error, mean_squared_error,
    )

    metrics: dict = {"n_total": int(len(df))}

    dep = df.dropna(subset=["dep_pred"])
    metrics["depC"] = {
        "n_parsed": int(len(dep)),
        "n_unparsed": int(len(df) - len(dep)),
    }
    if len(dep) >= 2 and dep["PHQ_Binary_true"].nunique() == 2:
        y_true = dep["PHQ_Binary_true"].astype(int).to_numpy()
        y_pred = dep["dep_pred"].astype(int).to_numpy()
        metrics["depC"].update({
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred)),
            "auc_hard": float(roc_auc_score(y_true, y_pred)),
        })

    phq = df.dropna(subset=["phq_pred"])
    metrics["phqR"] = {
        "n_parsed": int(len(phq)),
        "n_unparsed": int(len(df) - len(phq)),
    }
    if len(phq) >= 2:
        y_true = phq["PHQ_Score_true"].astype(float).to_numpy()
        y_pred = phq["phq_pred"].astype(float).to_numpy()
        metrics["phqR"].update({
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "pearson": float(np.corrcoef(y_true, y_pred)[0, 1])
            if np.std(y_pred) > 0 else None,
        })

    return metrics


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="data/edaic/processed/edaic.csv")
    p.add_argument("--audio-root", default="data/edaic/processed/audio")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--out", default="exps/qwen3omni_edaic_vllm/predictions.csv")
    p.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Thinking")
    p.add_argument("--max-duration", type=float, default=600.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-num-seqs", type=int, default=16,
                   help="vLLM concurrent sequences (was 1 in the original script)")
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-mem-util", type=float, default=0.92)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
