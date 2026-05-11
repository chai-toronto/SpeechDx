"""Zero-shot evaluation of Qwen3-Omni-Thinking on the two EDAIC tasks.

Tasks
-----
1. edaic_depC : binary depression / healthy classification (target: PHQ_Binary)
2. edaic_phqR : PHQ-8 score regression in [0, 24]            (target: PHQ_Score)

Audio inputs are the participant-only WAV files referenced in
``data/edaic/processed/edaic.csv``. Long interviews are trimmed to
``--max-duration`` seconds to stay within the model's audio context.

The cookbook this is patterned after:
    https://github.com/QwenLM/Qwen3-Omni/blob/main/cookbooks/sound_analysis.ipynb

Default backend is vLLM (matches the cookbook); pass ``--use-transformers`` to
fall back to the HF generate path.

Example
-------
    python scripts/test_qwen3omni_edaic.py \\
        --csv data/edaic/processed/edaic.csv \\
        --audio-root data/edaic/processed/audio \\
        --out exps/qwen3omni_edaic/predictions.csv \\
        --max-duration 600
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


# -------- I/O helpers --------------------------------------------------------

def load_test_rows(csv_path: Path, audio_root: Path, split: str, limit: int | None):
    df = pd.read_csv(csv_path)
    split_map = {"train": 0, "val": 1, "test": 2}
    df = df[df["split"] == split_map[split]].reset_index(drop=True)
    if limit:
        df = df.head(limit).reset_index(drop=True)
    df["abs_path"] = df["path"].map(lambda p: str((audio_root / p).resolve()))
    return df


def load_and_trim_audio(path: str, max_duration: float, target_sr: int = 16000):
    """Load mono audio at target_sr and trim to max_duration seconds (0 disables)."""
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


# -------- response parsing ---------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by the Thinking model."""
    return _THINK_RE.sub("", text).strip()


def parse_yes_no(text: str) -> int | None:
    cleaned = strip_thinking(text)
    m = re.search(r"ANSWER\s*:\s*(yes|no)", cleaned, re.IGNORECASE)
    if m:
        return 1 if m.group(1).lower() == "yes" else 0
    # fallback: last yes/no token
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


# -------- model backends -----------------------------------------------------

class QwenOmniBackend:
    """Thin wrapper exposing .generate(messages) -> str over vLLM or HF."""

    def __init__(self, model_path: str, use_transformers: bool, flash_attn2: bool):
        self.model_path = model_path
        self.use_transformers = use_transformers
        self._setup_env()

        from transformers import Qwen3OmniMoeProcessor
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)

        if use_transformers:
            import torch
            from transformers import Qwen3OmniMoeForConditionalGeneration
            kwargs = dict(device_map="auto", dtype="auto")
            if flash_attn2:
                kwargs["attn_implementation"] = "flash_attention_2"
            self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                model_path, **kwargs
            )
            self.torch = torch
        else:
            import torch
            from vllm import LLM
            self.model = LLM(
                model=model_path,
                trust_remote_code=True,
                gpu_memory_utilization=0.95,
                tensor_parallel_size=torch.cuda.device_count() or 1,
                limit_mm_per_prompt={"image": 1, "video": 3, "audio": 3},
                max_num_seqs=1,
                max_model_len=32768,
                seed=1234,
            )

    @staticmethod
    def _setup_env():
        os.environ.setdefault("VLLM_USE_V1", "0")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

    def generate(self, messages: list) -> str:
        from qwen_omni_utils import process_mm_info
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)

        if self.use_transformers:
            inputs = self.processor(
                text=text, audio=audios, images=images, videos=videos,
                return_tensors="pt", padding=True, use_audio_in_video=False,
            ).to(self.model.device).to(self.model.dtype)
            text_ids, _ = self.model.generate(
                **inputs,
                thinker_return_dict_in_generate=True,
                thinker_max_new_tokens=8192,
                thinker_do_sample=False,
                use_audio_in_video=False,
                return_audio=False,
            )
            new_ids = text_ids.sequences[:, inputs["input_ids"].shape[1]:]
            return self.processor.batch_decode(
                new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=1e-2, top_p=0.1, top_k=1, max_tokens=8192
        )
        inputs = {"prompt": text, "multi_modal_data": {},
                  "mm_processor_kwargs": {"use_audio_in_video": False}}
        if images is not None:
            inputs["multi_modal_data"]["image"] = images
        if videos is not None:
            inputs["multi_modal_data"]["video"] = videos
        if audios is not None:
            inputs["multi_modal_data"]["audio"] = audios
        outputs = self.model.generate(inputs, sampling_params=sampling_params)
        return outputs[0].outputs[0].text


# -------- evaluation loop ----------------------------------------------------

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
    print(f"[edaic] split={args.split}  n={len(rows)}  model={args.model}")

    backend = QwenOmniBackend(
        model_path=args.model,
        use_transformers=args.use_transformers,
        flash_attn2=args.flash_attn2,
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix="edaic_qwen_"))
    records = []
    t0 = time.time()

    for i, row in rows.iterrows():
        wav = load_and_trim_audio(row["abs_path"], args.max_duration)
        clip_path = tmp_dir / f"{row['uid']}.wav"
        sf.write(str(clip_path), wav, 16000, subtype="PCM_16")

        # Task 1: binary depression
        dep_resp = backend.generate(build_messages(str(clip_path), DEP_PROMPT))
        dep_pred = parse_yes_no(dep_resp)

        # Task 2: PHQ-8 regression
        phq_resp = backend.generate(build_messages(str(clip_path), PHQ_PROMPT))
        phq_pred = parse_phq_score(phq_resp)

        records.append({
            "uid": row["uid"],
            "Participant_ID": row["Participant_ID"],
            "duration_sec": float(len(wav) / 16000),
            "PHQ_Binary_true": int(row["PHQ_Binary"]),
            "PHQ_Score_true": float(row["PHQ_Score"]),
            "dep_pred": dep_pred,
            "dep_response": dep_resp,
            "phq_pred": phq_pred,
            "phq_response": phq_resp,
        })

        elapsed = time.time() - t0
        print(f"  [{i+1:>3}/{len(rows)}] uid={row['uid']} "
              f"dep_true={int(row['PHQ_Binary'])} dep_pred={dep_pred} "
              f"phq_true={float(row['PHQ_Score']):.0f} phq_pred={phq_pred} "
              f"({elapsed:.0f}s)")

        clip_path.unlink(missing_ok=True)
        if (i + 1) % 5 == 0 or i + 1 == len(rows):
            pd.DataFrame(records).to_csv(out_path, index=False)

    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)
    print(f"\n[edaic] wrote {len(df)} rows -> {out_path}")

    metrics = compute_metrics(df)
    metrics_path = out_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[edaic] metrics -> {metrics_path}")
    print(json.dumps(metrics, indent=2))


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


# -------- entry --------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="data/edaic/processed/edaic.csv")
    p.add_argument("--audio-root", default="data/edaic/processed/audio")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--out", default="exps/qwen3omni_edaic/predictions.csv")
    p.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Thinking",
                   help="HF model id; defaults to the Thinking variant")
    p.add_argument("--max-duration", type=float, default=600.0,
                   help="Trim audio to this many seconds (0 = no trim)")
    p.add_argument("--limit", type=int, default=None,
                   help="Only run on the first N rows (smoke test)")
    p.add_argument("--use-transformers", action="store_true",
                   help="Use HF generate path instead of vLLM")
    p.add_argument("--flash-attn2", action="store_true",
                   help="Enable flash_attention_2 in the transformers backend")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())