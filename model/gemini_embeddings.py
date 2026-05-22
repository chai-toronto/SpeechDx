"""Gemini Embeddings 2 as a SpeechDx encoder.

The API caps a single request at 180s of audio (empirically true for the
*total* audio across parts, not per-part), so the yaml sets
``max_length: 180`` and the pipeline splitter (``make_split_signal``) hands
the forward a single chunk of <=180s. The cache writer (sdx/warm.py)
loops over chunks and concatenates the resulting per-chunk embeddings
along the time dim, so a long sample ends up with one row per 180s of
audio in the HDF5 cache.

Audio is encoded to in-memory WAV bytes (no temp files). A process-wide
token bucket throttles concurrent calls to ~3K RPM, and transient API
errors retry with exponential backoff. ``is_api_encoder = True`` lets
the warm runner drive this encoder with parallel forwards.
"""
from __future__ import annotations

import io
import os
import random
import subprocess
import sys
import threading
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn


_DEFAULT_RPM = 3000
_DEFAULT_MODEL = "gemini-embedding-2"
_MAX_SEGMENT_SEC = 180.0


def _get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["security", "find-generic-password", "-s", "gemini-api-key", "-w"],
                capture_output=True, text=True, check=True,
            )
            key = out.stdout.strip()
            if key:
                return key
        except Exception:
            pass
    raise RuntimeError(
        "GeminiEmbeddings: no API key (set GEMINI_API_KEY or store under "
        "macOS keychain service 'gemini-api-key')"
    )


class _TokenBucket:
    """Process-wide token bucket. ``rpm`` requests per minute, with a burst
    equal to one second's worth of tokens. Thread-safe; shared across all
    GeminiEmbeddings instances using the same model id."""

    _instances: dict[tuple[str, int], "_TokenBucket"] = {}
    _instances_lock = threading.Lock()

    @classmethod
    def get(cls, key: str, rpm: int) -> "_TokenBucket":
        with cls._instances_lock:
            inst = cls._instances.get((key, rpm))
            if inst is None:
                inst = cls(rpm)
                cls._instances[(key, rpm)] = inst
            return inst

    def __init__(self, rpm: int):
        self.rate_per_sec = rpm / 60.0
        self.capacity = max(1.0, self.rate_per_sec)
        self.tokens = self.capacity
        self.last = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self, n: float = 1.0) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(
                    self.capacity,
                    self.tokens + (now - self.last) * self.rate_per_sec,
                )
                self.last = now
                if self.tokens >= n:
                    self.tokens -= n
                    return
                wait = (n - self.tokens) / self.rate_per_sec
            time.sleep(wait)


_RETRYABLE_MARKERS = (
    "429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "INTERNAL",
    "504", "DEADLINE_EXCEEDED", "ConnectError", "TimeoutError", "ReadError",
    "RemoteProtocolError",
)


def _is_retryable(exc: BaseException) -> bool:
    msg = f"{type(exc).__name__}: {exc!s}"
    return any(m in msg for m in _RETRYABLE_MARKERS)


class GeminiEmbeddings(nn.Module):
    """Gemini Embeddings 2 wrapped as an SSL-style audio encoder.

    Args:
        model: Gemini embedding model id.
        sample_rate: input waveform rate.
        output_dim: Matryoshka truncation -- 768 / 1536 / 3072.
        rpm: requests per minute the rate limiter targets.
        max_retries: per-request retry budget for transient errors.
        freeze_encoder / output_hidden_states: kept for interface compatibility.

    Forward:
        x: ``(B, T)`` waveform at ``sample_rate``; T must correspond to <=180s.
        lengths: ``(B,)`` relative lengths in [0, 1] (SpeechBrain convention).
        returns: ``(B, 1, output_dim)``.
    """

    is_api_encoder = True

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        sample_rate: int = 16000,
        output_dim: int = 3072,
        rpm: int = _DEFAULT_RPM,
        max_retries: int = 6,
        freeze_encoder: bool = True,
        output_hidden_states: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if output_hidden_states:
            raise ValueError("GeminiEmbeddings does not expose hidden_states")
        if output_dim not in (768, 1536, 3072):
            raise ValueError(f"output_dim must be 768/1536/3072 (got {output_dim})")

        self.model_name = model
        self.sample_rate = int(sample_rate)
        self.output_dim = int(output_dim)
        self.max_retries = int(max_retries)
        self.freeze_encoder = bool(freeze_encoder)
        self.output_hidden_states = False

        from google import genai
        self._client = genai.Client(api_key=_get_api_key())
        self._bucket = _TokenBucket.get(model, rpm)

        # No-op buffer so ``next(self.parameters()/buffers()).device`` resolves
        # for the warm cache writer's device inference (sdx/warm.py:190-193).
        self.register_buffer("_device_anchor", torch.zeros(1), persistent=False)

    def _wav_bytes(self, audio: np.ndarray) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, audio, self.sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def _embed_one(self, audio: np.ndarray) -> np.ndarray:
        from google.genai import types

        # The yaml is supposed to keep us at <=180s; clip defensively if a
        # caller passes a longer slice (e.g. probe runs).
        max_samples = int(_MAX_SEGMENT_SEC * self.sample_rate)
        if audio.shape[0] > max_samples:
            audio = audio[:max_samples]

        part = types.Part.from_bytes(
            data=self._wav_bytes(audio), mime_type="audio/wav",
        )
        config = types.EmbedContentConfig(output_dimensionality=self.output_dim)

        for attempt in range(self.max_retries + 1):
            self._bucket.acquire()
            try:
                resp = self._client.models.embed_content(
                    model=self.model_name, contents=[part], config=config,
                )
                return np.asarray(resp.embeddings[0].values, dtype=np.float32)
            except Exception as e:  # noqa: BLE001
                if not _is_retryable(e) or attempt == self.max_retries:
                    raise
                delay = min(32.0, (2 ** attempt) + random.uniform(0, 1.0))
                time.sleep(delay)
        raise RuntimeError("unreachable")

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B, T = x.shape

        if lengths is None:
            abs_lengths = [T] * B
        else:
            abs_lengths = (lengths.float() * T).round().clamp(min=1).long().tolist()

        wave_np = x.detach().cpu().numpy().astype(np.float32, copy=False)
        embs = [self._embed_one(wave_np[i, : abs_lengths[i]]) for i in range(B)]
        out = torch.from_numpy(np.stack(embs, axis=0)).unsqueeze(1)  # (B, 1, D)
        return out.to(device=x.device, dtype=torch.float32)


if __name__ == "__main__":
    enc = GeminiEmbeddings()
    sr = enc.sample_rate
    for tag, secs in (("5s", 5), ("60s", 60), ("180s", 180)):
        out = enc(torch.randn(1, sr * secs))
        print(f"{tag}: shape={tuple(out.shape)}  norm={out.norm().item():.3f}")
