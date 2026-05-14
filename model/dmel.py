import torch
import torch.nn as nn

from dmel import LogMelFbank


class DMEL(nn.Module):
    """dMEL log-mel encoder with optional uniform quantization (dequantized to float).

    When use_discrete=True (default) the log-mel spectrogram is quantized to
    n_bits levels and then mapped back to float bin-centre values, matching the
    information content seen by the dMEL speech tokenizer.  When False, raw
    log-mel features are returned.

    Paper: https://arxiv.org/pdf/2407.15835
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_filterbank: int = 80,
        n_bits: int = 4,
        n_fft: int = 1024,
        frame_size_ms: float = 50.0,
        frame_stride_ms: float = 12.5,
        quantize_min_value: float = -7.0,
        quantize_max_value: float = 2.0,
        use_discrete: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.output_hidden_states = False
        self.sample_rate = sample_rate
        self.n_bits = n_bits
        self.use_discrete = use_discrete
        self.q_min = float(quantize_min_value)
        self.q_max = float(quantize_max_value)

        self.logmelfbank = LogMelFbank(
            sampling_freq=sample_rate,
            n_fft=n_fft,
            frame_size_ms=frame_size_ms,
            frame_stride_ms=frame_stride_ms,
            n_filterbank=n_filterbank,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        """
        Args:
            x: (B, T) raw waveform
            lengths: (B,) relative lengths in [0, 1] — unused by mel (no masking
                     needed; downstream pooling handles variable lengths)
        Returns:
            (B, T', n_filterbank) float features
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)

        features, _ = self.logmelfbank(x.float(), None)  # (B, T', F)

        if self.use_discrete:
            n_levels = 2**self.n_bits - 1
            step = (self.q_max - self.q_min) / n_levels
            clamped = features.clamp(self.q_min, self.q_max)
            tokens = ((clamped - self.q_min) / step).round().long()
            features = self.q_min + tokens.float() * step

        return features  # (B, T', n_filterbank)