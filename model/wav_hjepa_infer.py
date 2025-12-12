"""
Barebone inference module for JEPA model.
Just loads checkpoint and provides forward pass.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Union
import sys
from model.models.WavHJepa.jepa import JEPA

# Handle legacy checkpoint compatibility: remap old 'models' module to 'model.models'
import model.models
sys.modules['models'] = model.models

class WavHJepaInfer(nn.Module):
    """
    Simple inference wrapper for JEPA model.

    Usage:
        model = WavHJepa("path/to/checkpoint.ckpt")
        embeddings = model(waveform)  # (B, C, T) -> (B, n_patches, D)
    """

    def __init__(self, model_ckpt: Union[str, Path], freeze_encoder: bool = True, output_hidden_states: bool = False):
        """
        Load JEPA model from checkpoint.

        Args:
            model_ckpt: Path to Lightning checkpoint (.ckpt file)
            freeze_encoder: Whether to freeze encoder parameters
            output_hidden_states: Whether to return all layer outputs (not supported, must be False)
        """
        super().__init__()

        model_ckpt = Path(model_ckpt)
        if not model_ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_ckpt}")

        # Load checkpoint
        checkpoint = torch.load(model_ckpt, map_location="cpu", weights_only=False)

        if "hyper_parameters" not in checkpoint:
            raise ValueError("Checkpoint missing hyperparameters")

        # WavHJepa does not support output_hidden_states
        if output_hidden_states:
            raise NotImplementedError("WavHJepa does not support output_hidden_states=True")
        self.output_hidden_states = False

        # Instantiate model from saved hyperparameters
        self.model = JEPA(**checkpoint["hyper_parameters"])

        # Load weights
        self.model.load_state_dict(checkpoint["state_dict"])

        # Set to eval mode
        if freeze_encoder:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, waveform: torch.Tensor, lengths=None) -> torch.Tensor:
        """
        Extract audio embeddings.

        Args:
            waveform: Audio tensor of shape (B, C, T) or (B, T)
                     where B=batch, C=channels, T=time samples
            lengths: Optional tensor of shape (B,) indicating the relative lengths of each sample

        Returns:
            Embeddings of shape (B, n_patches, embedding_dim)
        """
        # Handle (B, T) -> (B, C, T)
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)

        # Crop or pad to match target length
        target_length = self.model.target_length
        current_length = waveform.shape[-1]

        if current_length < target_length:
            # Pad on the right (time dimension)
            pad_amount = target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount), mode='constant', value=0)

            # Adjust relative lengths proportionally
            if lengths is not None:
                # lengths are relative, so scale them: new_relative = (old_relative * old_T) / new_T
                lengths = lengths * (current_length / target_length)
        elif current_length > target_length:
            # Crop to target length (take first target_length samples)
            waveform = waveform[..., :target_length]

            # Adjust relative lengths - since we're cropping, effective length is capped at 1.0
            if lengths is not None:
                # Scale down: if original was 1.0 at current_length, it's now 1.0 at target_length
                # But some samples might have been shorter, so we need to scale
                lengths = torch.clamp(lengths * (current_length / target_length), max=1.0)

        waveform = self.normalize(waveform)

        # Use model's inference method
        return self.model.get_audio_representation(waveform, lengths=lengths)

    def normalize(self, audio):
        mean = audio.mean(dim=(-2, -1), keepdim=True)
        std = audio.std(dim=(-2, -1), keepdim=True)
        audio = (audio - mean) / (std + 1e-5)  # Add epsilon for stability
        return audio

# if __name__ == "__main__":
#     # Simple test
#     model = WavHJepa("step-step=110000.ckpt")
#     length = torch.Tensor([0.8, 1])  # 10 seconds at 16kHz
#
#     # Test with random audio (10 seconds at 16kHz as expected by the model)
#     target_length = model.model.target_length - 1000  # Test with shorter length to trigger padding
#     waveform = torch.randn(2, target_length)
#
#     with torch.no_grad():
#         embeddings = model(waveform, length)
#
#     print(f"Input shape: {waveform.shape}")
#     print(f"Output shape: {embeddings.shape}")
#     print(f"Embedding dim: {model.model.encoder_embedding_dim}")
