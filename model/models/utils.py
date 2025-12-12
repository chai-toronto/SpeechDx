"""
Utility functions for model operations.

Includes multiview batch creation for JEPA-style training.
"""

import torch
from einops import repeat, rearrange


def create_multiview_batch(
    features: torch.Tensor,
    nr_samples_per_audio: int,
) -> torch.Tensor:
    """
    Create multi-view batch from encoded features.

    For each feature tensor in the batch, creates nr_samples_per_audio random views
    by duplicating and shuffling across the batch dimension.

    This is used in JEPA-style training where we want multiple views of the same
    encoded representation to train with different masks.

    Args:
        features: (B, T, D) - encoded features from feature extractor
        nr_samples_per_audio: Number of views to generate per sample

    Returns:
        features_multiview: (B * nr_samples, T, D) - features replicated and shuffled

    Example:
        >>> features = torch.randn(4, 100, 768)  # B=4, T=100, D=768
        >>> multiview = create_multiview_batch(features, nr_samples_per_audio=16)
        >>> multiview.shape
        torch.Size([64, 100, 768])  # 4 * 16 = 64
    """

    # Replicate each sample nr_samples_per_audio times
    # Shape: (B, nr_samples, T, D)
    features_expanded = repeat(
        features,
        "B T D -> B N T D",
        N=nr_samples_per_audio
    )

    # Flatten batch and views: (B, nr_samples, T, D) -> (B * nr_samples, T, D)
    features_flattened = rearrange(
        features_expanded,
        "B N T D -> (B N) T D"
    )

    # Shuffle across the batch dimension
    # idx = torch.randperm(features_flattened.size(0), device=features.device)
    # features_shuffled = features_flattened[idx]
    #
    # return features_shuffled

    return features_flattened
