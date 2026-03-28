"""
AudioMAE++ (Audio Masked Autoencoder Plus Plus) wrapper.

AudioMAE++ improves upon AudioMAE with SwiGLU FFNs and other enhancements.

Paper: "AudioMAE++: learning better masked audio representations with SwiGLU FFNs"
       IEEE Workshop on Machine Learning for Signal Processing (MLSP) 2025
GitHub: https://github.com/SarthakYadav/audiomae-plusplus-official

This implementation uses the original AudioMAE++ code from third_party/audiomae-plusplus-official.
Clone the repo first: git clone https://github.com/SarthakYadav/audiomae-plusplus-official third_party/audiomae-plusplus-official

Input: Raw audio at 16kHz
Output dimensions depend on model variant:
- tiny: 192
- base: 768
- large: 1024
"""

import os
import sys
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T

# Add AudioMAE++ repo to path for imports
AUDIOMAEPP_PATH = os.path.join(os.path.dirname(__file__), '..', 'third_party', 'audiomae-plusplus-official')
if AUDIOMAEPP_PATH not in sys.path:
    sys.path.insert(0, AUDIOMAEPP_PATH)

# Import from AudioMAE++ repo (requires third_party/audiomae-plusplus-official to be cloned)
try:
    from src.models.mae_pp import (
        mae_plusplus_tiny,
        mae_plusplus_base,
        mae_plusplus_large,
        MAE_PlusPlus
    )
    from src.data.features import LogMelSpec
    AUDIOMAEPP_AVAILABLE = True
except ImportError as e:
    AUDIOMAEPP_AVAILABLE = False
    AUDIOMAEPP_IMPORT_ERROR = str(e)


# Model factory functions and their output dimensions
MODEL_CONFIGS = {
    "tiny": {"factory": "mae_plusplus_tiny", "embed_dim": 192},
    "base": {"factory": "mae_plusplus_base", "embed_dim": 768},
    "large": {"factory": "mae_plusplus_large", "embed_dim": 1024},
}


class AudioMAEPlusPlus(nn.Module):
    """
    AudioMAE++ model wrapper for the Audio-Health-Benchmark.
    
    Uses the exact AudioMAE++ implementation from third_party/audiomae-plusplus-official.
    
    Supports model variants: tiny, base, large
    """
    
    def __init__(
        self,
        ssl_encoder_source="base",  # Model variant: tiny, base, large
        freeze_encoder=True,
        output_hidden_states=False,
        sample_rate=16000,
        checkpoint_path=None,  # Path to .pth checkpoint file
        # AudioMAE++ specific configs
        img_size=(200, 80),
        patch_size=(4, 16),
        encoder_plusplus_block=True,
        decoder_plusplus_block=True,
        encoder_use_swiglu_final_ffn=True,
        decoder_use_swiglu_final_ffn=True,
        encoder_use_rope=False,
        decoder_use_rope=False,
        frequency_first=False,
        *args,
        **kwargs
    ):
        super().__init__()
        
        if not AUDIOMAEPP_AVAILABLE:
            raise ImportError(
                f"AudioMAE++ repo not found. Please clone it first:\n"
                f"  git clone https://github.com/SarthakYadav/audiomae-plusplus-official third_party/audiomae-plusplus-official\n"
                f"Original error: {AUDIOMAEPP_IMPORT_ERROR}"
            )
        
        self.model_variant = ssl_encoder_source.lower()
        self.freeze_encoder = freeze_encoder
        self.output_hidden_states = output_hidden_states
        self.sample_rate = sample_rate
        self.target_sr = 16000
        self.img_size = img_size
        self.frequency_first = frequency_first
        
        # Validate model variant
        if self.model_variant not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model variant: {self.model_variant}. "
                f"Choose from: {list(MODEL_CONFIGS.keys())}"
            )
        
        self._embed_dim = MODEL_CONFIGS[self.model_variant]["embed_dim"]
        
        # Get the model factory function
        factory_name = MODEL_CONFIGS[self.model_variant]["factory"]
        factory_fn = {
            "mae_plusplus_tiny": mae_plusplus_tiny,
            "mae_plusplus_base": mae_plusplus_base,
            "mae_plusplus_large": mae_plusplus_large,
        }[factory_name]
        
        # Initialize model with config (matching the official configs)
        self.model = factory_fn(
            img_size=img_size,
            patch_size=patch_size,
            encoder_plusplus_block=encoder_plusplus_block,
            decoder_plusplus_block=decoder_plusplus_block,
            encoder_use_swiglu_final_ffn=encoder_use_swiglu_final_ffn,
            decoder_use_swiglu_final_ffn=decoder_use_swiglu_final_ffn,
            encoder_use_rope=encoder_use_rope,
            decoder_use_rope=decoder_use_rope,
            frequency_first=frequency_first,
        )
        
        # Audio preprocessor (matching AudioMAE++ LogMelSpec)
        self.log_mel_spec = LogMelSpec(
            sr=self.target_sr,
            n_mels=80,  # AudioMAE++ uses 80 mels
            n_fft=400,
            win_len=400,
            hop_len=160,
            f_min=50.,
            f_max=8000.,
            normalize=True,
            flip_ft=not frequency_first
        )
        
        # Resampler for non-16kHz audio
        self.resampler = None
        if sample_rate != self.target_sr:
            self.resampler = T.Resample(
                orig_freq=sample_rate,
                new_freq=self.target_sr
            )
        
        # Load pretrained weights if provided
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)
        
        # Freeze encoder if needed
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def _load_checkpoint(self, checkpoint_path):
        """Load pretrained weights from checkpoint file."""
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
            print("Using randomly initialized weights.")
            return
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            print(f"Loaded pretrained weights from {checkpoint_path}")
            
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Using randomly initialized weights.")
    
    @property
    def output_dim(self):
        """Return the output dimension of the model."""
        # AudioMAE++ pools frequency dimension, so output is embed_dim * grid_size[freq]
        grid_size = self.model.grid_size()
        if self.frequency_first:
            freq_patches = grid_size[0]
        else:
            freq_patches = grid_size[1]
        return self._embed_dim * freq_patches
    
    def _preprocess_audio(self, x):
        """Convert audio tensor to log mel-spectrogram.
        
        Args:
            x: Audio tensor [B, T] at self.sample_rate
            
        Returns:
            Log mel-spectrogram [B, 1, T', F] where T'=time, F=80 mels
        """
        device = x.device
        
        # Resample to 16kHz if needed
        if self.resampler is not None:
            self.resampler = self.resampler.to(device)
            x = self.resampler(x)
        
        # Move log_mel_spec to correct device
        self.log_mel_spec = self.log_mel_spec.to(device)
        
        # Convert to log mel spectrogram
        spec = self.log_mel_spec(x)  # [B, 1, T, F] or [B, 1, F, T] depending on flip_ft
        
        return spec
    
    def _encode(self, lms):
        """Encode log mel-spectrogram to features.
        
        Follows the RuntimeMAE.encode() logic from hear_api/runtime.py.
        
        Args:
            lms: Log mel spectrogram [B, 1, T, F] or [B, 1, F, T]
            
        Returns:
            Features [B, T', D] where D = embed_dim * freq_patches
        """
        x = lms
        
        if self.frequency_first:
            patch_fbins = self.model.grid_size()[0]
            unit_frames = self.img_size[1]
            cur_frames = x.shape[-1]
        else:
            patch_fbins = self.model.grid_size()[1]
            unit_frames = self.img_size[0]
            cur_frames = x.shape[-2]
        
        # Pad to be divisible by unit_frames
        pad_frames = unit_frames - (cur_frames % unit_frames)
        if pad_frames > 0:
            if self.frequency_first:
                x = torch.nn.functional.pad(x, (0, pad_frames))
            else:
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_frames))
        
        # Process in chunks
        if self.frequency_first:
            r = x.shape[-1] // unit_frames
        else:
            r = x.shape[-2] // unit_frames
        
        embeddings = []
        for i in range(r):
            if self.frequency_first:
                sub_x = x[..., i*unit_frames:(i+1)*unit_frames]
            else:
                sub_x = x[:, :, i*unit_frames:(i+1)*unit_frames, :]
            
            emb = self.model.forward_features(sub_x)  # [B, time_patches, embed_dim * freq_patches]
            embeddings.append(emb)
        
        # Stack embeddings
        x = torch.cat(embeddings, dim=1)
        
        # Remove padded tail embeddings
        if pad_frames > 0:
            pad_emb_frames = int(embeddings[0].shape[1] * pad_frames / unit_frames)
            if pad_emb_frames > 0:
                x = x[:, :-pad_emb_frames]
        
        return x
    
    def forward(self, x, lengths=None):
        """
        Forward pass through AudioMAE++.
        
        Args:
            x: Raw audio tensor [B, T] at self.sample_rate
            lengths: Relative lengths [B] (not used, kept for compatibility)
        
        Returns:
            Features [B, T', D] where D = embed_dim * freq_patches
        """
        # Preprocess audio to log mel-spectrogram
        spec = self._preprocess_audio(x)  # [B, 1, T, F]
        
        with (torch.no_grad() if self.freeze_encoder else torch.enable_grad()):
            self.model.eval()
            features = self._encode(spec)  # [B, T', D]
        
        if self.output_hidden_states:
            # Return tuple for layer pooling compatibility
            return (features,)
        else:
            return features


# ============================================================================
# Convenience classes for common variants
# ============================================================================

class AudioMAEPP_Tiny(AudioMAEPlusPlus):
    """AudioMAE++ Tiny model (192 dim encoder)."""
    
    def __init__(self, freeze_encoder=True, output_hidden_states=False,
                 sample_rate=16000, checkpoint_path=None, **kwargs):
        super().__init__(
            ssl_encoder_source="tiny",
            freeze_encoder=freeze_encoder,
            output_hidden_states=output_hidden_states,
            sample_rate=sample_rate,
            checkpoint_path=checkpoint_path,
            **kwargs
        )


class AudioMAEPP_Base(AudioMAEPlusPlus):
    """AudioMAE++ Base model (768 dim encoder)."""
    
    def __init__(self, freeze_encoder=True, output_hidden_states=False,
                 sample_rate=16000, checkpoint_path=None, **kwargs):
        super().__init__(
            ssl_encoder_source="base",
            freeze_encoder=freeze_encoder,
            output_hidden_states=output_hidden_states,
            sample_rate=sample_rate,
            checkpoint_path=checkpoint_path,
            **kwargs
        )


class AudioMAEPP_Large(AudioMAEPlusPlus):
    """AudioMAE++ Large model (1024 dim encoder)."""
    
    def __init__(self, freeze_encoder=True, output_hidden_states=False,
                 sample_rate=16000, checkpoint_path=None, **kwargs):
        super().__init__(
            ssl_encoder_source="large",
            freeze_encoder=freeze_encoder,
            output_hidden_states=output_hidden_states,
            sample_rate=sample_rate,
            checkpoint_path=checkpoint_path,
            **kwargs
        )
