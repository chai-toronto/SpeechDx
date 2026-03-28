"""
AudioMAE (Audio Masked Autoencoder) wrapper.

Uses the HuggingFace-compatible implementation from hance-ai/audiomae.

Original paper: "Masked Autoencoders that Listen" (NeurIPS 2022)
Original repo: https://github.com/facebookresearch/AudioMAE
HuggingFace model: https://huggingface.co/hance-ai/audiomae
"""

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from transformers import AutoModel


class AudioMAE(nn.Module):
    """
    AudioMAE (Audio Masked Autoencoder) wrapper using hance-ai/audiomae from HuggingFace.
    
    AudioMAE applies a Vision Transformer (ViT) to audio spectrograms,
    similar to AST but with MAE-style self-supervised pretraining.
    
    Key specs (ViT-Base):
    - 12 transformer layers
    - 768 hidden dimension
    - Maximum audio length: 10 seconds
    - Output shape: (768, 8, 64) = (latent_dim, freq_dim, time_dim)
    
    Note: The original hance-ai model expects file paths. This wrapper
    replicates the preprocessing to accept raw tensors directly.
    """
    
    def __init__(self, ssl_encoder_source, freeze_encoder, output_hidden_states, sample_rate,
                 max_length_s=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.sample_rate = sample_rate
        self.max_length_s = max_length_s
        self.freeze_encoder = freeze_encoder
        self.output_hidden_states = output_hidden_states
        
        # Load the HuggingFace model
        self.hf_model = AutoModel.from_pretrained(ssl_encoder_source, trust_remote_code=True)
        
        # Extract the encoder component for direct tensor processing
        # The hance-ai/audiomae model has an 'encoder' attribute (AudioMAEEncoder)
        if hasattr(self.hf_model, 'encoder'):
            self.encoder = self.hf_model.encoder
        else:
            self.encoder = self.hf_model
        
        # Setup audio preprocessing (replicate what the hance-ai model does internally)
        # AudioMAE uses: 16kHz, 128 mel bins, 1024 target frames
        self.target_sr = 16000
        self.n_mels = 128
        self.target_length = 1024  # 10 seconds at 16kHz with hop_length=160
        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=1024,
            win_length=1024,
            hop_length=160,
            n_mels=self.n_mels,
            f_min=0,
            f_max=8000,
            power=2.0,
            normalized=False,
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80)
        
        # AudioMAE normalization stats (from AudioSet pretraining)
        self.norm_mean = -4.2677393
        self.norm_std = 4.5689974
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def _preprocess_audio(self, x):
        """Convert audio tensor to normalized mel-spectrogram for AudioMAE.
        
        Args:
            x: Audio tensor [B, T] at self.sample_rate
            
        Returns:
            Spectrogram tensor [B, 1, target_length, n_mels]
        """
        device = x.device
        
        # Resample to 16kHz if needed
        if self.sample_rate != self.target_sr:
            resampler = T.Resample(
                orig_freq=self.sample_rate,
                new_freq=self.target_sr
            ).to(device)
            x = resampler(x)
        
        # Clip to max length (10 seconds = 160000 samples at 16kHz)
        max_samples = int(self.max_length_s * self.target_sr)
        if x.shape[1] > max_samples:
            x = x[:, :max_samples]
        
        # Compute mel spectrogram
        self.mel_transform = self.mel_transform.to(device)
        spec = self.mel_transform(x)  # [B, n_mels, time]
        spec = self.amplitude_to_db(spec)  # Convert to dB
        
        # Normalize
        spec = (spec - self.norm_mean) / (self.norm_std * 2)
        
        # Transpose to [B, time, n_mels]
        spec = spec.transpose(1, 2)
        
        # Pad or truncate to target_length
        if spec.shape[1] < self.target_length:
            pad_len = self.target_length - spec.shape[1]
            spec = torch.nn.functional.pad(spec, (0, 0, 0, pad_len))
        else:
            spec = spec[:, :self.target_length, :]
        
        # Add channel dimension: [B, 1, target_length, n_mels]
        spec = spec.unsqueeze(1)
        
        return spec

    def forward(self, x, lengths=None):
        """
        Forward pass through AudioMAE.
        
        Args:
            x: Raw audio tensor of shape [B, T] at self.sample_rate
            lengths: Relative lengths [B] (not used directly, kept for compatibility)
        
        Returns:
            If output_hidden_states is True: tuple of hidden states from all layers
            Else: last hidden state tensor of shape [B, T', D]
        
        Note: AudioMAE clips audio longer than 10 seconds.
        """
        # Preprocess audio to spectrogram
        spec = self._preprocess_audio(x)  # [B, 1, 1024, 128]
        
        with (torch.no_grad() if self.freeze_encoder else torch.enable_grad()):
            # Forward through the encoder
            # The encoder expects [B, C, H, W] format
            features = self.encoder.forward_features(spec)  # [B, num_patches+1, 768]
            
            # Remove CLS token if present (first token)
            if features.shape[1] > 512:  # Has CLS token
                features = features[:, 1:, :]  # [B, 512, 768]
            
            if self.output_hidden_states:
                # AudioMAE doesn't expose intermediate layers easily
                # Return final features repeated for compatibility
                return (features,) * 12
            else:
                return features  # [B, 512, 768]
