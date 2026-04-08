"""
AudioMAE (Audio Masked Autoencoder) wrapper.

Uses the HuggingFace-compatible implementation from hance-ai/audiomae.

Original paper: "Masked Autoencoders that Listen" (NeurIPS 2022)
Original repo: https://github.com/facebookresearch/AudioMAE
HuggingFace model: https://huggingface.co/hance-ai/audiomae
"""

import torch
import torch.nn as nn
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
    - Output shape: [B, 512, 768] = [batch, num_patches, hidden_dim]
    
    Uses the encoder's native waveform_to_melspec() for preprocessing.
    """
    
    def __init__(self, ssl_encoder_source, freeze_encoder, output_hidden_states, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.freeze_encoder = freeze_encoder
        self.output_hidden_states = output_hidden_states
        
        # Load the HuggingFace model and extract encoder
        self.hf_model = AutoModel.from_pretrained(ssl_encoder_source, trust_remote_code=True)
        self.encoder = self.hf_model.encoder if hasattr(self.hf_model, 'encoder') else self.hf_model
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x, lengths=None):
        """
        Forward pass through AudioMAE.
        
        Args:
            x: Raw audio tensor [B, T] at 16kHz (resampled by dataloader)
            lengths: Relative lengths [B] (not used, kept for interface compatibility)
        
        Returns:
            If output_hidden_states: tuple of hidden states (repeated for compatibility)
            Else: last hidden state [B, 512, 768]
        
        Note: Expects audio already resampled to 16kHz and cropped by dataloader.
              waveform_to_melspec internally pads/truncates to 1024 frames.
        """
        device = x.device
        
        # Process each sample using encoder's native waveform_to_melspec
        # encoder.waveform_to_melspec expects [1, T] and returns [1024, 128]
        specs = []
        for i in range(x.shape[0]):
            waveform = x[i:i+1]  # [1, T]
            spec = self.encoder.waveform_to_melspec(waveform)  # [1024, 128]
            specs.append(spec)
        
        # Stack and add channel dim: [B, 1, 1024, 128]
        spec = torch.stack(specs, dim=0).unsqueeze(1).to(device)
        
        with (torch.no_grad() if self.freeze_encoder else torch.enable_grad()):
            features = self.encoder.forward_features(spec)  # [B, 513, 768]
            
            # Remove CLS token (first token)
            if features.shape[1] > 512:
                features = features[:, 1:, :]  # [B, 512, 768]
            
            if self.output_hidden_states:
                return (features,) * 12  # Repeat for layer compatibility
            return features
