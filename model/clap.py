import torch
import torch.nn as nn
from transformers import ClapAudioModel, ClapFeatureExtractor


class CLAP(nn.Module):
    """
    CLAP (Contrastive Language-Audio Pretraining) audio encoder wrapper.
    
    CLAP is a multimodal model trained with contrastive learning on audio-text pairs.
    This wrapper exposes only the audio encoder for feature extraction.
    
    Key differences from wav2vec2-style models:
    - Default sample rate is 48kHz (not 16kHz)
    - Uses mel-spectrograms internally
    - Uses a Swin Transformer-based architecture (HTSAT)
    - Output hidden_states structure differs from standard transformers
    
    Truncation modes:
    - For models with enable_fusion=True (e.g., clap-htsat-fused):
      Uses "fusion" truncation which creates 4 mel views for long audio
    - For models with enable_fusion=False (e.g., larger_clap_general):
      Uses "rand_trunc" truncation which takes a random 10s crop
    
    Note: CLAP's architecture has 4 main stages with hierarchical structure,
    so num_layers=4 refers to these stages, not individual transformer layers.
    """
    
    def __init__(self, ssl_encoder_source, freeze_encoder, output_hidden_states, sample_rate,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        # CLAP uses its own feature extractor that converts audio to mel-spectrograms
        self.processor = ClapFeatureExtractor.from_pretrained(ssl_encoder_source)
        self.model = ClapAudioModel.from_pretrained(ssl_encoder_source)
        
        # Check if model supports fusion
        self.enable_fusion = self.model.config.enable_fusion
        # Use appropriate truncation mode based on fusion support
        self.truncation = "fusion" if self.enable_fusion else "rand_trunc"
        
        if freeze_encoder:
            self.model.requires_grad = False

        for param in self.model.parameters():
            param.requires_grad = not freeze_encoder

        self.freeze_encoder = freeze_encoder
        self.output_hidden_states = output_hidden_states
        self.sample_rate = sample_rate
        
        # Update processor's expected sample rate if different
        if self.processor.sampling_rate != sample_rate:
            print(f"Warning: CLAP expects {self.processor.sampling_rate}Hz audio, "
                  f"but sample_rate={sample_rate} was specified. "
                  f"Audio will be processed at {self.processor.sampling_rate}Hz.")
        
        print(f"CLAP initialized: enable_fusion={self.enable_fusion}, truncation={self.truncation}")

    def forward(self, x, lengths=None):
        """
        Forward pass through CLAP audio encoder.
        
        Args:
            x: Raw audio tensor of shape [B, T] or list of audio arrays
            lengths: Relative lengths [B] (not directly used by CLAP, 
                     but kept for interface compatibility)
        
        Returns:
            If output_hidden_states is True: tuple of hidden states
            Else: last hidden state tensor of shape [B, seq_len, hidden_size]
        
        Note: CLAP internally handles variable-length audio through its
        feature extractor (padding/truncation based on max_length_s).
        """
        # Handle both tensor and list inputs
        if isinstance(x, torch.Tensor):
            # Convert to list of numpy arrays for the processor
            x_list = [xi.cpu().numpy() for xi in x]
        else:
            x_list = x
            
        # Extract mel-spectrogram features
        # - truncation="fusion": creates 4 mel views (downsampled full + 3 crops) for long audio
        # - truncation="rand_trunc": takes random 10s crop for long audio
        # Feature extractor also computes is_longer flag (True if original > 10s)
        inputs = self.processor(
            x_list,
            sampling_rate=self.processor.sampling_rate,
            truncation=self.truncation,
            return_tensors="pt"
        )
        device = x.device if isinstance(x, torch.Tensor) else 'cpu'
        input_features = inputs.input_features.to(device)
        is_longer = inputs.is_longer.to(device) if self.enable_fusion else None

        with (torch.no_grad() if self.freeze_encoder else torch.enable_grad()):
            outputs = self.model(
                input_features,
                is_longer=is_longer,
                output_hidden_states=self.output_hidden_states
            )
            
            if self.output_hidden_states:
                # CLAP's hidden_states include embedding output + outputs from each stage
                # Return all hidden states (tuple)
                return outputs.hidden_states[1:]  # Skip embedding layer output
            else:
                # last_hidden_state is 4D: (B, C, H, W) from Swin transformer
                # Reshape to 3D: (B, H*W, C) for temporal probes
                x = outputs.last_hidden_state  # (B, C, H, W)
                B, C, H, W = x.shape
                x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, H*W, C)
                return x
