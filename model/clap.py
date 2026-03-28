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
    
    Note: CLAP's architecture has 4 main stages with hierarchical structure,
    so num_layers=4 refers to these stages, not individual transformer layers.
    """
    
    def __init__(self, ssl_encoder_source, freeze_encoder, output_hidden_states, sample_rate,
                 max_length_s=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # CLAP uses its own feature extractor that converts audio to mel-spectrograms
        self.processor = ClapFeatureExtractor.from_pretrained(ssl_encoder_source)
        self.model = ClapAudioModel.from_pretrained(ssl_encoder_source)
        
        if freeze_encoder:
            self.model.requires_grad = False

        for param in self.model.parameters():
            param.requires_grad = not freeze_encoder

        self.freeze_encoder = freeze_encoder
        self.output_hidden_states = output_hidden_states
        self.sample_rate = sample_rate
        self.max_length_s = max_length_s
        
        # Update processor's expected sample rate if different
        if self.processor.sampling_rate != sample_rate:
            print(f"Warning: CLAP expects {self.processor.sampling_rate}Hz audio, "
                  f"but sample_rate={sample_rate} was specified. "
                  f"Audio will be processed at {self.processor.sampling_rate}Hz.")

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
        
        # Check if audio is longer than max_length_s for fusion feature
        # This enables the model's built-in feature fusion for long audio
        is_longer = None
        if lengths is not None:
            # Compute actual duration in seconds
            # Note: lengths are relative (0-1), T is total samples
            T = x.shape[1] if isinstance(x, torch.Tensor) else len(x_list[0])
            actual_lengths_s = (lengths * T) / self.processor.sampling_rate
            is_longer = actual_lengths_s > self.max_length_s
            is_longer = is_longer.to(x.device if isinstance(x, torch.Tensor) else 'cpu')
            
        # Extract mel-spectrogram features
        inputs = self.processor(
            x_list,
            sampling_rate=self.processor.sampling_rate,
            return_tensors="pt"
        )
        input_features = inputs.input_features.to(
            x.device if isinstance(x, torch.Tensor) else 'cpu'
        )

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
