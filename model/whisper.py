import torch
import torch.nn as nn
from transformers import WhisperFeatureExtractor, WhisperModel


class Whisper(nn.Module):
    """
    Whisper encoder wrapper for extracting audio representations.
    
    Uses only the encoder part of OpenAI's Whisper model for feature extraction.
    The encoder converts audio into a sequence of hidden state representations.
    
    Whisper models available on HuggingFace:
    - openai/whisper-tiny:    39M params, 384 dim, 4 layers
    - openai/whisper-base:    74M params, 512 dim, 6 layers
    - openai/whisper-small:   244M params, 768 dim, 12 layers
    - openai/whisper-medium:  769M params, 1024 dim, 24 layers
    - openai/whisper-large:   1550M params, 1280 dim, 32 layers
    - openai/whisper-large-v2: Same architecture as large
    - openai/whisper-large-v3: Same architecture as large
    
    Input: Raw audio at 16kHz
    Output: Hidden states from encoder (B, T, D) where T depends on audio length
            T = audio_samples / 160 (Whisper uses 10ms hop length at 16kHz)
    """
    
    def __init__(self, ssl_encoder_source, freeze_encoder, output_hidden_states, sample_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Whisper uses its own feature extractor that converts audio to mel spectrograms
        self.processor = WhisperFeatureExtractor.from_pretrained(ssl_encoder_source)
        
        # Load the full model but we'll only use the encoder
        full_model = WhisperModel.from_pretrained(ssl_encoder_source)
        self.encoder = full_model.encoder
        
        # Free up memory from decoder
        del full_model.decoder
        del full_model
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.freeze_encoder = freeze_encoder
        self.output_hidden_states = output_hidden_states
        self.sample_rate = sample_rate
        
    def forward(self, x, lengths=None):
        """
        Forward pass through Whisper encoder.
        
        Args:
            x: Raw audio waveform (B, T) at sample_rate Hz
            lengths: Relative lengths of each audio in batch (B,), values in [0, 1]
            
        Returns:
            If output_hidden_states is True: tuple of hidden states from all encoder layers
            Else: last hidden state tensor (B, T', D)
        """
        # Process audio to mel spectrogram features
        # Note: WhisperFeatureExtractor expects list of numpy arrays or torch tensors
        if isinstance(x, torch.Tensor):
            x_list = [xi.cpu().numpy() for xi in x]
        else:
            x_list = x
            
        inputs = self.processor(
            x_list,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
            padding="max_length",  # Pad to 3000 mel frames (30s) as Whisper expects
            truncation=True,       # Truncate if longer than 30s
        )
        
        device = next(self.encoder.parameters()).device
        input_features = inputs.input_features.to(device)
        
        # Processor returns mask of shape [B, 3000] for mel frames
        # Downsample by 2 to match encoder output (conv subsampling halves time dimension)
        attention_mask = None
        if inputs.attention_mask is not None:
            # Take every other frame to match encoder's time resolution
            attention_mask = inputs.attention_mask[:, ::2].to(device)
        
        with (torch.no_grad() if self.freeze_encoder else torch.enable_grad()):
            outputs = self.encoder(
                input_features,
                attention_mask=attention_mask,
                output_hidden_states=self.output_hidden_states,
                return_dict=True,
            )
            
            if self.output_hidden_states:
                # Return all hidden states except the initial embedding (similar to other models)
                return outputs.hidden_states[1:]
            else:
                return outputs.last_hidden_state


if __name__ == "__main__":
    # Simple test
    model = Whisper(
        ssl_encoder_source="openai/whisper-base",
        freeze_encoder=True,
        output_hidden_states=False,
        sample_rate=16000
    )
    
    dummy_wav = torch.randn(2, 16000 * 5)  # batch of 2, 5 seconds each
    dummy_lengths = torch.tensor([1.0, 0.8])  # first is full length, second is 80%
    
    features = model(dummy_wav, dummy_lengths)
    print(f"Output shape: {features.shape}")
    # Expected: (2, T', D) where T' = 250 (5s * 100 / 2) and D = 512 for base model
