import torch
import torch.nn as nn
from speechbrain.dataio.dataio import length_to_mask
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


class MMS(nn.Module):
    """
    MMS (Massively Multilingual Speech) encoder wrapper for extracting audio representations.
    
    MMS is a wav2vec2-based model trained on 1,400+ languages.
    Source: https://arxiv.org/abs/2305.13516
    
    Available models:
    - facebook/mms-300m: 300M params, 1024 dim, 24 layers
    - facebook/mms-1b:   1B params, 1280 dim, 48 layers
    
    Input: Raw audio at 16kHz
    Output: Hidden states (B, T, D) where T = audio_samples / 320 (20ms stride)
    """
    
    def __init__(self, ssl_encoder_source, freeze_encoder, output_hidden_states, sample_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # MMS uses Wav2Vec2FeatureExtractor (not full Processor which requires tokenizer)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(ssl_encoder_source)
        self.model = Wav2Vec2Model.from_pretrained(ssl_encoder_source)

        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False

        self.freeze_encoder = freeze_encoder
        self.output_hidden_states = output_hidden_states
        self.model.config.output_hidden_states = self.output_hidden_states
        self.sample_rate = sample_rate

    def forward(self, x, lengths=None):
        """
        Forward pass through MMS encoder.
        
        Args:
            x: Raw audio waveform (B, T) at sample_rate Hz
            lengths: Relative lengths of each audio in batch (B,), values in [0, 1]
            
        Returns:
            If output_hidden_states is True: tuple of hidden states from all encoder layers
            Else: last hidden state tensor (B, T', D)
        """
        # Convert to list for feature extractor
        if isinstance(x, torch.Tensor):
            x_list = [xi.cpu().numpy() for xi in x]
        else:
            x_list = x
        
        inputs = self.feature_extractor(
            x_list,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        
        input_values = inputs.input_values.to(next(self.model.parameters()).device)
        
        # Create attention mask if lengths provided
        attention_mask = None
        if lengths is not None:
            T = input_values.shape[1]
            abs_lengths = (lengths * T).long()
            attention_mask = length_to_mask(abs_lengths, max_len=T)
            attention_mask = attention_mask.to(input_values.device)

        with (torch.no_grad() if self.freeze_encoder else torch.enable_grad()):
            outputs = self.model(input_values, attention_mask=attention_mask)
            
            if self.output_hidden_states:
                return outputs.hidden_states[1:]  # Skip embedding layer
            else:
                return outputs.last_hidden_state


if __name__ == "__main__":
    # Simple test
    model = MMS(
        ssl_encoder_source="facebook/mms-300m",
        freeze_encoder=True,
        output_hidden_states=False,
        sample_rate=16000
    )
    
    dummy_wav = torch.randn(2, 16000 * 5)  # batch of 2, 5 seconds each
    dummy_lengths = torch.tensor([1.0, 0.8])
    
    features = model(dummy_wav, dummy_lengths)
    print(f"Output shape: {features.shape}")
    # Expected: (2, ~250, 1024) for 5s audio with mms-300m
