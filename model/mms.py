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
        # Slice each row to its TRUE length before the feature extractor so
        # per-sample normalization (and the conv frontend) never see a ragged
        # batch's zero-padding — otherwise a padded batch's valid frames
        # diverge from the per-chunk serial forward (validated: layer-norm
        # feat-extract still contaminates without this). Mirrors WavLM.forward.
        # The serial (lengths=None) path is unchanged, so existing caches match.
        model_device = next(self.model.parameters()).device
        if isinstance(x, torch.Tensor):
            if x.dim() == 1:
                x = x.unsqueeze(0)
            B, T = x.shape
            if lengths is None:
                abs_len = None
                x_list = [x[i].detach().cpu().numpy() for i in range(B)]
            else:
                abs_len = (lengths.to(x.device).float() * T).round().long().clamp(1, T)
                x_list = [x[i, :int(abs_len[i])].detach().cpu().numpy() for i in range(B)]
        else:
            abs_len = None
            x_list = x

        inputs = self.feature_extractor(
            x_list,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )

        # Cast to the model's weight dtype (e.g. an encoder cast to fp16) so
        # the conv frontend doesn't hit an fp32/fp16 mismatch. No-op in fp32.
        input_values = inputs.input_values.to(
            model_device, dtype=next(self.model.parameters()).dtype)

        # Attention mask from the true lengths, over the padded frame axis.
        attention_mask = None
        if abs_len is not None:
            attention_mask = length_to_mask(
                abs_len, max_len=input_values.shape[1]).to(model_device)

        with (torch.no_grad() if self.freeze_encoder else torch.enable_grad()):
            outputs = self.model(input_values, attention_mask=attention_mask)
            
            if self.output_hidden_states:
                return outputs.hidden_states[1:]  # Skip embedding layer
            else:
                return outputs.last_hidden_state

    def feature_lengths(self, sample_lengths):
        """Exact encoder output frame count per input sample length.

        wav2vec2-family conv-stack downsampling is a deterministic function of
        input samples, so the batched warmer can crop a padded ``(B, T', D)``
        output back to each chunk's true frame count exactly (bit-identical to
        the per-chunk serial path) instead of the proportional approximation.
        """
        return self.model._get_feat_extract_output_lengths(
            torch.as_tensor(sample_lengths)).long()


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
