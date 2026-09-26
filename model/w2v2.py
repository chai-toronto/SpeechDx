import torch
import torch.nn as nn
from speechbrain.dataio.dataio import length_to_mask
from transformers import Wav2Vec2Processor, Wav2Vec2Model

class Wav2Vec2(nn.Module):
    def __init__(self, ssl_encoder_source, freeze_encoder, output_hidden_states, sample_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = Wav2Vec2Processor.from_pretrained(ssl_encoder_source, sampling_rate=sample_rate)
        self.model = Wav2Vec2Model.from_pretrained(ssl_encoder_source)

        if freeze_encoder:
            self.model.requires_grad = False

        for param in self.model.parameters():
            param.requires_grad = not freeze_encoder

        self.freeze_encoder = freeze_encoder

        self.output_hidden_states = output_hidden_states
        self.model.config.output_hidden_states = self.output_hidden_states

        self.sample_rate = sample_rate

    def forward(self, x, lengths=None):
        """
        If output_hidden_states is True, returns a tuple of hidden state tensors from all transformer layers.
        Else, returns the last hidden state as a tensor.

        Note: processor already normalizes the input waveform.
        """
        # Slice each row to its TRUE length before the feature extractor so
        # per-sample normalization (and the conv frontend) never see a ragged
        # batch's zero-padding — otherwise a padded batch's valid frames
        # diverge from the per-chunk serial forward (validated: layer-norm
        # feat-extract still contaminates without this). Mirrors WavLM.forward.
        # The serial (lengths=None) path is unchanged, so existing caches match.
        device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")
        # Match the model's weight dtype (e.g. an encoder cast to fp16) so the
        # conv frontend doesn't hit an fp32-input / fp16-weight mismatch. In
        # fp32 (the default warm) this is a no-op.
        in_dtype = next(self.model.parameters()).dtype
        if isinstance(x, torch.Tensor):
            if x.dim() == 1:
                x = x.unsqueeze(0)
            B, T = x.shape
            if lengths is None:
                abs_len = None
                x_list = [x[i].detach().cpu().numpy() for i in range(B)]
            else:
                abs_len = (lengths.to(device).float() * T).round().long().clamp(1, T)
                x_list = [x[i, :int(abs_len[i])].detach().cpu().numpy() for i in range(B)]
        else:
            abs_len = None
            x_list = x
        feature = self.processor(x_list,
                            return_tensors="pt",
                            sampling_rate=self.sample_rate,
                            padding=True).input_values
        xin = feature.to(device, dtype=in_dtype)

        mask = None
        if abs_len is not None:
            mask = length_to_mask(abs_len, max_len=xin.shape[1]).to(device) # (B, T)

        with (torch.no_grad() if self.freeze_encoder else torch.enable_grad()):
            if self.output_hidden_states:
                return self.model(xin, attention_mask=mask).hidden_states[1:] # tuple of 24 (B, T, D)
            else:
                return self.model(xin, attention_mask=mask).last_hidden_state # (B, T, D) matrix

    def feature_lengths(self, sample_lengths):
        """Exact encoder output frame count per input sample length.

        wav2vec2-family conv-stack downsampling is a deterministic function of
        input samples, so the batched warmer can crop a padded ``(B, T', D)``
        output back to each chunk's true frame count exactly (bit-identical to
        the per-chunk serial path) instead of the proportional approximation.
        """
        return self.model._get_feat_extract_output_lengths(
            torch.as_tensor(sample_lengths)).long()



