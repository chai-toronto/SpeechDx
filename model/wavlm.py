from contextlib import nullcontext

import torch
from speechbrain.dataio.dataio import length_to_mask
from transformers import AutoFeatureExtractor, WavLMModel
import torch.nn as nn
from transformers.models.wavlm.modeling_wavlm import WavLMBaseModelOutput

from model.pool import ChunkPool
from model.probe import chunk_stat


class WavLM(nn.Module):
    def __init__(self,
                 ssl_encoder_source,
                 freeze_encoder,
                 output_hidden_states,
                 sample_rate,
                 threshold = 0.5,
                 min_chunk_size = 10,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = AutoFeatureExtractor.from_pretrained(ssl_encoder_source)
        self.wavlm = WavLMModel.from_pretrained(ssl_encoder_source)

        for param in self.wavlm.parameters():
            param.requires_grad = not freeze_encoder

        if freeze_encoder:
            self.wavlm.eval()

        # default to have grad
        hidden_size = self.wavlm.config.hidden_size
        self.chunker = ChunkPool(hidden_size, hidden_size, threshold=threshold, min_chunk_size=min_chunk_size)

        self.freeze_encoder = freeze_encoder

        self.output_hidden_states = output_hidden_states
        self.sample_rate = sample_rate

        self.last_chunk_stat = None
        self.last_reduction = None

    def forward(self, x, lengths=None):
        input_values = self.processor(x, sampling_rate=self.sample_rate, return_tensors="pt").input_values[0]
        input_values = input_values.to(device=x.device, dtype=x.dtype)

        mask = None
        if lengths is not None:
            T = input_values.shape[1]
            lengths = (lengths * T).long() # Convert to absolute lengths
            mask = length_to_mask(lengths)

        if self.output_hidden_states:
            features = self.wavlm_forward(input_values,
                                          output_hidden_states=True,
                                          attention_mask=mask,
                                          ).hidden_states[1:]
        else:
            features = self.wavlm_forward(input_values,
                                          attention_mask=mask
                                          ).last_hidden_state # (B, T, D)
        return features

    def wavlm_forward(self, input_values, attention_mask=None, output_hidden_states=False):
        r"""
        mask_time_indices (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices to mask extracted features for contrastive loss. When in training mode, model learns to predict
            masked extracted features in *config.proj_codevector_dim* space.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.wavlm.config.output_hidden_states
        )
        return_dict = self.wavlm.config.use_return_dict

        extract_features = self.wavlm.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self.wavlm._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.wavlm.feature_projection(extract_features)
        hidden_states = self.wavlm._mask_hidden_states(
            hidden_states, attention_mask=attention_mask
        )

        # New: pass through chunker
        chunked_hidden_states, nonchunk_mask, nonboundary_mask, boundary_prob = self.chunker(hidden_states, pad_mask=~attention_mask)

        reduction = chunk_stat(hidden_states, chunked_hidden_states, attention_mask.float().mean(1), nonchunk_mask)

        self.last_reduction = reduction

        attention_mask = ~nonchunk_mask

        hidden_states = chunked_hidden_states

        self.last_chunk_stat = (nonboundary_mask, boundary_prob)

        encoder_outputs = self.wavlm.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.wavlm.adapter is not None:
            hidden_states = self.wavlm.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        return WavLMBaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

if __name__ == "__main__":
    # simple test
    model = WavLM(
        ssl_encoder_source="microsoft/wavlm-base-plus",
        freeze_encoder=True,
        output_hidden_states=False,
        sample_rate=16000
    )

    dummy_wav = torch.randn(2, 16000 * 5)  # batch of 2, 5 seconds each
    dummy_lengths = torch.tensor([1.0, 0.8])  # first is full length, second is 80%

    features = model(dummy_wav, dummy_lengths)
    print(features.shape)  # should print (2, T', D) where T' depends on the model's downsampling