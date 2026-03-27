import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoFeatureExtractor
from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model

from qwen_tts import Qwen3TTSTokenizer

class Qwen3Voice(nn.Module):
    def __init__(self, source, freeze_encoder, sample_rate, output_hidden_states, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register the custom model type so AutoModel can load it
        AutoConfig.register("qwen3_tts_tokenizer_12hz", Qwen3TTSTokenizerV2Config)
        AutoModel.register(Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model)

        self.model = AutoModel.from_pretrained(source)
        self.processor = AutoFeatureExtractor.from_pretrained(source)

        # The encoder is a MimiModel subclass (Qwen3TTSTokenizerV2Encoder)
        # It contains: .encoder (conv), .encoder_transformer, .quantizer
        # We want continuous features before quantization
        self.mimi_encoder = self.model.encoder

        if freeze_encoder:
            for param in self.mimi_encoder.parameters():
                param.requires_grad = False

        self.freeze_encoder = freeze_encoder
        self.sample_rate = sample_rate
        self.output_hidden_states = output_hidden_states
        self.d_transformer = 512

    def forward(self, x, lengths=None):
        # x: [B, T] waveform at self.sample_rate
        input_values = self.processor(
            list(x.cpu().numpy()), sampling_rate=self.sample_rate, return_tensors="pt"
        ).input_values  # [B, 1, T]

        input_values = input_values.to(device=x.device, dtype=x.dtype)

        # Conv encoder: [B, 1, T] -> [B, D, T']
        conv_output = self.mimi_encoder.encoder(input_values)

        # Transformer encoder: [B, D, T'] -> [B, T', D]
        transformer_out = self.mimi_encoder.encoder_transformer(
            conv_output.transpose(1, 2),
            output_hidden_states=self.output_hidden_states,
        )
        if self.output_hidden_states:
            # Skip layer 0 (conv projection input) to match WavLM convention
            return transformer_out.hidden_states[1:]
        else:
            return transformer_out.last_hidden_state  # [B, T', D]
