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

        # Chunker hook: set via register_transformer_pre_hook
        self._chunk_hook_fn = None
        self._chunk_at = None

    def forward(self, x, lengths=None):
        # x: [B, T] waveform at self.sample_rate
        input_values = self.processor(
            list(x.cpu().numpy()), sampling_rate=self.sample_rate, return_tensors="pt"
        ).input_values  # [B, 1, T]

        input_values = input_values.to(device=x.device, dtype=x.dtype)

        # Conv encoder: [B, 1, T] -> [B, D, T']
        conv_output = self.mimi_encoder.encoder(input_values)

        if self._chunk_hook_fn is not None:
            # Run transformer manually so we can apply chunking mid-stream
            hidden_states = self._run_transformer_with_chunking(
                conv_output.transpose(1, 2)
            )
            if self.output_hidden_states:
                return hidden_states  # already a tuple from manual run
            else:
                return hidden_states[-1] if isinstance(hidden_states, tuple) else hidden_states
        else:
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

    def _run_transformer_with_chunking(self, hidden_states):
        """Manually run transformer layers, applying chunker at self._chunk_at."""
        et = self.mimi_encoder.encoder_transformer
        device = hidden_states.device
        T = hidden_states.shape[1]

        # Build position_ids and cache_position for the current sequence
        cache_position = torch.arange(T, device=device)
        position_ids = cache_position.unsqueeze(0)

        pre_chunk_states = []  # hidden states before chunking (T_orig)
        post_chunk_states = []  # hidden states after chunking (T_new)
        chunker = None

        for i, layer in enumerate(et.layers):
            if i == self._chunk_at and self._chunk_hook_fn is not None:
                # Apply chunker: hook_fn(module, (hidden_states,)) -> (x_chunk,)
                new_args = self._chunk_hook_fn(layer, (hidden_states,))
                hidden_states = new_args[0]
                # Access the chunker module to get chunk_ids for pre-chunk aggregation
                # The hook_fn is Model.chunker_forward_hook, Model.chunker is ChunkPool
                chunker = getattr(self._chunk_hook_fn.__self__, 'chunker', None)
                # Recompute positional info for new sequence length
                T = hidden_states.shape[1]
                cache_position = torch.arange(T, device=device)
                position_ids = cache_position.unsqueeze(0)

            layer_out = layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                cache_position=cache_position,
            )
            hidden_states = layer_out[0]

            if self.output_hidden_states:
                if i < self._chunk_at:
                    pre_chunk_states.append(hidden_states)
                else:
                    post_chunk_states.append(hidden_states)

        if self.output_hidden_states:
            # Aggregate pre-chunk states to T_new using saved chunk_ids
            if chunker is not None and hasattr(chunker, '_last_chunk_ids'):
                chunk_ids = chunker._last_chunk_ids  # (B, T_orig)
                chunk_count = chunker._last_chunk_count  # (B, max_chunks)
                max_chunks = chunker._last_max_chunks
                aggregated_pre = []
                for hs in pre_chunk_states:
                    B, T_orig, D = hs.shape
                    agg = torch.zeros(B, max_chunks + 1, D, device=device, dtype=hs.dtype)
                    agg.scatter_add_(1, chunk_ids.unsqueeze(-1).expand_as(hs), hs)
                    agg = agg[:, 1:]  # drop bucket 0
                    agg = agg / chunk_count.unsqueeze(-1).clamp(min=1)
                    aggregated_pre.append(agg)
            else:
                # Fallback: repeat the first post-chunk state for pre-chunk layers
                aggregated_pre = [post_chunk_states[0]] * len(pre_chunk_states)
            return tuple(aggregated_pre + post_chunk_states)
        return hidden_states

    def register_transformer_pre_hook(self, hook_fn, layer_idx):
        """Store the hook function for manual application during forward pass."""
        self._chunk_hook_fn = hook_fn
        self._chunk_at = layer_idx
