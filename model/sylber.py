from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from sylber.utils.segment_utils import get_segment

from transformers import HubertModel, HubertConfig, BertModel, BertConfig
import torch.nn as nn

from model.utils import masked_normalize


class Sylber(nn.Module):
    def __init__(self,
                 model_ckpt="sylber",
                 ssl_encoder_source="facebook/hubert-base-ls960",
                 encoding_layer=9,
                 merge_threshold=0.8,
                 norm_threshold=2.6,
                 freeze_encoder=False,
                 output_hidden_states=False,
                 output_segment=False, # Mean of segments instead of frames
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        # Initialize the HuBERT model
        config = HubertConfig.from_pretrained(ssl_encoder_source, num_hidden_layers=encoding_layer)
        self.speech_model = HubertModel(config)

        # Model parameters
        self.enc_dim = self.speech_model.config.hidden_size
        self.encoding_layer = encoding_layer
        self.norm_threshold = norm_threshold
        self.merge_threshold = merge_threshold

        # Load pre-trained checkpoint if specified
        if model_ckpt is not None:
            if model_ckpt == "sylber":
                model_ckpt = "sylber.ckpt"
            if not Path(model_ckpt).exists():
                model_ckpt = hf_hub_download(repo_id="cheoljun95/sylber", filename=model_ckpt)
            state_dict = torch.load(model_ckpt, map_location='cpu')
            self.speech_model.load_state_dict(state_dict, strict=False)
            print("Pre-trained checkpoint loaded")

        # Freeze the encoder if specified
        for param in self.speech_model.parameters():
            param.requires_grad = not freeze_encoder
        self.freeze_encoder = freeze_encoder
        self.output_hidden_states = output_hidden_states
        self.output_segment = output_segment # Mean of segments instead of frames

    def forward(self, x, lengths):
        """
        x: (B, T) matrix of batch x time (raw waveform)
        lengths: (B,) relative lengths (to T_max) per sequence. If None, we assume no padding.
        Must be sampled at 16kHz. Either z-scored or not normalized.
        Returns:
          output: (B, T, D) or L's (B, T, D)
        """
        B, T = x.shape

        mask = None
        if lengths is not None:
            assert x.size(0) == lengths.size(0)
            # Convert lengths to absolute
            lengths = (lengths * T).long()
            # Build mask: 1 for valid, 0 for padded
            t = torch.arange(T, device=x.device).unsqueeze(0) # (1, T)
            mask = (t < lengths.unsqueeze(1)) * 1 # (B, T)

            # Normalize
            x = masked_normalize(x, mask)

        if self.output_hidden_states:
            layer_hidden_states = self.speech_model(x, attention_mask=mask, output_hidden_states=True).hidden_states[1:]
        else:
            layer_hidden_states = (self.speech_model(x, attention_mask=mask, output_hidden_states=False).last_hidden_state,)

        if not self.output_segment:
            if not self.output_hidden_states:
                return layer_hidden_states[0]
            return layer_hidden_states # tuple of L (B, T, D)'s

        # Compute segment-level features per layer
        layer_batch_features = []  # will hold tensors of shape [B, S_l, D] per layer
        B = x.shape[0]
        for hidden_states in layer_hidden_states:  # hidden_states: (B, T, D)
            hs_np = hidden_states.detach().to('cpu').numpy()  # (B, T, D)

            # Build segments per utterance
            batch_segments = [get_segment(hs_np[i], self.merge_threshold, self.norm_threshold)
                              for i in range(B)] # roughly shape (B, S_i, 2)

            # Compute per-utterance features [S_i, D], with S_i>=1
            batch_features_np = []
            max_num_segments_layer = 1
            for i, segs in enumerate(batch_segments):
                states = hs_np[i]  # (T, D)
                if len(segs) > 0:
                    feats = np.stack([states[s:e].mean(0) for s, e in segs], axis=0)  # (S_i, D)
                    max_num_segments_layer = max(max_num_segments_layer, len(segs))
                else:
                    # ensure 2D shape (1, D)
                    feats = states.mean(0, keepdims=True)  # (1, D)
                batch_features_np.append(feats)

            # Pad each utterance in this layer to max_num_segments_layer
            padded_batch_features = []
            for feats in batch_features_np:
                if feats.shape[0] < max_num_segments_layer:
                    pad = np.zeros((max_num_segments_layer - feats.shape[0], feats.shape[1]), dtype=feats.dtype)
                    feats = np.concatenate([feats, pad], axis=0)  # (S_layer, D)
                t = torch.from_numpy(feats).to(device=x.device, dtype=x.dtype)  # (S_layer, D)
                padded_batch_features.append(t)

            # Stack utterances for this layer -> (B, S_layer, D)
            layer_batch = torch.stack(padded_batch_features, dim=0)
            layer_batch_features.append(layer_batch)

        # Now unify S across layers (global pad) so we can stack on L
        S_global = max(t.shape[1] for t in layer_batch_features)
        for i, t in enumerate(layer_batch_features):
            if t.shape[1] < S_global:
                pad = torch.zeros((B, S_global - t.shape[1], t.shape[2]), device=x.device, dtype=x.dtype)
                layer_batch_features[i] = torch.cat([t, pad], dim=1)  # (B, S_global, D)

        # Final shape: L (B, S_global, D)'s
        if not self.output_hidden_states:
            return layer_batch_features[0]
        return tuple(layer_batch_features)









