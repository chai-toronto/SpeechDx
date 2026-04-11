"""
OPERA-GT: MAE ViT Small encoder for respiratory audio.

Paper: "Towards Open Respiratory Acoustic Foundation Models"
GitHub: https://github.com/evelyn0414/OPERA

Requires: git clone https://github.com/evelyn0414/OPERA third_party/OPERA

Input: 16kHz, 8.18 seconds (130,880 samples) → 256×64 spectrogram
Output: [B, 1, 384]
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Add OPERA repo to path
OPERA_PATH = os.path.join(os.path.dirname(__file__), '..', 'third_party', 'OPERA')
if OPERA_PATH not in sys.path:
    sys.path.insert(0, OPERA_PATH)

from src.benchmark.model_util import initialize_pretrained_model, get_encoder_path
from src.util import pre_process_audio_mel_t

SAMPLE_RATE = 16000
INPUT_SEC = 8.18  # 8.18s → T=256 frames
EXPECTED_SAMPLES = int(INPUT_SEC * SAMPLE_RATE)  # 130,880
OUTPUT_DIM = 384


class OPERA_GT(nn.Module):
    """OPERA-GT: MAE ViT Small encoder, 384 dim output."""
    
    def __init__(self, freeze_encoder=True, output_hidden_states=False, sample_rate=16000, **kwargs):
        super().__init__()
        
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"OPERA requires 16kHz audio, got {sample_rate}Hz")
        
        self.freeze_encoder = freeze_encoder
        self.output_hidden_states = output_hidden_states
        self.expected_samples = EXPECTED_SAMPLES
        
        # Initialize MAE model
        self.model = initialize_pretrained_model("operaGT")
        
        # Load weights
        ckpt_path = get_encoder_path("operaGT")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"Loaded OPERA-GT weights from {ckpt_path}")
        
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
    
    @property
    def output_dim(self):
        return OUTPUT_DIM
    
    def forward(self, x, lengths=None):
        """
        Args:
            x: Audio [B, T] at 16kHz, T=130880 (8.18 seconds)
        Returns:
            Features [B, 1, 384]
        """
        device = x.device
        batch_size = x.shape[0]
        
        # Convert to mel-spectrograms
        specs = []
        for i in range(batch_size):
            audio = x[i].cpu().numpy()
            spec = pre_process_audio_mel_t(audio, f_max=8000)
            specs.append(spec)
        
        specs = np.stack(specs, axis=0)
        specs = torch.tensor(specs, dtype=torch.float32, device=device)
        
        with (torch.no_grad() if self.freeze_encoder else torch.enable_grad()):
            features = self.model.forward_feature(specs)
        
        features = features.unsqueeze(1)  # [B, 1, 384]
        
        if self.output_hidden_states:
            return (features,)
        return features
