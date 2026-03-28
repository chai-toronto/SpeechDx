"""
OPERA (OPEn Respiratory Acoustic) Foundation Models.

OPERA provides three pretrained models for respiratory health audio analysis:
- OPERA-CT: Cola model with HTS-AT (Hierarchical Token-Semantic Audio Transformer) encoder
- OPERA-CE: Cola model with EfficientNet encoder  
- OPERA-GT: MAE ViT Small (Masked Autoencoder Vision Transformer)

Paper: "Towards Open Respiratory Acoustic Foundation Models: Pretraining and Benchmarking"
GitHub: https://github.com/evelyn0414/OPERA
HuggingFace: https://huggingface.co/evelyn0414/OPERA

This implementation uses the original OPERA code from third_party/OPERA.
Clone the repo first: git clone https://github.com/evelyn0414/OPERA third_party/OPERA

Input: Raw audio at 16kHz (converted to mel-spectrograms internally)
Output dimensions:
- OPERA-CT: 768
- OPERA-CE: 1280
- OPERA-GT: 384
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import librosa
from huggingface_hub import hf_hub_download

# Add OPERA repo to path for imports
OPERA_PATH = os.path.join(os.path.dirname(__file__), '..', 'third_party', 'OPERA')
if OPERA_PATH not in sys.path:
    sys.path.insert(0, OPERA_PATH)

# Import from OPERA repo (requires third_party/OPERA to be cloned)
try:
    from src.model.models_cola import Cola as OPERACola
    from src.model.models_mae import mae_vit_small
    OPERA_AVAILABLE = True
except ImportError as e:
    OPERA_AVAILABLE = False
    OPERA_IMPORT_ERROR = str(e)


# ============================================================================
# Audio Preprocessor
# ============================================================================

class OPERAAudioPreprocessor(nn.Module):
    """
    Preprocessor for OPERA models - converts waveform to mel-spectrogram.
    Matches the original OPERA pre_process_audio_mel_t function.
    
    For OPERA-GT (MAE), audio is split into fixed-length chunks (with 50% overlap)
    to produce exactly 256 time frames per chunk, matching img_size[0]=256.
    Features from all chunks are averaged (matching original OPERA behavior).
    """
    
    def __init__(self, sample_rate=16000, n_mels=64, f_min=50, f_max=2000,
                 n_fft=1024, hop_length=512, input_sec=8, expected_time_frames=None):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.input_sec = input_sec
        
        # If expected_time_frames is set (e.g., 256 for MAE), compute exact audio length needed
        # time_frames = (num_samples / hop_length) + 1
        # num_samples = (time_frames - 1) * hop_length
        if expected_time_frames is not None:
            self.expected_frames = expected_time_frames
            self.expected_samples = (expected_time_frames - 1) * hop_length
            self.split_chunks = True  # Enable chunk splitting for MAE
        else:
            # Fallback to input_sec-based calculation
            self.expected_frames = int(sample_rate * input_sec / hop_length) + 1
            self.expected_samples = int(sample_rate * input_sec)
            self.split_chunks = False
    
    def _duplicate_pad(self, audio, target_length):
        """Pad short audio by duplicating it (matching OPERA's _duplicate_padding)."""
        if len(audio) >= target_length:
            return audio[:target_length]
        
        # Repeat audio until we have enough
        n_repeats = (target_length // len(audio)) + 1
        padded = np.tile(audio, n_repeats)[:target_length]
        return padded
    
    def _split_into_chunks(self, audio):
        """
        Split audio into fixed-length overlapping chunks (matching OPERA's split_pad_sample).
        
        - If audio > chunk_length: split with 50% overlap, pad last chunk
        - If audio < chunk_length: duplicate-pad to chunk_length
        
        Returns: list of audio chunks, each of length self.expected_samples
        """
        chunk_length = self.expected_samples
        
        if len(audio) <= chunk_length:
            # Short audio: duplicate-pad to fill
            return [self._duplicate_pad(audio, chunk_length)]
        
        # Long audio: split into overlapping chunks (50% overlap)
        hop = chunk_length // 2
        chunks = []
        
        # Full chunks
        start = 0
        while start + chunk_length <= len(audio):
            chunks.append(audio[start:start + chunk_length])
            start += hop
        
        # Handle remaining audio (last partial chunk)
        if start < len(audio):
            last_chunk = audio[start:]
            # Pad the last chunk by duplicating
            last_chunk = self._duplicate_pad(last_chunk, chunk_length)
            chunks.append(last_chunk)
        
        return chunks
    
    def _audio_to_mel(self, audio):
        """Convert a single audio chunk to mel-spectrogram."""
        S = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sample_rate, 
            n_mels=self.n_mels,
            fmin=self.f_min, 
            fmax=self.f_max, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        
        # Convert to dB scale
        S = librosa.power_to_db(S, ref=np.max)
        
        # Normalize to [0, 1]
        if S.max() != S.min():
            mel_db = (S - S.min()) / (S.max() - S.min())
        else:
            mel_db = S
        
        # Transpose to [T, F] (matching OPERA's output)
        return mel_db.T
    
    def forward(self, x):
        """
        Convert waveform to mel-spectrogram(s).
        
        Args:
            x: Raw audio tensor [B, T] at self.sample_rate
        
        Returns:
            If split_chunks=False: Mel-spectrogram [B, T, F]
            If split_chunks=True: List of [num_chunks, T, F] tensors per batch item
                                  (for averaging features later)
        """
        device = x.device
        batch_size = x.shape[0]
        
        if not self.split_chunks:
            # Simple case: truncate/pad to fixed length
            specs = []
            for i in range(batch_size):
                audio = x[i].cpu().numpy()
                
                if len(audio) > self.expected_samples:
                    audio = audio[:self.expected_samples]
                elif len(audio) < self.expected_samples:
                    audio = np.pad(audio, (0, self.expected_samples - len(audio)), mode='constant')
                
                mel_db = self._audio_to_mel(audio)
                specs.append(mel_db)
            
            specs = np.stack(specs, axis=0)
            return torch.tensor(specs, dtype=torch.float32, device=device)
        
        else:
            # MAE case: split into chunks, return list for later averaging
            batch_chunks = []
            for i in range(batch_size):
                audio = x[i].cpu().numpy()
                chunks = self._split_into_chunks(audio)
                
                chunk_specs = []
                for chunk in chunks:
                    mel_db = self._audio_to_mel(chunk)
                    chunk_specs.append(mel_db)
                
                # Stack chunks for this sample: [num_chunks, T, F]
                chunk_specs = np.stack(chunk_specs, axis=0)
                batch_chunks.append(torch.tensor(chunk_specs, dtype=torch.float32, device=device))
            
            return batch_chunks  # List of tensors, one per batch item


# ============================================================================
# OPERA Model Wrapper
# ============================================================================

class OPERA(nn.Module):
    """
    OPERA model wrapper for the Audio-Health-Benchmark.
    
    Uses the exact OPERA implementation from third_party/OPERA.
    
    Supports three model variants:
    - operaCT: HTS-AT encoder (768 dim)
    - operaCE: EfficientNet encoder (1280 dim)
    - operaGT: MAE ViT Small (384 dim)
    """
    
    ENCODER_PATHS = {
        "operaCT": "encoder-operaCT.ckpt",
        "operaCE": "encoder-operaCE.ckpt",
        "operaGT": "encoder-operaGT.ckpt",
    }
    
    OUTPUT_DIMS = {
        "operaCT": 768,
        "operaCE": 1280,
        "operaGT": 384,
    }
    
    def __init__(self, ssl_encoder_source="operaCT", freeze_encoder=True, 
                 output_hidden_states=False, sample_rate=16000, 
                 input_sec=8, local_ckpt_dir="cks/model", *args, **kwargs):
        super().__init__()
        
        if not OPERA_AVAILABLE:
            raise ImportError(
                f"OPERA repo not found. Please clone it first:\n"
                f"  git clone https://github.com/evelyn0414/OPERA third_party/OPERA\n"
                f"Original error: {OPERA_IMPORT_ERROR}"
            )
        
        self.model_name = ssl_encoder_source
        self.freeze_encoder = freeze_encoder
        self.output_hidden_states = output_hidden_states
        self.sample_rate = sample_rate
        self.input_sec = input_sec
        self.local_ckpt_dir = local_ckpt_dir
        
        # Validate model name
        if self.model_name not in self.ENCODER_PATHS:
            raise ValueError(
                f"Unknown model: {self.model_name}. "
                f"Choose from: {list(self.ENCODER_PATHS.keys())}"
            )
        
        # Audio preprocessor (matches OPERA's pre_process_audio_mel_t)
        # For operaGT (MAE), we need exactly 256 time frames to match img_size[0]
        expected_time_frames = 256 if self.model_name == "operaGT" else None
        self.preprocessor = OPERAAudioPreprocessor(
            sample_rate=sample_rate,
            input_sec=input_sec,
            expected_time_frames=expected_time_frames
        )
        
        # Initialize model using exact OPERA code
        self._init_model()
        
        # Load pretrained weights
        self._load_pretrained()
        
        # Freeze if needed
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def _init_model(self):
        """Initialize the appropriate model architecture using exact OPERA code."""
        if self.model_name == "operaCT":
            # HTS-AT encoder (matching OPERA's initialize_pretrained_model)
            self.model = OPERACola(encoder="htsat")
        elif self.model_name == "operaCE":
            # EfficientNet encoder (matching OPERA's initialize_pretrained_model)
            self.model = OPERACola(encoder="efficientnet")
        elif self.model_name == "operaGT":
            # MAE ViT Small (matching OPERA's initialize_pretrained_model exactly)
            self.model = mae_vit_small(
                norm_pix_loss=False,
                in_chans=1,
                audio_exp=True,
                img_size=(256, 64),
                alpha=0.0,
                mode=0,
                use_custom_patch=False,
                split_pos=False,
                pos_trainable=False,
                use_nce=False,
                decoder_mode=1,  # decoder mode 0: global attn, 1: swin local attn
                mask_2d=False,
                mask_t_prob=0.7,
                mask_f_prob=0.3,
                no_shift=False
            ).float()
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def _load_pretrained(self):
        """Load pretrained weights from HuggingFace (matching OPERA's loading)."""
        ckpt_name = self.ENCODER_PATHS[self.model_name]
        ckpt_path = os.path.join(self.local_ckpt_dir, ckpt_name)
        
        # Download if not exists
        if not os.path.exists(ckpt_path):
            os.makedirs(self.local_ckpt_dir, exist_ok=True)
            print(f"Downloading {ckpt_name} from HuggingFace...")
            try:
                hf_hub_download(
                    repo_id="evelyn0414/OPERA",
                    filename=ckpt_name,
                    local_dir=self.local_ckpt_dir
                )
            except Exception as e:
                print(f"Warning: Could not download pretrained weights: {e}")
                print("Using randomly initialized weights.")
                return
        
        # Load checkpoint (matching OPERA's extract_opera_feature)
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
            
            # Load weights (OPERA uses strict=False)
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            print(f"Loaded pretrained weights for {self.model_name}")
                
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Using randomly initialized weights.")
    
    @property
    def output_dim(self):
        """Return the output dimension of the model."""
        return self.OUTPUT_DIMS[self.model_name]
    
    def forward(self, x, lengths=None):
        """
        Forward pass through OPERA model.
        
        Args:
            x: Raw audio tensor [B, T] at self.sample_rate
            lengths: Relative lengths [B] (not used, kept for compatibility)
        
        Returns:
            Features [B, 1, D] where D depends on model variant
        """
        # Preprocess audio to mel-spectrogram
        spec = self.preprocessor(x)
        
        with (torch.no_grad() if self.freeze_encoder else torch.enable_grad()):
            if self.model_name == "operaGT":
                # MAE model: spec is a list of chunk tensors per batch item
                # Process each chunk and average features (matching OPERA's original behavior)
                batch_features = []
                for sample_chunks in spec:  # sample_chunks: [num_chunks, T, F]
                    chunk_features = []
                    for chunk_idx in range(sample_chunks.shape[0]):
                        chunk = sample_chunks[chunk_idx:chunk_idx+1]  # [1, T, F]
                        # Skip chunks that are too small (matching OPERA: x.shape[0]>=16)
                        if chunk.shape[1] >= 16:
                            fea = self.model.forward_feature(chunk)  # [1, 384]
                            chunk_features.append(fea)
                    
                    if chunk_features:
                        # Average features across all chunks (matching OPERA's np.mean(features, axis=0))
                        stacked = torch.stack(chunk_features, dim=0)  # [num_chunks, 1, 384]
                        avg_fea = stacked.mean(dim=0)  # [1, 384]
                        batch_features.append(avg_fea)
                    else:
                        # Fallback: return zeros if no valid chunks
                        batch_features.append(torch.zeros(1, self.output_dim, device=x.device))
                
                features = torch.cat(batch_features, dim=0)  # [B, 384]
            else:
                # Cola models: spec is [B, T, F] tensor
                # Using dim=768 for CT, dim=1280 for CE
                features = self.model.extract_feature(spec, dim=self.output_dim)  # [B, dim]
        
        # OPERA models output a single embedding vector per audio
        # Expand to [B, 1, D] for compatibility with temporal probes
        features = features.unsqueeze(1)  # [B, 1, D]
        
        if self.output_hidden_states:
            # Return tuple for layer pooling compatibility
            return (features,)
        else:
            return features


# ============================================================================
# Convenience classes for each variant
# ============================================================================

class OPERA_CT(OPERA):
    """OPERA-CT model (HTS-AT encoder, 768 dim output)."""
    
    def __init__(self, freeze_encoder=True, output_hidden_states=False, 
                 sample_rate=16000, **kwargs):
        super().__init__(
            ssl_encoder_source="operaCT",
            freeze_encoder=freeze_encoder,
            output_hidden_states=output_hidden_states,
            sample_rate=sample_rate,
            **kwargs
        )


class OPERA_CE(OPERA):
    """OPERA-CE model (EfficientNet encoder, 1280 dim output)."""
    
    def __init__(self, freeze_encoder=True, output_hidden_states=False,
                 sample_rate=16000, **kwargs):
        super().__init__(
            ssl_encoder_source="operaCE",
            freeze_encoder=freeze_encoder,
            output_hidden_states=output_hidden_states,
            sample_rate=sample_rate,
            **kwargs
        )


class OPERA_GT(OPERA):
    """OPERA-GT model (MAE ViT Small encoder, 384 dim output)."""
    
    def __init__(self, freeze_encoder=True, output_hidden_states=False,
                 sample_rate=16000, **kwargs):
        super().__init__(
            ssl_encoder_source="operaGT",
            freeze_encoder=freeze_encoder,
            output_hidden_states=output_hidden_states,
            sample_rate=sample_rate,
            **kwargs
        )
