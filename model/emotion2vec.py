import os
import torch
import torch.nn as nn
import numpy as np

try:
    from funasr import AutoModel as FunASRAutoModel
    FUNASR_AVAILABLE = True
except ImportError:
    FUNASR_AVAILABLE = False


class Emotion2Vec(nn.Module):
    """
    emotion2vec encoder wrapper for extracting speech emotion representations.
    
    emotion2vec is a self-supervised model for speech emotion recognition.
    Source: https://arxiv.org/abs/2312.15185
    GitHub: https://github.com/ddlBoJack/emotion2vec
    
    Available models (via FunASR/ModelScope):
    - iic/emotion2vec_base:        ~90M params, 768 dim
    - iic/emotion2vec_plus_large:  ~300M params, 1024 dim (fine-tuned on emotion data)
    
    Note: Requires FunASR library: pip install funasr
    
    Input: Raw audio at 16kHz
    Output: Frame-level embeddings (B, T, D) at 50 Hz, where D=768 or 1024
    """
    
    def __init__(self, ssl_encoder_source, freeze_encoder, output_hidden_states, sample_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if not FUNASR_AVAILABLE:
            raise ImportError(
                "emotion2vec requires the FunASR library. "
                "Install it with: pip install funasr"
            )
        
        # FunASR loads models from ModelScope
        # Map HuggingFace-style names to ModelScope names
        model_mapping = {
            "emotion2vec/emotion2vec_base": "iic/emotion2vec_base",
            "emotion2vec/emotion2vec_plus_large": "iic/emotion2vec_plus_large",
            "emotion2vec/emotion2vec_plus_base": "iic/emotion2vec_plus_base",
            "emotion2vec/emotion2vec_plus_seed": "iic/emotion2vec_plus_seed",
        }
        
        model_id = model_mapping.get(ssl_encoder_source, ssl_encoder_source)
        
        # Initialize device - will be set properly on first forward pass
        self._device = None
        
        # Check if model is cached locally to avoid ModelScope server check
        local_cache_path = os.path.expanduser(f"~/.cache/modelscope/hub/models/{model_id}")
        if os.path.isdir(local_cache_path) and os.path.exists(os.path.join(local_cache_path, "model.pt")):
            # Use local path directly to avoid network check
            model_path = local_cache_path
        else:
            model_path = model_id
        
        # Load the model via FunASR (disable update check for speed)
        self.model = FunASRAutoModel(
            model=model_path,
            disable_update=True,
        )
        
        self.freeze_encoder = freeze_encoder
        self.output_hidden_states = output_hidden_states
        self.sample_rate = sample_rate
        
        # emotion2vec outputs 768-dim (base) or 1024-dim (plus_large) embeddings
        # Actual dim will be determined at runtime from model output
        self.feature_dim = None  # Set after first forward pass
        
        # FunASR's AutoModel is not a proper nn.Module submodule, so parameters()
        # would be empty. Add a dummy parameter so next(parameters()).device works
        # in training pipelines that detect device this way.
        self._dummy_param = nn.Parameter(torch.empty(0), requires_grad=False)
        
    def forward(self, x, lengths=None):
        """
        Forward pass through emotion2vec encoder.
        
        Args:
            x: Raw audio waveform (B, T) at sample_rate Hz
            lengths: Relative lengths of each audio in batch (B,), values in [0, 1]
                    (Note: emotion2vec processes variable lengths internally)
            
        Returns:
            Frame-level embeddings tensor (B, T', D) where T' = audio_samples / 320
            (50 Hz frame rate for 16kHz audio), D=768 or 1024 depending on model
        """
        device = x.device
        
        # Convert to list of numpy arrays for FunASR
        if isinstance(x, torch.Tensor):
            x_list = [xi.cpu().numpy() for xi in x]
        else:
            x_list = x
        
        # Convert relative lengths to absolute sample counts for FunASR
        input_len = None
        if lengths is not None:
            T = x.shape[1]  # Total samples per audio (all same due to padding)
            input_len = (lengths * T).long().tolist()  # [48000, 32000, ...]
        
        with torch.set_grad_enabled(not self.freeze_encoder):
            results = self.model.generate(
                x_list,
                input_len=input_len,
                granularity="frame",
                extract_embedding=True,
            )
            
            # Extract embeddings from each result
            embeddings = []
            for result in results:
                feats = result.get("feats", None)
                if feats is None:
                    raise RuntimeError("emotion2vec failed to extract embeddings")
                if isinstance(feats, np.ndarray):
                    feats = torch.from_numpy(feats)
                embeddings.append(feats)
        
        output = torch.stack(embeddings, dim=0).to(device)
        
        if self.output_hidden_states:
            # emotion2vec doesn't expose intermediate layers via FunASR
            # Return single tensor wrapped in tuple for compatibility
            return (output,)
        
        return output
