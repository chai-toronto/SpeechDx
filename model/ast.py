import torch
import torch.nn as nn
from transformers import ASTFeatureExtractor, ASTModel


class AST(nn.Module):
    """
    Audio Spectrogram Transformer (AST) wrapper.
    
    Unlike wav2vec2-style models that operate on raw waveforms, AST converts
    audio to mel-spectrograms and processes them as image patches using a
    Vision Transformer architecture.
    
    Note: AST does not use attention masks in the same way as wav2vec2 models.
    The feature extractor handles padding/truncation to a fixed length.
    """
    
    def __init__(self, ssl_encoder_source, freeze_encoder, output_hidden_states, sample_rate, 
                 max_length=1024, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # AST uses its own feature extractor that converts audio to mel-spectrograms
        self.processor = ASTFeatureExtractor.from_pretrained(
            ssl_encoder_source,
            max_length=max_length  # temporal dimension of spectrograms
        )
        self.model = ASTModel.from_pretrained(ssl_encoder_source)

        if freeze_encoder:
            self.model.requires_grad = False

        for param in self.model.parameters():
            param.requires_grad = not freeze_encoder

        self.freeze_encoder = freeze_encoder

        self.output_hidden_states = output_hidden_states
        self.model.config.output_hidden_states = self.output_hidden_states

        self.sample_rate = sample_rate
        self.max_length = max_length

    def forward(self, x, lengths=None):
        """
        Forward pass through AST.
        
        Args:
            x: Raw audio tensor of shape [B, T] or list of audio arrays
            lengths: Relative lengths [B] (not used by AST, kept for interface compatibility)
        
        Returns:
            If output_hidden_states is True: tuple of hidden states from all layers
            Else: last hidden state tensor of shape [B, num_patches, hidden_size]
        
        Note: AST's output sequence length depends on the spectrogram size and patch stride,
        not directly on audio length. The feature extractor pads/truncates to max_length.
        """
        # Handle both tensor and list inputs
        if isinstance(x, torch.Tensor):
            # Convert to list of numpy arrays for the processor
            x_list = [xi.cpu().numpy() for xi in x]
        else:
            x_list = x
            
        # Extract mel-spectrogram features
        inputs = self.processor(
            x_list,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )
        input_values = inputs.input_values.to(x.device if isinstance(x, torch.Tensor) else 'cpu')

        with (torch.no_grad() if self.freeze_encoder else torch.enable_grad()):
            outputs = self.model(input_values, output_hidden_states=self.output_hidden_states)
            
            if self.output_hidden_states:
                # Return all hidden states except the embedding layer (index 0)
                # hidden_states is tuple of (embedding_output, layer1, layer2, ..., layerN)
                return outputs.hidden_states[1:]  # tuple of layers (B, num_patches, D)
            else:
                return outputs.last_hidden_state  # (B, num_patches, D)
