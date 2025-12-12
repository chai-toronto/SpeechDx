import copy
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from speechbrain.dataio.dataio import length_to_mask
from torch import nn
from torch.utils.checkpoint import checkpoint

from model.models.base_model import BaseAudioPretrainModel
from model.models.WavHJepa.extractors.audio_extractor import Extractor
from model.models.WavHJepa.functions import trunc_normal_
from model.models.WavHJepa.pos_embed import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
)
from model.models.WavHJepa.types import ForwardReturn, TransformerEncoderCFG, TransformerLayerCFG
from model.models.WavHJepa.chunking_encoder import ChunkingEncoder, ChunkingEncoderConfig


torch._dynamo.config.capture_dynamic_output_shape_ops = True


class JEPA(BaseAudioPretrainModel):
    """
    Joint-Embedding Predictive Architecture (JEPA) for audio pretraining.

    Adapted from original WavJEPA for the audio-pretrainer framework.
    Inherits from BaseAudioPretrainModel for consistent training interface.

    This implementation is inspired by:
        * I-JEPA http://arxiv.org/abs/2301.08243
        * Data2vec 2.0 http://arxiv.org/abs/2212.07525

    Architecture:
        - Feature extraction on raw audio (B, C, T) -> (B, n_patches, D)
        - Multiview batch creation from encoded features (B, n_patches, D) -> (B*nr_samples, n_patches, D)
        - Masking applied by DataModule to generate context and target masks
        - Student encoder processes visible context tokens
        - Decoder predicts masked target tokens
        - Teacher encoder (EMA) provides targets for prediction

    Args:
        feature_extractor: Audio feature extractor (e.g., ConvFeatureExtractor)
        transformer_encoder_layers_cfg: Config for encoder transformer layers
        transformer_encoder_cfg: Config for encoder transformer
        transformer_decoder_layers_cfg: Config for decoder transformer layers
        transformer_decoder_cfg: Config for decoder transformer
        ema_decay: Initial EMA decay rate for teacher
        ema_end_decay: Final EMA decay rate
        ema_anneal_end_step: Step to finish EMA decay annealing
        average_top_k_layers: Number of top layers to average for targets
        sample_rate: Audio sample rate (default 16kHz)
        crop_seconds: Duration of audio crops in seconds (used for calculating num_patches)
        nr_samples_per_audio: Number of views to create per audio sample
    """

    teacher_encoder: nn.Module

    def __init__(
        self,
        feature_extractor: Extractor,
        transformer_encoder_layers_cfg: TransformerLayerCFG | dict,
        transformer_encoder_cfg: TransformerEncoderCFG | dict,
        transformer_decoder_layers_cfg: TransformerLayerCFG | dict,
        transformer_decoder_cfg: TransformerEncoderCFG | dict,
        loss_fn: nn.Module = None,
        ema_decay: float = 0.999,
        ema_end_decay: float = 0.9999,
        ema_anneal_end_step: int = 100000,
        average_top_k_layers: int = 12,
        sample_rate: int = 16000,
        crop_seconds: float = 10.0,
        in_channels: int = 1,
        nr_samples_per_audio: int = 16,
        use_gradient_checkpointing: bool = False,
        compile_modules: bool = False,
        is_spectrogram: bool = False,
        size: str = "base",
        # Chunking encoder parameters
        use_chunking_encoder: bool = False,
        chunking_encoder_cfg: ChunkingEncoderConfig | dict | None = None,
        lb_loss_weight: float = 1,
        # BaseAudioPretrainModel parameters
        learning_rate: float = 1e-4,
        weight_decay: float = 0.04,
        warmup_steps: int = 100000,
        max_steps: int = 400000,
        optimizer: str = "adamw",
        scheduler: str = "linear_warmup_cosine",
        **kwargs: dict[str, Any],
    ):
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        self.sr = sample_rate
        self.is_spectrogram = is_spectrogram
        # Please dont leave vars hanging
        self.ema_decay = ema_decay
        self.ema_end_decay = ema_end_decay
        self.average_top_k_layers = average_top_k_layers
        self.ema_end_step = ema_anneal_end_step
        self.target_length = int(sample_rate * crop_seconds)
        self.total_patches = feature_extractor.total_patches(self.target_length)
        self.use_compiled_forward = compile_modules
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.in_channels = in_channels
        self.nr_samples_per_audio = nr_samples_per_audio
        self.save_hyperparameters(
            ignore=["feature_extractor", "loss_fn"]
        )
        self.extract_audio = feature_extractor
        self.feature_norms: nn.Module = nn.LayerNorm(self.extract_audio.embedding_dim)
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss(reduction="none")

        # Convert dict/DictConfig configs to dataclass instances (Hydra compatibility)
        from omegaconf import DictConfig

        if not isinstance(transformer_encoder_layers_cfg, TransformerLayerCFG):
            # Convert DictConfig or dict to dataclass
            cfg_dict = dict(transformer_encoder_layers_cfg) if isinstance(transformer_encoder_layers_cfg, DictConfig) else transformer_encoder_layers_cfg
            transformer_encoder_layers_cfg = TransformerLayerCFG(**cfg_dict)

        if not isinstance(transformer_encoder_cfg, TransformerEncoderCFG):
            cfg_dict = dict(transformer_encoder_cfg) if isinstance(transformer_encoder_cfg, DictConfig) else transformer_encoder_cfg
            transformer_encoder_cfg = TransformerEncoderCFG(**cfg_dict)

        if not isinstance(transformer_decoder_layers_cfg, TransformerLayerCFG):
            cfg_dict = dict(transformer_decoder_layers_cfg) if isinstance(transformer_decoder_layers_cfg, DictConfig) else transformer_decoder_layers_cfg
            transformer_decoder_layers_cfg = TransformerLayerCFG(**cfg_dict)

        if not isinstance(transformer_decoder_cfg, TransformerEncoderCFG):
            cfg_dict = dict(transformer_decoder_cfg) if isinstance(transformer_decoder_cfg, DictConfig) else transformer_decoder_cfg
            transformer_decoder_cfg = TransformerEncoderCFG(**cfg_dict)

        # if chunking_encoder_cfg is not None and not isinstance(chunking_encoder_cfg, ChunkingEncoderConfig):
        #     cfg_dict = dict(chunking_encoder_cfg) if isinstance(chunking_encoder_cfg, DictConfig) else chunking_encoder_cfg
        #     chunking_encoder_cfg = ChunkingEncoderConfig(**cfg_dict)

        # If size is large, then alter the encoder parameters to mimic VIT-Large. Should results in ~300m parameters.
        if size == "large":
            transformer_encoder_layers_cfg.nhead = 16
            transformer_encoder_layers_cfg.d_model = 1024
            transformer_encoder_layers_cfg.mlp_ratio = 4.0
            transformer_encoder_cfg.num_layers = 24

        self.n_encoder_heads = transformer_encoder_layers_cfg.nhead
        self.encoder_embedding_dim = transformer_encoder_layers_cfg.d_model
        self.n_decoder_heads = transformer_decoder_layers_cfg.nhead
        self.decoder_embedding_dim = transformer_decoder_layers_cfg.d_model

        # Encoder: optionally use ChunkingEncoder
        self.use_chunking_encoder = use_chunking_encoder
        self.lb_loss_weight = lb_loss_weight

        if use_chunking_encoder:
            # Create ChunkingEncoder config if not provided
            if chunking_encoder_cfg is None:
                chunking_encoder_cfg = ChunkingEncoderConfig(
                    d_model=[self.encoder_embedding_dim, self.encoder_embedding_dim],
                    num_heads=[self.n_encoder_heads, self.n_encoder_heads],
                    encoder_layers=[transformer_encoder_cfg.num_layers // 2,
                                    transformer_encoder_cfg.num_layers // 2],
                    decoder_layers=[transformer_encoder_cfg.num_layers // 2,
                                    transformer_encoder_cfg.num_layers // 2],
                    main_layers=[0, transformer_encoder_cfg.num_layers],
                    mlp_ratio=transformer_encoder_layers_cfg.mlp_ratio,
                    norm_first=transformer_encoder_layers_cfg.norm_first,
                    dropout=transformer_encoder_layers_cfg.dropout,
                    activation=transformer_encoder_layers_cfg.activation,
                    layer_norm_eps=transformer_encoder_layers_cfg.layer_norm_eps,
                )
            self.encoder = ChunkingEncoder(chunking_encoder_cfg)
        else:
            assert "Use ChunkingEncoder"
        self.post_extraction_mapper: nn.Module | None = (
            nn.Linear(feature_extractor.embedding_dim, self.encoder_embedding_dim)
            if feature_extractor.embedding_dim != self.encoder_embedding_dim
            else None
        )
        decoder_layer = nn.TransformerEncoderLayer(**transformer_decoder_layers_cfg.to_layer_kwargs())
        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            norm=nn.LayerNorm(self.decoder_embedding_dim),
            **transformer_decoder_cfg.to_encoder_kwargs(),
        )

        self.decoder_to_encoder_mapper = nn.Linear(
            self.decoder_embedding_dim, self.encoder_embedding_dim, bias=True
        )
        self.encoder_to_decoder_mapper = nn.Linear(
            self.encoder_embedding_dim, self.decoder_embedding_dim
        )

        # For the autocast add batch dimensions.
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, self.decoder_embedding_dim, requires_grad=True)
        )
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.pos_encoding_encoder = self._get_pos_embed_params(
            self.encoder_embedding_dim
        )
        self.pos_encoding_decoder = self._get_pos_embed_params(
            self.decoder_embedding_dim
        )

        self.apply(self._init_weights)
        self._init_teacher()
        if compile_modules:
            self._compile_operations()

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:  # type: ignore
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _get_pos_embed_params(self, embedding_dim):
        """Calculates the positional embedding parameters and returns them."""
        pos_embed = nn.Parameter(
            torch.zeros(1, self.total_patches, embedding_dim),
            requires_grad=False,
        )
        positions = np.arange(self.total_patches, dtype=np.float64)

        if self.is_spectrogram:
            # For spectrogram input, use 2D sincos embeddings
            pos_embed_data = get_2d_sincos_pos_embed(
                embedding_dim, self.extract_audio.grid_size, cls_token_num=0
            )
        else:
            # For waveform input (ConvFeatureExtractor), use 1D sincos embeddings
            pos_embed_data = get_1d_sincos_pos_embed_from_grid(
                embedding_dim, positions
            )

        pos_embed.data.copy_(torch.from_numpy(pos_embed_data).float().unsqueeze(0))
        return pos_embed

    def _init_teacher(self):
        self.teacher_encoder = copy.deepcopy(self.encoder)
        self.teacher_encoder.requires_grad_(False)

    def _get_ema_decay(self):
        if self.global_step >= self.ema_end_step:
            return self.hparams.ema_end_decay
        r = self.hparams.ema_end_decay - self.hparams.ema_decay
        pct_remaining = 1 - self.global_step / self.ema_end_step
        return self.hparams.ema_end_decay - r * pct_remaining

    @torch.no_grad()
    def _step_teacher(self):
        r = self._get_ema_decay()
        for student, teacher in zip(
            self.encoder.parameters(), self.teacher_encoder.parameters()
        ):
            teacher.data.mul_(r).add_((1 - r) * student.detach().data)

    def _compile_operations(self):
        """
        Use torch.compile on the extractor, encoder and decoder blocks for faster forward
        """
        try:
            self.encoder_forward = torch.compile(self.encoder_forward, fullgraph=True)
            self.decoder_forward = torch.compile(self.decoder_forward, fullgraph=True)
            self._forward_teacher = torch.compile(self._forward_teacher, fullgraph=True)
            self.extract_audio = torch.compile(self.extract_audio)
            self.masked_loss = torch.compile(self.masked_loss)

        except Exception as e:
            print(f"Warning: Could not compile operations: {e}")
            self.use_compiled_forward = False

    def _make_targets(self, layer_outputs: list[torch.Tensor]):
        """
        Predicting targets which are the average of multiple layers is more robust than
        predicting only the top most layer (K = 1) for most modalities.
        Args:
            layer_outputs: average_top_k_layers * (batch_size, n_patches, emb_dim)

        Returns:
            array of shape (batch_size, n_patches, emb_dim)
        """

        # They have for audioset -> instance_norm_target_layer: true
        # They have for audioset -> layer_norm_targets : true
        # So this is the way following the data2vec2 paper for audio.
        stacked_outputs = torch.stack(
            layer_outputs,
        )  # [num_layers, batch, seq_len, features]
        transposed = stacked_outputs.transpose(
            2, 3
        )  # [num_layers, batch, features, seq_len]

        # Apply instance norm to all layers simultaneously
        normalized = F.instance_norm(
            transposed
        )  # [num_layers, batch, features, seq_len]
        normalized = normalized.transpose(
            2, 3
        )  # [num_layers, batch, seq_len, features]

        # Compute mean across layers
        y = normalized.mean(dim=0)  # [batch, seq_len, features]
        return y

    @torch.no_grad()
    def _forward_teacher(self, x: torch.Tensor) -> torch.Tensor:
        # ChunkingEncoder doesn't expose layers, just use final output
        # TODO: try to get intermediate layers from chunking encoder
        if self.use_chunking_encoder:
            targets = self.teacher_encoder(x)
            return targets

        # Standard TransformerEncoder: iterate through layers
        layer_outputs = []
        for i, bl in enumerate(self.teacher_encoder.layers):  # type: ignore
            x: torch.Tensor = bl(x)
            if (
                len(self.teacher_encoder.layers) - i
                <= self.hparams.average_top_k_layers
            ):
                layer_outputs.append(x)

        if self.hparams.average_top_k_layers > 1:
            targets = self._make_targets(
                layer_outputs
            )  # (batch_size, n_patches, emb_dim)
        else:
            targets = layer_outputs[-1]
        return targets

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step with EMA teacher update."""
        outputs = self(**batch)
        loss = self.compute_loss(batch, outputs)

        # Logging
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/ema_decay", self._get_ema_decay(), on_step=True, sync_dist=True)
        # TODO: log more detailed metrics if needed

        # NaN check
        if torch.isnan(loss) or torch.isinf(loss):
            self.log("train/nan_loss", 1.0, on_step=True)
            return None

        # EMA teacher update (FP32 for stability)
        with torch.amp.autocast("cuda", enabled=False):
            self._step_teacher()

        return loss

    def compute_loss(self, batch: Dict[str, Any], outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract pre-computed loss from forward outputs and add load balancing loss.

        Args:
            batch: Input batch
            outputs: Model outputs including 'loss' and 'lb_loss'

        Returns:
            Total loss combining main loss and load balancing loss
        """
        main_loss = outputs["loss"]
        lb_loss = outputs.get("lb_loss", torch.tensor(0.0, device=main_loss.device))

        # Combine losses
        total_loss = main_loss + self.lb_loss_weight * lb_loss

        # Log separately
        if self.training:
            self.log("train/main_loss", main_loss, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/lb_loss", lb_loss, on_step=True, on_epoch=True, sync_dist=True)

        return total_loss

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.

        Overrides base class to support per-parameter learning rates when
        using ChunkingEncoder with lr_multipliers.
        """
        from model.models.WavHJepa.optim_utils import group_params_by_optim

        # Check if we're using chunking encoder with LR multipliers
        use_param_groups = (
            isinstance(self.encoder, ChunkingEncoder) and
            self.encoder.config.lr_multipliers is not None
        )

        if use_param_groups:
            # Use parameter groups for per-stage learning rates
            param_groups = group_params_by_optim(
                self,
                base_lr=self.learning_rate,
                base_wd=self.weight_decay
            )
        else:
            # Default: single parameter group (same as base class)
            param_groups = [p for p in self.parameters() if p.requires_grad]

        # Create optimizer
        # Note: When param_groups is a list of dicts, lr/weight_decay from groups take precedence
        # We still pass default values for any groups that don't specify them
        if self.optimizer_name.lower() == "adamw":
            if use_param_groups:
                # Parameter groups already have lr/weight_decay set, don't override
                optimizer = torch.optim.AdamW(
                    param_groups,
                    betas=(0.9, 0.999),
                    eps=1e-8
                )
            else:
                # Single parameter group, use global lr/weight_decay
                optimizer = torch.optim.AdamW(
                    param_groups,
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                    betas=(0.9, 0.999),
                    eps=1e-8
                )
        elif self.optimizer_name.lower() == "adam":
            if use_param_groups:
                optimizer = torch.optim.Adam(param_groups)
            else:
                optimizer = torch.optim.Adam(
                    param_groups,
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay
                )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        # Create scheduler (same as base class)
        if self.scheduler_name.lower() == "linear_warmup_cosine":
            def lr_lambda(step):
                if step < self.warmup_steps:
                    return float(step) / float(max(1, self.warmup_steps))
                progress = float(step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
                return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265359))))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            }
        elif self.scheduler_name.lower() == "constant":
            return optimizer
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler_name}")

    def masked_loss(self, pred, target, target_indices):
        """
        Calculates the masked loss using broadcasting to avoid memory-heavy repeats.

        pred:   Tensor of shape [(B * N), T, D]
        target: Tensor of shape [B, T, D]
        mask:   Tensor of shape [B, N, T]
        """

        B, N, _ = target_indices.shape
        D = pred.shape[-1]

        pred_reshaped = pred.view(B, N, -1, D)

        # This makes it broadcastable to the shape of pred_reshaped [B, N, T, D] during the loss

        target = repeat(target, "B T D -> B N T D", N=N)
        loss = self.loss_fn(pred_reshaped, target)  # -> Shape: [B, N, T, D]

        loss_per_timestep = loss.mean(dim=-1)  # -> Shape: [B, N, T]

        # No rearrange is needed for the mask.
        masked_loss_tensor = loss_per_timestep * target_indices  # -> Shape: [B, N, T]

        # Calculate the final mean loss over only the masked elements.
        total_loss = masked_loss_tensor.sum()
        indices_count = target_indices.sum()

        return total_loss / (indices_count + 1e-8)

    def forward(
        self,
        waveform: torch.Tensor,
        mask: torch.Tensor = None,
        context_mask: torch.Tensor = None,
        target_indices: torch.Tensor = None,
        ctx_and_target_masks: torch.Tensor = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for JEPA model with multiview support.

        Args:
            waveform: (B, C, T) or (B, T) - raw audio at sample_rate
            mask: (B, T) - optional padding mask (True = valid), unused for now
            context_mask: (B * nr_samples, n_patches) - pre-computed context mask from DataModule
            target_indices: (B * nr_samples, n_targets, n_patches) - pre-computed target indices from DataModule
            ctx_and_target_masks: (B * nr_samples, n_targets, n_patches) - pre-computed combined masks from DataModule
            **kwargs: Additional arguments

        Returns:
            Dict with keys:
                * loss: scalar loss value
                * local_features: (B, n_patches, emb_dim) - feature extractor output
                * contextual_features: encoder output for context tokens
                * preds: decoder predictions for target tokens
                * targets: teacher encoder targets
        """
        # Handle input shape: (B, T) -> (B, 1, T)
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)

        # Extract local features from waveform
        local_features = self.extract_audio(waveform)
        local_features = self.feature_norms(local_features)
        if self.post_extraction_mapper is not None:
            local_features = self.post_extraction_mapper(local_features)

        # NOW create multiview batch from encoded features
        # local_features: (B, T, D) -> (B * nr_samples, T, D)
        from model.models.utils import create_multiview_batch
        local_features_multiview = create_multiview_batch(
            local_features,
            nr_samples_per_audio=self.nr_samples_per_audio
        )

        # Use pre-computed masks from DataModule
        ctx_masks = context_mask
        target_indices_mask = target_indices
        ctx_and_target_masks_arg = ctx_and_target_masks

        # Add positional encoding
        local_features_multiview = local_features_multiview + self.pos_encoding_encoder

        # Student encoder forward
        contextual_features = self.encoder_forward(
            local_features_multiview, src_key_padding_mask=ctx_masks, output_hidden_states=self.output_hidden_states
        )

        # Get load balancing loss if using chunking encoder
        if self.use_chunking_encoder:
            lb_loss = self.encoder.get_load_balancing_loss(N=5.0)
        else:
            lb_loss = torch.tensor(0.0, device=waveform.device)

        # Extract context features and project to decoder dimension
        contextual_features_ctx = contextual_features[~ctx_masks]
        contextual_features_ctx = self.encoder_to_decoder_mapper(contextual_features_ctx)

        # Decoder forward
        preds = self.decoder_forward(
            contextual_features_ctx,
            ctx_masks,
            nr_targets=target_indices_mask.shape[1],
            src_key_padding_mask=ctx_and_target_masks_arg,
        )

        # Teacher forward (no gradients)
        x_targets = local_features_multiview.detach()
        targets = self._forward_teacher(x_targets)

        # Compute loss
        loss = self.masked_loss(preds, targets, target_indices_mask)

        return {
            "loss": loss,
            "lb_loss": lb_loss,
            "local_features": local_features,
            "contextual_features": contextual_features,
            "preds": preds,
            "targets": targets,
        }

    def decoder_forward(
        self,
        contextual_features: torch.Tensor,
        ctx_mask: torch.BoolTensor,
        nr_targets: int,
        src_key_padding_mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        B = ctx_mask.shape[0]
        # Prepare the mask tokens.
        tgt = self.mask_token.repeat(B, self.total_patches, 1).type_as(
            contextual_features
        )  # (B, seq_len, decoder_dim)
        tgt[~ctx_mask, :] = contextual_features.reshape(
            (-1, self.decoder_embedding_dim)
        )
        tgt = tgt.reshape((B, -1, self.decoder_embedding_dim))
        # Add positional encoding to the decoder
        tgt = tgt + self.pos_encoding_decoder

        # Repeat the context for every target, and absorb into batch dimension
        tgt = repeat(tgt, "B Seq Emb -> B T Seq Emb", T=nr_targets)
        tgt = rearrange(tgt, "B T Seq Emb -> (B T) Seq Emb")
        src_key_padding_mask = rearrange(src_key_padding_mask, "B T Seq1 -> (B T) Seq1")

        # Decoder only attends to context tokens and target mask tokens.
        tgt = self.decoder(tgt, src_key_padding_mask=src_key_padding_mask)
        preds = self.decoder_to_encoder_mapper(tgt)
        return preds

    # TODO use flex attention
    def encoder_forward(
        self,
        x_contexts: torch.Tensor,
        src_key_padding_mask: torch.BoolTensor | None = None,
        output_hidden_states: bool = False,
    ) -> torch.Tensor:
        if self.use_gradient_checkpointing and self.training:
            contextual_features = checkpoint(
                self.encoder, x_contexts, use_reentrant=False
            )
        else:
            contextual_features = self.encoder(
                x_contexts,
                src_key_padding_mask=src_key_padding_mask,
                output_hidden_states=output_hidden_states
            )

        return contextual_features

    @torch.inference_mode()
    def get_audio_representation(self, audio: torch.Tensor, lengths: torch.tensor, output_hidden_states: bool = False) -> torch.Tensor:
        # lengths is relative to the total length of audio input
        # Get the audio representatin of waveform x.
        self.eval()
        hidden_states = []
        local_features = self.extract_audio(audio)
        local_features = self.feature_norms(local_features)
        if self.post_extraction_mapper:
            local_features = self.post_extraction_mapper(local_features)
        local_features = local_features + self.pos_encoding_encoder
        hidden_states.append(local_features)

        if lengths is not None:
            T = local_features.shape[1]
            lengths = (lengths * T).long()  # Convert to absolute lengths
            mask = length_to_mask(lengths, T) # (B, T)
            mask = ~mask.bool()
        else:
            mask = None

        # Encoder and decoder forward
        contextual_features = self.encoder_forward(
            local_features,
            src_key_padding_mask=mask,
            output_hidden_states=output_hidden_states
        )
        return contextual_features
