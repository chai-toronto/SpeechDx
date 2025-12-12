from model.models import BaseAudioPretrainModel
from model.models.Sejal.Chunker import RoutingModule, ChunkLayer
from model.models.Sejal.WavLM import WavLM, WavLMConfig
import torch
import torch.nn.functional as F

from model.models.Sejal.sejal_modules import (
    MaskedChunkPredictor,
    mask_chunks,
    info_nce,
    local_smoothness,
    boundary_regularizers
)


class Config:
    def __init__(self, cfg=None):
        self.tau_info_nce: float = 0.1         # temperature for contrastive
        self.mask_ratio: float = 0.3         # fraction of chunks to mask
        self.lambda_ratio: float = 0.01
        self.lambda_entropy: float = 0.01
        self.lambda_smooth: float = 0.5
        self.lambda_coverage: float = 0.01
        self.lambda_masked: float = 1.0         # masked-chunk weight
        self.lambda_contr: float = 1.0        # contrastive weight

        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        self.__dict__.update(cfg)

class WavLMSejal(BaseAudioPretrainModel):
    def __init__(self, wavlm_cfg, sejal_cfg, **kwargs):
        super().__init__(**kwargs)
        # Config setup
        self.encoder = WavLM(wavlm_cfg)
        self.tau_info_nce = sejal_cfg.tau_info_nce
        self.mask_ratio = sejal_cfg.mask_ratio
        self.lambda_ratio = sejal_cfg.lambda_ratio
        self.lambda_smooth = sejal_cfg.lambda_smooth
        self.lambda_entropy = sejal_cfg.lambda_entropy
        self.lambda_coverage = sejal_cfg.lambda_coverage
        self.lambda_masked = sejal_cfg.lambda_masked
        self.lambda_contr = sejal_cfg.lambda_contr

        self.hidden = self.encoder.cfg.encoder_embed_dim

        self.router = RoutingModule(self.hidden)
        self.downsampler = ChunkLayer()
        self.masked_chunk_predictor = MaskedChunkPredictor(self.hidden)

    def forward(self, waveform: torch.Tensor, mask: torch.Tensor = None, **kwargs):
        # audio: (B, C, T) raw waveform
        # mask: (B, T) bool tensor where True indicates valid (non-padding) positions
        if waveform.ndim == 3:
            waveform = waveform.squeeze(1)  # (B, T)

        # Forward pass through WavLM encoder
        features, feat_mask = self.encoder.extract_features(waveform, mask)
        # features: (B, T_feat, D_feat), feat_mask: (B, T_feat)

        # Router determines chunk boundaries
        router_output = self.router(features, mask=feat_mask)
        boundary_prob = router_output.boundary_prob # (B, T_feat, 2)
        boundary_mask = router_output.boundary_mask # (B, T_feat)

        # Downsample features into chunks
        z_chunks, _, _, chunk_mask = self.downsampler(features, boundary_mask, mask=feat_mask)
        # z_chunks: (B, L, D), chunk_mask: (B, L) where True = valid chunk

        # Create masked input for prediction (will be done in compute_loss)
        # Here we just return the clean chunks

        return {
            "features": features,              # (B, T_feat, D) - frame embeddings before chunking
            "boundary_prob": boundary_prob[:, :, 1],  # (B, T_feat) - probability of boundary after frame
            "z_chunks": z_chunks,              # (B, L, D) - chunk embeddings
            "chunk_mask": chunk_mask,          # (B, L) - True = valid, False = padding
        }

    def compute_loss(self, batch, outputs):
        """
        Compute the full SSL loss with all components:
        1. Contrastive loss (InfoNCE between chunk views)
        2. Masked chunk prediction loss (MSE)
        3. Local smoothness loss (adjacent frame similarity)
        4. Boundary regularizers (entropy, coverage, smoothness)
        """
        features = outputs["features"]           # (B, T_feat, D)
        boundary_prob = outputs["boundary_prob"] # (B, T_feat)
        z_chunks = outputs["z_chunks"]           # (B, L, D)
        chunk_mask = outputs["chunk_mask"]       # (B, L) - True = valid, False = padding

        # Invert mask: sejal_modules expects True = padding
        z_mask = ~chunk_mask

        # LOSS 1: Contrastive loss between chunk views
        # Split chunks in half and create two views
        B, L, D = z_chunks.shape
        if L >= 2:
            z_q = z_chunks[:, :L//2, :].mean(dim=1)  # (B, D)
            z_k = z_chunks[:, L//2:, :].mean(dim=1)  # (B, D)
        else:
            # Fallback: use same representation (no contrast)
            z_q = z_k = z_chunks.mean(dim=1)

        loss_contr = info_nce(z_q, z_k, tau=self.tau_info_nce)

        # LOSS 2: Masked chunk prediction
        z_in, mask_bool = mask_chunks(z_chunks, z_mask, self.mask_ratio)  # (B, L, D), (B, L)
        z_pred = self.masked_chunk_predictor(z_in, z_mask)                # (B, L, D)
        mse = (z_pred - z_chunks).pow(2).sum(dim=-1)                      # (B, L)

        # Only compute loss on masked positions
        masked_den = mask_bool.sum().clamp_min(1)
        loss_masked = (mse[mask_bool]).sum() / masked_den

        # LOSS 3: Local smoothness - encourages similar adjacent frames to be in same chunk
        loss_smooth = local_smoothness(features, boundary_prob)

        # LOSS 4: Boundary regularizers
        boundary_hard = (boundary_prob > 0.5).float()
        regs = boundary_regularizers(
            boundary_prob,
            boundary_hard,
            length_target=20.0  # Target ~20 frames per chunk
        )
        loss_entropy = regs["loss_entropy"]
        loss_coverage = regs["loss_coverage"]
        loss_smooth_reg = regs["loss_smooth"]

        # Combine all losses
        total_loss = (
            self.lambda_contr * loss_contr +
            self.lambda_masked * loss_masked +
            self.lambda_smooth * loss_smooth +
            self.lambda_ratio * loss_coverage +
            self.lambda_entropy * loss_entropy +
            0.1 * loss_smooth_reg  # smoothness regularizer weight
        )

        # Store metrics for logging (accessed in training_step)
        self._last_metrics = {
            "loss_contr": loss_contr.detach(),
            "loss_masked": loss_masked.detach(),
            "loss_smooth": loss_smooth.detach(),
            "loss_entropy": loss_entropy.detach(),
            "loss_coverage": loss_coverage.detach(),
            "avg_chunks": (~z_mask).float().sum(dim=1).mean().detach(),  # Average chunks per sample
        }

        return total_loss

    def training_step(self, batch, batch_idx):
        """Override training step to log detailed metrics."""
        import sys

        rank = 0
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()

        # Heartbeat: entering training step
        if batch_idx % 10 == 0:
            print(f"[TRAIN_STEP] Rank {rank} | Step {batch_idx} | Entering training_step",
                  file=sys.stderr, flush=True)

        outputs = self(**batch)

        # Heartbeat: after forward pass
        if batch_idx % 10 == 0:
            print(f"[TRAIN_STEP] Rank {rank} | Step {batch_idx} | Forward pass complete",
                  file=sys.stderr, flush=True)

        loss = self.compute_loss(batch, outputs)

        # Log main loss
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # Log component losses
        if hasattr(self, '_last_metrics'):
            for key, value in self._last_metrics.items():
                self.log(f"train/{key}", value, on_step=True, on_epoch=True, sync_dist=True)

        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            self.log("train/nan_loss", 1.0, on_step=True)
            loss = next(self.parameters()).sum() * 0.0

        # Heartbeat: returning from training step (before backward)
        if batch_idx % 10 == 0:
            print(f"[TRAIN_STEP] Rank {rank} | Step {batch_idx} | Exiting training_step (loss={loss.item():.4f})",
                  file=sys.stderr, flush=True)

        return loss







