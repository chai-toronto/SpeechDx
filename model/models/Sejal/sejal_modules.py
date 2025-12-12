import torch, torch.nn as nn, torch.nn.functional as F
import random

# --- PRETRAIN PARAMS ---
SSL_EPOCHS       = 20

LR_SSL           = 1e-3
TAU_INFO_NCE     = 0.1          # temperature for contrastive
MASK_RATIO       = 0.3          # fraction of chunks to mask
HIDDEN           = 256          # chunk d_model
LAMBDA_SMOOTH    = 0.5
LAMBDA_RATIO     = 0.01
LAMBDA_ENTROPY   = 0.01
LAMBDA_COVERAGE  = 0.01
LAMBDA_MASKED    = 1.0          # masked-chunk weight
LAMBDA_CONTR     = 1.0          # contrastive weight


## loss functions and utilities
def l2_normalize(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def info_nce(z_q, z_k, tau=0.1):
    """
    In-batch InfoNCE. z_q, z_k: (B, D) normalized.
    """
    z_q = l2_normalize(z_q)
    z_k = l2_normalize(z_k)
    logits = (z_q @ z_k.t()) / tau        # (B,B)
    labels = torch.arange(z_q.size(0), device=z_q.device)
    return F.cross_entropy(logits, labels)

def mask_chunks(z_chunks, pad_mask, mask_ratio=0.3):
    """
    z_chunks: (B, L, D)
    pad_mask: (B, L)  True = PAD
    Returns masked_z, mask_bool (both exclude padding).
    """
    B, L, _ = z_chunks.shape
    device = z_chunks.device

    mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    valid = ~pad_mask  # only mask non-padding locations

    for b in range(B):
        valid_idx = valid[b].nonzero(as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        k = max(1, int(valid_idx.numel() * mask_ratio))
        perm = valid_idx[torch.randperm(valid_idx.numel(), device=device)[:k]]
        mask[b, perm] = True

    z_masked = z_chunks.clone()
    z_masked[mask] = 0.0
    return z_masked, mask

def local_smoothness(frames, p):
    """
    frames: (B,T,D) frame embeddings BEFORE chunking
    p:      (B,T)   boundary prob after frame t
    loss: mean_t (1 - p_t) * (1 - cosine(h_t, h_{t+1}))
    """
    if frames.size(1) < 2:
        return frames.new_zeros(())
    h0, h1 = frames[:, :-1, :], frames[:, 1:, :] # shifted versions of frames array for computing similarity between adjacent frames
    cos = F.cosine_similarity(h0, h1, dim=-1)        # (B,T-1)
    smooth = (1.0 - cos).clamp_min(0.0)
    w = 1.0 - p[:, :-1].clamp(0, 1)                  # (B,T-1)
    return (w * smooth).mean()

def boundary_regularizers(start_prob: torch.Tensor,
                        start_hard: torch.Tensor,
                        length_target: float | None = 20.0,
                        eps: float = 1e-9):
    #Compute three auxiliary losses for stability: entropy, coverage, smoothness.
    B, T = start_prob.shape
    p = start_prob[:, 1:].clamp(eps, 1 - eps)  #avoid log(0); ignore first frame

    ent = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))  # [B,T]
    loss_entropy = -ent.mean() #Binary entropy per frame; we maximize entropy - equivalent to minimizing -entropy
    num_starts = start_hard.sum(dim=1)  # [B] number of segments per sequence

    if length_target is not None and length_target > 0:
        seg_target = T / length_target #desired number of segments based on target length
        loss_coverage = ((num_starts - seg_target) ** 2).mean() / (seg_target + 1e-6) #MSE between actual and target number of segments, normalized
    else:
        loss_coverage = torch.tensor(0.0, device=start_prob.device)


    smooth = F.mse_loss(p[:, 1:], p[:, :-1])  # encourage smoothness over time and discourage rapid fluctuations

    return {
        "loss_entropy": loss_entropy,
        "loss_coverage": loss_coverage,
        "loss_smooth": smooth,
    }

def encode_to_chunks(model, wav):
    """
    Change this according to your model structure (I made this for VGGish to match Maryam's model).
    wav: (B, T_wav).
    Returns:
      frames:     (B, T, Df)  encoder frame embeddings (pre-chunk)
      start_prob: (B, T)
      chunk_embs: (B, L, Dc)
      chunk_mask: (B, L)  True=PAD
    """
    # ---- replicate your model’s forward stages but stop before classifier
    mel   = model.frontend(wav)                # (B,1,T_mel,64)
    frames = model.encoder(mel)                # (B,T,Df)  (VGGish patches or SSM frames)
    start_prob = model.router(frames)          # (B,T)
    
    chunk_embs, chunk_mask = model.soft_pool(frames, start_prob)  # implements the downsampler
    return frames, start_prob, chunk_embs, chunk_mask


## lightweight transformer for masked chunk prediction
class MaskedChunkPredictor(nn.Module):
    """
    Predicts original chunk embeddings from context with masked chunks = 0.
    Input/Output dim == chunk dim.
    """
    def __init__(self, d_model=HIDDEN, nhead=4, num_layers=2, ff=2):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=d_model*ff, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, z, pad_mask):
        # z: (B,L,D) with masked positions already zeroed
        pred = self.proj(self.enc(z, src_key_padding_mask=pad_mask)) # provide a src_key_padding_mask according to padding if needed
        return pred


## SSL training step
def ssl_step(model, predictor, batch_wav):
    """
    model: main model (encoder+router+downsampler)
    predictor: MaskedChunkPredictor
    batch_wav: (B, T_wav), T_wav is raw waveform length
    Returns total_ssl_loss, dict_of_terms
    """
    # ----- Forward to frames/chunks
    frames, start_prob, z_chunks, z_mask = encode_to_chunks(model, batch_wav)  # (B,T,Df), (B,T), (B,L,Dc), (B,L)

    # LOSS 1: Within-chunk contrastive loss ----- 
    # Make two “views” of each chunk by simple dropout on tokens (cheap) OR split even/odd chunks.
    # OR (here) average first half and second half separately (if L>=2); else fallback to mean.
    B, L, D = z_chunks.shape
    if L >= 2:
        z_q = z_chunks[:, :L//2, :].mean(dim=1)
        z_k = z_chunks[:, L//2:, :].mean(dim=1)
    else:
        z_q = z_k = z_chunks.mean(dim=1)

    loss_contr = info_nce(z_q, z_k, tau=TAU_INFO_NCE)

    # LOSS 2: Masked-chunk prediction (MSE on masked positions) ----- 
    z_in, mask_bool = mask_chunks(z_chunks, z_mask, MASK_RATIO)               # (B,L,D), (B,L)
    z_pred = predictor(z_in, z_mask)                               # (B,L,D)
    mse = (z_pred - z_chunks).pow(2).sum(dim=-1)                      # (B,L)
    # Only masked positions
    masked_den = mask_bool.sum().clamp_min(1)
    loss_masked = (mse[mask_bool]).sum() / masked_den

    # LOSS 3: Local smoothness on frame embeddings ----- 
    loss_smooth = local_smoothness(frames, start_prob)

    # LOSS 4: Router regularizers (reuse your function if available) ----- 
    # If you have boundary_regularizers(start_prob, start_hard, length_target=...), plug it here.

    regs = boundary_regularizers(start_prob, (start_prob > 0.5).float(), length_target=24)
    loss_ratio    = regs["loss_coverage"]      # proxy for length/coverage
    loss_entropy  = regs["loss_entropy"]
    loss_cov_smooth = 0.1 * regs["loss_smooth"]

    # ----- (5) Total SSL loss
    total = (
        LAMBDA_CONTR   * loss_contr +
        LAMBDA_MASKED  * loss_masked +
        LAMBDA_SMOOTH  * loss_smooth +
        LAMBDA_RATIO   * loss_ratio +
        LAMBDA_ENTROPY * loss_entropy +
        LAMBDA_COVERAGE* loss_cov_smooth
    )

    logs = dict(
        loss_total=float(total.item()),
        loss_contr=float(loss_contr.item()),
        loss_masked=float(loss_masked.item()),
        loss_smooth=float(loss_smooth.item()),
        loss_ratio=float(loss_ratio.item()),
        loss_entropy=float(loss_entropy.item()),
    )
    return total, logs



