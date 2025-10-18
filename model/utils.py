import torch
def masked_normalize(x, mask, eps=1e-8):
    """
    Normalize x (B, D) along the batch axis using only valid rows indicated by mask.

    mask: (B,D) tensor of 0/1s
    returns normalized x (same shape)
    """
    assert mask.shape == x.shape
    mask_f = mask.float()

    # compute masked mean and std
    valid_sum = mask_f.sum(dim=1)        # (B, 1)
    mean = (x * mask_f).sum(dim=1) / valid_sum.clamp_min(1.0) # (B)
    mean = mean.unsqueeze(1)             # (B, 1)

    var = ((x - mean) ** 2 * mask_f).sum(dim=1) / valid_sum.clamp_min(1.0) # (B)
    std = var.sqrt().clamp_min(eps) # (B)
    std = std.unsqueeze(1)             # (B, 1)

    # normalize
    x_norm = (x - mean) / std
    return x_norm
