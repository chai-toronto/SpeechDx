import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.dataio.dataio import length_to_mask


class ChunkPool(nn.Module):
    def __init__(self, d_model, d_out, threshold=0.5, aggregate="mean"):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(d_model))
            self.k_proj.weight.copy_(torch.eye(d_model))
        self.q_proj.weight._no_reinit = True
        self.k_proj.weight._no_reinit = True
        self.up_proj = nn.Linear(d_model, d_out)
        self.d_out = d_out
        self.d_in = d_model
        self.threshold = threshold

        assert aggregate in ["mean"], "Only mean aggregation is currently implemented."
        self.aggregate = aggregate

    def forward(self, x, lengths = None, pad_mask = None):
        """
        x: (B, T_max, D) padded with zeros in the tail
        lengths: (B,) relative lengths (to T_max) per sequence. If None, we assume no padding.

        Returns:
            sorted_hidden: (B, T_chunk, D) reordered hidden states with boundaries first
            chunk_mask: (B, T_chunk) boolean mask where True indicates non-boundary tokens
            boundary_mask: (B, T_max) boolean mask where True indicates non-boundary tokens
            boundary_prob: (B, T_max, 2) probabilities for non-boundary and boundary classes, respectively
        """
        B, T, D = x.shape

        if lengths is not None and pad_mask is not None:
            raise ValueError("Cannot provide both lengths and pad_mask; they are redundant.")

        if lengths is not None and pad_mask is None:
            T = x.shape[1]
            lengths = (lengths * T).long()  # Convert to absolute lengths
            pad_mask = ~length_to_mask(lengths).bool() # (B, T), True for pads

        # Cosine similarity between consecutive tokens
        q = F.normalize(self.q_proj(x[:, :-1]), dim=-1)  # (B, L-1, D)
        k = F.normalize(self.k_proj(x[:, 1:]), dim=-1)  # (B, L-1, D)
        cos_sim = (q * k).sum(dim=-1)  # (B, L-1)

        # Convert to boundary score: high cosine sim � low boundary prob
        boundary_score = (1 - cos_sim) / 2  # (B, L-1), range [0, 1]

        boundary_score = F.pad(boundary_score, (1, 0), value=1.0)  # First token always boundary

        boundary_prob = torch.stack(((1 - boundary_score), boundary_score), dim=-1)

        selected_idx = boundary_prob[:, :, 1] > self.threshold  # (B, L) bool

        # selected_idx = pick_with_spacing_mask_batched(
        #     boundary_prob[:, :, 1],
        #     threshold=self.threshold,
        #     k=self.min_chunk_size
        # )  # (B, L) bool

        nonboundary_mask = selected_idx != 1  # (shape hidden_states.shape[:-1])

        # Handle padding: force padded positions to NOT be boundaries
        if pad_mask is not None:
            nonboundary_mask = nonboundary_mask | pad_mask

        # Assign chunk IDs via cumsum on boundary mask
        boundary_mask = ~nonboundary_mask  # True at boundaries
        chunk_ids = boundary_mask.long().cumsum(dim=1)  # (B, T), 1-indexed

        # Zero out padded positions so they don't contribute
        if pad_mask is not None:
            chunk_ids = chunk_ids * (~pad_mask).long()

        # Count chunks per batch
        num_chunks = chunk_ids.max(dim=1).values  # (B,)
        max_chunks = num_chunks.max().item()

        if max_chunks == 0:
            raise ValueError("No boundaries detected in any sequence.")

        # Mean-pool all timesteps within each chunk via scatter
        chunk_sum = torch.zeros(B, max_chunks + 1, D, device=x.device, dtype=x.dtype)
        chunk_sum.scatter_add_(1, chunk_ids.unsqueeze(-1).expand_as(x), x)
        chunk_sum = chunk_sum[:, 1:]  # drop the 0-bucket (padding)

        chunk_count = torch.zeros(B, max_chunks + 1, device=x.device, dtype=x.dtype)
        chunk_count.scatter_add_(1, chunk_ids, torch.ones_like(chunk_ids, dtype=x.dtype))
        chunk_count = chunk_count[:, 1:]  # drop the 0-bucket

        sorted_hidden = chunk_sum / chunk_count.unsqueeze(-1).clamp(min=1)

        # True = padding chunk slots (to be masked out)
        nonchunk_mask = torch.arange(max_chunks, device=x.device)[None, :] >= num_chunks[:, None]

        sorted_hidden = self.up_proj(sorted_hidden * (~nonchunk_mask).unsqueeze(-1).float())

        return sorted_hidden, nonchunk_mask, nonboundary_mask, boundary_prob


def pick_with_spacing_mask_batched(x: torch.Tensor, threshold: float, k: int) -> torch.Tensor:
    """
    x: (B, T)
    Returns: (B, T) bool mask of picked indices.

    Rule per row:
      - pick index 0 (always)
      - then greedily pick the next index i such that:
           x[b, i] > threshold  AND  i - last_pick >= k
        (equivalently i >= last_pick + k)

    Note: if k == 0, we still force progress to the right by using step=max(k,1),
    otherwise the greedy rule could re-pick the same index forever.
    """
    assert x.dim() == 2
    B, T = x.shape
    device = x.device

    step = max(int(k), 1)

    # Candidates: (B, T)
    cand = x > threshold

    # Build next-true lookup:
    # next_idx[b, t] = smallest j >= t with cand[b, j]=True, else T
    idx = torch.arange(T, device=device).view(1, T).expand(B, T)  # (B, T)
    cand_pos = torch.where(cand, idx, torch.full_like(idx, T))    # True -> its index, False -> T
    next_idx = torch.cummin(cand_pos.flip(-1), dim=-1).values.flip(-1)  # (B, T)

    # Output mask
    picked = torch.zeros((B, T), device=device, dtype=torch.bool)
    picked[:, 0] = True  # always pick first one

    cur = torch.zeros(B, device=device, dtype=torch.long)   # last picked index
    start = cur + step                                     # next allowed starting point

    done = torch.zeros(B, device=device, dtype=torch.bool)

    # Safe upper bound on number of iterations (loop over picks, not over T)
    max_steps = (T + step - 1) // step + 2

    for _ in range(max_steps):
        done = done | (start >= T)
        if done.all():
            break

        # gather next candidate at or after start (clamp only to keep gather in-bounds)
        start_clamped = start.clamp(max=T - 1)
        nxt = next_idx.gather(1, start_clamped[:, None]).squeeze(1)  # (B,)

        has = (nxt < T) & (~done)
        if not has.any():
            break

        # mark picks
        picked[has, nxt[has]] = True

        # update last pick and next start
        cur = torch.where(has, nxt, cur)
        start = cur + step

        # rows that failed to find a next candidate are done
        done = done | (~has)

    return picked

if __name__ == "__main__":
    # Simple test
    # B, T, D = 2, 10, 4
    # x = torch.randn(B, T, D)
    x = torch.Tensor([[1, 0],
                      [-1, 0]])
    x = x.unsqueeze(0) # (1, 2, 2)

    # lengths = torch.tensor([1.0, 0.8])
    lengths = None

    chunk_pool = ChunkPool(d_model=2, d_out=4)
    sorted_hidden, chunk_mask, boundary_mask, boundary_prob = chunk_pool(x, lengths)

    print("Input shape:", x.shape)
    print("Sorted hidden shape:", sorted_hidden.shape)
    print("Chunk mask shape:", chunk_mask.shape)
    print("Boundary mask shape:", boundary_mask.shape)
    print("Boundary prob shape:", boundary_prob.shape)
