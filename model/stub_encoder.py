import torch.nn as nn


class StubEncoder(nn.Module):
    """No-op encoder used by reader-only jobs whose embeddings come from cache.

    Satisfies HyperPyYAML !new: instantiation and the `output_hidden_states`
    attribute reads in dataio_prep / brain.compute_forward without paying the
    cost of loading the real (often multi-GB) encoder. forward() is intentionally
    fatal — if it ever fires, the caller mis-classified the job as a reader.
    """

    def __init__(self, output_hidden_states: bool = False):
        super().__init__()
        self.output_hidden_states = output_hidden_states

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "StubEncoder.forward called — this job was launched as a reader "
            "(warm_cache=False) but something tried to extract embeddings. "
            "Re-run with warm_cache=True to use the real encoder."
        )
