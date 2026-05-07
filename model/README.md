# `model/` — encoders, probes, pooling

This directory holds three families of `nn.Module`s:

| Family            | Files                                                    | Role                                              |
|-------------------|----------------------------------------------------------|---------------------------------------------------|
| Encoder wrappers  | `wavlm.py`, `w2v2.py`, `hubert.py`, `ast.py`, `whisper.py`, `clap.py`, `mms.py`, `audiomae.py`, `emotion2vec.py`, `opera.py`, `qwen3_voice.py`, `wav_jepa.py`, … | Wrap a frozen pretrained encoder; expose `(B, T, D)` features (or layer-wise tuples). |
| Probe heads       | `probe.py`                                               | `LinearProbe`, `TemporalProbe`, `LayerTemporalProbe`, `EyeProbe`, `XTTSProbe`. |
| Poolers           | `pool.py`                                                | `AvgTPool`, `AttentiveTemporalPool`, `ASP`, `LayerWeightedAvgPool`. |

`stub_encoder.py` is a placeholder used by the orchestrator's reader-only
fast path: when training reads from a warm cache, the heavy real encoder
is replaced with this no-op so the multi-GB weights never load.

## Encoder contract

The expected interface is intentionally minimal:

```python
class MyEncoder(nn.Module):
    def __init__(self, ssl_encoder_source, freeze_encoder=True,
                 output_hidden_states=False, sample_rate=16000):
        ...

    def forward(self, waveform, lengths=None):
        # waveform: (B, T_audio) at self.sample_rate
        # lengths:  (B,) relative lengths in [0, 1] — fraction of the padded
        #           batch length (SpeechBrain convention), not absolute samples
        # returns:  (B, T, D)  if output_hidden_states is False
        #           tuple of (B, T, D) per layer  if output_hidden_states is True
        ...
```

The probe head consumes the output and is responsible for any temporal
or layer pooling; the encoder does not pool.

## Adding a new encoder

1. Create `model/<name>.py` implementing the contract above. Most
   encoders boil down to a `transformers.AutoModel.from_pretrained(...)`
   call followed by a `.last_hidden_state` extraction; see
   `model/wavlm.py` for the reference pattern.
2. Create the matching yaml at `sdx/configs/encoders/<name>.yaml`.
   It declares the encoder's metadata and constructs the module:

   ```yaml
   sample_rate:       16000
   feature_dim:       1024
   num_layers:        24
   layer_dim:         1
   max_length:        300
   min_length:        1

   encoder: !new:model.<name>.<MyEncoder>
     ssl_encoder_source: "<huggingface-id>"
     freeze_encoder:     True
     output_hidden_states: False
     sample_rate: !ref <sample_rate>
   ```

3. Register the encoder in `sdx/configs/registry.yaml` under
   `encoders:`:

   ```yaml
   encoders:
     ...
     <model_name>: <name>.yaml
   ```

   `<model_name>` is the public name — it's what `--encoder` accepts and
   what shows up in the experiment folder name.

4. Verify:

   ```bash
   python -c "from sdx import registry; print('<model_name>' in registry.encoders())"
   python -m sdx status   # the new encoder column appears
   ```

You do **not** need to edit the orchestrator scripts.

## Probe / pool contract

A probe takes encoder output and `lengths` (relative wrt batch's audio length), returns `(B, num_labels)`
logits. Probes that pool temporally accept either a single tensor or a
layer-wise tuple; layer probes additionally accept the layer dimension
and apply a learned softmax. See `probe.py` and `pool.py` for the
exact signatures — these are stable.

