# LoRA adaptation track

The benchmark freezes every encoder and trains only a linear probe. This track
asks a different question: how much does an encoder improve if its **last few
transformer blocks** are allowed to adapt to the task? It adds rank-8 LoRA
adapters to those blocks' query and value projections and trains them together
with the benchmark's linear head. Every original weight stays frozen.

Everything lives in `sdx/lora/` and `scripts/lora_*.py`. Nothing here is
imported by the frozen-probe pipeline, so the benchmark's own numbers cannot
change because of it.

## Supported encoders

Model-specific support ships for the three best encoders on the main board.
The `--encoder` names are the ones in `sdx/configs/registry.yaml`.

| `--encoder`  | model                         | blocks | width | adapted blocks | cut | adapter params | runner       |
|--------------|-------------------------------|-------:|------:|---------------:|----:|---------------:|--------------|
| `whisper`    | Whisper-large-v3 encoder      | 32     | 1280  | 2              | 30  | 81,920         | `sequential` |
| `wavlm`      | WavLM-Large                   | 24     | 1024  | 3              | 21  | 98,304         | `wavlm`      |
| `qwen3voice` | Qwen3-TTS-Tokenizer-12Hz      | 8      | 512   | 6              | 2   | 98,304         | `delegate`   |

Each one passes `scripts/lora_parity.py` on the real weights: the split
encoder reproduces the frozen encoder with max |diff| = 0 in fp32 on CPU.
Other encoders need a spec first; see [Adding an encoder](#adding-an-encoder).

## How it works

```
audio ─► frozen prefix: blocks[0 : cut]  ─►  boundary cache (fp16, per chunk)
                                              │
                   training reads only this ◄─┘
                                              ▼
                  adapted tail: blocks[cut :] + final norm   (LoRA on q, v)
                                              ▼
              recording-level mean over frames ─► linear head ─► loss
```

1. **Plan** (`targets.plan_adapters`). Walk backward from the last block and
   add blocks while `rank * (in + out)` summed over q and v stays within
   100,000 parameters. The first adapted block is the **cut**.
2. **Warm the boundary** (`scripts/lora_warm.py`). Run the encoder only up to
   the cut, using the same audio → augmentation → chunking pipeline as the
   benchmark warm (`sdx.warm._build_warm_dynamic_items`). Store the activation
   entering block `cut` for every chunk, in
   `<cache_dir>/{train,val}/boundary_L<cut>/cache.hdf5`, next to the
   benchmark's `single_avg/`. The prefix never runs again.
3. **Train** (`scripts/lora_train.py`). Replay `blocks[cut:]` plus the
   post-loop steps (final layer norm) per chunk, with adapters injected.
   Concatenate the chunk outputs, take an unmasked mean over frames exactly as
   the frozen warm does, then apply the linear head. A runtime hook raises if
   any prefix block is ever executed during training.

A fresh adapter has `B = 0`, so at step 0 the tail is bit-identical to the
frozen encoder, and training starts from the frozen-probe setting.

## Protocol

These are the values behind the leaderboard's LoRA rows. They are defaults in
the code, not suggestions to tune per task.

| knob | value |
|---|---|
| adapters | rank 8, alpha 16 (scale 2), dropout 0.05, no bias; q and v projections only (K is never adapted) |
| budget | at most 100,000 adapter parameters, allocated backward from the final block |
| readout | recording-level global mean pool + linear head, same as the benchmark |
| data | the benchmark's manifests, preprocessing, augmentation versions, losses, auto class weights (unweighted loss for validation and test) |
| optimizer | AdamW, two groups: LoRA lr `1e-4`, weight decay 0; head lr `9.476e-4`, weight decay `0.012272` |
| schedule | per-group linear decay to lr/10 over the run |
| training | 15 epochs, batch 16 (per-recording gradient accumulation), no early stopping, no HP search |
| selection | checkpoint with the best validation loss; one test evaluation |
| CV tasks | the same, per fold; report the plain mean ± std of each fold's best-validation metric, as `sdx/train_cv.py` does |

The head settings are the most frequent winner of the benchmark's own
5-trial probe search (307 of 412 frozen-probe cells).

## Running it

```bash
# 0. Check the split once per encoder (CPU, no data needed).
uv run python scripts/lora_parity.py --encoder whisper

# 1. Build the boundary cache for a (task, encoder). Tasks on the same dataset
#    share it: pass the siblings with --also-task so one cache covers all uids.
uv run python scripts/lora_warm.py --task T25 --also-task T26 --encoder wavlm --device cuda

# 2. Train + evaluate one cell. Held-out tasks write result.json,
#    best_adapter.pt and test predictions to exps/lora/<task>/<encoder>-lr<lr>/.
uv run python scripts/lora_train.py --task T25 --encoder wavlm --device cuda
```

Useful flags:

- `lora_warm.py --precision fp16` autocasts the prefix on CUDA. The default,
  `fp32`, matches the benchmark warm; the leaderboard rows were warmed in
  fp16. Precision is part of the cache identity, so the two never mix in one
  file.
- `lora_warm.py --limit N` / `lora_train.py --limit N` cap the uids for smoke
  tests.
- `lora_train.py --control` keeps the adapters frozen at `B = 0`, which trains
  only the head under this protocol. `LoRA − control` isolates the effect of
  adaptation. `control − frozen board` measures the protocol difference
  (15 epochs and one configuration here, vs 50 epochs and a 5-trial search on
  the board).
- `lora_train.py --no-test` skips the test pass. Use it when comparing LoRA
  learning rates on validation loss, so the test set is touched once.
- `lora_train.py --cache-root DIR` reads the cache from
  `DIR/<dataset>/<encoder>/{train,val}/`, e.g. a node-local SSD copy. Training
  reads every frame of every recording each epoch, so network filesystems can
  dominate the run time.
- `SDX_ONLY_FOLD=<i>` (CV tasks) trains only fold `i` and saves it to
  `fold_<i>.json`. Run one job per fold in parallel, then run once without the
  variable to aggregate them into `result.json`.
- `SDX_LORA_PIN_REVISION=<sha>` records the checkpoint revision in the cache
  metadata.

Things to plan for:

- **Disk.** The boundary cache keeps every frame at the cut layer (fp16), so
  it is about as large as an ASP `single/` cache: small for short-clip
  corpora, hundreds of GB or more for the largest ones.
- **Regression at 15 epochs.** T2 (PHQ-8) and T8 (MMSE) underfit this budget
  in both the LoRA and control arms, so their numbers measure the budget, not
  adaptation.
- **CV folds.** CV fold assignment depends on the scikit-learn version
  (1.7.x and 1.8.x give different folds for the same seed). Use the pinned
  environment (`uv.lock` / `requirements.txt`, scikit-learn 1.8.0) so LoRA
  cells share folds with the board, or compare against a frozen probe you ran
  in the same environment.

## Adding an encoder

Adding an encoder is usually a spec, not new code. An `EncoderTargetSpec`
declares where the blocks and projections live and how to replay the tail.

1. **Find the pieces.** Load the wrapper from its encoder yaml and
   `print(wrapper)`. Paths are resolved against the `!new:model.*` wrapper
   object:
   - `blocks`: the `ModuleList` of transformer blocks, e.g. `model.encoder.layers`.
   - `attn`: the self-attention module inside one block, e.g. `attention`.
   - `q` + `v`: separate query/value `nn.Linear`s, **or** `qkv` for one fused
     `[q; k; v]` linear (timm ViTs). Fused projections get a slice-aware
     adapter that leaves K alone. A fused layer declared as `q=` is refused
     loudly.
   - `post`: what the wrapper applies between the last block and the tensor it
     returns. Usually the final layer norm (a module path), plus stateless ops
     such as `DropTokens(n)` (drop CLS/extra tokens) or `MeanTokens()`.
   - `post_if`: a boolean attribute that gates `post`, for encoders that apply
     the final norm after the loop only in their pre-LN variant.
   - `block_kwargs`: static keyword arguments every block call needs.
2. **Pick a runner.** `sequential` covers any block that you can call as
   `block(hidden, **block_kwargs)` and whose output is the hidden state (or a
   tuple whose first item is the hidden state). Blocks that need state their
   parent builds need their own runner:
   - `wavlm`: rebuilds WavLM's relative position bias from sequence length.
   - `delegate`: truncates the parent's `layers` and calls the parent, which
     rebuilds masks / rotary embeddings itself (Mimi-style transformers).
   - Anything else (e.g. ALiBi biases in data2vec 2.0 / emotion2vec): subclass
     `EncoderTail`, override `run_blocks`, and add it to `RUNNERS` in
     `sdx/lora/tails.py`. Rebuild per-length state in `run_blocks` instead of
     caching it; it is a pure function of length and far too large to store.
3. **Make sure attention calls the projection modules.** LoRA wraps `q_proj` /
   `v_proj`. If the attention reads `q_proj.weight` directly (HF's WavLM does,
   via `F.multi_head_attention_forward`), the adapter crashes or is silently
   bypassed. `WavLMTail` rebinds its blocks' attention to call the modules. Do
   the same in your runner if needed.
4. **Register it** in `SPECS` (`sdx/lora/targets.py`) under the encoder's
   `registry.yaml` name. Optionally add a realistic chunk length to
   `DEFAULT_SECONDS` in `scripts/lora_parity.py`.
5. **Run the parity check** until every line passes:

   ```bash
   uv run python scripts/lora_parity.py --encoder <name>
   ```

   It checks that `tail(capture(x)) == encoder(x)` within fp32 tolerance; that
   a fresh adapter is an exact no-op; that the plan is within budget; that
   gradients reach every adapter and nothing else (this is what catches an
   attention path that bypasses the adapters); and that an adapter checkpoint
   round-trips. An encoder is supported only once this passes. Build caches
   after that, never before: the cut decides what the cache stores.

Starting points for other benchmark encoders. These specs passed the same
check against the wrappers they were written for. Re-run it against yours.

```python
# AST: HF ViT-style naming; sequential runner.
"ast": EncoderTargetSpec(
    blocks="model.encoder.layer", attn="attention.attention",
    q="query", v="value", block_kwargs={"head_mask": None},
    post=("model.layernorm",)),

# wav2vec2 family (w2v2 / HuBERT / MMS): stable-layer-norm encoders apply the
# final norm after the loop; the post-LN variant applies it before (prefix).
"hubert": EncoderTargetSpec(
    blocks="model.encoder.layers", attn="attention", q="q_proj", v="v_proj",
    block_kwargs={"attention_mask": None},
    post=("model.encoder.layer_norm",),
    post_if="model.config.do_stable_layer_norm"),

# AudioMAE (timm ViT, fused qkv): the wrapper drops the CLS token after norm.
"audiomae": EncoderTargetSpec(
    blocks="encoder.blocks", attn="attn", qkv="qkv",
    post=("encoder.norm", DropTokens(1))),

# OPERA-GT (timm blocks, fused qkv): pools BEFORE its norm.
"opera_gt": EncoderTargetSpec(
    blocks="model.blocks", attn="attn", qkv="qkv",
    post=(DropTokens(1), MeanTokens(), "model.norm")),
```

emotion2vec (data2vec 2.0 `AltBlock`s, fused qkv) also needs a runner that
rebuilds the ALiBi bias from sequence length (step 2).

## Tests

```bash
uv run python -m pytest tests/test_lora_*.py      # synthetic modules, no downloads
uv run python scripts/lora_parity.py              # real weights for every registered spec
```
