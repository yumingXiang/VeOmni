# Usage of dyn_bsz

## Background of VeOmni data packing

In LLM or VLM training, we usually adopt batch training, and the sequence lengths of different samples often vary, for example:

* S1 = [A B C D] (len 4)
* S2 = [E F] (len 2)
* S3 = [G H I] (len 3)
* S4 = [J] (len 1)

To form a batch, a common practice is to pad all sequences to the maximum length within the batch:

* S1_padding = [A B C D]
* S2_padding = [E F PAD PAD]
* S3_padding = [G H I PAD]
* S4_padding = [J PAD PAD PAD]

This approach introduces several serious problems:
1. **A large amount of meaningless computation**. Transformer attention and FFN layers still perform computations on PAD tokens.
2. **Unnecessary memory consumption**. Padding tokens also occupy GPU memory.

The core idea of VeOmni data packing is:

Before computation, remove padding tokens from the tensors and perform computation only on real tokens.

The specific procedure is as follows:

1. Within a batch, flatten all real tokens into a single contiguous sequence.
2. Record the token boundaries of each sample using cu_seq_lens_q and cu_seq_lens_k.
3. In Attention (using FlashAttention or FlexAttention) and FFN, compute only on the real tokens.

## Background of dyn_bsz

In distributed training, data packing may cause the sequence lengths to be different across ranks. Suppose there are two ranks: S1 and S2 are on rank0, and S3 and S4 are on rank1:

* Rank0: [A B C D E F]
* Rank1: [G H I J]

Rank1 will finish the forward and backward computation earlier than rank0, and will wait for rank0 to complete before performing gradient communication. This introduces unnecessary idle time. A common solution is to introduce a dynamic batchsize (dyn_bsz), where different numbers of samples are concatenated according to a target budget so that the total sequence length of a batch is around the target budget. This approach is very common in pretraining.

However, for SFT tasks, we care more about how many epochs are completed. In this case, it is recommended to disable dyn_bsz.

So the difference is:

**dyn_bsz ON: batch size varies to hit a token budget.**

**dyn_bsz OFF: batch size is fixed, but still uses position_ids packing (no padding).**

## Usage

### dyn_bsz = True (Recommended for Pretraining)

Goal: pack multiple samples into one long sequence and choose how many samples to fit per batch based on token budget.

Example packed batch (one micro‑batch):

* input_ids = [A B C D E F G H I J]
* position_ids = [0 1 2 3 0 1 0 1 2 0]
* (No padding added.)

dyn_bsz decides how many samples to pack so total tokens ~ target budget. If your target is, say, 10 tokens, it packs all 4 here.

`train.dyn_bsz_count_mode` controls how that budget is measured:

* `total` (default): count all packed tokens via `attention_mask.sum()`. This matches the physical sequence budget and preserves legacy behavior.
* `effective`: count only tokens with `labels != IGNORE_INDEX`. This is useful for prompt-heavy SFT data where DP ranks can have similar packed lengths but very different amounts of loss-contributing tokens.

When `dyn_bsz_count_mode=effective`, sample selection targets the effective-token budget while also enforcing a hard physical-token cap of `ceil(micro_batch_size * max_seq_len * train.dyn_bsz_physical_overflow_ratio)` for each micro batch. The ratio defaults to `1.5`, so effective-token batches can be physically longer than the legacy total-token budget while still bounding prompt-heavy examples with very few labels. Set the ratio closer to `1.0` for a tighter OOM guard, or higher for more effective-token balancing headroom. A single sample can still be longer than this cap by itself, so keep preprocessing/truncation aligned with `data.max_seq_len`.

### dyn_bsz = False (Recommended for SFT)

Goal: still pack with position_ids, but batch size is fixed in number of samples, not by tokens. The num of samples in a batch is determined by the `micro_batch_size`.

Example fixed batch size = 2 samples:

* Batch 1: S1 + S2
* input_ids = [A B C D E F]
* position_ids = [0 1 2 3 0 1]
* Batch 2: S3 + S4
* input_ids = [G H I J]
* position_ids = [0 1 2 0]

## Padding packed inputs (pad_packed_input)
`pad_to_length` can pad the packed sequence to a fixed length. This is useful to avoid uneven lengths that can trigger kernel recompilation.
- Related GitHub issue: [#402](https://github.com/ByteDance-Seed/VeOmni/issues/402)
`pad_to_length` is useful only when `dyn_bsz` = True, all the packed sequences in a batch will be padded to the `max_seq_len`.
It is also required when `model.accelerator.torch_compile.enable=True`, because per-block
`torch.compile` needs stable packed shapes to avoid repeated recompilation. If
CUDA Graphs are explicitly enabled, stable shapes are also required for useful
graph replay.

Important details:

* Padding is applied only after packing; labels are padded with `IGNORE_INDEX`.
* FlashAttention kwargs (`cu_seq_lens_q/k` and `max_length_q/k`) are recomputed after padding so they match
  the padded length and avoid numerical instability.

Examples (data level):

1) SFT-style packing with fixed-length padding:

* Packed tokens (two samples): input_ids = [A B C D E F], position_ids = [0 1 2 3 0 1]
* cu_seqlens (from position_ids): [0 4 6]
* With `pad_packed_to_length=8`:
  * input_ids = [A B C D E F 0 0]
  * position_ids = [0 1 2 3 0 1 0 1]
  * cu_seqlens (padded): [0 4 6 8]
  * labels padded with `IGNORE_INDEX` for the last 2 positions

2) dyn_bsz packing with fixed-length padding:

* Packed tokens (three samples): input_ids = [A B C D E F G], position_ids = [0 1 2 0 1 0 1]
* cu_seqlens (from position_ids): [0 3 5 7]
* With `pad_packed_to_length=10`:
  * input_ids = [A B C D E F G 0 0 0]
  * position_ids = [0 1 2 0 1 0 1 0 1 2]
  * cu_seqlens (padded): [0 3 5 7 10]
  * labels padded with `IGNORE_INDEX` for the last 3 positions

3) Inferred pad length (micro_batch_size * max_seq_len):

* micro_batch_size=2, max_seq_len=4 -> pad_packed_to_length=8
* Packed tokens (two samples): input_ids = [A B C D E], position_ids = [0 1 2 3 0]
* cu_seqlens (from position_ids): [0 4 5]
* After padding:
  * input_ids = [A B C D E 0 0 0]
  * position_ids = [0 1 2 3 0 1 2 3]
  * cu_seqlens (padded): [0 4 5 8]

## DiT (diffusion) packing

### Why a DiT reuses only half of this machinery

`PackingCollator` packs **1-D token streams** and derives `cu_seq_lens` from
`position_ids == 0`. A DiT sample is not one: it is a `[C, T, H, W]` latent whose
geometry differs per clip, its prompt is tokenized *after* the dataloader (by the
condition model), and the model's two attention passes need two different segment
layouts. So the **tensor-level** pack lives in the model
(`veomni/models/diffusers/cosmos3/cosmos3_core/packed_layout.py`: per-sample `cu_seqlens`
+ `Cosmos3MoTTransformer._forward_packed`), and only the **grouping** half is reused:
`DynBszBuffer` / `TextBatchingStrategy` / `DynamicBatchSizeDataLoader` decide how
many samples share a micro-batch, exactly as they do for text. The DiT-specific piece
the grouping needs is a per-sample length function
(`veomni/data/multimodal/dit/packing.py::dit_packed_length`), because the shared
fallbacks price a DiT sample (no `attention_mask`, no `input_ids` yet) as one token.

### Turning it on

```yaml
data:
  pack_mode: token_budget              # "mbs" (default) or "token_budget"
  max_num_tokens_after_packing: 45056  # token budget of one packed micro batch
train:
  micro_batch_size: 1                  # counts packed micro batches in token_budget mode
  global_batch_size: 16
  dyn_bsz: true                        # required by token_budget mode
```

`mbs` is the default and is what every run had before this existed: a fixed
`micro_batch_size` samples per micro-batch, `global_batch_size` counting samples.
`token_budget` is the open-source Cosmos3 `PackingDataLoader` semantics: samples are
greedily appended until the next one would cross `max_num_tokens_after_packing`, so
`micro_batch_size` counts *packed micro batches* (keep it 1) and `global_batch_size`
counts micro batches. `train.gradient_accumulation_steps` is still derived from
`global_batch_size // (micro_batch_size * dp_size)`; there is no `grad_accum` field.
The model-side switch `config.pack_samples` (`auto` / `always` / `never`) is separate:
`auto` runs any multi-sample micro-batch as one packed sequence, `never` keeps the
one-forward-per-sample loop.

### `dyn_bsz_length_fn`: what a new modality must provide

`build_native_dataloader(..., dyn_bsz_length_fn=None)` is the injection point added
for this: a picklable callable returning **how many tokens the sample costs under the
token budget**. `None` (the default) keeps
`get_length_fn_by_count_mode(dyn_bsz_count_mode)`, i.e. today's behaviour for every
existing pipeline. A modality whose samples do not look like token ids must pass its
own; the DiT one is built from the run's own declarations
(`build_dit_length_fn(mm_configs, model_config=...)`) and mirrors the open-source
`SampleTokenCounter`: prompt tokens `+ 1`, plus
`ceil(h_lat / patch) * ceil(w_lat / patch) * t_lat + 2`. Its **text term is an
estimate** (characters / `pack_text_chars_per_token`) until the prompt is tokenized,
so the grouping can differ from a tokenizer-exact packer by a couple of percent of a
sample.

### Things to watch

* `num_micro_batch = global_batch_size // (micro_batch_size * dp_size)`, and every
  packed micro batch is filled to the token budget -- so tokens/step is
  `num_micro_batch * max_num_tokens_after_packing * world_size` regardless of how
  many clips that turns out to be (16 x 45056 = 720,896 for 8 ranks, `global=16`,
  `micro=1`).
* Cross-sample isolation is **structural, not mask-based**: each sample's K/V is one
  contiguous `[und_i ; gen_i]` block (`interleave_und_gen`) and both attention passes
  take per-segment `cu_seqlens`, while position ids restart per sample. Packing
  therefore does not add a mask anywhere; it makes leakage unrepresentable.
* Packed attention is served by `eager` (block-diagonal mask), `sdpa` /
  `veomni_sdpa` (per-segment split) and the flash rows (one varlen kernel). Any other
  attention row raises `NotImplementedError` instead of silently attending across the
  whole pack.
