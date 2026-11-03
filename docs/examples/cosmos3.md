# Cosmos3-Nano SFT Quick Start

This guide walks through fine-tuning **Cosmos3-Nano** (a mixture-of-towers
text-to-video diffusion transformer) with VeOmni. The recipes mirror the
open-source `cosmos-framework` `vision_sft_nano` configuration, so a VeOmni run
trains the same objective on the same data.

- Verified environment: 8 Ascend 950DT NPUs, `torch_npu` + `torchrun`
- Verified flows: online VAE training, two-stage offline training
  (embedding -> offline), and joint training of both towers

---

## 0. Architecture at a glance

Everything lives under `veomni/models/diffusers/cosmos3/`:

| sub-package | owns |
| --- | --- |
| `cosmos3_core/` | joint two-pass attention, the decoder block, 3-D mRoPE, timestep embedder, packed (TND) layout, rectified-flow sampling and loss |
| `cosmos3_transformer/` | `Cosmos3MoTTransformer`: the freeze switch, packed / per-sample forward and the loss |
| `cosmos3_condition/` | `Cosmos3ConditionModel`: frozen Wan2.2 VAE, Qwen3-VL tokenizer, sigma / noise / conditioning draws, latent-cache identity, per-sample token cost |

Module names are the released checkpoint's (reasoner `to_q`/`to_k`/`to_v`/`to_out`,
generator `add_q_proj`/`add_k_proj`/`add_v_proj`/`to_add_out` plus `proj_in`/`proj_out`),
so a checkpoint loads with no key remapping. A micro batch is greedily packed up to
45 056 tokens and runs as one TND sequence with per-sample `cu_seqlens`: no padding and
no cross-sample mask.

---

## Table of Contents

0. [Architecture at a glance](#0-architecture-at-a-glance)
1. [Model](#1-model)
2. [The two towers, and the freeze switch](#2-the-two-towers-and-the-freeze-switch)
3. [Data](#3-data)
4. [Online training](#4-online-training)
5. [Offline training (two stages)](#5-offline-training-two-stages)
6. [Packed batching](#6-packed-batching)
7. [What the port does not support yet](#7-what-the-port-does-not-support-yet)

---

## 1. Model

Cosmos3-Nano is one checkpoint holding two towers that share a single attention
softmax:

| Half | Role | Trained by the SFT recipe |
| --- | --- | --- |
| reasoner (`und`) | Qwen3-VL text tower: reads the prompt | frozen |
| generation (`gen`) | diffusion tower: owns the flow-matching prediction | yes |

```shell
# Hugging Face layout: transformer/ + vae/ + text_tokenizer/
# Every path in the three recipe YAMLs is a relative placeholder of this shape.
modelscope download --model nvidia/Cosmos3-Nano --local_dir pretrained_models/Cosmos3-Nano
```

Environment: this port needs nothing beyond the standard Ascend install — follow
[install_ascend_x86.md](../get_started/installation/install_ascend_x86.md) (or the
`aarch64` variant) and pick an image from
[AscendDockerUsage](../hardware_support/AscendDockerUsage/overview.md). Two things are
worth calling out because they are not specific to Cosmos3: the recipes leave
`rms_norm` / `rotary_pos_emb` unset so the ops registry picks the platform default
(`torch_npu` on Ascend, `eager` elsewhere), and `ops_implementation.attn_implementation:
flash_attention_2` resolves to the fused TND kernel on NPU. The published Ascend image
tags cover 910B (A2) and A3; a 950DT host needs an image built against its own CANN
release, which is what the runs quoted in this page used.

The recipe's `model.config_path` points at
`configs/model_configs/cosmos3/cosmos3_nano.json`, a derived copy of the released
`transformer/config.json`: the released file declares `model_type:
qwen3_vl_text` (upstream builds the MoT on top of the Qwen3-VL text tower), while
VeOmni needs its own `model_type` to dispatch to `Cosmos3MoTTransformer`. The
copy also drops the export's audio/action modality flags, which this training
shell does not build.

`model.condition_model_path` points at the checkpoint *root*: the frozen
condition model (Wan2.2 VAE + Qwen3-VL tokenizer) resolves `vae/` and the
tokenizer relative to it.

## 2. The two towers, and the freeze switch

`train_joint_reasoner` is the only knob:

| Value | Meaning | Recipe |
| --- | --- | --- |
| `false` (default) | frozen reasoner; only the generation tower trains (upstream's `keys_to_select = ["moe_gen", "time_embedder", "vae2llm", "llm2vae"]`) | `vision_sft_nano` |
| `true` | both towers train from the single diffusion loss, propagated through joint attention | joint fine-tuning |

Set it as a `model_config` override — it is one key, not a second architecture
file:

```yaml
model:
  config_path: configs/model_configs/cosmos3/cosmos3_nano.json
  model_config:
    train_joint_reasoner: true
```

The `lm_head` of the reasoner tower stays frozen either way: v1 trains no
autoregressive loss, so it never enters the graph.

## 3. Data

The transform reads a JSONL in the open-source SFT dump format: one row per clip,
decoded through `mm_configs.data_dir`, with the prompt passed through the checkpoint's
own chat template. The authoritative schema is `cosmos-framework/docs/dataset_jsonl.md`
in the reference repository; the fields this transform actually reads are:

| Field | Required | Meaning |
|---|---|---|
| `vision_path` (or `video_path`) | yes | clip path; relative paths resolve against `mm_configs.data_dir`, absolute paths and `://` URIs are used as-is |
| `t2w_windows` | yes | non-empty list of time windows; one window is trained per sample |
| `t2w_windows[i].caption_json` | preferred | structured caption, serialised verbatim (string is stripped, list takes its first element, dict is `json.dumps`-ed) |
| `t2w_windows[i].caption` | fallback | dense caption string; `duration`/`fps`/resolution suffixes are appended |
| row-level `caption` / `prompt` | fallback | same, when the window has no caption of its own |
| `duration`, `height`, `width`, `fps` | optional | only feed the caption suffixes (window-level value wins) |
| `t2w_windows[i].resolution{H,W}` | optional | resolution suffix |
| `start_frame` / `end_frame` | ignored | v1 trains the whole clip; windows are not cropped |

Example row (the smallest one that trains):

```json
{"vision_path": "videos/clip_0001.mp4",
 "t2w_windows": [{"caption_json": "A bridge at sunset.",
                  "duration": 4.0, "fps": 5,
                  "resolution": {"H": 256, "W": 256}}]}
```

Three `mm_configs` values are worth knowing:

- `max_frames` must match the dataset's longest clip. The shipped value (117) is the
  `4k+1` alignment of the measured maximum on the 1222-clip Bridge subset (p99 = 96,
  mean = 83.6); a smaller cap silently truncates the longest clips and shifts the mean
  latent-frame count. **There is no rule that derives it for a new dataset — measure
  `end_frame - start_frame + 1` and round down to `4k+1`.**
- `cfg_dropout_rate` is the reference's `cfg_dropout_rate=0.1`: one sample in ten
  trains on an *empty* caption, which is what teaches the diffusion tower to run
  without text. `cfg_dropout_seed` makes the draw resumable across restarts.
- `resolution` selects the sigma-`shift` bucket only. Nothing in this port resizes
  frames to that bucket, so training 480p data with `resolution: "256"` means "480p
  geometry with the 256 bucket's shift", not "resized to 256".

Dataset acquisition is outside this repository: the recipes were validated on the
`BridgeData2` SFT subset with synthetic captions, distributed as a JSONL plus a
`videos/` tree; `data.data_dir` is the directory that holds both. Point the three
recipe YAMLs at your own copy — every `data_dir` / `train_path` / `model_path` /
`output_dir` in them is a placeholder path.

### 3.1 Training on a different dataset

The adapter is bound to the **format**, not to that dataset: `data.source_name:
cosmos3_sft_jsonl` accepts any JSONL in the shape above, and `data.mm_configs.data_dir`
is the only place the dataset location appears. Switching datasets means re-measuring
five numbers in the recipe, not editing code:

| Knob | How to set it for a new dataset |
|---|---|
| `mm_configs.fps` | the rate the clips were captured at; it is what the loader samples *down* to. Per-clip container rates are read separately and drive mRoPE, so a mixed-fps dataset is handled per sample |
| `mm_configs.max_frames` | `4 * floor((longest_clip_frames - 1) / 4) + 1`. Measure it — a value below the longest clip silently truncates those clips (`video_utils.py:118-121` caps, it does not raise) |
| `mm_configs.min_frames` | the shortest clip you want to keep; shorter clips are stretched up to it |
| `mm_configs.frame_factor` / `frame_factor_remainder` | the VAE's temporal compression — `4` / `1` for Wan2.2. Only changes if you swap the VAE |
| `data.max_num_tokens_after_packing` | the token budget of one packed micro batch; 45056 is the reference recipe's, and the step time scales with it |

Three limits of this port that a new dataset can hit:

1. **One window per clip, and it must span the clip.** The adapter picks a window
   (`mm_configs.window_selection`) but does not crop the clip to that window's
   `start_frame`/`end_frame`. A dataset whose rows carry windows covering only parts of
   a clip will train window *i*'s caption against the whole clip; the preprocessor logs
   a one-time warning when it sees such rows.
2. **One clip per sample.** Multi-clip rows and image-only rows are not supported
   (`get_condition` rejects more than one clip; the conditioning draw expects a clip).
3. **JSONL in, JSONL out.** Parquet/CSV/other layouts need either a conversion step or a
   small `@PREPROCESSOR_REGISTRY.register("<name>")` adapter returning
   `(prompt, {}, [], [clip_path])` — the same shape `cosmos3_sft_jsonl` returns.

## 4. Online training

Encodes each clip through the VAE inside the training step.

```shell
# 8 cards; the batch shape in the recipe assumes exactly this DP size
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash train.sh tasks/train_dit.py configs/dit/cosmos3_sft.yaml
```

`train.sh` wraps `torchrun` (and adds `--standalone` on a single node), so both forms
work; the direct equivalent is
`torchrun --nproc_per_node=8 --standalone tasks/train_dit.py configs/dit/cosmos3_sft.yaml`.
`global_batch_size: 16` with `micro_batch_size: 1` over 8 ranks means
`gradient_accumulation_steps = 2`, computed from the DP size — changing the card count
changes the accumulation, not the recipe.

Measured on 8×Ascend 950DT: **13.2–13.5 s/step** (offline, frozen reasoner),
**23.0–23.5 s/step** (online), **14.4–14.5 s/step** (offline, both towers), at
720,896 tokens/step and `max_memory_allocated` 34.1 / 36.0 / 39.1 GB per card.

## 5. Offline training (two stages)

Stage 1 encodes every clip once into a latent cache; stage 2 trains from it without
loading the VAE parameters. This is the faster path whenever the same data is trained
for more than a few hundred steps.

```shell
# stage 1 -- fill the cache (resumable) into data.offline_embedding_save_dir:
#   rank_<r>_shard_<i>.parquet, latent_cache/ and .latent_cache_manifest.json
bash train.sh tasks/train_dit.py configs/dit/cosmos3_sft_embedding.yaml

# stage 2 -- data.train_path points at that same directory
bash train.sh tasks/train_dit.py configs/dit/cosmos3_sft_offline.yaml
```

Things that must hold:

1. `data.datasets_type` must be `mapping`.
2. Do not pass `--data.train_size`: the dyn-bsz step count reads it as a *total
   token count*, not a sample count. The yaml default (`1e12`) is correct.
3. Keep the `vae/` directory in place. Stage 2 does not load the VAE *parameters*, but
   it reads `vae/config.json`, and the cache recipe hashes the VAE weight files.

Stage 2 runs `max_steps` steps over the cache, **cycling the data** as needed;
`num_train_epochs: 1` does not mean "one pass over the cache".

The cache is content-addressed: every row is keyed by a SHA-256 over the clip
bytes, the prompt, the geometry handed to the VAE and a run-level recipe (VAE
dtype, weights and statistics, chat template, latent grid and a schema version
bumped whenever the encoding code changes). A re-run skips every sample whose key
is already present, and `offline_training` refuses a directory whose
`.latent_cache_manifest.json` records a different recipe.

## 6. Packed batching

The open-source recipe trains on a fixed number of *tokens* per micro batch, not
a fixed number of clips:

```yaml
data:
  pack_mode: token_budget
  max_num_tokens_after_packing: 45056
train:
  micro_batch_size: 1      # one *packed* micro batch
  global_batch_size: 16    # 8 ranks x 2 micro batches -> 16 x 45056 = 720,896 tokens/step
  dyn_bsz: true
```

`pack_mode: mbs` (the default) keeps the older meaning: `micro_batch_size`
counts samples. Either way the samples are run as one packed TND sequence, so
cross-sample isolation is structural rather than mask-based. See
[DiT packing](../usage/data_packing_and_dyn_bsz.md#dit-diffusion-packing) for the
per-sample length function and the grouping rule.

## 7. What the port does not support yet

- **Ulysses sequence parallelism** is refused by joint attention: both attention
  passes span whole token groups, so an SP run would have to rebuild the two
  K/V concatenations per rank slice first.
- **Sparse-MoE feed-forwards**: no released Cosmos3 checkpoint is sparse, so only
  dense blocks are built.
- **MFU**: the shared FLOPs estimator does not price the two-tower layout; derive it
  from the logged step time and the 720,896 tokens/step. That figure is the packing
  *budget*, not an exact count (packs fill to ~99-100%, and online mode estimates
  prompt length from characters), and the reasoner tower costs 2N FLOPs/token when
  frozen vs 6N when `train_joint_reasoner: true`, so an MFU derived this way should
  not be compared as-is between the two modes.
- **Autoregressive training of the reasoner tower**: joint training fine-tunes it
  through the diffusion loss only; there is no next-token loss, and `lm_head`
  stays frozen.
