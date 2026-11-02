"""Numeric core of the Cosmos3 mixture-of-towers transformer.

The modules here are self-contained on purpose: they depend on ``torch`` and on
VeOmni's op registry only, never on ``cosmos_framework``, ``flash_attn`` or
``torch_npu`` at import time, so they can be unit-tested on a CPU-only machine and
reused by both the transformer and the condition model.

Public surface:

* :mod:`.flow_matching` -- training-time sigma sampling (``uniform`` /
  ``logitnormal`` / ``waver`` + resolution shift), interpolation and the
  rectified-flow loss.
* :mod:`.timestep_embedder` -- sinusoidal + MLP timestep conditioning.
* :mod:`.mot_rope` -- 3-D interleaved mRoPE position ids and rotation.
* :mod:`.mot_attention` -- the two-pass joint attention and its reference.
* :mod:`.mot_decoder_layer` -- the two-tower decoder block.
* :mod:`.packed_layout` -- the TND segment layout of a multi-sample micro-batch.
"""

from .flow_matching import (
    WAVER_MODE_S,
    TrainTimeSampler,
    flow_matching_loss,
    interpolate,
    shift_times,
)
from .mot_attention import (
    MotAttention,
    MotRMSNorm,
    bind_mot_attention,
    eager_attention_forward,
    joint_attention,
    packed_visibility_mask,
    sdpa_reference_joint_attention,
    sdpa_reference_packed_joint_attention,
)
from .mot_decoder_layer import MoTDecoderLayer, MotMLP, MotSparseMoeBlock, moe_layer_enabled
from .mot_rope import (
    DEFAULT_BASE_FPS,
    DEFAULT_TEMPORAL_COMPRESSION_FACTOR,
    MotRope,
    build_latent_grid,
    build_latent_position_ids,
    build_segmented_latent_position_ids,
    build_segmented_text_position_ids,
    build_text_position_ids,
    interleave_mrope_frequencies,
    latent_temporal_positions,
)
from .packed_layout import PackedLayout, build_packed_layout, interleave_und_gen
from .timestep_embedder import TimestepEmbedder, add_time_conditioning, timestep_embedding


__all__ = [
    "DEFAULT_BASE_FPS",
    "DEFAULT_TEMPORAL_COMPRESSION_FACTOR",
    "WAVER_MODE_S",
    "MoTDecoderLayer",
    "MotAttention",
    "MotMLP",
    "MotRMSNorm",
    "MotRope",
    "MotSparseMoeBlock",
    "PackedLayout",
    "TimestepEmbedder",
    "TrainTimeSampler",
    "add_time_conditioning",
    "bind_mot_attention",
    "build_latent_grid",
    "build_latent_position_ids",
    "build_packed_layout",
    "build_segmented_latent_position_ids",
    "build_segmented_text_position_ids",
    "build_text_position_ids",
    "eager_attention_forward",
    "flow_matching_loss",
    "interleave_mrope_frequencies",
    "interleave_und_gen",
    "interpolate",
    "joint_attention",
    "latent_temporal_positions",
    "moe_layer_enabled",
    "packed_visibility_mask",
    "sdpa_reference_joint_attention",
    "sdpa_reference_packed_joint_attention",
    "shift_times",
    "timestep_embedding",
]
