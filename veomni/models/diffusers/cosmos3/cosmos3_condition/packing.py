"""Per-sample token cost for ``data.pack_mode: token_budget`` (the reference's ``PackingDataLoader``).

Mirrors the reference's ``SampleTokenCounter``: ``und = prompt tokens + 1`` (eos)
and ``gen = ceil(h_lat / p) * ceil(w_lat / p) * t_lat + 2`` (vision markers). The
prompt is tokenized only inside the condition model, so before that its length is
estimated from characters (``mm_configs.pack_text_chars_per_token``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from functools import partial
from typing import Any, Callable, Optional

import torch


def packed_length_from_counts(
    text_tokens: int,
    num_frames: int,
    height: int,
    width: int,
    *,
    spatial_compression_factor: int,
    temporal_compression_factor: int,
    latent_patch_size: int,
) -> int:
    latent_t = 1 + (num_frames - 1) // temporal_compression_factor
    patches_h = math.ceil((height // spatial_compression_factor) / latent_patch_size)
    patches_w = math.ceil((width // spatial_compression_factor) / latent_patch_size)
    return (text_tokens + 1) + patches_h * patches_w * latent_t + 2


def _prompt_tokens(sample: Mapping[str, Any], chars_per_token: float) -> int:
    """Exact once ``input_ids`` exist, else estimated; an empty (CFG-dropped) prompt costs at least one."""
    input_ids = sample.get("input_ids")
    if isinstance(input_ids, torch.Tensor) and input_ids.numel():
        return int(input_ids.numel())
    if isinstance(input_ids, (list, tuple)) and input_ids and isinstance(input_ids[0], torch.Tensor):
        return int(sum(tensor.numel() for tensor in input_ids))
    for key in ("inputs", "prompt"):
        value = sample.get(key)
        if isinstance(value, (list, tuple)):
            value = value[0] if value else ""
        if isinstance(value, str):
            return max(1, math.ceil(len(value) / chars_per_token))
    raise ValueError(f"cannot price a Cosmos3 sample without `input_ids` or a prompt; keys are {sorted(sample)}.")


def _frame_geometry(sample: Mapping[str, Any]) -> tuple[int, int, int]:
    """``(T, H, W)`` of the sample's decoded clip."""
    videos = sample.get("videos")
    if isinstance(videos, (list, tuple)):
        videos = videos[0]
    if isinstance(videos, torch.Tensor) and videos.dim() == 4:  # [T, C, H, W]
        return int(videos.shape[0]), int(videos.shape[2]), int(videos.shape[3])
    raise ValueError(f"cannot price a Cosmos3 sample: expected one decoded [T, C, H, W] clip, got {type(videos)}.")


def _latent_geometry(sample: Mapping[str, Any]) -> Optional[tuple[int, int, int]]:
    """``(T_lat, H_lat, W_lat)`` of an ``offline_training`` sample, which carries latents instead of frames."""
    latents = sample.get("latents")
    if isinstance(latents, (list, tuple)):
        latents = latents[0] if latents else None
    if not isinstance(latents, torch.Tensor):
        return None
    return int(latents.shape[1]), int(latents.shape[2]), int(latents.shape[3])


def cosmos3_packed_length(
    sample: Mapping[str, Any],
    *,
    spatial_compression_factor: int,
    temporal_compression_factor: int,
    latent_patch_size: int,
    chars_per_token: float,
) -> int:
    text_tokens = _prompt_tokens(sample, chars_per_token)
    latent_grid = _latent_geometry(sample)
    if latent_grid is not None:
        latent_t, latent_h, latent_w = latent_grid
        patches = math.ceil(latent_h / latent_patch_size) * math.ceil(latent_w / latent_patch_size)
        return (text_tokens + 1) + patches * latent_t + 2
    frames, height, width = _frame_geometry(sample)
    return packed_length_from_counts(
        text_tokens,
        frames,
        height,
        width,
        spatial_compression_factor=spatial_compression_factor,
        temporal_compression_factor=temporal_compression_factor,
        latent_patch_size=latent_patch_size,
    )


def build_cosmos3_length_fn(mm_configs: Optional[Mapping[str, Any]], model_config: Any) -> Callable[[Any], int]:
    """A picklable per-sample length fn; ``mm_configs`` overrides the DiT config, which overrides the defaults."""
    configs = dict(mm_configs or {})

    def pick(key: str, model_attr: str, default: int) -> int:
        for candidate in (configs.get(key), getattr(model_config, model_attr, None)):
            if candidate is not None and int(candidate) > 0:
                return int(candidate)
        return default

    temporal = int(configs.get("temporal_compression_factor", 0) or 0) or int(configs.get("frame_factor", 0) or 0) or 4
    return partial(
        cosmos3_packed_length,
        spatial_compression_factor=pick("spatial_compression_factor", "latent_downsample_factor", 16),
        temporal_compression_factor=temporal,
        latent_patch_size=pick("latent_patch_size", "latent_patch_size", 2),
        chars_per_token=float(configs.get("pack_text_chars_per_token", 4.0)),
    )
