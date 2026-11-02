"""3-D interleaved mRoPE for the Cosmos3 towers.

Both towers attend jointly, so their keys have to live in one positional frame.
Cosmos3 gives every token three position ids: reasoner tokens use a single increasing
id on all three axes (the text case of ``get_3d_mrope_ids_text_tokens``), diffusion
tokens use the latent's patched 3-D grid (the vision case of
``get_3d_mrope_ids_vae_tokens`` with ``reset_spatial_indices``).

**FPS modulation** (``enable_fps_modulation`` / ``base_fps``) is the part of the vision
case that is easy to omit and impossible to catch with a shape check: the reference
indexes latent frames by *real time*, so a latent frame advances the temporal axis by
``base_fps / fps`` (the temporal compression factor cancels). For the SFT recipe
(``base_fps=24``, Bridge clips at 5 fps) that is 4.8, not 1.0; without it a 5 fps clip
is read as 24 fps and temporal attention degenerates into neighbour averaging, which
lowers the loss with no error anywhere. Reference: ``mrope.py:164-180``,
``sequence.py:596-608``, ``nano_model_config.py:42-43``.

The temporal axis is a cursor, not a per-segment restart: the text span sits on
``0 .. L-1`` and the vision span continues from ``L`` (``sequence.py:271-275``,
``:596-608``, plus the optional margin at ``packers.py:570``), which is what
``temporal_offset`` is. Spatial axes do restart per segment
(``unified_3d_mrope_reset_spatial_ids=True``).

The three axes fold into one frequency table by Qwen3-VL's interleaved rule
(``mrope_interleaved=True``, ``mrope_section=[24, 20, 20]``): axis ``t`` owns the slots
outside the H/W stripes, ``h`` owns ``1, 4, 7, ... < 3 * section[1]``, ``w`` owns
``2, 5, 8, ... < 3 * section[2]``. The rotation itself is the ``rope/full`` op, so no
NPU import lives here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from veomni.ops import VeomniOp
from veomni.ops.config import resolve_op_impl


__all__ = [
    "DEFAULT_BASE_FPS",
    "DEFAULT_TEMPORAL_COMPRESSION_FACTOR",
    "MotRope",
    "build_latent_grid",
    "build_latent_position_ids",
    "build_segmented_latent_position_ids",
    "build_segmented_text_position_ids",
    "build_text_position_ids",
    "interleave_mrope_frequencies",
    "latent_temporal_positions",
]


# ``nano_model_config.py:42-43`` and ``vision_sft_nano.py:271``: the reference's
# mRoPE time base is 24 fps of *pixel* time, and the Wan2.2 VAE compresses 4 pixel
# frames into one latent frame.
DEFAULT_BASE_FPS = 24.0
DEFAULT_TEMPORAL_COMPRESSION_FACTOR = 4


def build_text_position_ids(
    num_tokens: int,
    *,
    temporal_offset: int | float = 0,
    device: torch.device | str | None = None,
) -> Tensor:
    """Position ids of ``num_tokens`` reasoner tokens, shape ``(3, num_tokens)``.

    Text has no geometry, so all three axes carry the same increasing id, exactly
    like the reference's ``text_mrope_ids``.
    """
    ids = torch.arange(num_tokens, dtype=torch.float32, device=device) + temporal_offset
    return ids.unsqueeze(0).expand(3, -1).contiguous()


def build_latent_grid(latent_shape: Sequence[int], patch_size: Sequence[int] = (1, 2, 2)) -> tuple[int, int, int]:
    """Patch grid ``(t, h, w)`` of a ``(C, F, H, W)`` latent for a ``(pt, ph, pw)`` patch.

    Args:
        latent_shape: VAE latent shape, channel first, as fed to the patchifier.
        patch_size: ``(pt, ph, pw)``; Cosmos3 uses ``(1, 2, 2)``.

    Returns:
        ``(F // pt, H // ph, W // pw)``.
    """
    if len(latent_shape) != 4:
        raise ValueError(f"latent_shape must be (C, F, H, W), got {tuple(latent_shape)}")
    _, frames, height, width = (int(size) for size in latent_shape)
    pt, ph, pw = (int(size) for size in patch_size)
    for extent, patch, axis in ((frames, pt, "F"), (height, ph, "H"), (width, pw, "W")):
        if patch <= 0 or extent % patch:
            raise ValueError(f"{axis}={extent} is not divisible by patch {patch}")
    return frames // pt, height // ph, width // pw


def latent_temporal_positions(
    grid_t: int,
    *,
    fps: Optional[float] = None,
    base_fps: float = DEFAULT_BASE_FPS,
    temporal_compression_factor: int = DEFAULT_TEMPORAL_COMPRESSION_FACTOR,
    enable_fps_modulation: Optional[bool] = None,
    temporal_offset: float = 0.0,
    device: torch.device | str | None = None,
) -> Tensor:
    """Temporal coordinates of ``grid_t`` latent frames, shape ``(grid_t,)`` float32.

    This is the temporal half of ``get_3d_mrope_ids_vae_tokens``
    (``mrope.py:164-180``) with ``start_frame_offset=0`` and
    ``base_temporal_compression_factor == temporal_compression_factor``:

        scaled_t[i] = i / (fps / tcf) * (base_fps / tcf) + temporal_offset
                    = i * base_fps / fps + temporal_offset

    The compression factor cancels algebraically; it is still a parameter because
    the reference's formula is written in tokens-per-second and a caller that
    overrides ``base_fps`` per stream needs the same spelling to stay auditable.

    Args:
        fps: The clip's real frame rate. ``None`` falls back to ``base_fps`` --
            which is the reference's own fallback (``packers.py:274``:
            ``fps = data.fps_vision[sample] if ... else base_fps``) and makes the
            step exactly ``1.0``, i.e. the pre-FPS-modulation layout.
        base_fps: The mRoPE time base (``nano_model_config.py:42``).
        enable_fps_modulation: Tri-state. ``None`` (default) enables modulation
            only when ``fps`` is given; ``False`` forces the integer layout even
            when an fps is supplied; ``True`` requires an fps.
        temporal_offset: Added to every coordinate -- the temporal cursor the
            segment starts from (the prompt length for a generation segment).

    Returns:
        ``(grid_t,)`` float32 positions.

    Raises:
        ValueError: on a non-positive or non-finite ``fps``, a non-positive
            ``base_fps``/``temporal_compression_factor``, or
            ``enable_fps_modulation=True`` without an ``fps``.
    """
    if grid_t < 1:
        raise ValueError(f"grid_t must be >= 1, got {grid_t}.")
    if base_fps <= 0 or not math.isfinite(float(base_fps)):
        raise ValueError(f"base_fps must be positive and finite, got {base_fps!r}.")
    if temporal_compression_factor < 1:
        raise ValueError(f"temporal_compression_factor must be >= 1, got {temporal_compression_factor!r}.")

    if enable_fps_modulation is None:
        modulated = fps is not None
    else:
        modulated = bool(enable_fps_modulation)
        if modulated and fps is None:
            raise ValueError(
                "enable_fps_modulation=True needs the clip's fps: without it the temporal axis would be "
                "indexed by latent-frame number and the clip would be trained as if it were base_fps."
            )
    if fps is not None and (fps <= 0 or not math.isfinite(float(fps))):
        raise ValueError(f"fps must be positive and finite, got {fps!r}.")

    index = torch.arange(grid_t, dtype=torch.float32, device=device)
    if not modulated:
        return index + float(temporal_offset)

    # ``fps is None`` with modulation on cannot reach here (validated above); the
    # fallback keeps the step at exactly 1.0, which is the reference's behaviour.
    effective_fps = float(base_fps if fps is None else fps)
    tps = effective_fps / float(temporal_compression_factor)
    base_tps = float(base_fps) / float(temporal_compression_factor)
    return index / tps * base_tps + float(temporal_offset)


def build_latent_position_ids(
    grid_t: int,
    grid_h: int,
    grid_w: int,
    *,
    fps: Optional[float] = None,
    base_fps: float = DEFAULT_BASE_FPS,
    temporal_compression_factor: int = DEFAULT_TEMPORAL_COMPRESSION_FACTOR,
    enable_fps_modulation: Optional[bool] = None,
    temporal_offset: float = 0.0,
    device: torch.device | str | None = None,
) -> Tensor:
    """Position ids of a patched latent grid, shape ``(3, grid_t * grid_h * grid_w)``.

    The flattening order is ``t`` outer, ``h`` middle, ``w`` inner -- the same order
    the patchifier emits rows in, so position rows line up with latent tokens.
    Spatial axes always restart at zero; the temporal axis follows
    :func:`latent_temporal_positions`. With the defaults (``fps=None``,
    ``temporal_offset=0``) this is the plain ``arange`` grid it has always been.
    """
    t_index = latent_temporal_positions(
        grid_t,
        fps=fps,
        base_fps=base_fps,
        temporal_compression_factor=temporal_compression_factor,
        enable_fps_modulation=enable_fps_modulation,
        temporal_offset=temporal_offset,
        device=device,
    ).view(-1, 1, 1)
    h_index = torch.arange(grid_h, dtype=torch.float32, device=device).view(1, -1, 1)
    w_index = torch.arange(grid_w, dtype=torch.float32, device=device).view(1, 1, -1)
    return torch.stack(
        [
            t_index.expand(grid_t, grid_h, grid_w).reshape(-1),
            h_index.expand(grid_t, grid_h, grid_w).reshape(-1),
            w_index.expand(grid_t, grid_h, grid_w).reshape(-1),
        ],
        dim=0,
    )


def build_segmented_text_position_ids(
    num_tokens_per_segment: Sequence[int],
    *,
    temporal_offset: int | float = 0,
    device: torch.device | str | None = None,
) -> Tensor:
    """Concatenate :func:`build_text_position_ids` over the segments of a pack.

    A packed micro-batch holds several samples, and each one is its own positional
    document: numbering must restart at zero per sample. Letting the ids run on
    across a pack boundary would put every sample after the first into a positional
    frame it was never trained in -- and, unlike a shape error, would still train.

    Returns:
        ``(3, sum(num_tokens_per_segment))`` position ids.
    """
    lengths = tuple(int(length) for length in num_tokens_per_segment)
    if not lengths:
        raise ValueError("a packed position-id table needs at least one segment.")
    if any(length <= 0 for length in lengths):
        raise ValueError(f"every segment needs at least one token, got {lengths}")
    segments = [build_text_position_ids(length, temporal_offset=temporal_offset, device=device) for length in lengths]
    return torch.cat(segments, dim=1)


def build_segmented_latent_position_ids(
    grids: Sequence[Sequence[int]],
    *,
    fps: Optional[Sequence[Optional[float]] | float] = None,
    base_fps: float = DEFAULT_BASE_FPS,
    temporal_compression_factor: int = DEFAULT_TEMPORAL_COMPRESSION_FACTOR,
    enable_fps_modulation: Optional[bool] = None,
    temporal_offsets: Optional[Sequence[float] | float] = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Concatenate :func:`build_latent_position_ids` over the segments of a pack.

    Same argument as :func:`build_segmented_text_position_ids`: each sample's latent
    grid is its own positional document, so its spatial ``(h, w)`` ids restart at
    zero. The *temporal* axis still takes each this-sample's own start cursor and
    its own fps: a pack of mixed-fps clips cannot share one scale factor, and the
    generation segment of a sample starts after that sample's prompt, not at 0.

    Args:
        grids: ``(grid_t, grid_h, grid_w)`` per sample, in pack order.
        fps: Per-sample frame rates (one entry per grid, or one scalar for all).
            ``None`` (or a ``None`` entry) falls back to ``base_fps`` per sample.
        temporal_offsets: Per-sample temporal cursor (one entry per grid, or one
            scalar for all). Defaults to ``0`` for every sample.

    Returns:
        ``(3, sum(grid_t * grid_h * grid_w))`` position ids.

    Raises:
        ValueError: when the per-sample lists do not line up with ``grids``.
    """
    if not grids:
        raise ValueError("a packed position-id table needs at least one segment.")
    count = len(grids)
    fps_per_sample = _per_sample(fps, count, "fps", default=None)
    offset_per_sample = _per_sample(temporal_offsets, count, "temporal_offsets", default=0.0)
    segments = [
        build_latent_position_ids(
            int(t),
            int(h),
            int(w),
            fps=fps_per_sample[index],
            base_fps=base_fps,
            temporal_compression_factor=temporal_compression_factor,
            enable_fps_modulation=enable_fps_modulation,
            temporal_offset=offset_per_sample[index],
            device=device,
        )
        for index, (t, h, w) in enumerate(grids)
    ]
    return torch.cat(segments, dim=1)


def _per_sample(value, count: int, name: str, *, default) -> list:
    """Broadcast a scalar (or ``None``) or validate a per-sample sequence of length ``count``."""
    if value is None:
        return [default] * count
    if isinstance(value, (int, float)):
        return [value] * count
    values = list(value)
    if len(values) != count:
        raise ValueError(f"got {len(values)} {name} entries for {count} latent grids; they must match.")
    return values


def interleave_mrope_frequencies(
    axis_frequencies: Tensor,
    mrope_section: Sequence[int],
    *,
    interleaved: bool = True,
) -> Tensor:
    """Fold per-axis angles ``(3, L, F)`` into one ``(L, F)`` table.

    ``interleaved=True`` reproduces Qwen3-VL's ``apply_interleaved_mrope``: start
    from the temporal axis and overwrite the H/W stripes, leaving the tail on the
    temporal axis (``[24, 20, 20]`` on a 64-slot table means 20 interleaved H, 20
    interleaved W and the remaining 24 slots for T). ``interleaved=False`` is the
    contiguous section split of ``rope/mrope`` (chunk ``i`` takes axis ``i % 3``).

    Args:
        axis_frequencies: ``(3, L, F)`` angles for the ``(t, h, w)`` axes.
        mrope_section: Per-axis frequency counts; must sum to ``F``.

    Returns:
        ``(L, F)`` mixed angles.
    """
    if axis_frequencies.ndim != 3 or axis_frequencies.shape[0] != 3:
        raise ValueError(f"axis_frequencies must be (3, L, F), got {tuple(axis_frequencies.shape)}")
    sections = tuple(int(size) for size in mrope_section)
    if len(sections) != 3 or sum(sections) != axis_frequencies.shape[-1]:
        raise ValueError(
            f"mrope_section {sections} must hold three axes summing to {axis_frequencies.shape[-1]} frequency slots"
        )
    if not interleaved:
        chunks = []
        offset = 0
        for axis, size in enumerate(sections):
            chunks.append(axis_frequencies[axis, :, offset : offset + size])
            offset += size
        return torch.cat(chunks, dim=-1)

    mixed = axis_frequencies[0].clone()
    for axis, offset in ((1, 1), (2, 2)):
        stop = sections[axis] * 3
        if stop > mixed.shape[-1]:
            raise ValueError(f"mrope_section[{axis}]={sections[axis]} does not fit a {mixed.shape[-1]}-slot table")
        index = slice(offset, stop, 3)
        mixed[:, index] = axis_frequencies[axis][:, index]
    return mixed


class MotRope(nn.Module):
    """Shared mRoPE table builder plus the ``rope/full`` application handle."""

    def __init__(
        self,
        head_dim: int,
        *,
        rope_theta: float = 5_000_000.0,
        mrope_section: Sequence[int] = (24, 20, 20),
        interleaved: bool = True,
    ) -> None:
        """Pre-compute the inverse frequencies of ``head_dim`` and bind the rope op."""
        super().__init__()
        sections = tuple(int(size) for size in mrope_section)
        if len(sections) != 3 or sum(sections) != head_dim // 2:
            raise ValueError(f"mrope_section {sections} must hold three axes summing to head_dim/2 = {head_dim // 2}")
        self.head_dim = head_dim
        self.mrope_section = sections
        self.interleaved = interleaved
        steps = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        self.register_buffer("inv_freq", 1.0 / (rope_theta**steps), persistent=False)
        self.veomni_rope = VeomniOp("rope", "full", resolve_op_impl("rotary_pos_emb_implementation"))

    def forward(self, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Build ``(cos, sin)`` of shape ``(L, head_dim)`` from ``(3, L)`` position ids."""
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0).expand(3, -1)
        if position_ids.ndim != 2 or position_ids.shape[0] != 3:
            raise ValueError(f"position_ids must be (3, L), got {tuple(position_ids.shape)}")
        inv_freq = self.inv_freq.to(device=position_ids.device, dtype=torch.float32)
        angles = position_ids.to(torch.float32).unsqueeze(-1) * inv_freq.view(1, 1, -1)  # (3, L, head_dim/2)
        mixed = interleave_mrope_frequencies(angles, self.mrope_section, interleaved=self.interleaved)
        emb = torch.cat((mixed, mixed), dim=-1)  # (L, head_dim), the rotate-half layout
        return emb.cos(), emb.sin()

    def apply(self, query: Tensor, key: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
        """Rotate ``(L, heads, head_dim)`` query/key with per-token ``(L, head_dim)`` tables."""
        return self.veomni_rope(query, key, cos, sin)
