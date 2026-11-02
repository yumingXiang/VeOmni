"""Token-budget packing for the DiT data path (``data.pack_mode``).

Why this file exists: the open-source Cosmos3 recipe does not train on a fixed
number of samples per micro-batch, it trains on a fixed number of *tokens*. Its
``PackingDataLoader`` (``cosmos_framework/data/generator/joint_dataloader.py:1363``)
greedily appends samples to a micro-batch until the next one would cross
``model.max_num_tokens_after_packing`` (45056 in ``vision_sft_nano``), so a step
holds a varying number of clips but a near-constant amount of work. VeOmni's text
side has the same idea (``docs/usage/data_packing_and_dyn_bsz.md``, ``train.dyn_bsz``)
and its grouping half -- ``DynBszBuffer`` / ``TextBatchingStrategy`` /
``DynamicBatchSizeDataLoader`` -- is reusable by DiT as is, *provided something tells
it how long a DiT sample is*. That "something" is :func:`dit_packed_length`.

What cannot be reused is ``PackingCollator``: it concatenates 1-D token streams
(``input_ids`` / ``labels`` / ``position_ids``) along a pack dim and derives
``cu_seq_lens`` from ``position_ids == 0``. A DiT sample is a ``[C, T, H, W]``
latent whose geometry differs per clip, its prompt is tokenized *after* the
dataloader (``Cosmos3ConditionModel.get_condition``), and its two attention passes
need two different segment layouts. The tensor-level pack therefore happens in the
model (``cosmos3_core.packed_layout`` + ``Cosmos3MoTTransformer._forward_packed``), and
this module only answers "how many tokens is this sample worth?" plus a readable
statement of the grouping rule.

Token accounting mirrors ``SampleTokenCounter._compute_token_split_per_sample``
(``cosmos_framework/data/generator/sample_token_counter.py``):

======================  =====================================================
term                    count
======================  =====================================================
reasoner (``und``)      prompt tokens ``+ 1`` (the ``<eos>`` delimiter)
generation (``gen``)    ``ceil(h_lat / patch) * ceil(w_lat / patch) * t_lat + 2``
                        (the ``<vision_start>`` / ``<vision_end>`` markers)
======================  =====================================================

with ``h_lat = H // spatial_compression``, ``w_lat = W // spatial_compression`` and
``t_lat = 1 + (T - 1) // temporal_compression`` -- the Wan2.2 VAE's 16x spatial and
4x temporal compression, which is what ``configs/dit/cosmos3_sft.yaml`` trains with
(``mm_configs.frame_factor: 4`` / ``frame_factor_remainder: 1``).

**Known approximation:** the *media* term is exact (it is derived from the decoded
frame tensor), but the *text* term is estimated from the prompt string because the
prompt is tokenized later, inside the condition model. For the structured-JSON
captions of this recipe (a few hundred tokens against ~1.5-2.5k media tokens) the
error is a couple of percent of a sample, which can move a greedy boundary by one
sample; it is reported rather than hidden, and ``pack_text_chars_per_token`` is the
knob that closes it. A pipeline that tokenizes in the dataset worker (as the
reference's does) can instead return ``len(input_ids)`` by passing the post-tokenizer
sample: :func:`dit_packed_length` prefers ``input_ids`` whenever they are present.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any, Callable, Optional

import torch


__all__ = [
    "DIT_DEFAULT_LATENT_PATCH_SIZE",
    "DIT_DEFAULT_SPATIAL_COMPRESSION",
    "DIT_DEFAULT_TEMPORAL_COMPRESSION",
    "DIT_DEFAULT_TEXT_CHARS_PER_TOKEN",
    "build_dit_length_fn",
    "dit_packed_length",
    "greedy_token_budget_batches",
    "packed_length_from_counts",
]


# Cosmos3-Nano's recipe: Wan2.2 VAE (16x spatial, 4x temporal) and a 2x2 latent
# patch. ``mm_configs.frame_factor`` / ``latent_patch_size`` override the temporal and
# patch terms; the spatial factor belongs to the VAE and is not in the run config.
DIT_DEFAULT_SPATIAL_COMPRESSION = 16
DIT_DEFAULT_TEMPORAL_COMPRESSION = 4
DIT_DEFAULT_LATENT_PATCH_SIZE = 2
# English/JSON prompt text averages ~4 characters per tokenizer token. Only used
# before the tokenizer has run; see the module docstring.
DIT_DEFAULT_TEXT_CHARS_PER_TOKEN = 4.0

# The three structural tokens the reference charges for on top of the real content:
# ``<eos>`` closes the reasoner stream, ``<vision_start>`` / ``<vision_end>`` bracket
# the generation stream (``sample_token_counter.py:253`` and ``:309``).
_DELIMITER_TOKENS = 1
_VISION_BOUNDARY_TOKENS = 2


def packed_length_from_counts(
    text_tokens: int,
    num_frames: int,
    height: int,
    width: int,
    *,
    spatial_compression_factor: int = DIT_DEFAULT_SPATIAL_COMPRESSION,
    temporal_compression_factor: int = DIT_DEFAULT_TEMPORAL_COMPRESSION,
    latent_patch_size: int = DIT_DEFAULT_LATENT_PATCH_SIZE,
) -> int:
    """Packed length of one sample from its caption length and pixel geometry.

    The frame/pixel side is arithmetic on shapes only -- nothing is decoded or
    encoded -- so this can price a sample in a dataloader worker.

    Args:
        num_frames: Pixel frame count ``T`` of the clip.
        height: Pixel height ``H``.
        width: Pixel width ``W``.

    Returns:
        ``und + gen`` tokens, i.e. the length this sample adds to a packed sequence.
    """
    if text_tokens <= 0:
        raise ValueError(f"a sample needs at least one prompt token, got {text_tokens}")
    if num_frames <= 0 or height <= 0 or width <= 0:
        raise ValueError(f"a clip needs positive geometry, got frames={num_frames} height={height} width={width}")
    for name, factor in (
        ("spatial_compression_factor", spatial_compression_factor),
        ("temporal_compression_factor", temporal_compression_factor),
        ("latent_patch_size", latent_patch_size),
    ):
        if factor <= 0:
            raise ValueError(f"{name} must be positive, got {factor}")

    latent_t = 1 + (num_frames - 1) // temporal_compression_factor
    patches_h = math.ceil((height // spatial_compression_factor) / latent_patch_size)
    patches_w = math.ceil((width // spatial_compression_factor) / latent_patch_size)
    und_tokens = text_tokens + _DELIMITER_TOKENS
    gen_tokens = patches_h * patches_w * latent_t + _VISION_BOUNDARY_TOKENS
    return und_tokens + gen_tokens


def _prompt_text(sample: Mapping[str, Any]) -> Optional[str]:
    """The sample's prompt as a string, or ``None`` when it carries no prompt field.

    ``None`` and ``""`` are deliberately different answers. An **empty** prompt is a
    supported sample -- CFG dropout (``vision_sft_nano.py:260`` ->
    ``sft_dataset.py:324-326``) trains one clip in ten on exactly that, and the
    reference tokenizes it to the bare chat template -- so it must price out as "a few
    tokens", not as an error. An **absent** prompt field is a data bug and still raises.
    """
    for key in ("inputs", "prompt"):
        value = sample.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)) and value and isinstance(value[0], str):
            return value[0]
        if isinstance(value, (list, tuple)) and not value and key in sample:
            # A collated batch that arrived with an empty prompt column: still a
            # present-but-empty prompt, not a missing one.
            return ""
    return None


def _prompt_tokens(sample: Mapping[str, Any], chars_per_token: float) -> int:
    """Prompt token count: exact when tokenized, otherwise a character estimate.

    Truncation happens before the generation markers are appended
    (``sft_dataset.py:174-177`` then ``sequence.py:257-260``), so this is the
    *content* count; :func:`packed_length_from_counts` adds the structural tokens.
    """
    input_ids = sample.get("input_ids")
    if isinstance(input_ids, torch.Tensor) and input_ids.numel():
        return int(input_ids.numel())
    if isinstance(input_ids, (list, tuple)) and input_ids and isinstance(input_ids[0], torch.Tensor):
        return int(sum(tensor.numel() for tensor in input_ids))
    text = _prompt_text(sample)
    if text is None:
        raise ValueError(
            "cannot price a DiT sample: it carries neither `input_ids` nor a prompt string "
            f"(`inputs`/`prompt`); keys are {sorted(sample)}."
        )
    if chars_per_token <= 0:
        raise ValueError(f"chars_per_token must be positive, got {chars_per_token}")
    # ``max(1, ...)`` is what makes an empty (CFG-dropped) caption price at the
    # template floor instead of 0; ``packed_length_from_counts`` then rejects a
    # non-positive total, so this floor is load-bearing.
    return max(1, math.ceil(len(text) / chars_per_token))


def _frame_geometry(sample: Mapping[str, Any]) -> tuple[int, int, int]:
    """``(T, H, W)`` of the sample's clip, from whatever shape the transform left."""
    videos = sample.get("videos")
    if videos is None:
        videos = sample.get("video")
    if videos is None:
        raise ValueError(f"cannot price a DiT sample: it carries no `videos`/`video`; keys are {sorted(sample)}")
    if isinstance(videos, torch.Tensor):
        if videos.dim() == 4:  # one clip, [T, C, H, W]
            return int(videos.shape[0]), int(videos.shape[2]), int(videos.shape[3])
        if videos.dim() == 3:  # a single frame, [C, H, W]
            return 1, int(videos.shape[1]), int(videos.shape[2])
        raise ValueError(f"unsupported video tensor rank {videos.dim()}")
    if isinstance(videos, (list, tuple)) and videos:
        first = videos[0]
        if isinstance(first, torch.Tensor):
            if first.dim() == 4:  # one decoded clip wrapped in a list
                return int(first.shape[0]), int(first.shape[2]), int(first.shape[3])
            if first.dim() == 3:  # one tensor per frame
                return len(videos), int(first.shape[1]), int(first.shape[2])
        if isinstance(first, (list, tuple)) and first and isinstance(first[0], torch.Tensor):
            return _frame_geometry({"videos": first})
    raise ValueError(f"cannot price a DiT sample: unsupported `videos` entry {type(videos).__name__}")

def _latent_geometry(sample: Mapping[str, Any]) -> Optional[tuple[int, int, int]]:
    """``(T_lat, H_lat, W_lat)`` of a cached sample, or ``None`` when it has no latent.

    ``offline_training`` replays ``get_condition``'s output, which carries ``latents``
    as ``[C, T, H, W]`` and no pixel clip at all, so the pixel-side pricing below
    cannot run. Returning ``None`` keeps the online path byte-for-byte unchanged.
    """
    latents = sample.get("latents")
    if latents is None:
        return None
    if isinstance(latents, (list, tuple)):
        if not latents:
            return None
        latents = latents[0]
    if not isinstance(latents, torch.Tensor):
        return None
    if latents.dim() != 4:
        raise ValueError(
            f"cannot price a cached DiT sample: expected latents of rank 4 [C, T, H, W], got {tuple(latents.shape)}"
        )
    return int(latents.shape[1]), int(latents.shape[2]), int(latents.shape[3])


def dit_packed_length(
    sample: Mapping[str, Any],
    *,
    spatial_compression_factor: int = DIT_DEFAULT_SPATIAL_COMPRESSION,
    temporal_compression_factor: int = DIT_DEFAULT_TEMPORAL_COMPRESSION,
    latent_patch_size: int = DIT_DEFAULT_LATENT_PATCH_SIZE,
    chars_per_token: float = DIT_DEFAULT_TEXT_CHARS_PER_TOKEN,
) -> int:
    """Packed token length of one DiT sample -- the ``get_length_fn`` for dyn batching.

    Shaped like ``veomni.data.dataset.get_length_fn`` (one sample in, an ``int`` out),
    because that is the hook the token-budget dataloader calls. Buildings it as a
    ``functools.partial`` of this module-level function keeps it picklable, which
    ``dyn_bsz_runtime="worker"`` requires.

    Args:
        sample: One transformed DiT sample (``inputs`` / ``videos`` and, once the
            prompt is tokenized, ``input_ids``).
        chars_per_token: Character-to-token ratio used while the prompt is still a
            string.

    Returns:
        The sample's length in packed tokens; raises when it cannot be priced rather
        than returning a placeholder, because a wrong length silently over- or
        under-packs every micro-batch.
    """
    text_tokens = _prompt_tokens(sample, chars_per_token)
    latent_grid = _latent_geometry(sample)
    if latent_grid is not None:
        # offline_training replays the VAE's latents, so there is no pixel clip to
        # read. The two quantities packed_length_from_counts derives from pixel
        # geometry are exactly what the encoder produced, so read them off the grid
        # instead of inventing a second measure (height // spatial == H_lat).
        latent_t, latent_h, latent_w = latent_grid
        patches_h = math.ceil(latent_h / latent_patch_size)
        patches_w = math.ceil(latent_w / latent_patch_size)
        return (text_tokens + _DELIMITER_TOKENS) + patches_h * patches_w * latent_t + _VISION_BOUNDARY_TOKENS
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


def build_dit_length_fn(
    mm_configs: Mapping[str, Any] | None = None, *, model_config: Any = None
) -> Callable[[Mapping[str, Any]], int]:
    """Build the picklable ``get_length_fn`` from the run's data and model config.

    The run already declares the media geometry twice, so the token measure reads
    those declarations instead of inventing a third copy: ``data.mm_configs`` carries
    ``frame_factor`` / ``frame_factor_remainder`` (the VAE's temporal compression,
    which ``dit_online`` also uses to pick the frame count), and the DiT config JSON
    carries ``latent_downsample_factor`` (16) and ``latent_patch_size`` (2). Precedence
    is ``mm_configs`` -> model config -> documented default, so a run can always
    override.

    Args:
        mm_configs: ``data.mm_configs`` of the run; ``None`` uses the other sources.
        model_config: The DiT config object (``self.model_config`` of the runtime).

    Returns:
        A one-argument callable suitable for ``get_length_fn`` / ``dyn_bsz_length_fn``.
    """
    configs = dict(mm_configs or {})

    def pick(key: str, model_attr: str, default: int) -> int:
        for candidate in (configs.get(key), getattr(model_config, model_attr, None)):
            if candidate is not None and int(candidate) > 0:
                return int(candidate)
        return default

    temporal = int(configs.get("temporal_compression_factor", 0) or 0)
    if temporal <= 0:
        # ``frame_factor`` + ``frame_factor_remainder`` describe the same compression:
        # ``n_latent = 1 + (T - r) // frame_factor`` (Wan2.2 keeps ``r == 1``).
        temporal = int(configs.get("frame_factor", 0) or 0)
    if temporal <= 0:
        temporal = DIT_DEFAULT_TEMPORAL_COMPRESSION
    return partial(
        dit_packed_length,
        spatial_compression_factor=pick(
            "spatial_compression_factor", "latent_downsample_factor", DIT_DEFAULT_SPATIAL_COMPRESSION
        ),
        temporal_compression_factor=temporal,
        latent_patch_size=pick("latent_patch_size", "latent_patch_size", DIT_DEFAULT_LATENT_PATCH_SIZE),
        chars_per_token=float(configs.get("pack_text_chars_per_token", DIT_DEFAULT_TEXT_CHARS_PER_TOKEN)),
    )


_DEFAULT_LOOKAHEAD = 10  # JointDataLoader._DEFAULT_LOOKAHEAD_LIMIT


def greedy_token_budget_batches(
    samples: Sequence[Any],
    max_num_tokens: int,
    *,
    length_fn: Callable[[Any], int],
    max_samples: int | None = None,
    lookahead: int | None = _DEFAULT_LOOKAHEAD,
) -> list[list[Any]]:
    """Group ``samples`` into token-budget micro-batches, the reference way.

    This is ``PackingDataLoader.__iter__`` (``joint_dataloader.py:1400-1480``) written
    as a pure function, kept as the *executable specification* of what
    ``data.pack_mode: token_budget`` means:

    * a candidate joins the batch while ``packed_tokens + candidate < max_num_tokens``
      -- the reference rejects on ``>=`` (``_sample_fits``), so a batch stays strictly
      under the budget;
    * a sample that does not fit is set aside and the scan keeps going, so a small clip
      behind a large one still fills the budget; set-aside samples are re-queued for
      the next batch, ahead of the samples never examined;
    * a sample that does not fit an *empty* batch starts one anyway: the reference logs
      an error and drops it. Dropping would silently lose training data, so this is the
      one place the spec differs from the reference on purpose;
    * ``max_samples`` caps the sample count independently of the token budget.

    The runtime path is VeOmni's own ``TextBatchingStrategy`` / ``DynBszBuffer``, which
    applies the same rule to the same :func:`dit_packed_length`
    (``tests/models/diffusers/test_cosmos3.py`` covers the model side; the greedy rule here
    is the data-side spec).

    Args:
        length_fn: Per-sample token count; :func:`dit_packed_length` (or a partial of
            it) in production.
        max_samples: Optional cap on the number of samples per batch.
        lookahead: How many non-fitting candidates to look past before closing the
            batch; the default is the reference's ``_DEFAULT_LOOKAHEAD_LIMIT`` (10).
            ``None`` keeps scanning until the budget is reached, which is what
            ``DynBszBuffer.get_samples`` does, so the runtime is more aggressive than
            the reference here.

    Returns:
        One list of samples per micro-batch, in emission order.
    """
    if max_num_tokens <= 0:
        raise ValueError(f"max_num_tokens must be positive, got {max_num_tokens}")
    pending = list(samples)
    batches: list[list[Any]] = []
    while pending:
        batch: list[Any] = []
        packed_tokens = 0
        deferred: list[Any] = []
        rest: list[Any] = []
        stop = False
        for item in pending:
            if stop:
                rest.append(item)
                continue
            if max_samples is not None and len(batch) >= max_samples:
                stop = True
                rest.append(item)
                continue
            length = int(length_fn(item))
            if packed_tokens + length >= max_num_tokens and batch:
                deferred.append(item)
                if lookahead is not None and len(deferred) > int(lookahead):
                    stop = True
                continue
            batch.append(item)
            packed_tokens += length
            if packed_tokens >= max_num_tokens:
                stop = True
        if not batch:
            raise RuntimeError(
                f"no sample fits the token budget of {max_num_tokens}; the first candidate of an empty batch is "
                "always admitted, so this means the stream contained no samples."
            )
        # Set-aside samples come back first, in arrival order, then the unexamined tail:
        # the reference re-queues with ``appendleft`` in reverse, which is the same order.
        pending = deferred + rest
        batches.append(batch)
    return batches
