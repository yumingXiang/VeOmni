"""Preprocessor for the open-source Cosmos SFT JSONL (``cosmos3_sft_jsonl``).

Any dataset in that format works: the reference dump ships one JSONL row per video
clip and *one or more* ``t2w_windows`` per row (see
``cosmos-framework/docs/dataset_jsonl.md``). VeOmni's DiT pipeline
consumes ``(prompt, outputs, images, videos)`` per sample, so this module does
the three translations the generic transform cannot:

1. **window → sample**: a row collapses to a single window. The open-source
   ``SFTDataset`` also crops the clip to that window's ``start_frame`` /
   ``end_frame``; here the video reference stays the whole file, which is exact
   for this dataset (one window covering the clip) and is the reason the default
   ``window_selection`` is ``"first"`` rather than a per-epoch random draw.
2. **caption → prompt**: ``caption_json`` (the canonical structured caption) is
   serialized verbatim, because the model is trained *and* run against the exact
   same JSON string. The dense ``caption`` string is the fallback and gets the
   reference's media suffixes appended, so a row that predates ``caption_json``
   still describes its own duration/fps/resolution.
3. **media → reference**: the clip becomes a path handed to the shared video
   loader. Nothing is decoded here: decoding belongs to the dataloader workers
   (``fetch_videos``) and re-encoding frames to JPEG inside a dataset worker, as
   the first cut of this adapter did, burns worker CPU on a lossy round-trip the
   VAE then sees.

**No conditioning field here, on purpose.** The upstream T2V/I2V/V2V draw happens in the dataset (``sft_dataset.py:348-359`` builds a ``SequencePlan`` whose
``condition_frame_indexes_vision`` is ``range(num_cond)``), and the packer turns
that into ``condition_mask`` (``sequence_packing/sequence.py:572-577``). It needs
exactly one dataset-side number for it: ``t_latent = 1 + (num_frames_pixel - 1) //
temporal_compression_factor`` (``sft_dataset.py:350``), used only for the clamp
``num_cond = min(num_cond, t_latent - 1)``. By the time the draw happens in this
stack, ``Cosmos3ConditionModel.process_condition`` already holds the encoded
latent, so ``t_latent`` is ``latent.shape[1]`` -- deriving it from the tensor is
exact and cannot disagree with the VAE. Sampling there also keeps the mixture on
the resumable ``rng_state_dict`` stream instead of a dataloader worker's global
RNG, which no checkpoint of ours restores. So this preprocessor stays a pure
row → ``(prompt, _, [], [video_path])`` mapping; adding a ``conditioning`` field
here would duplicate the draw and desynchronize it from the noise stream.

**CFG dropout *is* here, though**, and that asymmetry is the reference's: clearing a
caption is a property of the caption (``sft_dataset.py:324-326``), so it happens
where the caption is built, while the T2V/I2V/V2V mixture is a property of the
training tensors and happens where they are built. The reference's knob is
``cfg_dropout_rate=0.1`` (``vision_sft_nano.py:260``) and its draw uses the dataset
worker's ``random``, which is why this module owns a dedicated
``random.Random`` stream instead of borrowing the condition model's
``torch.Generator`` -- see :func:`cfg_dropout_rng` for what borrowing it would break.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Optional

from ....utils import logging
from ..preprocess import PREPROCESSOR_REGISTRY


logger = logging.get_logger(__name__)


# Copied from ``cosmos_framework/data/generator/local_datasets/sft_dataset.py`` so
# a dense-caption row produces the same prompt text as the open-source recipe.
_DURATION_TEMPLATE = "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
_RESOLUTION_TEMPLATE = "This video is of {height}x{width} resolution."

_CAPTION_JSON_KEY = "caption_json"

# CFG dropout: the open-source vision SFT recipe clears the *whole* caption on a
# fraction of samples (``vision_sft_nano.py:260`` -> ``sft_dataset.py:324-326``),
# which is what trains the diffusion tower to work without text as well as with it.
# ``packers.py:510`` records the exact scope: "CFG dropout only drops text *content*,
# not the structural text split" -- so the chat template, ``[eos]`` and
# ``<|vision_start|>`` all survive; only the caption string becomes empty.
#
# The draw must NOT come from the condition model's ``torch.Generator``: that CPU
# stream drives sigma / noise / the conditioning mixture and is snapshotted into the
# job checkpoint (``modeling_cosmos3_condition.py`` ``rng_state_dict``), so borrowing
# a value from it would shift every later sigma/noise draw and make a resumed run's
# noise disagree with an uninterrupted one. It gets its own ``random.Random`` here,
# which is also the module the reference uses (``sft_dataset.py:324``).
_DEFAULT_CFG_DROPOUT_SEED = 42

# One stream per (seed, dataloader-worker) pair. Dataloader workers are separate
# processes, so a per-process dict is the whole registry; the worker id is in the key
# because two workers must not draw the *same* values in the same order (that would
# correlate dropout with shard position).
_cfg_dropout_streams: dict[tuple[int, int], random.Random] = {}


def _worker_id() -> int:
    """This process's dataloader worker id (``0`` outside a worker / without torch)."""
    try:
        from torch.utils.data import get_worker_info
    except Exception:  # pragma: no cover - torch is a hard dependency, belt and braces
        return 0
    info = get_worker_info()
    return 0 if info is None else int(info.id)


def cfg_dropout_rng(seed: int = _DEFAULT_CFG_DROPOUT_SEED) -> random.Random:
    """The CFG-dropout stream for this process and worker, created on first use.

    Deliberately ``random.Random`` and not a torch generator: it keeps the draw off
    the condition model's resumable sigma/noise stream (see the module comment), and
    it matches the reference's own ``random.random()`` call in the dataset worker.

    Returns:
        The process-wide stream for ``(seed, worker id)``.
    """
    key = (int(seed), _worker_id())
    stream = _cfg_dropout_streams.get(key)
    if stream is None:
        stream = random.Random(key[0] + key[1])
        _cfg_dropout_streams[key] = stream
    return stream


def _maybe_drop_caption(
    prompt: Optional[str],
    rate: float,
    seed: int,
) -> Optional[str]:
    """Apply the recipe's CFG dropout to one already-built caption.

    Called *after* the caption is fully assembled (structured JSON, or the dense
    caption with its duration/fps/resolution suffixes), which is where the reference
    does it when ``cfg_dropout_keep_metadata=False`` -- the recipe's setting
    (``vision_sft_nano.py:259``). So the metadata goes too: the emptied prompt is a
    bare chat template, not "a bare caption plus its suffix".

    Args:
        rate: Dropout probability. ``0`` returns the input unchanged and, crucially,
            draws nothing at all -- so turning the knob off is bit-identical to the
            pre-dropout pipeline, not merely equal in distribution.
        seed: Base seed of the stream (see :func:`cfg_dropout_rng`).

    Returns:
        ``""`` when the draw fires, else ``prompt``.
    """
    if rate <= 0.0 or prompt is None:
        return prompt
    if rate >= 1.0:
        return ""
    if cfg_dropout_rng(seed).random() < rate:
        return ""
    return prompt


def _select_window(example: dict, window_selection: str, window_index: int) -> dict:
    """Pick one ``t2w_window`` from a dataset row."""
    windows = example.get("t2w_windows")
    if not windows:
        raise ValueError(
            "cosmos3_sft_jsonl requires a non-empty 't2w_windows' list; "
            f"got keys {sorted(example)}. Is this really the Cosmos3 SFT JSONL?"
        )
    if window_selection == "first":
        index = 0
    elif window_selection == "random":
        index = random.randrange(len(windows))
    elif window_selection == "index":
        index = window_index
    else:
        raise ValueError(f"window_selection must be 'first', 'random' or 'index'; got {window_selection!r}.")
    if not -len(windows) <= index < len(windows):
        raise ValueError(f"window_index {index} is out of range for a row with {len(windows)} windows.")
    if len(windows) > 1:
        # v1 hands the whole clip to the loader, so a row whose windows cover only
        # parts of it would pair this window's caption with the full clip. That is
        # silent in the loss, so say it once and let the user pick a dataset whose
        # windows span the clip (or crop upstream).
        logger.warning_once(
            "This dataset has rows with more than one t2w_window. The preprocessor does not crop the clip "
            "to the chosen window, so the caption of window %s is trained against the whole clip. Rows are "
            "expected to have one window spanning the clip (see docs/examples/cosmos3.md, section 3).",
            index,
        )
    return windows[index]


def _structured_caption(window: dict) -> Any:
    """The window's structured caption object, or ``None`` when it has none."""
    caption = window.get(_CAPTION_JSON_KEY)
    if caption is None:
        return None
    if isinstance(caption, str):
        # Some converters store the pre-serialized JSON string; keep it verbatim
        # rather than re-serializing (byte-identity with inference prompts).
        return caption.strip()
    if isinstance(caption, list):
        caption = caption[0] if caption else None
        if caption is None:
            return None
        return caption if isinstance(caption, str) else json.dumps(caption, ensure_ascii=False)
    return json.dumps(caption, ensure_ascii=False)


def _dense_caption(window: dict, example: dict, media_suffix: bool, fps: Any) -> str:
    """Fallback prompt from the dense ``caption`` string, with the media suffixes."""
    caption = window.get("caption") or example.get("caption") or example.get("prompt")
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError(
            "cosmos3_sft_jsonl needs either a 'caption_json' object or a dense 'caption' "
            f"string per t2w_window; window keys: {sorted(window)}."
        )
    caption = caption.strip().rstrip(".") + "."
    if not media_suffix:
        return caption

    # The preprocessor deliberately does not decode the clip, so duration/fps come
    # from the annotations instead of a frame count: the window's structured
    # caption records them, the row's ``duration`` is the clip-level fallback.
    structured = window.get(_CAPTION_JSON_KEY)
    structured = structured if isinstance(structured, dict) else {}
    duration = window.get("duration") or example.get("duration") or structured.get("duration")
    if isinstance(duration, str):
        duration = duration.rstrip("sS")
    effective_fps = structured.get("fps") or window.get("fps") or fps
    if duration is not None and effective_fps:
        caption = f"{caption} {_DURATION_TEMPLATE.format(duration=float(duration), fps=float(effective_fps))}"

    resolution = structured.get("resolution")
    resolution = resolution if isinstance(resolution, dict) else {}
    height = resolution.get("H") or window.get("height") or example.get("height")
    width = resolution.get("W") or window.get("width") or example.get("width")
    if height and width:
        caption = f"{caption} {_RESOLUTION_TEMPLATE.format(height=int(height), width=int(width))}"
    return caption.strip()


def _resolve_vision_path(example: dict, data_dir: str) -> str:
    """Resolve ``vision_path`` against the dataset root unless it is absolute/remote."""
    vision_path = example.get("vision_path") or example.get("video_path")
    if not isinstance(vision_path, str) or not vision_path:
        raise ValueError(
            f"cosmos3_sft_jsonl requires 'vision_path' (or 'video_path') per row; got keys {sorted(example)}."
        )
    if data_dir and not os.path.isabs(vision_path) and "://" not in vision_path:
        return os.path.join(data_dir, vision_path)
    return vision_path


@PREPROCESSOR_REGISTRY.register("cosmos3_sft_jsonl")
def cosmos3_sft_jsonl_preprocess(
    conversations: dict,
    data_dir: str = "",
    window_selection: str = "first",
    window_index: int = 0,
    media_suffix: bool = True,
    fps: Any = None,
    cfg_dropout_rate: float = 0.0,
    cfg_dropout_seed: int = _DEFAULT_CFG_DROPOUT_SEED,
    **kwargs,
):
    """One JSONL row → ``(prompt, outputs, images, videos)`` for ``dit_online``.

    Args:
        conversations: the raw dataset row (``uuid`` / ``duration`` / ``width`` /
            ``height`` / ``vision_path`` / ``t2w_windows``).
        data_dir: dataset root the relative ``vision_path`` resolves against →             set it as ``data.mm_configs.data_dir`` (the directory that holds
            ``videos/`` next to ``video_dataset_file.jsonl``).
        window_selection: ``"first"`` (default), ``"random"`` (matches the
            open-source recipe's per-epoch window draw) or ``"index"``.
        window_index: window chosen when ``window_selection="index"``.
        media_suffix: append the duration/fps/resolution sentences to a *dense*
            caption. Structured ``caption_json`` captions already carry those
            fields, so they never receive a suffix (the reference guards on
            ``used_structured_json`` the same way).
        fps: dataset fps, from ``data.mm_configs.fps``; used for the dense-caption
            duration suffix.
        cfg_dropout_rate: probability of replacing this sample's caption with the
            empty string, i.e. the reference's ``cfg_dropout_rate``
            (``vision_sft_nano.py:260``, applied at ``sft_dataset.py:324-326``). The
            structure around the caption is untouched -- the chat template, the
            ``[eos]`` and the ``<|vision_start|>`` marker still surround an empty
            prompt, which is what ``packers.py:510`` means by "CFG dropout only
            drops text *content*, not the structural text split". ``0.0`` (the
            default) draws nothing at all.
        cfg_dropout_seed: base seed of the dedicated dropout stream. It is *not* the
            condition model's sigma/noise generator: that stream is snapshotted into
            the job checkpoint and must not be disturbed here (see
            :func:`cfg_dropout_rng`).
        **kwargs: forwarded by the transform; unknown keys are ignored.

    Returns:
        ``(prompt, {}, [], [video_path])`` → VeOmni's DiT shape, with the video
        left as a reference for the loader.
    """
    window = _select_window(conversations, window_selection, window_index)
    prompt = _structured_caption(window)
    if prompt is None:
        prompt = _dense_caption(window, conversations, media_suffix, fps)
    # After the caption is complete, like the reference: ``sft_dataset.py:322``
    # strips, then ``:324-326`` clears, then ``:327`` tokenizes.
    prompt = _maybe_drop_caption(prompt, float(cfg_dropout_rate or 0.0), int(cfg_dropout_seed))
    video_path = _resolve_vision_path(conversations, data_dir)
    return prompt, {}, [], [video_path]
