"""Preprocessor for the open-source Cosmos SFT JSONL (``cosmos3_sft_jsonl``).

One row is one clip with one or more ``t2w_windows``. The row becomes
``(prompt, {}, [], [video_path])``: one window's caption (``caption_json`` verbatim,
else the dense ``caption`` plus the reference's media suffixes, then CFG dropout)
and the whole clip as a path for the dataloader to decode. The T2V/I2V/V2V draw is
not made here but in ``Cosmos3ConditionModel.process_condition``, on the resumable
noise stream.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Optional

from ....utils import logging
from ..preprocess import PREPROCESSOR_REGISTRY


logger = logging.get_logger(__name__)


# The reference ``SFTDataset``'s suffixes for a dense caption.
_DURATION_TEMPLATE = "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
_RESOLUTION_TEMPLATE = "This video is of {height}x{width} resolution."

_DEFAULT_CFG_DROPOUT_SEED = 42

# CFG dropout draws from its own ``random.Random`` per (seed, dataloader worker), never
# from the condition model's generator, whose stream is checkpointed for resume.
_cfg_dropout_streams: dict[tuple[int, int], random.Random] = {}


def _worker_id() -> int:
    from torch.utils.data import get_worker_info

    info = get_worker_info()
    return 0 if info is None else int(info.id)


def cfg_dropout_rng(seed: int = _DEFAULT_CFG_DROPOUT_SEED) -> random.Random:
    key = (int(seed), _worker_id())
    stream = _cfg_dropout_streams.get(key)
    if stream is None:
        stream = random.Random(key[0] + key[1])
        _cfg_dropout_streams[key] = stream
    return stream


def _maybe_drop_caption(prompt: Optional[str], rate: float, seed: int) -> Optional[str]:
    """Empty the whole caption with probability ``rate``; ``rate=0`` draws nothing."""
    if rate <= 0.0 or prompt is None:
        return prompt
    if rate >= 1.0:
        return ""
    if cfg_dropout_rng(seed).random() < rate:
        return ""
    return prompt


def _select_window(example: dict, window_selection: str, window_index: int) -> dict:
    windows = example.get("t2w_windows")
    if not windows:
        raise ValueError(f"cosmos3_sft_jsonl requires a non-empty 't2w_windows' list; got keys {sorted(example)}.")
    if window_selection == "first":
        index = 0
    elif window_selection == "random":
        index = random.randrange(len(windows))
    elif window_selection == "index":
        index = window_index
    else:
        raise ValueError(f"window_selection must be 'first', 'random' or 'index'; got {window_selection!r}.")
    if len(windows) > 1:
        logger.warning_once(
            "Rows with several t2w_windows: the clip is not cropped to the chosen window, so its caption is "
            "trained against the whole clip."
        )
    return windows[index]


def _structured_caption(window: dict) -> Any:
    """``caption_json`` serialized verbatim, or ``None`` when the window has none."""
    caption = window.get("caption_json")
    if caption is None:
        return None
    if isinstance(caption, str):
        return caption.strip()
    if isinstance(caption, list):
        caption = caption[0] if caption else None
        if caption is None:
            return None
        return caption if isinstance(caption, str) else json.dumps(caption, ensure_ascii=False)
    return json.dumps(caption, ensure_ascii=False)


def _dense_caption(window: dict, example: dict, media_suffix: bool, fps: Any) -> str:
    caption = window.get("caption") or example.get("caption") or example.get("prompt")
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError(f"cosmos3_sft_jsonl needs 'caption_json' or a dense 'caption'; window keys {sorted(window)}.")
    caption = caption.strip().rstrip(".") + "."
    if not media_suffix:
        return caption

    structured = window.get("caption_json")
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
    vision_path = example.get("vision_path") or example.get("video_path")
    if not isinstance(vision_path, str) or not vision_path:
        raise ValueError(f"cosmos3_sft_jsonl requires 'vision_path' (or 'video_path'); got keys {sorted(example)}.")
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
    """One JSONL row -> ``(prompt, {}, [], [video_path])``; relative paths resolve against ``mm_configs.data_dir``."""
    window = _select_window(conversations, window_selection, window_index)
    prompt = _structured_caption(window)
    if prompt is None:
        prompt = _dense_caption(window, conversations, media_suffix, fps)
    prompt = _maybe_drop_caption(prompt, float(cfg_dropout_rate or 0.0), int(cfg_dropout_seed))
    return prompt, {}, [], [_resolve_vision_path(conversations, data_dir)]
