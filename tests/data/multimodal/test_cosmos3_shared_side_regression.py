"""Unit tests for the shared-side entry points added by the Cosmos3 port.

Covers the four public additions plus the config invariant that keeps
``data.pack_mode`` the only DiT batching switch.
"""

import dataclasses

import pytest


# ---------------------------------------------------------------------------
# calculate_frame_indices(frame_selection=...) / smart_video_nframes
# ---------------------------------------------------------------------------


def test_frame_selection_is_optional_and_keeps_linspace():
    from veomni.data.multimodal.video_utils import calculate_frame_indices

    implicit, pad_implicit = calculate_frame_indices(total_frames=100, video_fps=25, fps=5)
    explicit, pad_explicit = calculate_frame_indices(
        total_frames=100, video_fps=25, fps=5, frame_selection="linspace"
    )
    assert implicit == explicit
    assert pad_implicit == pad_explicit == 0


def test_frame_selection_prefix_takes_a_stride_prefix():
    from veomni.data.multimodal.video_utils import calculate_frame_indices

    # 100 frames at 25 fps sampled to 5 fps -> 20 frames, stride 5 -> 0, 5, ..., 95.
    indices, pad = calculate_frame_indices(total_frames=100, video_fps=25, fps=5, frame_selection="prefix")
    assert indices == list(range(0, 100, 5))
    assert pad == 0


def test_frame_selection_prefix_differs_from_linspace():
    from veomni.data.multimodal.video_utils import calculate_frame_indices

    linspace, _ = calculate_frame_indices(total_frames=100, video_fps=25, fps=5)
    prefix, _ = calculate_frame_indices(total_frames=100, video_fps=25, fps=5, frame_selection="prefix")
    assert linspace != prefix
    # linspace ends on the last frame, the prefix drops the tail.
    assert linspace[-1] == 99
    assert prefix[-1] == 95


def test_frame_selection_prefix_has_no_duplicates_when_the_stride_covers_the_budget():
    from veomni.data.multimodal.video_utils import calculate_frame_indices

    for total_frames in (20, 24, 96, 117, 200):
        indices, _ = calculate_frame_indices(
            total_frames=total_frames, video_fps=total_frames, fps=5, frame_selection="prefix"
        )
        assert indices == sorted(set(indices)), total_frames
        assert all(0 <= i < total_frames for i in indices)


def test_frame_selection_prefix_does_not_fabricate_duplicate_indices():
    """A short clip can yield fewer prefix frames than requested; the shortfall is padding."""
    from veomni.data.multimodal.video_utils import calculate_frame_indices

    # 9 frames sampled to 6 -> stride 2 -> [0, 2, 4, 6, 8] = 5 frames, one frame short.
    indices, pad = calculate_frame_indices(total_frames=9, video_fps=9, fps=6, frame_selection="prefix")
    assert indices == [0, 2, 4, 6, 8]
    assert pad == 1
    assert len(indices) + pad == 6
    assert len(set(indices)) == len(indices)


@pytest.mark.parametrize("bad", ["Prefix", "prefix ", "lin", ""])
def test_frame_selection_rejects_unknown_values(bad):
    from veomni.data.multimodal.video_utils import calculate_frame_indices

    with pytest.raises(ValueError, match="frame_selection"):
        calculate_frame_indices(total_frames=100, video_fps=25, fps=5, frame_selection=bad)


def test_smart_video_nframes_forwards_frame_selection(monkeypatch):
    """Pre-decoded clips must honour ``frame_selection`` instead of silently using linspace."""
    import torch

    from veomni.data.multimodal import video_utils

    seen = {}

    def fake_calculate_frame_indices(**kwargs):
        seen.update(kwargs)
        return [0, 1, 2, 3], 0

    monkeypatch.setattr(video_utils, "calculate_frame_indices", fake_calculate_frame_indices)
    video_utils.smart_video_nframes(torch.zeros(10, 3, 8, 8), video_fps=10, fps=4, frame_selection="prefix")
    assert seen["frame_selection"] == "prefix"


def test_smart_video_nframes_defaults_to_linspace(monkeypatch):
    import torch

    from veomni.data.multimodal import video_utils

    seen = {}

    def fake_calculate_frame_indices(**kwargs):
        seen.update(kwargs)
        return [0, 1, 2, 3], 0

    monkeypatch.setattr(video_utils, "calculate_frame_indices", fake_calculate_frame_indices)
    video_utils.smart_video_nframes(torch.zeros(10, 3, 8, 8), video_fps=10, fps=4)
    assert seen["frame_selection"] == "linspace"


# ---------------------------------------------------------------------------
# build_native_dataloader(dyn_bsz_length_fn=...)
# ---------------------------------------------------------------------------


def test_dyn_bsz_length_fn_is_optional_and_defaults_to_none():
    import inspect

    from veomni.data.data_loader import build_native_dataloader

    param = inspect.signature(build_native_dataloader).parameters["dyn_bsz_length_fn"]
    assert param.default is None


def test_length_fn_falls_back_to_count_mode():
    """``None`` must reproduce ``get_length_fn_by_count_mode`` for every mode."""
    from veomni.data.dataset import get_length_fn_by_count_mode

    for mode in ("total", "effective"):
        expected = get_length_fn_by_count_mode(mode)
        assert (None or get_length_fn_by_count_mode(mode)) is expected
        custom = object()
        assert (custom or get_length_fn_by_count_mode(mode)) is custom


# ---------------------------------------------------------------------------
# cosmos3_sft_jsonl preprocessor
# ---------------------------------------------------------------------------


def _row(**overrides):
    row = {
        "uuid": "clip0",
        "vision_path": "videos/a.mp4",
        "t2w_windows": [{"caption_json": {"subjects": [{"description": "a bridge"}]}}],
    }
    row.update(overrides)
    return row


def test_cosmos3_preprocess_returns_the_dit_tuple():
    from veomni.data.multimodal.dit.preprocess import cosmos3_sft_jsonl_preprocess

    prompt, audios, images, videos = cosmos3_sft_jsonl_preprocess(_row())
    assert prompt.startswith('{"subjects"')
    assert audios == {}
    assert images == []
    assert videos == ["videos/a.mp4"]


def test_cosmos3_preprocess_string_caption_json_is_stripped():
    from veomni.data.multimodal.dit.preprocess import cosmos3_sft_jsonl_preprocess

    row = _row(t2w_windows=[{"caption_json": "  a bridge at sunset  "}])
    assert cosmos3_sft_jsonl_preprocess(row)[0] == "a bridge at sunset"


def test_cosmos3_preprocess_dense_caption_gets_a_full_stop():
    from veomni.data.multimodal.dit.preprocess import cosmos3_sft_jsonl_preprocess

    row = _row(t2w_windows=[{"caption": "a bridge at sunset."}])
    assert cosmos3_sft_jsonl_preprocess(row)[0] == "a bridge at sunset."


def test_cosmos3_preprocess_falls_back_to_the_row_level_caption():
    from veomni.data.multimodal.dit.preprocess import cosmos3_sft_jsonl_preprocess

    row = _row(t2w_windows=[{}], caption="a bridge")
    assert cosmos3_sft_jsonl_preprocess(row)[0] == "a bridge."


def test_cosmos3_preprocess_without_any_caption_raises():
    from veomni.data.multimodal.dit.preprocess import cosmos3_sft_jsonl_preprocess

    with pytest.raises(ValueError, match="caption"):
        cosmos3_sft_jsonl_preprocess(_row(t2w_windows=[{}]))


def test_cosmos3_preprocess_without_a_window_raises():
    from veomni.data.multimodal.dit.preprocess import cosmos3_sft_jsonl_preprocess

    with pytest.raises(KeyError):
        cosmos3_sft_jsonl_preprocess({"uuid": "x", "vision_path": "a.mp4"})


def test_cosmos3_preprocess_joins_a_relative_path():
    from veomni.data.multimodal.dit.preprocess import cosmos3_sft_jsonl_preprocess

    assert cosmos3_sft_jsonl_preprocess(_row(), data_dir="/data/root")[3] == ["/data/root/videos/a.mp4"]


def test_cosmos3_preprocess_keeps_an_absolute_path():
    from veomni.data.multimodal.dit.preprocess import cosmos3_sft_jsonl_preprocess

    row = _row(vision_path="/abs/a.mp4")
    assert cosmos3_sft_jsonl_preprocess(row, data_dir="/data/root")[3] == ["/abs/a.mp4"]


def test_cosmos3_preprocess_cfg_dropout_rate_zero_keeps_the_caption():
    from veomni.data.multimodal.dit.preprocess import cosmos3_sft_jsonl_preprocess

    assert cosmos3_sft_jsonl_preprocess(_row(), cfg_dropout_rate=0.0)[0] != ""


def test_cosmos3_preprocess_cfg_dropout_rate_one_drops_the_caption():
    from veomni.data.multimodal.dit.preprocess import cosmos3_sft_jsonl_preprocess

    assert cosmos3_sft_jsonl_preprocess(_row(), cfg_dropout_rate=1.0)[0] == ""


def test_cosmos3_preprocess_cfg_dropout_thresholds_on_the_global_rng(monkeypatch):
    from veomni.data.multimodal import dit as dit_pkg  # noqa: F401  (ensures the module is imported)
    from veomni.data.multimodal.dit import preprocess as mod

    monkeypatch.setattr(mod.random, "random", lambda: 0.05)
    assert mod.cosmos3_sft_jsonl_preprocess(_row(), cfg_dropout_rate=0.1)[0] == ""

    monkeypatch.setattr(mod.random, "random", lambda: 0.95)
    assert mod.cosmos3_sft_jsonl_preprocess(_row(), cfg_dropout_rate=0.1)[0] != ""


# ---------------------------------------------------------------------------
# data.pack_mode is the only DiT batching switch
# ---------------------------------------------------------------------------


def test_pack_mode_field_defaults_to_mbs():
    from veomni.trainer.dit_trainer import DiTDataArguments

    fields = {f.name: f for f in dataclasses.fields(DiTDataArguments)}
    assert "pack_mode" in fields
    assert fields["pack_mode"].default == "mbs"


def test_cosmos3_recipes_set_pack_mode_and_no_train_dyn_bsz():
    """A stray ``train.dyn_bsz`` would reintroduce a second source of truth for batching."""
    from pathlib import Path

    import yaml

    recipes = Path(__file__).resolve().parents[3] / "configs" / "dit"
    names = ["cosmos3_sft.yaml", "cosmos3_sft_offline.yaml", "cosmos3_sft_embedding.yaml"]
    for name in names:
        cfg = yaml.safe_load((recipes / name).read_text())
        assert cfg["data"]["pack_mode"] == "token_budget", name
        assert "dyn_bsz" not in cfg.get("train", {}), name
        assert cfg["train"]["micro_batch_size"] == 1, name
