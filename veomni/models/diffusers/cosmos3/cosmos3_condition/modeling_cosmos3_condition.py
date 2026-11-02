"""Cosmos3 condition model: online video encoding and rectified-flow targets.

Two stages, both called by ``DiTTrainer``:

  1. :meth:`Cosmos3ConditionModel.get_condition` decodes each sample's video, runs the
     frozen Wan2.2 VAE, normalizes the latents and tokenizes the prompt for the frozen
     Qwen3-VL tower. Returns per-sample lists.
  2. :meth:`Cosmos3ConditionModel.process_condition` samples one sigma per sample
     (``waver`` + per-resolution shift warp), draws the T2V/I2V/V2V conditioning
     mixture and the noise, and builds ``(x_t, timestep, target, condition_mask)``.

Numerical contract, copied from ``cosmos_framework``:

  * sigma is sampled by ``TrainTimeSampler`` (``waver``) and then shifted by
    ``sigma = shift * t / (1 + (shift - 1) * t)``; this module only supplies the
    per-sample shift.
  * ``x_t = noise * sigma + clean * (1 - sigma)`` with ``target = noise - clean``. The
    reference names the clean latent ``x_1`` and the noise ``x_0``, the opposite of the
    diffusion literature's convention.
  * the transformer receives ``sigma * num_train_timesteps`` (``omni_mot_model.py``:
    ``timesteps = sigmas * max_timestep``), because its timestep embedder was pretrained
    on a 1000-step scale and applies ``timestep_scale`` (1e-3) before the sinusoid.
  * the loss is computed on the transformer side, where the prediction exists.

``flow_matching`` and ``diffusers`` are optional for code that only imports this module
(config tooling, CPU tests), so they are imported inside the functions that need them.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from .....distributed.parallel_state import get_parallel_state
from .....utils import logging
from .configuration_cosmos3_condition import Cosmos3ConditionModelConfig
from .latent_cache import LatentRecipe
from .wan22_vae import Wan22VAE


logger = logging.get_logger(__name__)


def _import_flow_matching():
    """Return ``(TrainTimeSampler, interpolate)`` from the shared Cosmos3 core.

    Imported lazily: ``cosmos3_core.flow_matching`` is shared with the
    transformer side, and it pulls in ``diffusers`` scheduling helpers that a
    config-only or CPU test path should not need.
    """
    from ..cosmos3_core.flow_matching import TrainTimeSampler, interpolate

    return TrainTimeSampler, interpolate


class Cosmos3ConditionModel(PreTrainedModel):
    """Frozen VAE + tokenizer + rectified-flow sampler for Cosmos3 SFT.

    The model holds no trainable parameters: ``DiTTrainer`` freezes it and never
    shards it. Its state is the VAE weights (needed only when encoding) and the
    RNG stream that makes noise/timestep draws resumable.
    """

    config_class = Cosmos3ConditionModelConfig
    supports_gradient_checkpointing = False

    def __init__(self, config: Cosmos3ConditionModelConfig, meta_init: bool = False, **kwargs):
        super().__init__(config, **kwargs)
        self.meta_init = meta_init
        # Components are plain attributes until ``_load_components`` assigns them;
        # the VAE is an ``nn.Module`` so the trainer's ``.to(device)`` moves it.
        self.vae: Optional[Wan22VAE] = None
        self.tokenizer = None
        self.processor = None
        self.train_time_sampler = None
        self.flow_interpolate = None

        self.seed = config.seed
        # A CPU generator, deliberately: ``TrainTimeSampler`` draws with
        # ``torch.rand`` on CPU, and a device-independent stream keeps a saved
        # RNG state restorable on a different accelerator (resume on GPU鈫擭PU).
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(self.seed if self.seed is not None else 42) + get_parallel_state().dp_rank)

        self._load_components()

    # 鈹€鈹€ construction 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def _load_components(self) -> None:
        config = self.config
        base = config.base_model_path
        if not base:
            raise ValueError(
                "Cosmos3ConditionModelConfig.base_model_path is empty; the condition model cannot resolve "
                "the VAE / tokenizer. Set model.condition_model_path in the run config."
            )
        logger.info_rank0(f"Loading Cosmos3 condition components from {base}.")

        self._init_flow_matching()
        self._init_vae()
        self._init_text_components()

    def _init_flow_matching(self) -> None:
        TrainTimeSampler, interpolate = _import_flow_matching()
        self.train_time_sampler = TrainTimeSampler(self.config.train_time_video_distribution)
        self.flow_interpolate = interpolate

    def _init_vae(self) -> None:
        config = self.config
        if self.meta_init:
            # ``offline_training`` replays precomputed latents and never runs the
            # encoder, so only the config is needed 鈥?loading 250M frozen VAE
            # parameters per rank would be pure waste.
            from diffusers import AutoencoderKLWan

            vae_config = AutoencoderKLWan.load_config(config.base_model_path, subfolder=config.vae_subfolder or None)
            vae = AutoencoderKLWan.from_config(vae_config)
            vae.eval()
            vae.requires_grad_(False)
            self.vae = Wan22VAE(vae, component_name=config.component_dir(config.vae_subfolder))
        else:
            self.vae = Wan22VAE.from_pretrained(
                config.base_model_path,
                subfolder=config.vae_subfolder,
                torch_dtype=getattr(torch, config.vae_dtype),
                temporal_chunk_frames=config.vae_temporal_chunk_frames,
            )
        if config.vae_temporal_chunk_frames:
            logger.info_rank0(
                "Cosmos3 VAE encodes in %d-frame temporal chunks (approximate only at chunk warm-up boundaries).",
                config.vae_temporal_chunk_frames,
            )

    def _init_text_components(self) -> None:
        """Load the Qwen3-VL tokenizer (required) and processor (best effort).

        SFT needs token ids only, so a checkpoint without a processor is usable.
        I2V/V2V do not need one either: their conditioning frames are the leading
        latent frames of the *same* clip (:meth:`_sample_condition_mask`), not a
        separate reference image that would have to be encoded through the
        processor -- which is why the mixture can be turned on without one.
        """
        from transformers import AutoProcessor, AutoTokenizer

        config = self.config
        tokenizer_dir = config.component_dir(config.text_tokenizer_subfolder)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load the Cosmos3 text tokenizer from {tokenizer_dir}. The checkpoint must ship "
                f"Qwen3-VL tokenizer files at the root or under {config.text_tokenizer_subfolder!r}."
            ) from exc

        processor_dir = config.component_dir(config.processor_subfolder)
        try:
            self.processor = AutoProcessor.from_pretrained(processor_dir)
        except Exception:
            self.processor = None
            logger.warning_rank0(
                "No AutoProcessor found at %s; SFT tokenizes captions directly and does not need one "
                "(image/video conditioning here reuses the clip's own leading latent frames).",
                processor_dir,
            )
        if self.processor is not None and getattr(self.processor, "tokenizer", None) is not None:
            self.tokenizer = self.processor.tokenizer

    # 鈹€鈹€ RNG state (resume contract) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def rng_state_dict(self) -> dict[str, torch.Tensor]:
        """Snapshot the noise/sigma stream for the job-level checkpoint.

        ``process_condition`` draws both sigma and the noise from
        ``self.generator``, so a resumed run that does not restore it replays the
        stream from the construction-time seed instead of continuing it, and the
        first steps after resume reuse the pre-resume noise.
        """
        return {"generator": self.generator.get_state()}

    def load_rng_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore a snapshot produced by :meth:`rng_state_dict`."""
        self.generator.set_state(state["generator"])

    # 鈹€鈹€ latent cache identity (offline_embedding / offline_training) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def vae_component_dir(self) -> str:
        """Directory holding the VAE checkpoint files, whether or not they loaded.

        ``meta_init`` (``offline_training``) builds the VAE from its config and
        never reads the weights, but the weight *files* still define which latents
        the training run is replaying -- so the cache identity reads them from disk
        rather than from the (possibly empty) module.
        """
        component_name = getattr(self.vae, "component_name", None)
        if component_name and os.path.isdir(component_name):
            return component_name
        return self.config.component_dir(self.config.vae_subfolder)

    def latent_cache_recipe(self) -> "LatentRecipe":
        """The run-level half of the latent cache key, computed once.

        Memoized because it hashes the VAE checkpoint files (~0.5-1 GB): doing that
        per sample would dominate the fill.
        """
        cached = getattr(self, "_latent_cache_recipe", None)
        if cached is not None:
            return cached
        from .latent_cache import LatentRecipe, checkpoint_digest, code_digest, latent_stats_digest, tokenizer_digest

        vae = self.vae
        recipe = LatentRecipe(
            vae_dtype=str(self.config.vae_dtype),
            vae_weights=checkpoint_digest(self.vae_component_dir()),
            latents_stats=latent_stats_digest(vae),
            temporal_chunk_frames=int(
                getattr(vae, "temporal_chunk_frames", self.config.vae_temporal_chunk_frames) or 0
            ),
            temporal_chunk_warmup=int(getattr(vae, "temporal_chunk_warmup", 0) or 0),
            code=code_digest(),
            tokenizer=tokenizer_digest(self.tokenizer, max_sequence_length=self.config.max_sequence_length),
            latent_patch_size=int(self.config.latent_patch_size),
            spatial_scale=int(getattr(vae, "spatial_scale", 0) or 0),
            temporal_scale=int(getattr(vae, "temporal_scale", 0) or 0),
            resolution=str(self.config.resolution),
        )
        self._latent_cache_recipe = recipe
        logger.info_rank0(
            "Latent cache recipe: %s (schema=%s, vae_dtype=%s, vae_weights=%s, code=%s).",
            recipe.digest()[:16],
            recipe.schema,
            recipe.vae_dtype,
            recipe.vae_weights[:16],
            recipe.code[:16],
        )
        return recipe

    def latent_cache_key(self, prompt: str, frames: torch.Tensor) -> str:
        """Content-addressed key of one sample's cached latent.

        Args:
            frames: The decoded clip (``[T, C, H, W]``) as the transform produced
                it. Its dtype, shape and bytes are the clip's identity -- see
                ``latent_cache`` for why a content hash replaces a file path.

        Returns:
            A 64-character SHA-256 hex digest.
        """
        from .latent_cache import latent_cache_key

        if frames.dim() != 4:
            raise ValueError(f"latent_cache_key expects a [T, C, H, W] clip, got {tuple(frames.shape)}")
        target_h, target_w = self.latent_grid_target(int(frames.shape[2]), int(frames.shape[3]))
        return latent_cache_key(
            self.latent_cache_recipe(),
            prompt=prompt,
            frames=frames,
            fed_geometry=(int(frames.shape[0]), int(frames.shape[1]), target_h, target_w),
        )

    def latent_cache_keys(self, micro_batch: dict[str, Any]) -> list[str]:
        """One cache key per sample of a collated micro batch, in batch order.

        Must be called while the batch is still on the CPU (``DiTDataCollator``
        hands over lists): the clip digest reads the decoded pixels, and a
        device-resident tensor would turn that into a host round-trip per sample.

        Args:
            micro_batch: The collated batch, with ``inputs`` and ``videos`` lists.

        Returns:
            Keys in the same order as ``micro_batch["inputs"]``.
        """
        inputs = micro_batch.get("inputs")
        videos = micro_batch.get("videos")
        if inputs is None or videos is None:
            raise ValueError(
                "latent_cache_keys needs the raw batch (`inputs` + `videos`); got keys "
                f"{sorted(micro_batch)}. Call it before `get_condition` replaces the batch."
            )
        if len(inputs) != len(videos):
            raise ValueError(f"Got {len(inputs)} prompts but {len(videos)} video entries; they must match.")
        keys = []
        for index, prompt in enumerate(inputs):
            frames = self._frames_from_batch_entry(videos[index], index)
            keys.append(self.latent_cache_key(prompt, frames))
        return keys

    # 鈹€鈹€ get_condition: raw data 鈫?per-sample condition 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    @torch.no_grad()
    def get_condition(
        self,
        inputs: list[str],
        videos: list,
        images: Optional[list] = None,
        resolution: Any = None,
        video_fps: Optional[list] = None,
        **kwargs: Any,
    ) -> dict[str, list]:
        """Encode prompts and videos into the per-sample condition lists.

        Args:
            inputs: ``list[str]`` 鈥?one prompt (the structured-JSON caption) per
                sample.
            videos: one entry per sample. Either a decoded frame tensor
                ``[T, C, H, W]`` (what the ``dit_online`` transform produces via
                ``fetch_videos``), a list of ``[C, H, W]`` frames, or a path /
                bytes / list-of-paths that the shared video loader decodes on the
                spot.
            images: unused for text-to-video; accepted so the multi-modal batch
                keys the collator always emits do not have to be special-cased.
            resolution: optional per-sample resolution bucket (``str`` or
                ``list[str]``, e.g. ``"256"``) overriding ``config.resolution``;
                it selects the sigma shift in :meth:`process_condition`.
            video_fps: optional per-sample frame rate, forwarded from the data
                transform (``dit_online`` reads it off the container). Carried
                straight through to :meth:`process_condition` and then to the
                transformer, where it scales the mRoPE temporal axis by
                ``base_fps / fps`` (``mrope.py:164-180``). It is *conditioning
                geometry*, not a nicety: without it a 5 fps clip is positioned as if
                it ran at ``base_fps``. ``None`` (a batch that lost the column) is
                passed through as ``None`` and the transformer logs the fallback.

        Returns:
            ``dict[str, list]`` with one entry per sample:

            * ``latents``: normalized VAE latent ``[C, T_lat, H_lat, W_lat]``
              (no batch dimension 鈥?the transformer consumes one clip's latent
              grid per list element).
            * ``resolution``: the shift bucket (``str``) the sample was encoded
              for.
            * ``input_ids`` / ``ar_attention_mask``: the prompt tokenized with
              the Qwen3-VL chat template, plus the generation markers (see
              :meth:`_tokenize_prompt`). These are the *frozen tower's*
              conditioning, not a cross-entropy target.
            * ``fps``: the clip's frame rate (``float`` per sample), or ``None``
              per sample when the batch did not carry one.
        """
        if self.vae is None:
            raise RuntimeError("Cosmos3ConditionModel.vae is not loaded; cannot encode videos.")
        if len(inputs) != len(videos):
            raise ValueError(f"Got {len(inputs)} prompts but {len(videos)} video entries; they must match.")
        if not inputs:
            raise ValueError("Cosmos3ConditionModel.get_condition needs at least one sample.")

        buckets = self._resolve_resolutions(resolution, len(inputs))
        fps_per_sample = self._resolve_fps(video_fps, len(inputs))
        condition: dict[str, list] = {
            "latents": [],
            "resolution": [],
            "input_ids": [],
            "ar_attention_mask": [],
            "fps": [],
        }
        # Measurement probe (VEOMNI_VAE_PROBE=1): how much of the training step is
        # the frozen VAE encoder. Device-synchronised, so it is real device time and
        # not just enqueue time -- which also makes the probe itself cost a sync.
        _probe = os.environ.get("VEOMNI_VAE_PROBE", "0") == "1"
        _encode_s = 0.0
        _t_begin = time.perf_counter()
        for index, prompt in enumerate(inputs):
            frames = self._frames_from_batch_entry(videos[index], index)
            video = self._prepare_video(frames)
            # [1, C, T, H, W] -> [C, T, H, W]: one clip per list element.
            if _probe:
                _sync = getattr(getattr(torch, "npu", None), "synchronize", None)
                if callable(_sync):
                    _sync()
                _t_encode = time.perf_counter()
                _latent = self.vae.encode_latents(video)[0]
                if callable(_sync):
                    _sync()
                _encode_s += time.perf_counter() - _t_encode
            else:
                _latent = self.vae.encode_latents(video)[0]
            condition["latents"].append(_latent)
            condition["resolution"].append(buckets[index])
            input_ids, attention_mask = self._tokenize_prompt(prompt, index)
            condition["input_ids"].append(input_ids)
            condition["ar_attention_mask"].append(attention_mask)
            condition["fps"].append(fps_per_sample[index])
        if _probe:
            _total_s = time.perf_counter() - _t_begin
            _tokens = sum(int(ids.numel()) for ids in condition["input_ids"])
            logger.info_rank0(
                f"[VAE-PROBE] samples={len(inputs)} encode={_encode_s * 1e3:.1f}ms "
                f"get_condition_total={_total_s * 1e3:.1f}ms "
                f"encode_share_of_call={_encode_s / max(_total_s, 1e-9) * 100:.1f}% "
                f"prompt_tokens={_tokens}"
            )
        self._log_condition_geometry(condition, buckets)
        return condition

    def _resolve_fps(self, video_fps: Any, count: int) -> list[Optional[float]]:
        """Broadcast the batch's ``video_fps`` column to one entry per sample.

        Missing or malformed values become ``None`` (the transformer then falls back
        to ``base_fps``, exactly like the reference's ``packers.py:274``) rather than
        an exception: a batch from a transform that does not report fps is a
        supported shape (``offline_training`` replays latents, and shape-only tests
        build batches by hand). A *non-finite* or non-positive value is a data bug
        and does raise.
        """
        if video_fps is None:
            return [None] * count
        if isinstance(video_fps, (int, float)):
            values: list = [video_fps] * count
        else:
            values = list(video_fps)
            if len(values) != count:
                raise ValueError(f"Got {len(values)} fps entries but {count} samples; they must match.")
        resolved: list[Optional[float]] = []
        for index, value in enumerate(values):
            if value is None:
                resolved.append(None)
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                resolved.append(None)
                continue
            if not math.isfinite(numeric) or numeric <= 0:
                raise ValueError(f"Sample {index}: video_fps must be positive and finite, got {value!r}.")
            resolved.append(numeric)
        return resolved

    def _log_condition_geometry(self, condition: dict[str, list], buckets: list[str]) -> None:
        """Print what the mRoPE time base will actually see, once per run.

        The fps column is the one input that can go missing without a shape error
        (a collator that drops it, an offline replay, a transform that never reported
        it) and its absence degrades silently into "trained as if base_fps". Printing
        the observed range turns that into something visible in the log. The same
        applies to the prompt length the temporal cursor starts from, which is why it
        is logged next to it.
        """
        fps_values = [value for value in condition["fps"] if value is not None]
        lengths = [int(ids.numel()) for ids in condition["input_ids"]]
        if not fps_values:
            logger.warning_once(
                "Cosmos3 condition saw no fps for any sample (batch size %d); the mRoPE temporal axis will fall "
                "back to base_fps. If this is a training run, the data transform is not forwarding `video_fps`.",
                len(condition["fps"]),
            )
            return
        logger.info_once(
            "Cosmos3 condition geometry: fps min/max=%.4f/%.4f (%d/%d samples reported), prompt tokens "
            "min/max=%d/%d (generation markers %s), resolution bucket(s) %s.",
            min(fps_values),
            max(fps_values),
            len(fps_values),
            len(condition["fps"]),
            min(lengths),
            max(lengths),
            "appended" if self.config.append_generation_markers else "disabled",
            # Joined, not a list: ``logger.info_once`` is an ``lru_cache`` over the log
            # arguments, so an unhashable argument raises
            # ``TypeError: unhashable type: 'list'`` inside the *first* ``get_condition``
            # of the run -- which is before step 1, i.e. the entire run dies at the very
            # first batch. Same rule for ``warning_once``.
            ",".join(sorted(set(buckets))),
        )

    def _frames_from_batch_entry(self, entry: Any, index: int) -> torch.Tensor:
        """Normalize one batch entry to a ``[T, C, H, W]`` frame tensor.

        ``DiTDataCollator`` keeps per-key lists, so a sample's video arrives as a
        list holding one decoded clip (``fetch_videos`` returns one tensor per
        input reference), while MiniMax-H3-style transforms hand over a list of
        per-frame tensors. Both, plus a raw path/bytes reference, are accepted.
        """
        if isinstance(entry, torch.Tensor):
            frames = entry
        elif isinstance(entry, (list, tuple)) and entry:
            first = entry[0]
            if isinstance(first, torch.Tensor):
                if first.dim() == 4:
                    # One decoded clip per sample, wrapped in the collator's list.
                    if len(entry) != 1:
                        raise ValueError(f"Sample {index}: Cosmos3 accepts one video per sample, got {len(entry)}.")
                    frames = first
                elif first.dim() == 3:
                    # One tensor per frame, each [C, H, W].
                    frames = torch.stack(list(entry))
                else:
                    raise ValueError(f"Sample {index}: unsupported frame tensor rank {first.dim()}.")
            else:
                frames = self._decode_video_references(list(entry))
        elif isinstance(entry, (str, bytes)):
            frames = self._decode_video_references([entry])
        else:
            raise TypeError(f"Sample {index}: unsupported video entry type {type(entry).__name__}.")

        if frames.dim() == 3:
            frames = frames.unsqueeze(0)
        if frames.dim() != 4:
            raise ValueError(f"Sample {index}: expected frames shaped [T, C, H, W], got {tuple(frames.shape)}.")
        if frames.shape[1] != 3:
            # [T, H, W, C] from a raw decoder.
            if frames.shape[-1] != 3:
                raise ValueError(f"Sample {index}: frames must be 3-channel, got {tuple(frames.shape)}.")
            frames = frames.permute(0, 3, 1, 2)
        return frames

    @staticmethod
    def _decode_video_references(references: list) -> torch.Tensor:
        """Decode a path/bytes video reference with the shared loader.

        Only reached when the data transform did not decode (``data_transform``
        misconfigured); the shared loader keeps frame sampling consistent with
        ``dit_online``.
        """
        from .....data.multimodal.video_utils import fetch_videos

        decoded, _ = fetch_videos(references)
        return decoded[0]

    def _prepare_video(self, frames: torch.Tensor) -> torch.Tensor:
        """``[T, C, H, W]`` (uint8 or float) 鈫?``[1, 3, T, H, W]`` float in ``[-1, 1]``.

        The VAE contract is ``[-1, 1]`` (``VideoProcessor.preprocess_video``'s
        ``min_value=-1``), so the range is normalized here instead of being
        assumed 鈥?a mistyped ``[0, 255]`` tensor produces a plausible-looking but
        useless latent rather than an error.
        """
        source_dtype = frames.dtype
        video = frames.detach().to(dtype=torch.float32)
        if source_dtype == torch.uint8:
            video = video.div(255.0)
        elif video.numel():
            if float(video.max()) > 1.0 + 1e-3:
                video = video.div(255.0)  # [0, 255] floats
            elif float(video.min()) < -1e-3:
                video = video.add(1.0).div(2.0)  # already [-1, 1]
        video = video.mul(2.0).sub(1.0)  # [0, 1] -> [-1, 1]

        video = self._fit_latent_grid(video)
        # [T, C, H, W] -> [1, C, T, H, W]
        video = video.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
        return video.to(device=self.vae.device, dtype=self.vae.dtype)

    def _fit_latent_grid(self, video: torch.Tensor) -> torch.Tensor:
        """Round ``H, W`` to a multiple of ``vae.spatial_scale * latent_patch_size``.

        The transformer patchifies the latent by ``latent_patch_size`` on top of
        the VAE's 16x spatial compression, so a latent side that is not a
        multiple of the patch size would leave a partial patch. Bridge clips are
        already 256x256 (both divisors), hence a no-op there; other sources are
        rounded rather than rejected so a mixed-resolution dump still trains.
        """
        _, _, height, width = video.shape
        target_h, target_w = self.latent_grid_target(height, width)
        if (target_h, target_w) == (height, width):
            return video
        logger.info_once(
            "Cosmos3 condition resizes video frames to the VAE x patch grid (multiple of %d pixels); "
            "e.g. %dx%d -> %dx%d.",
            self.latent_grid_size,
            height,
            width,
            target_h,
            target_w,
        )
        resized = F.interpolate(
            video,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return resized

    @property
    def latent_grid_size(self) -> int:
        """Pixel multiple the encoder's input side is rounded to."""
        return int(self.vae.spatial_scale) * int(self.config.latent_patch_size)

    def latent_grid_target(self, height: int, width: int) -> tuple[int, int]:
        """``(H, W)`` the encoder will actually see for a ``height x width`` clip.

        Split out of :meth:`_fit_latent_grid` so the latent cache key can name the
        geometry fed to the VAE without materializing the resized tensor (the key
        is computed on the CPU-resident batch, before the H2D move).
        """
        grid = self.latent_grid_size
        target_h = max(grid, int(round(height / grid)) * grid)
        target_w = max(grid, int(round(width / grid)) * grid)
        return target_h, target_w

    def _tokenize_prompt(self, prompt: str, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize one prompt exactly as the reference dataset does.

        The reference tokenizes with the Qwen chat template, one ``user`` turn,
        ``add_generation_prompt=True`` and no system prompt
        (``cosmos_framework/model/generator/reasoner/qwen3_vl/utils.py::tokenize_caption``);
        reproducing it here keeps the training prompt byte-identical to
        inference.

        **Generation markers.** The reference's packer does not stop at the chat
        template: it appends ``[eos_token_id, start_of_generation]`` to the text span
        whenever something is generated after it (``sequence.py:257-260``, with
        ``start_of_generation == <|vision_start|>`` and *no* BOS because
        ``special_tokens`` carries no ``bos_token_id`` -- ``omni_mot_model.py:399-403``).
        The diffusion tower was pretrained with that boundary marker, so a prompt
        without it hands the generation tower a slightly out-of-distribution
        conditioning and loses the explicit "the generation segment starts here"
        signal. Ids: eos 151645, ``<|vision_start|>`` 151652.

        **Order: truncate first, then append** -- the reference clips ``text_ids`` to
        ``max_caption_tokens`` in the dataset (``sft_dataset.py:174-177``) and only
        adds the two markers later, in the packer, so the packed length can reach
        ``max_caption_tokens + 2``. Reserving the two slots here instead would
        silently shorten a long prompt relative to the reference.
        """
        if self.tokenizer is None:
            raise RuntimeError("Cosmos3ConditionModel.tokenizer is not loaded; cannot tokenize prompts.")
        if not isinstance(prompt, str):
            raise TypeError(f"Sample {index}: prompt must be a string, got {type(prompt).__name__}.")
        conversation = [{"role": "user", "content": prompt}]
        try:
            token_ids = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                add_vision_id=False,
                return_dict=False,
            )
        except TypeError:
            # Tokenizers whose template does not know ``add_vision_id``.
            token_ids = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
        input_ids = torch.tensor(list(token_ids), dtype=torch.long)
        if input_ids.numel() > self.config.max_sequence_length:
            logger.warning_once(
                "Cosmos3 condition truncates a prompt of %d tokens to max_sequence_length=%d.",
                input_ids.numel(),
                self.config.max_sequence_length,
            )
            input_ids = input_ids[: self.config.max_sequence_length]
        if self.config.append_generation_markers:
            input_ids = torch.cat([input_ids, self.generation_marker_ids()])
        return input_ids, torch.ones_like(input_ids)

    def generation_marker_ids(self) -> torch.Tensor:
        """``[eos_token_id, <|vision_start|>]`` as a ``long`` tensor (memoized).

        Resolved from the tokenizer rather than hard-coded so a checkpoint with a
        remapped special-token table cannot silently get the wrong ids; the known
        values (151645 / 151652) are what the reference's ``special_tokens`` dict
        holds (``omni_mot_model.py:399-403``, ``lidar_transfer_test.py:363``).
        """
        cached = getattr(self, "_generation_marker_ids", None)
        if cached is not None:
            return cached.to(device="cpu")
        if self.tokenizer is None:
            raise RuntimeError("Cosmos3ConditionModel.tokenizer is not loaded; cannot resolve special token ids.")

        def resolve(token: str, fallback: Optional[int], what: str) -> int:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            unk = getattr(self.tokenizer, "unk_token_id", None)
            if token_id is None or (unk is not None and token_id == unk):
                if fallback is None:
                    raise ValueError(
                        f"The tokenizer has no {what} token ({token!r}); the Cosmos3 generation boundary needs it."
                    )
                logger.warning_once(
                    "Cosmos3 condition: tokenizer has no %s token (%r); falling back to id %d.",
                    what,
                    token,
                    fallback,
                )
                return fallback
            return int(token_id)

        eos = self.tokenizer.eos_token_id
        if eos is None:
            eos = resolve("<|im_end|>", 151645, "eos")
        vision_start = resolve("<|vision_start|>", 151652, "vision_start")
        markers = torch.tensor([int(eos), int(vision_start)], dtype=torch.long)
        self._generation_marker_ids = markers
        return markers.to(device="cpu")

    # 鈹€鈹€ process_condition: sigma + noise 鈫?model inputs and targets 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    @torch.no_grad()
    def process_condition(
        self,
        latents: list[torch.Tensor],
        resolution: Any = None,
        input_ids: Optional[list[torch.Tensor]] = None,
        ar_attention_mask: Optional[list[torch.Tensor]] = None,
        fps: Optional[list] = None,
        **kwargs: Any,
    ) -> dict[str, list]:
        """Build the transformer's training inputs from the clean latents.

        Every returned key is a parameter name of the Cosmos3 transformer's
        ``forward``; the trainer unpacks this dict as ``model(**output)``, so the
        names here are the interface between the two halves of the model.

        Conditioning (T2V / I2V / V2V): one mixture weight per sample is drawn
        from ``config.conditioning_config`` (``{clean_prefix_latent_frames:
        probability}``, default ``{0: 0.7, 1: 0.2, 2: 0.1}``) and expanded into a
        per-frame ``condition_mask`` -- see :meth:`_sample_condition_mask` for the
        frame rule and its upstream source. The mask does two things here:

        * ``sigma_eff = sigma * (1 - condition_mask)`` -- the clean frames go into
          ``x_t`` as the **unmodified clean latent** (``sigma=0``), which is what
          ``posttrain/model/cosmos3.py:1156-1167`` does
          (``sigmas_list[i] = t * (1 - condition_mask[i])``). There is no
          conditioning noise augmentation: ``imgvid_cond_noise_aug`` does not
          exist in cosmos-framework (it is a MiniMax-H3 knob), and its value here
          is pinned to ``1.0``/disabled by the config's validation.
        * the velocity target stays **unmasked** (``noise - clean`` everywhere):
          upstream notes ``dot_x_t`` is "unmasked ... masking is applied only in
          the loss reduction" (``cosmos3.py:1157-1158``), and the loss then drops
          the conditioned frames' contribution.

        Returns:
            ``hidden_states`` ``[C, T_lat, H_lat, W_lat]`` (noised latent),
            ``timestep`` ``[1]`` (``sigma * num_train_timesteps``),
            ``training_target`` ``[C, T_lat, H_lat, W_lat]`` (``noise - clean``),
            ``condition_mask`` ``[T_lat, 1, 1]`` (``1`` = clean conditioning frame,
            ``0`` = generated frame -- the same axis convention and shape as
            ``sequence.py:574`` and ``modality.py:56``),
            plus the pass-through prompt tokens ``input_ids`` /
            ``ar_attention_mask`` for the frozen text tower, and the pass-through
            ``fps`` column the transformer needs for mRoPE FPS modulation.
        """
        if not latents:
            raise ValueError("Cosmos3ConditionModel.process_condition needs at least one sample.")
        if self.train_time_sampler is None or self.flow_interpolate is None:
            raise RuntimeError("Cosmos3 condition flow-matching components are not loaded.")
        count = len(latents)
        buckets = self._resolve_resolutions(resolution, count)
        sigmas = self._sample_sigmas(buckets)

        processed: dict[str, list] = {
            "hidden_states": [],
            "timestep": [],
            "training_target": [],
            "condition_mask": [],
        }
        for index, clean in enumerate(latents):
            if clean.dim() != 4:
                raise ValueError(
                    f"Sample {index}: expected a [C, T, H, W] latent, got {tuple(clean.shape)}. "
                    "get_condition returns one clip without a batch dimension."
                )
            sigma = sigmas[index : index + 1]  # [1]
            # Noise comes from the CPU generator so the stream (and its restored
            # state) does not depend on the training device.
            noise = torch.randn(clean.shape, generator=self.generator, dtype=torch.float32)
            noise = noise.to(device=clean.device, dtype=clean.dtype)
            # The conditioning draw comes *after* this sample's noise draw, so a
            # seed that produced a given noise tensor before conditioning existed
            # still produces it: the mask only consumes the stream after it.
            condition_mask = self._sample_condition_mask(clean)
            hidden_states, training_target = self._interpolate_sample(noise, clean, sigma, condition_mask)
            processed["hidden_states"].append(hidden_states)
            processed["timestep"].append((sigma * self.config.num_train_timesteps).to(torch.float32))
            processed["training_target"].append(training_target)
            processed["condition_mask"].append(condition_mask)
        # The frozen text tower's tokens pass through untouched; they are
        # conditioning, not a cross-entropy target, so no ``labels`` are built.
        if input_ids is not None:
            processed["input_ids"] = list(input_ids)
        if ar_attention_mask is not None:
            processed["ar_attention_mask"] = list(ar_attention_mask)
        # The frame rate travels with the batch to the transformer's mRoPE table (see
        # ``get_condition``): one float per sample, or ``None`` when the batch never
        # carried the column. Kept out of the ``**kwargs`` swallow so the transformer
        # is handed it explicitly instead of relying on the name matching by luck.
        if fps is not None:
            if len(fps) != count:
                raise ValueError(f"Got {len(fps)} fps entries but {count} samples; they must match.")
            processed["fps"] = list(fps)
        return processed

    def _sample_condition_mask(self, clean: torch.Tensor) -> torch.Tensor:
        """Draw this sample's clean-frame mask, shape ``[T_lat, 1, 1]``.

        **Which frames are clean: a prefix, ``0 .. num_clean-1``.** Upstream turns
        the sampled count into ``condition_frame_indexes_vision=list(range(n))``
        (``augmentors/sequence_plan.py:191`` and ``:206`` -- the named-mode branch
        says so explicitly: "Prefix conditioning is used for T2V, I2V, and V2V")
        and the packer sets ``condition_mask[frame_idx] = 1`` for those indexes
        (``sequence_packing/sequence.py:572-577``). So:

        * ``0`` clean frames -> T2V (nothing conditioned),
        * ``1`` clean frame  -> I2V, the *first* latent frame,
        * ``2`` clean frames -> V2V, the first two latent frames.

        **V2V vs I2V is only the count.** There is no second camera stream and no
        reference video on another input in this model: both are the leading
        latent frames of the same clip, and the recipe's comment spells the intent
        out as "20% I2V (first frame), 10% V2V (first 5 frames / 2 latent frames)"
        (``configs/base/experiment/sft/vision_sft_nano.py:261``). With the Wan2.2
        VAE's 4x temporal compression, latent frame 1 covers pixel frames 1-4, so
        "2 latent frames" == "the first 5 pixel frames".

        **The clamp** ``num_clean = min(num_clean, T_lat - 1)`` is upstream's
        (``sft_dataset.py:354``, ``sequence_plan.py:190``/``:204``): at least one
        latent frame must stay noisy or the sample has nothing to supervise (and
        a fully conditioned sample is a zero loss). It is applied *after* the
        draw, so a short clip degrades V2V -> I2V -> T2V rather than re-drawing.

        **RNG**: the draw uses ``self.generator`` -- the same CPU generator as
        sigma and noise, which is exactly the stream
        :meth:`rng_state_dict` / :meth:`load_rng_state_dict` snapshot. Upstream
        samples with Python's global ``random.choices`` inside a dataloader worker
        (``sft_dataset.py:353``), which no checkpoint of ours could resume; the
        distribution is identical (an inverse-CDF walk over the same normalized
        weights) but the stream is ours, so a resumed run continues the mixture
        instead of restarting it.

        Args:
            clean: One ``[C, T_lat, H_lat, W_lat]`` latent; only ``T_lat`` matters.

        Returns:
            ``float32`` (not the latent dtype: a bf16 mask would round ``1 - mask``
            and leak a non-zero sigma into a "clean" frame), on the latent's device.
        """
        latent_frames = int(clean.shape[1])
        if latent_frames < 1:
            raise ValueError(f"latent has no temporal frames: shape {tuple(clean.shape)}.")
        options = self.config.conditioning_probabilities()
        draw = float(torch.rand((), generator=self.generator, dtype=torch.float32))
        # Inverse-CDF walk, the discrete analogue of random.choices: the intervals
        # partition [0, 1) by the normalized weights, so the selected option has
        # probability equal to its weight.
        cumulative = 0.0
        num_clean = options[-1][0]
        for frames, probability in options:
            cumulative += probability
            if draw < cumulative:
                num_clean = frames
                break
        num_clean = max(0, min(num_clean, latent_frames - 1))
        condition_mask = torch.zeros((latent_frames, 1, 1), device=clean.device, dtype=torch.float32)
        if num_clean:
            condition_mask[:num_clean] = 1.0
        return condition_mask

    def _sample_sigmas(self, buckets: list[str]) -> torch.Tensor:
        """Draw one sigma per sample from the configured distribution + shift warp.

        ``TrainTimeSampler`` owns the warp (its ``shift`` argument is the
        per-sample shift); warping here as well would double-apply it.
        """
        shift = torch.tensor([self.config.shift_for_resolution(bucket) for bucket in buckets], dtype=torch.float32)
        sigmas = self.train_time_sampler(len(buckets), generator=self.generator, shift=shift)
        return sigmas.to(dtype=torch.float32).reshape(-1)

    def _interpolate_sample(
        self,
        noise: torch.Tensor,
        clean: torch.Tensor,
        sigma: torch.Tensor,
        condition_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(x_t, velocity_target)`` for one sample.

        Delegates to ``cosmos3_core.flow_matching.interpolate``, which owns the
        reference formulas (``x_t = noise * sigma + clean * (1 - sigma)``,
        ``target = noise - clean``). The shape check catches a sigma whose shape
        silently broadcast along the wrong axis 鈥?that would train on a target
        that no longer matches the prediction grid.

        ``condition_mask`` (``[T_lat, 1, 1]`` or ``None``) lowers sigma to exactly
        ``0`` on the clean frames, so ``x_t`` holds the clean latent there and the
        noise is invisible to them; the target is left unmasked on purpose (see
        :meth:`process_condition`).
        """
        effective_sigma = sigma
        if condition_mask is not None:
            # ``sigma`` is drawn on the CPU generator (device-independent stream) and
            # the mask lives with the latent, so the product has to be brought
            # together on the latent's device before interpolating.
            mask = condition_mask.to(device=noise.device, dtype=sigma.dtype)
            if mask.ndim != 3 or mask.shape[0] != noise.shape[1]:
                raise ValueError(
                    f"condition_mask must be [T={noise.shape[1]}, 1, 1] for a latent shaped "
                    f"{tuple(noise.shape)}, got {tuple(condition_mask.shape)}."
                )
            effective_sigma = sigma.to(device=noise.device).reshape(1, 1, 1) * (1.0 - mask)  # [T,1,1]
        hidden_states, training_target = self.flow_interpolate(noise, clean, effective_sigma)
        if training_target.shape != noise.shape or hidden_states.shape != noise.shape:
            raise ValueError(
                "cosmos3_core.flow_matching.interpolate returned hidden_states "
                f"{tuple(hidden_states.shape)} / target {tuple(training_target.shape)} for a noise tensor "
                f"shaped {tuple(noise.shape)}; sigma shape {tuple(effective_sigma.shape)} does not broadcast cleanly."
            )
        return hidden_states, training_target

    # 鈹€鈹€ helpers 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def _resolve_resolutions(self, resolution: Any, count: int) -> list[str]:
        """Broadcast the batch-level resolution bucket to one bucket per sample."""
        if resolution is None:
            return [self.config.resolution] * count
        if isinstance(resolution, str):
            return [resolution] * count
        buckets = [str(bucket) for bucket in resolution]
        if len(buckets) != count:
            raise ValueError(f"Got {len(buckets)} per-sample resolutions but {count} samples; they must match.")
        return buckets
