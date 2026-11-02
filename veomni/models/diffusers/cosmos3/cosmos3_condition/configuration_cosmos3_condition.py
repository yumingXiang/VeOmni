"""Cosmos3 condition-model configuration.

Mirrors ``MiniMaxH3ConditionModelConfig`` (the self-contained DiT family that
Cosmos3 follows) rather than ``WanTransformer3DConditionModelConfig``: Cosmos3's
text tower lives *inside* the diffusion transformer (a Mixture-of-Towers shares
attention between the frozen Qwen3-VL layers and the generation layers), so the
condition side has no text encoder to load — only the VAE and the tokenizer.

Every field here is a *knob of the online data path* (which video codec, which
rectified-flow schedule, which resolution bucket). The values are copied
verbatim from ``cosmos_framework``'s ``NANO_MODEL_CONFIG``
(``rectified_flow_training_config``) and ``vision_sft_nano`` so a VeOmni run
trains the same objective as the open-source recipe; anything they do not
constrain (tiling, chunking, tokenizer length) is ours.
"""

from __future__ import annotations

import os
from typing import Optional

from transformers import PretrainedConfig


DEFAULT_SHIFT = {"256": 3, "480": 5, "720": 10}

# The recipe's conditioning mixture: ``vision_sft_nano.py:262`` (and ``:317`` for
# the map-style dataloader) passes ``conditioning_config={0: 0.7, 1: 0.2, 2: 0.1}``
# with the comment "70% T2V, 20% I2V (first frame), 10% V2V (first 5 frames / 2
# latent frames)". Keys are latent-frame counts of the clean *prefix*, not mode
# names; ``sequence_plan.py:27-31`` names the same three numbers t2v/i2v/v2v.
DEFAULT_CONDITIONING_CONFIG = {0: 0.7, 1: 0.2, 2: 0.1}


class Cosmos3ConditionModelConfig(PretrainedConfig):
    """Configuration for :class:`Cosmos3ConditionModel`.

    ``base_model_path`` is the Cosmos3-Nano checkpoint root; it is filled in
    automatically by :meth:`get_config_dict` so a run config only has to name
    ``model.condition_model_path``.
    """

    model_type = "Cosmos3ConditionModelConfig"

    def __init__(
        self,
        base_model_path: Optional[str] = None,
        vae_subfolder: str = "vae",
        processor_subfolder: str = "",
        text_tokenizer_subfolder: str = "text_tokenizer",
        num_train_timesteps: int = 1000,
        shift: Optional[dict] = None,
        train_time_video_distribution: str = "waver",
        train_time_weight: str = "uniform",
        loss_scale: float = 10.0,
        resolution: str = "256",
        latent_patch_size: int = 2,
        latent_channel: int = 48,
        max_sequence_length: int = 2048,
        append_generation_markers: bool = True,
        # bfloat16, matching the reference implementation (wan2pt2_vae_4x16x16.py:1292).
        # This is not a speed/precision trade: against the reference's own dumped
        # latents, bf16 agrees to ~0.3% (mean |d| 2.7e-3 on a latent of std 0.86) while
        # fp32 agrees 5-10x worse (4.2e-3 .. 5.7e-3), because the reference's bf16
        # rounding is part of what is being compared against. It is also a latent-cache
        # key factor, so changing it invalidates an existing cache.
        vae_dtype: str = "bfloat16",
        vae_temporal_chunk_frames: int = 0,
        conditioning_config: Optional[dict] = None,
        imgvid_cond_noise_aug: float = 1.0,
        seed: Optional[int] = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_path = base_model_path
        self.vae_subfolder = vae_subfolder
        self.processor_subfolder = processor_subfolder
        self.text_tokenizer_subfolder = text_tokenizer_subfolder

        # The rectified-flow schedule. ``num_train_timesteps`` is the scale factor
        # applied to sigma before it reaches the transformer's timestep embedder
        # (``timestep = sigma * num_train_timesteps``), which is how the reference
        # model was pretrained; sigma itself stays continuous in [0, 1].
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = _normalize_shift(shift)
        self.train_time_video_distribution = train_time_video_distribution
        self.train_time_weight = train_time_weight
        # Not read here: the multiplier belongs to the DiT
        # (``Cosmos3MoTTransformerConfig.loss_scale``, which is where the prediction
        # exists). It is kept on this config because the open-source recipe's files
        # spell it under the condition model, and ``DiTModelRuntime`` forwards it
        # onto the DiT config (``dit_trainer.py::resolve_loss_scale``) so the two
        # spellings cannot diverge silently.
        self.loss_scale = float(loss_scale)
        self.resolution = str(resolution)
        self.latent_patch_size = int(latent_patch_size)
        self.latent_channel = int(latent_channel)
        self.max_sequence_length = int(max_sequence_length)
        # Append ``[eos, <|vision_start|>]`` to every prompt after truncation: the
        # reference's packer adds that generation boundary to the text span
        # (``sequence.py:257-260``) and the diffusion tower was pretrained with it.
        # Kept switchable so the "+2 token" term can be isolated in an A/B instead of
        # needing a code edit -- it is the one fix whose sign points the *wrong* way
        # (it should raise our loss slightly).
        self.append_generation_markers = bool(append_generation_markers)
        self.vae_dtype = vae_dtype
        self.vae_temporal_chunk_frames = int(vae_temporal_chunk_frames)

        # T2V / I2V / V2V mixture: ``{clean_prefix_latent_frames: probability}``.
        # Kept as the raw dict (keys coerced to int) and *normalized on demand*
        # rather than here, because ``PretrainedConfig.from_dict`` re-applies
        # saved fields with ``setattr`` and so arrives with JSON string keys.
        self.conditioning_config = _coerce_conditioning_config(conditioning_config)
        # Robustness knob kept for parity with the MiniMax-H3 family, whose config
        # carries ``imgvid_cond_noise_aug``. Cosmos3 has **no** conditioning noise
        # augmentation: ``grep -rn imgvid_cond_noise_aug`` over cosmos-framework
        # finds nothing, and the reference hard-codes the conditioning frames to
        # sigma = 0 (``posttrain/model/cosmos3.py:1156``: "conditioning frames stay
        # clean (sigma=0)"), i.e. the clean latent is placed into ``x_t`` verbatim.
        # 1.0 is that "disabled" value, so the field defaults to it and any other
        # value is rejected instead of silently inventing an unaudited recipe.
        self.imgvid_cond_noise_aug = float(imgvid_cond_noise_aug)
        self.seed = seed

        if self.num_train_timesteps < 2:
            raise ValueError("num_train_timesteps must be >= 2; it scales sigma into the timestep embedder.")
        if self.resolution not in self.shift:
            raise ValueError(
                f"resolution={self.resolution!r} has no entry in shift={self.shift!r}. "
                f"Configured buckets: {sorted(self.shift)}."
            )
        if self.train_time_weight != "uniform":
            raise ValueError(
                "train_time_weight must be 'uniform': the reference loss multiplies by time_weight == 1 "
                f"(cosmos_framework/model/generator/algorithm/loss/flow_matching.py). Got {self.train_time_weight!r}."
            )
        if self.latent_patch_size < 1:
            raise ValueError("latent_patch_size must be >= 1.")
        if self.imgvid_cond_noise_aug != 1.0:
            raise ValueError(
                "imgvid_cond_noise_aug must be 1.0 (no conditioning noise augmentation). The Cosmos3 "
                "reference puts the conditioning frames into x_t with sigma = 0 verbatim "
                f"(posttrain/model/cosmos3.py:1156); got {self.imgvid_cond_noise_aug!r}."
            )
        # Fail at config time, not at the first training step.
        self.conditioning_probabilities()

    @classmethod
    def get_config_dict(
        cls,
        pretrained_model_name_or_path,
        **kwargs,
    ):
        """Inject the checkpoint root as ``base_model_path``.

        ``DiTTrainer`` builds the condition model with
        ``config_class.from_pretrained(model.condition_model_path)``, so the
        directory handed to ``from_pretrained`` *is* the component root that
        ``vae_subfolder`` / ``text_tokenizer_subfolder`` resolve against.

        A Cosmos3-Nano checkpoint root may legitimately carry no ``config.json``
        (it is a diffusers-style bundle of ``transformer/`` + ``vae/`` +
        tokenizer files), so a missing base config is not an error — the defaults
        in ``__init__`` are the recipe's values.
        """
        try:
            config_dict, kwargs = super().get_config_dict(pretrained_model_name_or_path, **kwargs)
        except Exception:
            config_dict = {}
        config_dict["base_model_path"] = pretrained_model_name_or_path
        return config_dict, kwargs

    def component_dir(self, subfolder: str) -> str:
        """Resolve a component subfolder, falling back to the checkpoint root.

        Empty ``subfolder`` means "the root itself". A named subfolder that the
        checkpoint does not have also resolves to the root, because Cosmos3-Nano
        ships its Qwen3-VL tokenizer files both at the top level and under
        ``text_tokenizer/`` depending on the conversion revision.
        """
        base = self.base_model_path or ""
        if not subfolder:
            return base
        candidate = os.path.join(base, subfolder)
        return candidate if os.path.isdir(candidate) else base

    def conditioning_probabilities(self) -> list[tuple[int, float]]:
        """Normalized ``[(clean_prefix_latent_frames, probability), ...]`` to sample from.

        Why the normalization lives here instead of in ``__init__``: the reference
        normalizes once at dataset construction
        (``sft_dataset.py:148-152``, ``self.conditioning_config = {k: v / total}``)
        because its dataset object is long-lived; nothing there mutates the field.
        Our config object *is* mutated -- ``PretrainedConfig.from_dict`` re-applies
        ``condition_model_cfg`` with ``setattr``, which is how ``configs/dit/
        cosmos3_sft.yaml`` reaches this class -- so the invariant is recomputed on
        every read instead of being trusted from construction time.

        The ratio, not the absolute scale, is what matters:
        ``sequence_plan.py:185-187`` samples with ``random.choices(modes,
        weights=normalized.values())``, which only depends on the normalized
        weights, so ``{0: 7, 1: 2, 2: 1}`` and ``{0: 0.7, ...}`` are the same
        mixture. Order is the dict's own, which is what makes the draw
        reproducible across a YAML / JSON round trip.

        Returns:
            Probabilities summing to 1.0, in the configured order.

        Raises:
            ValueError: when the mixture is empty, holds a negative probability, or
                sums to a non-positive number.
        """
        return _normalize_conditioning_config(self.conditioning_config)

    def shift_for_resolution(self, resolution: Optional[str] = None) -> float:
        """Per-sample shift for a resolution bucket (``None`` → this config's bucket).

        Validated here as well as in ``__init__`` because
        ``PretrainedConfig.from_dict`` re-applies ``model.condition_model_cfg``
        with ``setattr``, which bypasses constructor validation.
        """
        bucket = str(resolution if resolution is not None else self.resolution)
        if bucket not in self.shift:
            raise ValueError(
                f"resolution={bucket!r} is not a configured shift bucket. Available: {sorted(self.shift)}. "
                "Add the bucket to model.condition_model_cfg.shift rather than guessing a shift."
            )
        shift = float(self.shift[bucket])
        if shift <= 0:
            raise ValueError(f"shift[{bucket!r}] must be positive, got {shift}.")
        return shift


def _normalize_shift(shift: Optional[dict]) -> dict:
    """Normalize the shift table to ``{str(bucket): float(shift)}``.

    Keys arrive as strings from JSON but as ints from a Python dict literal, and
    a single int is accepted as "one shift for every bucket" — the three shapes
    the reference configuration tolerates. Values are the sigma-warp strengths of
    ``sigma = shift * t / (1 + (shift - 1) * t)``.
    """
    if shift is None:
        return {key: float(value) for key, value in DEFAULT_SHIFT.items()}
    if isinstance(shift, (int, float)):
        if shift <= 0:
            raise ValueError(f"shift must be positive, got {shift}.")
        return {key: float(shift) for key in DEFAULT_SHIFT}
    if not isinstance(shift, dict) or not shift:
        raise ValueError(f"shift must be a non-empty dict of {{resolution: shift}}, got {shift!r}.")
    normalized = {}
    for key, value in shift.items():
        numeric = float(value)
        if numeric <= 0:
            raise ValueError(f"shift[{key!r}] must be positive, got {value}.")
        normalized[str(key)] = numeric
    return normalized


def _coerce_conditioning_config(conditioning_config: Optional[dict]) -> dict[int, float]:
    """Coerce a conditioning mixture to ``{int(frames): raw_weight}`` without normalizing.

    ``None`` means the recipe's default mixture, exactly like the constructor
    default: the reference's *absence* of the field means "no ``SequencePlan``,
    i.e. plain T2V" (``sft_dataset.py:348`` is the only writer), and reaching that
    from an unset config knob is the surprising behaviour, not the default. A run
    that really wants pure T2V says ``{0: 1.0}``.

    Keys are coerced because the same dict arrives as ints from Python, as strings
    from a ``config.json``, and as ints again from a YAML load. ``sequence_plan.py``
    accepts both spellings too (``isinstance(mode, int)`` or a named mode); the
    named modes (``"t2v"`` / ``"i2v"`` / ``"v2v"``) are *not* accepted here because
    the numeric form is the one the SFT recipe uses
    (``vision_sft_nano.py:262``) and inventing a second spelling for the same knob
    would make a typo silently change the mixture.
    """
    if conditioning_config is None:
        return {int(frames): float(weight) for frames, weight in DEFAULT_CONDITIONING_CONFIG.items()}
    if not isinstance(conditioning_config, dict) or not conditioning_config:
        raise ValueError(
            f"conditioning_config must be a non-empty dict of {{clean_latent_frames: probability}}, got "
            f"{conditioning_config!r}."
        )
    coerced: dict[int, float] = {}
    for frames, weight in conditioning_config.items():
        try:
            count = int(str(frames).strip())
        except (TypeError, ValueError):
            raise ValueError(
                f"conditioning_config keys must be latent-frame counts like 0/1/2, got {frames!r}."
            ) from None
        if count < 0:
            raise ValueError(f"conditioning_config keys must be non-negative, got {frames!r}.")
        coerced[count] = float(weight)
    return coerced


def _normalize_conditioning_config(conditioning_config: Optional[dict]) -> list[tuple[int, float]]:
    """Validate and normalize a mixture into ``[(frames, probability)]`` summing to 1.

    Mirrors ``sft_dataset.py:148-152`` (normalize, require a positive total) plus
    the non-negativity check ``sequence_plan.py:90-91`` performs.
    """
    coerced = _coerce_conditioning_config(conditioning_config)
    total = 0.0
    for frames, weight in coerced.items():
        if weight < 0:
            raise ValueError(f"conditioning_config[{frames}] must be non-negative, got {weight}.")
        total += weight
    if total <= 0:
        raise ValueError(f"conditioning_config probabilities must sum to a positive number, got {coerced!r}.")
    return [(frames, weight / total) for frames, weight in coerced.items()]
