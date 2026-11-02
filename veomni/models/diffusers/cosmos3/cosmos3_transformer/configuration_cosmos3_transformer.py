"""Cosmos3 mixture-of-towers transformer configuration.

Mirrors ``MiniMaxH3DiTModelConfig`` (the self-contained DiT family Cosmos3 follows)
rather than a diffusers config: the backbone of Cosmos3 is *not* a diffusers class,
it is :class:`~.modeling_cosmos3_transformer.Cosmos3MoTTransformer` built out of
``cosmos3_core``. There is therefore no third-party ``__init__`` signature to
project onto; :meth:`to_diffuser_dict` spells the projection out instead.

Two class attributes matter to the framework:

* ``model_type`` -- the registry key. It is also the ``_class_name`` /
  ``model_type`` pair in ``configs/model_configs/cosmos3/cosmos3_nano.json``, so
  the shipped JSON loads through this class unchanged.
* ``condition_model_type`` -- the registry key ``DiTTrainer`` resolves the frozen
  companion through (``veomni/trainer/dit_trainer.py``:
  ``_build_condition_model(dit_config.condition_model_type)``). The condition side
  registers this one key in both registries, so the model name is the key.

Defaults are Cosmos3-Nano's (``cosmos_framework/configs/base/defaults/model_config.py``
plus ``vision_sft_nano``). A run config overrides them; the values here are what a
config-only import (tooling, tests) sees.
"""

from __future__ import annotations

from typing import Optional, Sequence

from transformers import PretrainedConfig


# Qwen3-VL's interleaved mRoPE split, as shipped in ``cosmos3_nano.json``'s
# ``rope_scaling``. The three entries must sum to ``head_dim // 2`` (64 for Nano's
# ``head_dim=128``); ``MotRope`` re-checks that rather than trusting this comment.
DEFAULT_MROPE_SECTION = (24, 20, 20)

# ``cosmos3_nano.json``, ``vision_sft_nano`` and the ``proj_in``/``proj_out`` heads
# are sized for one 2x2 latent patch of the Wan2.2 VAE's 48 channels.
DEFAULT_PATCH_LATENT_DIM = 192
DEFAULT_LATENT_PATCH_SIZE = 2

# The recipe's ``loss_scale`` (``Cosmos3ConditionModelConfig.loss_scale``, the
# value ``vision_sft_nano`` trains with). The loss lives on the transformer side
# because that is where the prediction exists, so the field is repeated here; a
# run that overrides one should override both.
DEFAULT_LOSS_SCALE = 10.0

# Accepted ``pack_samples`` values: pack a micro-batch into one TND sequence when it
# holds several samples (``"auto"``), always, or never (one forward per sample).
PACK_SAMPLES_MODES = ("auto", "always", "never")

# ``nano_model_config.py:42-43`` (the SFT recipe's only model config): the mRoPE
# temporal axis is indexed in real time against a 24 fps base, and the Wan2.2 VAE
# compresses 4 pixel frames per latent frame (``vision_sft_nano.py:271``).
DEFAULT_BASE_FPS = 24.0
DEFAULT_TEMPORAL_COMPRESSION_FACTOR = 4

# ``nano_model_config.py:49``: the extra temporal distance inserted between the text
# span and the generation span (``packers.py:570``). ``cosmos3_nano.json`` ships the
# same 15000; ``model_config.py:80`` defaults it to 0.
DEFAULT_MROPE_TEMPORAL_MODALITY_MARGIN = 0


class Cosmos3MoTTransformerConfig(PretrainedConfig):
    """Configuration for :class:`Cosmos3MoTTransformer`.

    Field grouping follows the two towers:

    * shared: ``hidden_size`` / ``num_hidden_layers`` / ``vocab_size`` / the
      attention geometry / ``patch_latent_dim`` / the mRoPE knobs
      (``mrope_section`` / ``mrope_interleaved`` / ``base_fps`` /
      ``enable_fps_modulation`` / ``temporal_compression_factor`` /
      ``unified_3d_mrope_temporal_modality_margin``);
    * generation-tower-only: ``include_gen_pathway`` (build the diffusion tower),
      ``timestep_scale``, ``loss_scale``;
    * the freeze switch: ``train_joint_reasoner``;
    * the batching switch: ``pack_samples`` (packed TND forward vs. one call per
      sample).
    """

    model_type = "Cosmos3MoTTransformer"
    condition_model_type = "Cosmos3ConditionModel"

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        intermediate_size: int = 12288,
        moe_intermediate_size: Optional[int] = None,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        num_experts: int = 0,
        num_experts_per_tok: int = 8,
        decoder_sparse_step: int = 1,
        mlp_only_layers: Optional[Sequence[int]] = None,
        qk_norm_for_text: bool = True,
        qk_norm_for_diffusion: bool = True,
        use_und_k_norm_for_gen: bool = True,
        patch_latent_dim: int = DEFAULT_PATCH_LATENT_DIM,
        latent_patch_size: int = DEFAULT_LATENT_PATCH_SIZE,
        latent_channel: Optional[int] = None,
        timestep_scale: float = 0.001,
        rope_theta: float = 5_000_000.0,
        rope_scaling: Optional[dict] = None,
        mrope_section: Optional[Sequence[int]] = None,
        mrope_interleaved: Optional[bool] = None,
        base_fps: float = DEFAULT_BASE_FPS,
        enable_fps_modulation: bool = True,
        temporal_compression_factor: int = DEFAULT_TEMPORAL_COMPRESSION_FACTOR,
        unified_3d_mrope_temporal_modality_margin: int = DEFAULT_MROPE_TEMPORAL_MODALITY_MARGIN,
        include_gen_pathway: bool = True,
        train_joint_reasoner: bool = False,
        pack_samples: str = "auto",
        loss_scale: float = DEFAULT_LOSS_SCALE,
        max_position_embeddings: int = 262144,
        **kwargs,
    ):
        """Store the backbone geometry and validate the patch/latent relation.

        ``latent_channel`` is derivable (``patch_latent_dim / latent_patch_size ** 2``)
        but is accepted explicitly because the shipped JSON carries it; a
        disagreement between the two is a config bug that would otherwise surface
        as a matmul shape error inside ``proj_in``.
        """
        # ``super().__init__`` first: it consumes the framework kwargs
        # (``tie_word_embeddings``, ``pad_token_id``, ``_attn_implementation``, ...)
        # and stores leftover checkpoint keys as attributes, so the assignments
        # below are the authoritative values for every field this class declares.
        # There is no LM head, so weight tying is inert either way; the shipped JSON
        # says ``false`` and the default must not silently claim otherwise.
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)

        if hidden_size % num_attention_heads:
            raise ValueError(
                f"hidden_size={hidden_size} is not divisible by num_attention_heads={num_attention_heads}."
            )
        if num_key_value_heads > num_attention_heads or num_key_value_heads < 1:
            raise ValueError(
                f"num_key_value_heads={num_key_value_heads} must be in [1, num_attention_heads={num_attention_heads}]."
            )
        if latent_patch_size < 1:
            raise ValueError(f"latent_patch_size must be >= 1, got {latent_patch_size}.")
        derived_channel = patch_latent_dim // (latent_patch_size**2)
        if derived_channel * latent_patch_size**2 != patch_latent_dim:
            raise ValueError(
                f"patch_latent_dim={patch_latent_dim} is not a multiple of "
                f"latent_patch_size^2={latent_patch_size**2}; it must equal latent_channel * latent_patch_size ** 2."
            )
        if latent_channel is not None and int(latent_channel) != derived_channel:
            raise ValueError(
                f"latent_channel={latent_channel} disagrees with patch_latent_dim={patch_latent_dim} and "
                f"latent_patch_size={latent_patch_size}, which imply {derived_channel}."
            )

        # ``rope_scaling`` is where the checkpoint JSON keeps the mRoPE knobs
        # (``{"mrope_section": [...], "mrope_interleaved": true, "rope_type":
        # "default"}``). The flat fields are the ones the model reads; the nested
        # copy is only consulted when a flat field is left unset, so a config can
        # be written in either shape without the two silently diverging.
        self.rope_scaling = dict(rope_scaling) if rope_scaling else None
        scaling = self.rope_scaling or {}
        section = scaling.get("mrope_section", DEFAULT_MROPE_SECTION) if mrope_section is None else mrope_section
        interleaved = scaling.get("mrope_interleaved", True) if mrope_interleaved is None else mrope_interleaved
        self.mrope_section = tuple(int(size) for size in section)
        if len(self.mrope_section) != 3:
            raise ValueError(f"mrope_section must hold three axis sizes (t, h, w), got {self.mrope_section}.")
        self.mrope_interleaved = bool(interleaved)

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.intermediate_size = int(intermediate_size)
        self.moe_intermediate_size = int(moe_intermediate_size or intermediate_size)
        self.hidden_act = hidden_act
        self.rms_norm_eps = float(rms_norm_eps)
        self.attention_bias = bool(attention_bias)
        self.attention_dropout = float(attention_dropout)
        # ``num_experts == 0`` is Cosmos3-Nano: both towers are dense MLPs. A MoE
        # backbone swaps in ``MotSparseMoeBlock`` for every ``decoder_sparse_step``
        # layer (see ``cosmos3_core.mot_decoder_layer.moe_layer_enabled``).
        self.num_experts = int(num_experts)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.decoder_sparse_step = int(decoder_sparse_step)
        self.mlp_only_layers = tuple(int(index) for index in mlp_only_layers or ())
        self.qk_norm_for_text = bool(qk_norm_for_text)
        self.qk_norm_for_diffusion = bool(qk_norm_for_diffusion)
        self.use_und_k_norm_for_gen = bool(use_und_k_norm_for_gen)
        self.patch_latent_dim = int(patch_latent_dim)
        self.latent_patch_size = int(latent_patch_size)
        self.latent_channel = derived_channel
        self.timestep_scale = float(timestep_scale)
        self.rope_theta = float(rope_theta)

        # mRoPE temporal axis (``mrope.py:164-180``). Declared here so a config that
        # sets them cannot disagree with what the model does; the effective values are
        # logged once per run.
        self.base_fps = float(base_fps)
        self.enable_fps_modulation = bool(enable_fps_modulation)
        self.temporal_compression_factor = int(temporal_compression_factor)
        self.unified_3d_mrope_temporal_modality_margin = int(unified_3d_mrope_temporal_modality_margin)
        if self.base_fps <= 0:
            raise ValueError(f"base_fps must be positive, got {base_fps!r}; it is the mRoPE time base.")
        if self.temporal_compression_factor < 1:
            raise ValueError(f"temporal_compression_factor must be >= 1, got {temporal_compression_factor!r}.")
        if self.unified_3d_mrope_temporal_modality_margin < 0:
            raise ValueError(
                "unified_3d_mrope_temporal_modality_margin must be non-negative, got "
                f"{unified_3d_mrope_temporal_modality_margin!r}."
            )

        self.include_gen_pathway = bool(include_gen_pathway)
        self.train_joint_reasoner = bool(train_joint_reasoner)
        # Model-side packing switch: run the micro-batch as one packed (TND) sequence
        # ("auto": whenever it holds more than one sample, "always", "never") instead
        # of one forward per sample. Distinct from the data-side ``data.pack_mode``,
        # which decides how many samples share a micro-batch at all; see
        # ``modeling_cosmos3_transformer.Cosmos3MoTTransformer._packs_micro_batch``.
        self.pack_samples = str(pack_samples)
        if self.pack_samples not in PACK_SAMPLES_MODES:
            raise ValueError(
                f"pack_samples={self.pack_samples!r} is not one of {sorted(PACK_SAMPLES_MODES)}; "
                "'auto' packs a multi-sample micro-batch, 'never' keeps the per-sample loop, "
                "'always' packs even a single sample."
            )
        self.loss_scale = float(loss_scale)
        self.max_position_embeddings = int(max_position_embeddings)

    # ── config bridges ───────────────────────────────────────────────────────

    def to_diffuser_dict(self) -> dict:
        """Project this config onto the kwargs that build the backbone modules.

        Why this is not an ``inspect.signature`` sweep (as it is for Wan): Cosmos3
        has no diffusers source class to bridge to. The "diffusers dict" here means
        *the subset of config fields the module factories read* --
        ``MoTDecoderLayer`` / ``MotAttention`` / ``MotRope`` / ``TimestepEmbedder``
        take a config object and pull these names off it
        (``cosmos3_core.mot_attention.config_field``). Keeping the list in one place
        makes "which knob does the backbone actually consume" auditable, and lets a
        tool build a minimal config for a shape-only check.

        Framework-only fields (``model_type``, ``condition_model_type``,
        ``_attn_implementation``, ...) are deliberately absent: they steer the
        registry and the trainer, not the network.
        """
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "moe_intermediate_size": self.moe_intermediate_size,
            "hidden_act": self.hidden_act,
            "rms_norm_eps": self.rms_norm_eps,
            "attention_bias": self.attention_bias,
            "attention_dropout": self.attention_dropout,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "decoder_sparse_step": self.decoder_sparse_step,
            "mlp_only_layers": self.mlp_only_layers,
            "qk_norm_for_text": self.qk_norm_for_text,
            "qk_norm_for_diffusion": self.qk_norm_for_diffusion,
            "use_und_k_norm_for_gen": self.use_und_k_norm_for_gen,
            "patch_latent_dim": self.patch_latent_dim,
            "latent_patch_size": self.latent_patch_size,
            "latent_channel": self.latent_channel,
            "timestep_scale": self.timestep_scale,
            "rope_theta": self.rope_theta,
            "mrope_section": self.mrope_section,
            "mrope_interleaved": self.mrope_interleaved,
            "base_fps": self.base_fps,
            "enable_fps_modulation": self.enable_fps_modulation,
            "temporal_compression_factor": self.temporal_compression_factor,
            "unified_3d_mrope_temporal_modality_margin": self.unified_3d_mrope_temporal_modality_margin,
            "include_gen_pathway": self.include_gen_pathway,
            "train_joint_reasoner": self.train_joint_reasoner,
            "loss_scale": self.loss_scale,
        }

    def to_dict(self) -> dict:
        """Serialize with the ``_class_name`` a checkpoint loader looks for.

        ``cosmos3_nano.json`` carries both ``_class_name`` and ``model_type``, and
        both are the *model* key ``Cosmos3MoTTransformer`` -- not this class name,
        which carries the ``Config`` suffix. Injecting ``model_type`` keeps an
        exported ``config.json`` loadable by the same key that routes
        :func:`veomni.models.registry.get_model_config`.
        """
        config_dict = super().to_dict()
        config_dict["_class_name"] = self.model_type
        return config_dict
