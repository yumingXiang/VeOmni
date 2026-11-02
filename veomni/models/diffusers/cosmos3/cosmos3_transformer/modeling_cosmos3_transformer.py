"""HF ``PreTrainedModel`` shell around the Cosmos3 mixture-of-towers backbone.

``cosmos3_core`` is a pure-``torch`` numeric core and ``cosmos3_condition`` produces the
training triple, but neither is something ``DiTTrainer`` can hold: it needs a
``PreTrainedModel`` the FSDP2 / checkpoint machinery understands, that computes the
rectified-flow loss internally and can freeze itself. This module is that shell; it owns
no math that lives in the core.

The forward contract is the trainer's: ``DiTDataCollator`` keeps one list per batch key
(one element per sample, never stacked, because videos differ in geometry), so a micro
batch is ``model(**micro_batch)`` with the keys ``process_condition`` returns --
``hidden_states`` (``[C, T_lat, H_lat, W_lat]``), ``timestep`` (already
``sigma * num_train_timesteps``), ``training_target`` (``noise - clean``),
``input_ids`` / ``ar_attention_mask`` (the frozen reasoner's conditioning; there is no
AR loss in v1, so no ``lm_head`` is used). The loss comes back as
``{"dm_loss": scalar}``, which the trainer divides by the microbatch count.

Parameter names are the released checkpoint's, so it loads without a key-mapping table::

    embed_tokens.weight          layers.{i}.*            norm.weight
    norm_moe_gen.weight          proj_in.{weight,bias}   proj_out.{weight,bias}
    time_embedder.linear_{1,2}.{weight,bias}
    lm_head.weight

The diffusers spellings (not ``cosmos_framework``'s ``vae2llm`` / ``llm2vae`` /
``mlp.{0,2}``) are the ones NVIDIA's export uses; ``_convert_model_to_diffusers.py``
does that rename and ``cosmos_framework/inference/model.py:109`` maps it back, so the
two spellings denote the same tensors. ``IGNORED_CHECKPOINT_KEY_PREFIXES`` lists the
checkpoint tensors this shell deliberately does not own, with the reason for each.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from veomni.distributed.parallel_plan import ParallelPlan
from veomni.utils import logging

from ..cosmos3_core import (
    MoTDecoderLayer,
    MotRMSNorm,
    MotRope,
    PackedLayout,
    TimestepEmbedder,
    add_time_conditioning,
    build_latent_grid,
    build_latent_position_ids,
    build_packed_layout,
    build_segmented_latent_position_ids,
    build_segmented_text_position_ids,
    build_text_position_ids,
    flow_matching_loss,
)
from .configuration_cosmos3_transformer import Cosmos3MoTTransformerConfig


logger = logging.get_logger(__name__)


__all__ = [
    "ALWAYS_FROZEN_NAME_PREFIXES",
    "IGNORED_CHECKPOINT_KEY_PREFIXES",
    "TRAINABLE_NAME_KEYS",
    "Cosmos3MoTTransformer",
    "Cosmos3MoTTransformerOutput",
    "apply_cosmos3_transformer_patch",
    "patchify_latent",
    "unpatchify_latent",
]


# Substrings of a parameter name that stay trainable when the reasoner tower is
# frozen. Copied from the recipe rather than invented:
# ``cosmos_framework/utils/generator/optimizer.py:303-305`` freezes every parameter
# whose name does not contain one of ``keys_to_select``, and
# ``cosmos_framework/configs/base/experiment/sft/vision_sft_nano.py:106-111`` is the
# SFT config that sets ``keys_to_select = ["moe_gen", "time_embedder", "vae2llm",
# "llm2vae"]``. The semantics are substring matching, not prefix matching.
#
# The spellings differ from upstream's because the modules do: ``moe_gen`` still
# matches the generation tower's ``mlp_moe_gen`` / ``*_layernorm_moe_gen`` (and the
# final ``norm_moe_gen``), but the generation tower's *attention* is exported as
# ``add_*`` rather than ``*_proj_moe_gen`` (see ``mot_attention.MotAttention``), and
# ``vae2llm`` / ``llm2vae`` are exported as ``proj_in`` / ``proj_out``. Dropping the
# attention keys would silently freeze the generation tower's Q/K/V/O, so they are
# listed explicitly; ``"moe_gen"`` is kept as the substring that covers the MLPs,
# the layer norms and the final norm of that tower.
TRAINABLE_NAME_KEYS = (
    "moe_gen",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "norm_added_q",
    "norm_added_k",
    "time_embedder",
    "proj_in",
    "proj_out",
)


# Parameters that stay frozen for the whole life of this shell, even under
# ``train_joint_reasoner=True``. The reasoner tower's causal-LM head is the only
# entry: v1 has no autoregressive (next-token) loss, so ``lm_head`` never appears in
# a forward pass and can never receive a gradient. Releasing it would only put an
# empty, always-``None``-grad parameter into the optimizer state, which is why the
# joint-training branch has to skip it *explicitly* rather than lean on
# ``TRAINABLE_NAME_KEYS`` not matching it (``"lm_head"`` is not a whitelist key
# today, but a future key such as ``"head"`` would silently match).
#
# When AR training of the reasoner tower is introduced, this list is where the
# decision lives: remove ``lm_head.`` here, add the AR loss to ``forward``, and add
# the key to ``TRAINABLE_NAME_KEYS`` if the recipe wants it trained.
ALWAYS_FROZEN_NAME_PREFIXES = ("lm_head.",)


# Checkpoint keys this shell deliberately does not own, in the order of the
# released ``transformer/`` index. The index holds 814 tensors -- 36 layers x 22
# plus 22 network-level ones -- and this model owns 804 of them (everything except
# the 10 below). The prefixes are honoured twice: ``from_pretrained`` ignores them
# through ``_keys_to_ignore_on_load_unexpected`` (so they are not even reported as
# unexpected), and :func:`split_ignorable_checkpoint_keys` filters them out for a
# strict loader instead of raising.
IGNORED_CHECKPOINT_KEY_PREFIXES = (
    # The sound pathway. ``cosmos3_nano.json`` says ``sound_gen: false`` and
    # ``Cosmos3ConditionModel`` produces no audio latent, so there is no input to
    # ``audio_proj_in`` and no target for ``audio_proj_out``. The audio *tokenizer*
    # is a separate pipeline component (``sound_tokenizer/``), not a transformer
    # sub-module. ``audio_modality_embed`` is a bare tensor (no ``.weight`` suffix).
    "audio_proj_in.",
    "audio_proj_out.",
    "audio_modality_embed",
    # The action pathway, same argument: ``action_gen: false`` for Nano (the
    # released transformer ``config.json`` still carries ``action_gen: true`` and
    # ``action_dim: 64`` because it is the omni export, but this training shell's
    # condition model hands over no action state). ``action_proj_in``/``out`` are
    # ``{"fc", "bias"}.weight`` pairs, hence the ``.`` suffix; the modality embed is
    # again a bare tensor.
    "action_proj_in.",
    "action_proj_out.",
    "action_modality_embed",
    # The Qwen3-VL *vision encoder*: 27 ``blocks``, ``patch_embed``, ``pos_embed``,
    # the ``merger`` and three ``deepstack_merger_list`` entries. These keys live in
    # ``vision_encoder/model.safetensors`` (``Qwen3VLVisionModel``, depth 27) and are
    # only present in the repo-root ``model.safetensors.index.json``, which the
    # exporter writes as the *union* index over ``transformer/`` +
    # ``vision_encoder/`` (``_convert_model_to_diffusers.py:546-557``); the
    # transformer's own ``diffusion_pytorch_model.safetensors.index.json`` does not
    # contain a single one of them. The ViT is a separate pipeline component with its
    # own loader, so this shell must not claim its tensors. Note in particular that
    # ``patch_embed.proj`` is the ViT's ``Conv3d``-style patch convolution over RGB
    # pixels, *not* a replacement for the latent patchify: the latent side is the
    # pure reshape in :func:`patchify_latent` followed by ``proj_in``.
    "blocks.",
    "merger.",
    "deepstack_merger_list.",
    "patch_embed.",
    "pos_embed.",
)


@dataclass
class Cosmos3MoTTransformerOutput(ModelOutput):
    """``loss`` is the ``{name: scalar}`` dict the trainer sums; ``predictions`` is per sample."""

    loss: Optional[dict[str, torch.Tensor]] = None
    predictions: Optional[list[torch.Tensor]] = None


# ── latent <-> patch grid ────────────────────────────────────────────────────


def patchify_latent(latent: torch.Tensor, patch_size: tuple[int, int]) -> torch.Tensor:
    """Flatten a ``[C, F, H, W]`` latent into ``[F * H/p * W/p, p*p*C]`` patch rows.

    The flatten order is the reference's
    (``cosmos3_vfm_network.py::patchify_and_pack_latents``)::

        [C, F, H, W] -> [C, F, h, p, w, p] --einsum("cthpwq->thwpqc")-> [F*h*w, p*p*C]

    i.e. ``t`` outer, ``h`` middle, ``w`` inner, and the patch's last axis is
    ``(h_offset, w_offset, channel)``. That order is what ``proj_in``'s weight rows
    were trained against and what ``build_latent_position_ids`` emits, so it must
    not be "cleaned up" into channel-major.

    Args:
        patch_size: Spatial ``(ph, pw)``; Cosmos3's time axis is never patched
            (``latent_patch_size`` is spatial only, unlike the ``(pt, ph, pw)`` form
            ``build_latent_grid`` takes).

    Returns:
        One row per latent patch. A side that is not a multiple of the patch size is
        an error rather than zero-padding: ``unpatchify_latent`` has to reproduce the
        input shape exactly for the loss to compare prediction and target element by
        element, and the condition model already rounds the video to
        ``vae.spatial_scale * latent_patch_size`` (``_fit_latent_grid``).
    """
    channels, frames, height, width = (int(size) for size in latent.shape)
    ph, pw = (int(size) for size in patch_size)
    if height % ph or width % pw:
        raise ValueError(
            f"latent spatial size {(height, width)} is not divisible by the patch size {(ph, pw)}; "
            "the condition model must hand over a latent on the patch grid."
        )
    grid_h, grid_w = height // ph, width // pw
    patches = latent.reshape(channels, frames, grid_h, ph, grid_w, pw)
    patches = torch.einsum("cthpwq->thwpqc", patches)
    return patches.reshape(-1, ph * pw * channels)


def unpatchify_latent(
    patches: torch.Tensor,
    latent_shape: tuple[int, int, int, int],
    patch_size: tuple[int, int],
) -> torch.Tensor:
    """Invert :func:`patchify_latent` back to ``[C, F, H, W]``.

    ``latent_shape`` is the *input* latent's shape, so the prediction comes back in
    the same geometry as ``training_target`` by construction.
    """
    channels, frames, height, width = (int(size) for size in latent_shape)
    ph, pw = (int(size) for size in patch_size)
    grid_h, grid_w = height // ph, width // pw
    expected = frames * grid_h * grid_w
    if patches.shape[0] != expected:
        raise ValueError(
            f"got {patches.shape[0]} patch rows for a latent of shape {tuple(latent_shape)} on a "
            f"{(ph, pw)} patch, which holds {expected}."
        )
    if patches.shape[-1] != ph * pw * channels:
        raise ValueError(
            f"patch rows are {patches.shape[-1]} wide but {ph}x{pw} patches of a {channels}-channel "
            f"latent are {ph * pw * channels} wide."
        )
    shaped = patches.reshape(frames, grid_h, grid_w, ph, pw, channels)
    latent = torch.einsum("thwpqc->cthpwq", shaped)
    return latent.reshape(channels, frames, height, width)


def patch_row_count(latent_shape: tuple[int, int, int, int], patch_size: tuple[int, int]) -> int:
    """Number of patch rows :func:`patchify_latent` produces for ``latent_shape``."""
    _, frames, height, width = (int(size) for size in latent_shape)
    ph, pw = (int(size) for size in patch_size)
    return frames * (height // ph) * (width // pw)


# ── the shell ────────────────────────────────────────────────────────────────


class Cosmos3MoTTransformer(PreTrainedModel):
    """Cosmos3-Nano's mixture-of-towers backbone as a trainable HF model.

    Holds the network-level modules of ``Cosmos3VFMNetwork``: the token embedding of
    the reasoner tower, the mRoPE tables, the ``MoTDecoderLayer`` stack, one final
    norm per tower, the diffusion tower's three adapters (``proj_in``, ``proj_out``,
    ``time_embedder``), and the reasoner tower's causal-LM head (``lm_head``, built
    for key coverage but permanently frozen -- v1 has no AR loss).
    """

    config_class = Cosmos3MoTTransformerConfig
    supports_gradient_checkpointing = True
    # FSDP2 wraps at this granularity. A block has to be one unit because its two
    # towers exchange tensors inside ``self_attn``, so the wrap boundary is the
    # block and never a sub-module of it.
    _no_split_modules = ["MoTDecoderLayer"]
    # The checkpoint carries tensors this shell does not model; naming them here is
    # what keeps ``from_pretrained`` from reporting them as unexpected. Anchored with
    # ``^`` because Transformers matches with ``re.search``, not ``re.match``.
    _keys_to_ignore_on_load_unexpected = [f"^{re.escape(prefix)}" for prefix in IGNORED_CHECKPOINT_KEY_PREFIXES]

    def __init__(self, config: Cosmos3MoTTransformerConfig, **kwargs):
        """Build both towers, then apply the freeze policy.

        ``meta_init`` and the other loader kwargs land in ``**kwargs`` and are
        forwarded to :class:`PreTrainedModel` unchanged.
        """
        super().__init__(config, **kwargs)
        if not config.include_gen_pathway:
            raise ValueError(
                "Cosmos3MoTTransformer is the diffusion-training shell: without the generation pathway "
                "(include_gen_pathway=False) it has no timestep conditioning, no latent adapters and "
                "nothing to predict. Set include_gen_pathway=True."
            )

        # HF's marker. The recompute decision belongs to each block, not here: HF's
        # ``gradient_checkpointing_enable()`` walks ``modules()`` and writes every
        # block's ``gradient_checkpointing`` flag (which is also how
        # ``every_n_layers`` reaches them), so this shell must not override that with
        # one container-level boolean. ``MoTDecoderLayer`` then runs its own
        # ``torch.utils.checkpoint`` call -- it cannot hand the decision to
        # ``_gradient_checkpointing_func`` because rope tables are not tensors -- which
        # also means VeOmni's per-block recompute policies cannot bind onto Cosmos3
        # blocks; the boolean flag is the whole contract.
        self.gradient_checkpointing = False

        # Cosmos3's time axis is not patched; ``latent_patch_size`` is spatial.
        patch = int(config.latent_patch_size)
        self.patch_size = (patch, patch)
        # ``build_latent_grid`` speaks ``(pt, ph, pw)``; there is no temporal patch.
        self.latent_grid_patch = (1, patch, patch)

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            getattr(config, "pad_token_id", None),
        )
        self.layers = nn.ModuleList(
            [
                MoTDecoderLayer(config, layer_idx=layer_idx, include_gen_pathway=True)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # One final norm per tower (``unified_mot.py:1075`` and ``:1223-1224``).
        self.norm = MotRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_moe_gen = MotRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # The diffusion tower's entry/exit adapters. ``enable_input_bias`` is True in
        # the Nano recipe, so both projections carry a bias, as does the embedder's
        # MLP (``cosmos3_vfm_network.py:381-383``). ``proj_in`` / ``proj_out`` are the
        # checkpoint's names for upstream's ``vae2llm`` / ``llm2vae``
        # (``_convert_model_to_diffusers.py:1509-1512``).
        self.time_embedder = TimestepEmbedder(config.hidden_size)
        self.proj_in = nn.Linear(config.patch_latent_dim, config.hidden_size)
        self.proj_out = nn.Linear(config.hidden_size, config.patch_latent_dim)

        # The reasoner tower's causal-LM head. ``tie_word_embeddings`` is False in the
        # checkpoint, so it is a separate ``[vocab_size, hidden_size]`` matrix (622M
        # parameters for Nano) that the released index does carry -- building it is
        # what makes the key set cover all 814 transformer tensors. v1 never runs it:
        # there is no AR loss, so ``forward`` has no reason to touch it, and
        # ``freeze_model`` keeps it frozen even for joint training
        # (``ALWAYS_FROZEN_NAME_PREFIXES``). It costs no runtime work -- FSDP2 shards
        # it like any other frozen weight and the forward never reads it -- and it has
        # to exist the day the reasoner tower gets its own AR training.
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Shared mRoPE table builder. It owns no parameters (``inv_freq`` is a
        # non-persistent buffer), so it does not appear in the checkpoint or in the
        # weight-name contract.
        self.rotary_emb = MotRope(
            config.head_dim,
            rope_theta=config.rope_theta,
            mrope_section=config.mrope_section,
            interleaved=config.mrope_interleaved,
        )

        # Print the *effective* mRoPE time base once per run. These knobs are the ones
        # that can be silently inert: ``base_fps`` / ``enable_fps_modulation`` were
        # shipped in ``cosmos3_nano.json`` for months with nothing reading them, so a
        # run trained every 5 fps clip as if it ran at 24 fps and the only symptom was
        # a slightly-too-low loss. Same lesson as ``loss_scale``: a config field that
        # is not printed is a config field nobody can prove is wired.
        logger.info_rank0(
            "Cosmos3 mRoPE: enable_fps_modulation=%s base_fps=%g temporal_compression_factor=%d "
            "modality_margin=%d (generation t = i * base_fps / clip_fps + und_len + margin).",
            config.enable_fps_modulation,
            config.base_fps,
            config.temporal_compression_factor,
            config.unified_3d_mrope_temporal_modality_margin,
        )

        # NOTE: ``from_pretrained`` would need ``self.post_init()`` here (it registers
        # ``all_tied_weights_keys`` and friends); without it HF's loader raises
        # ``AttributeError``. It is deliberately not called: ``post_init`` also runs
        # ``init_weights()`` and rewrites ``_no_split_modules`` into a set, so adding
        # it belongs in its own change, not in the checkpoint-naming one. VeOmni builds
        # and loads through ``load_model_weights`` (meta init + per-tensor dispatch),
        # which does not need ``post_init``.
        self._init_modality_heads()
        # Same policy the trainer applies later through ``freeze_model()``: a model
        # instantiated and used without a trainer must behave identically.
        self.freeze_model()

    # ── weight initialisation ────────────────────────────────────────────────

    @torch.no_grad()
    def _init_modality_heads(self) -> None:
        """Initialise the adapters with the reference's truncated normals.

        Only matters for ``_from_config`` (a run without a checkpoint); loading a
        checkpoint overwrites all of it. Copied from
        ``cosmos3_vfm_network.py:455-462`` so a from-scratch run starts from the same
        scale the released weights have, and so the core's
        :meth:`TimestepEmbedder.reset_parameters` (which exists because the upstream
        ``_init_weights`` clashed with ``nn.Module.apply``) is actually used.
        """
        self.time_embedder.reset_parameters()

        std = 1.0 / math.sqrt(self.config.patch_latent_dim)
        nn.init.trunc_normal_(self.proj_in.weight, std=std, a=-3 * std, b=3 * std)
        if self.proj_in.bias is not None:
            nn.init.zeros_(self.proj_in.bias)

        std = 1.0 / math.sqrt(self.config.hidden_size)
        nn.init.trunc_normal_(self.proj_out.weight, std=std, a=-3 * std, b=3 * std)
        if self.proj_out.bias is not None:
            nn.init.zeros_(self.proj_out.bias)

    # ── the two-tower freeze switch ──────────────────────────────────────────

    def freeze_model(self) -> None:
        """Apply ``config.train_joint_reasoner`` to the parameters.

        Called by ``veomni/models/model_runtime.py:572-574``
        (``_freeze_model_module``: ``freeze_model_function = getattr(self.model,
        "freeze_model", None); if callable(...): freeze_model_function()``), i.e. as a
        bound method with no arguments, and once more at the end of ``__init__``.

        ``train_joint_reasoner=False`` is the SFT recipe: keep the frozen Qwen3-VL
        reasoner tower frozen and train only the diffusion tower, which the whitelist
        expresses by *parameter name* (``TRAINABLE_NAME_KEYS``), exactly as upstream
        does. ``True`` fine-tunes both towers jointly, so everything is released --
        except :data:`ALWAYS_FROZEN_NAME_PREFIXES`, which is not part of the
        join/freeze decision at all. ``lm_head`` is the one such parameter: v1 has no
        AR loss, so it is never in the graph, and releasing it would only create
        optimizer state whose gradient stays ``None`` forever.
        """
        train_joint = bool(getattr(self.config, "train_joint_reasoner", False))
        for name, parameter in self.named_parameters():
            if any(name.startswith(prefix) for prefix in ALWAYS_FROZEN_NAME_PREFIXES):
                parameter.requires_grad_(False)
                continue
            parameter.requires_grad_(train_joint or any(key in name for key in TRAINABLE_NAME_KEYS))

    # ── position ids and rope tables ─────────────────────────────────────────

    def gen_temporal_offset(self, und_len: int) -> float:
        """Temporal cursor the generation span of one sample starts from.

        The reference lays a sample's text span on ``0 .. und_len-1`` and keeps the
        running cursor (``sequence.py:271-275`` -> ``:596-608``, ``temporal_offset``
        threaded through ``_mrope_temporal_offset``), so the first latent frame sits
        at ``und_len`` -- the *post*-append prompt length, i.e. the same number the
        und rope table was built from, never recomputed from the caption alone.
        ``unified_3d_mrope_temporal_modality_margin`` is then added on top
        (``packers.py:570``), which is what the shipped ``cosmos3_nano.json``
        requests with ``= 15000``.

        Args:
            und_len: This sample's reasoner-token count (``input_ids.numel()``).

        Returns:
            ``und_len + margin`` as a float, ready for the mRoPE temporal axis.
        """
        return float(und_len) + float(self.config.unified_3d_mrope_temporal_modality_margin)

    def _sample_fps(self, fps: Optional[Sequence[float]], index: int, count: int) -> Optional[float]:
        """Pick one sample's fps out of an optional per-sample sequence.

        ``None`` means "the loader did not report fps". The reference falls back to
        ``base_fps`` there (``packers.py:274``: ``fps = data.fps_vision[sample] if
        ... else base_fps``), which puts the step back at exactly ``1.0`` -- so the
        fallback is returned explicitly rather than left to the core, and the core
        keeps rejecting "modulation on, no fps" instead of quietly meaning two
        different things. It is logged loudly because a *silent* fallback here would
        re-create the dead-field bug this wiring fixed: every clip would train as if
        it ran at ``base_fps`` with no error anywhere.
        """
        value = None
        if fps is not None:
            value = fps if isinstance(fps, (int, float)) else fps[index]
        if value is None:
            if not self.config.enable_fps_modulation:
                return None
            logger.warning_once(
                "Cosmos3 mRoPE: no per-sample fps reached the transformer, so the temporal axis falls back to "
                "base_fps=%s (step 1.0 per latent frame). That is the pre-FPS-modulation layout -- expected for "
                "offline replays and shape-only calls, but a *training* run that hits it is training every clip "
                "as if it were base_fps.",
                self.config.base_fps,
            )
            return float(self.config.base_fps)
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"sample {index}: fps must be positive and finite, got {value!r} ({count} samples).")
        return value

    def _rope_tables(
        self,
        latent_shape: tuple[int, int, int, int],
        num_text_tokens: int,
        device: torch.device,
        *,
        fps: Optional[Sequence[float]] = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """Build ``(und_rope, gen_rope)``, each a ``(cos, sin)`` pair of ``(L, head_dim)``.

        The two towers live in different positional frames and take their own tables
        (``mot_attention.MotAttention.forward`` takes ``und_rope`` and ``gen_rope``
        separately): the reasoner tokens get ``arange`` on all three axes, the
        diffusion tokens get the patched latent grid, in exactly the flatten order
        :func:`patchify_latent` produces.

        The generation tower's temporal axis is *not* the latent-frame index: it
        advances by ``base_fps / fps`` per latent frame and starts at this sample's
        own cursor (:meth:`gen_temporal_offset`). Both are in the reference
        (``mrope.py:164-180``); leaving them out trains a 5 fps clip as if it were
        ``base_fps`` and puts the generation span on top of the prompt.

        Args:
            latent_shape: ``[C, T_lat, H_lat, W_lat]`` of this sample.
            fps: Optional per-sample frame rate (scalar or one entry per pack member).
        """
        und_ids = build_text_position_ids(num_text_tokens, device=device)
        grid_t, grid_h, grid_w = build_latent_grid(latent_shape, self.latent_grid_patch)
        gen_ids = build_latent_position_ids(
            grid_t,
            grid_h,
            grid_w,
            fps=self._sample_fps(fps, 0, 1),
            base_fps=self.config.base_fps,
            temporal_compression_factor=self.config.temporal_compression_factor,
            enable_fps_modulation=self.config.enable_fps_modulation,
            temporal_offset=self.gen_temporal_offset(num_text_tokens),
            device=device,
        )
        und_rope = self.rotary_emb(und_ids)
        gen_rope = self.rotary_emb(gen_ids)
        expected = patch_row_count(latent_shape, self.patch_size)
        if gen_rope[0].shape[0] != expected:
            raise RuntimeError(
                "the generation-tower mRoPE table and the patchifier disagree on the token count "
                f"({gen_rope[0].shape[0]} vs {expected})."
            )
        return und_rope, gen_rope

    # ── one sample ───────────────────────────────────────────────────────────

    def _validate_latent(self, latent: object, index: int | None = None) -> None:
        """Reject a ``hidden_states`` entry that is not a 4-D latent tensor."""
        where = "" if index is None else f"sample {index}: "
        if not isinstance(latent, torch.Tensor) or latent.dim() != 4:
            raise ValueError(
                f"{where}each `hidden_states` entry must be a 4-D [C, T_lat, H_lat, W_lat] latent; got "
                f"{type(latent).__name__}"
                f"{'' if not isinstance(latent, torch.Tensor) else ' ' + str(tuple(latent.shape))}."
            )

    def _sample_tokens(self, input_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Flatten one sample's prompt ids to a ``(L,)`` long tensor."""
        tokens = input_ids.to(device=device, dtype=torch.long).reshape(-1)
        if tokens.numel() == 0:
            raise ValueError("Cosmos3 needs at least one prompt token per sample in `input_ids`.")
        return tokens

    def _sample_patches(self, latent: torch.Tensor) -> torch.Tensor:
        """Patchify one latent and check its width against ``patch_latent_dim``."""
        patches = patchify_latent(latent, self.patch_size)
        if patches.shape[-1] != self.config.patch_latent_dim:
            raise ValueError(
                f"a {tuple(latent.shape)} latent patchifies to {patches.shape[-1]} values per patch but "
                f"config.patch_latent_dim is {self.config.patch_latent_dim}; the latent channel count must be "
                f"patch_latent_dim / latent_patch_size ** 2 = {self.config.latent_channel}."
            )
        return patches

    def _forward_sample(
        self,
        input_ids: torch.Tensor,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        condition_mask: Optional[torch.Tensor] = None,
        fps: Optional[float] = None,
    ) -> torch.Tensor:
        """Run both towers over one sample and return its predicted velocity field.

        Args:
            input_ids: ``LongTensor(L,)`` prompt tokens (the reasoner tower's input).
            latent: ``[C, T_lat, H_lat, W_lat]`` noised latent ``x_t``, VAE dtype.
            timestep: ``[1]`` timestep -- already ``sigma * num_train_timesteps``.
            condition_mask: Optional ``[T_lat, 1, 1]`` mask, ``1`` on clean
                conditioning frames. Used only to suppress the timestep
                conditioning on those frames (see below); it does not change what
                is predicted, the loss does that.
            fps: Optional frame rate of this clip, for mRoPE FPS modulation. ``None``
                falls back to ``base_fps`` (step 1.0), see
                :meth:`_packed_rope_tables`.

        Returns:
            ``[C, T_lat, H_lat, W_lat]`` predicted velocity, same geometry as the
            input latent.
        """
        self._validate_latent(latent)
        tokens = self._sample_tokens(input_ids, latent.device)

        # Reasoner tower: the frozen Qwen3-VL token embedding.
        hidden_und = self.embed_tokens(tokens)

        # Diffusion tower: latent patches through ``proj_in``. The latents arrive in
        # the VAE dtype (float32) while the model may be bf16, so the cast happens
        # here -- FSDP does not cast activations, and ``proj_in`` would otherwise see
        # a dtype mismatch on its weight.
        patches = self._sample_patches(latent)
        hidden_gen = self.proj_in(patches.to(self.proj_in.weight.dtype))

        # Timestep conditioning. ``timestep`` is passed through untouched: it already
        # carries the ``sigma * num_train_timesteps`` scale the embedder was
        # pretrained on, and ``add_time_conditioning`` applies ``timestep_scale``
        # (1e-3) itself. Pre-scaling or dividing it here would shift the sinusoid
        # phase by that factor.
        #
        # Conditioning frames carry no timestep conditioning at all: the reference
        # scatter-adds the timestep embedding over ``noisy_frame_indexes`` only, so a
        # clean frame reaches the backbone with no timestep added
        # (``cosmos3_vfm_network.py:1096-1101``, ``:2841``). ``row_weight`` is that
        # scatter expressed as a per-patch-row multiplier; ``None`` (no mask) keeps
        # the all-noisy behaviour bit for bit.
        hidden_gen = add_time_conditioning(
            hidden_gen,
            timestep,
            self.time_embedder,
            self.config.timestep_scale,
            row_weight=self._noisy_patch_row_weight(condition_mask, latent.shape, latent.device),
        )

        und_rope, gen_rope = self._rope_tables(latent.shape, tokens.numel(), latent.device, fps=fps)
        for layer in self.layers:
            # No ``checkpointed=`` override: each block reads its own flag, which
            # ``gradient_checkpointing_enable()`` has already set per block.
            hidden_und, hidden_gen = layer(
                hidden_und,
                hidden_gen,
                und_rope=und_rope,
                gen_rope=gen_rope,
            )

        # Per-tower final norms (``unified_mot.py:1223-1224``). Only the generation
        # tower's output is decoded: v1 has no reasoner head (no AR loss), so the
        # reasoner tower's job is done once its K/V have conditioned the joint
        # attention. ``self.norm`` is still applied -- the checkpoint declares it, and
        # this shell exercises the parameter set it loads instead of leaving a
        # declared module silently unused.
        self.norm(hidden_und)
        hidden_gen = self.norm_moe_gen(hidden_gen)

        return unpatchify_latent(self.proj_out(hidden_gen), latent.shape, self.patch_size)

    # ── one packed micro-batch ───────────────────────────────────────────────

    def _packed_rope_tables(
        self,
        latent_shapes: list[tuple[int, int, int, int]],
        und_lengths: list[int],
        layout: PackedLayout,
        device: torch.device,
        fps: Optional[Sequence[float]] = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """Packed ``(und_rope, gen_rope)`` tables, one restart per sample.

        The position ids are built per segment and concatenated
        (:func:`build_segmented_text_position_ids` /
        :func:`build_segmented_latent_position_ids`) rather than built once over the
        packed length: a continuous numbering would put every sample after the first
        into a positional frame the checkpoint never saw, which trains silently
        wrong instead of raising.

        The same per-sample split applies to the mRoPE temporal axis, and there it is
        two numbers rather than one: each sample's own ``fps`` (a pack may mix frame
        rates, so a single pack-level scale factor would be wrong) and each sample's
        own cursor ``und_lengths[i] + margin`` (:meth:`gen_temporal_offset`). Both
        are threaded exactly like :meth:`_packed_time_conditioning` threads the
        per-sample sigma and mask.

        Args:
            latent_shapes: One ``[C, T_lat, H_lat, W_lat]`` per sample, in pack order.
            layout: The pack layout the tables must line up with.
            fps: Optional per-sample frame rate, in pack order.
        """
        und_ids = build_segmented_text_position_ids(und_lengths, device=device)
        grids = [build_latent_grid(shape, self.latent_grid_patch) for shape in latent_shapes]
        gen_ids = build_segmented_latent_position_ids(
            grids,
            fps=[self._sample_fps(fps, index, len(grids)) for index in range(len(grids))],
            base_fps=self.config.base_fps,
            temporal_compression_factor=self.config.temporal_compression_factor,
            enable_fps_modulation=self.config.enable_fps_modulation,
            temporal_offsets=[self.gen_temporal_offset(length) for length in und_lengths],
            device=device,
        )
        und_rope = self.rotary_emb(und_ids)
        gen_rope = self.rotary_emb(gen_ids)
        if gen_rope[0].shape[0] != layout.gen_total:
            raise RuntimeError(
                "the packed generation-tower mRoPE table and the patchifier disagree on the token count "
                f"({gen_rope[0].shape[0]} vs {layout.gen_total})."
            )
        return und_rope, gen_rope

    def _packed_time_conditioning(
        self,
        hidden_gen: torch.Tensor,
        timesteps: list[torch.Tensor],
        gen_lengths: list[int],
        condition_masks: list[Optional[torch.Tensor]],
        latent_shapes: list[tuple[int, int, int, int]],
        device: torch.device,
    ) -> torch.Tensor:
        """Apply each sample's *own* sigma and conditioning mask to the packed pack.

        Two things must not be shared across a pack boundary:

        * **sigma** -- every sample drew its own noise level, so the packed rows carry
          ``repeat_interleave(sigma_i, gen_len_i)`` and not one timestep for the whole
          sequence (which would condition every sample on the first one's noise
          level);
        * the **condition mask** -- per-frame, expanded to this sample's patch rows
          only, then concatenated in pack order.

        The embedding itself is computed once for the packed rows. That is the same
        elementwise conditioning as one call per sample (the MLP is row-independent),
        and it is one of the reasons packing is faster.
        """
        sigmas = []
        for index, timestep in enumerate(timesteps):
            value = torch.as_tensor(timestep, dtype=torch.float32, device=device).reshape(-1)
            if value.numel() != 1:
                raise ValueError(
                    f"sample {index}: `timestep` must hold exactly one value per sample, got {value.numel()}."
                )
            sigmas.append(value)
        sigma_per_row = torch.repeat_interleave(
            torch.cat(sigmas, dim=0),
            torch.tensor(gen_lengths, dtype=torch.long, device=device),
        )
        # A sample without a mask is "every frame generated", i.e. weight 1 on all of
        # its rows -- spelled out here so a batch that mixes masked and unmasked
        # samples concatenates, and so the all-unmasked batch keeps the unmasked
        # ``add_time_conditioning`` path bit for bit.
        weights: list[torch.Tensor] = []
        masked = False
        for mask, latent_shape in zip(condition_masks, latent_shapes):
            weight = self._noisy_patch_row_weight(mask, latent_shape, device)
            if weight is None:
                weight = torch.ones(
                    (patch_row_count(latent_shape, self.patch_size), 1), dtype=torch.float32, device=device
                )
            else:
                masked = True
            weights.append(weight)
        return add_time_conditioning(
            hidden_gen,
            sigma_per_row,
            self.time_embedder,
            self.config.timestep_scale,
            row_weight=torch.cat(weights, dim=0) if masked else None,
        )

    def _forward_packed(
        self,
        input_ids: list[torch.Tensor],
        hidden_states: list[torch.Tensor],
        timestep: list[torch.Tensor],
        condition_masks: list[Optional[torch.Tensor]],
        fps: Optional[Sequence[float]] = None,
    ) -> list[torch.Tensor]:
        """Run both towers over the whole micro-batch as one packed (TND) sequence.

        Every sample's reasoner tokens are concatenated into one ``und`` sequence and
        its latent patches into one ``gen`` sequence; the blocks then run **once** and
        the two attention passes take their sample boundaries from the
        :class:`PackedLayout` (see ``cosmos3_core.mot_attention``). Per-sample
        isolation is a property of that layout -- each sample's K/V is one contiguous
        segment -- so this is exactly equivalent to the per-sample loop in
        :meth:`_forward_sample`; the equivalence was verified element by element during
        development and is exercised by the training runs in the PR.

        Returns:
            One ``[C, T_lat, H_lat, W_lat]`` prediction per sample, in microbatch
            order.
        """
        sample_count = len(hidden_states)
        latent_shapes: list[tuple[int, int, int, int]] = []
        und_lengths: list[int] = []
        gen_lengths: list[int] = []
        token_rows: list[torch.Tensor] = []
        patch_rows: list[torch.Tensor] = []
        for index, latent in enumerate(hidden_states):
            self._validate_latent(latent, index)
            tokens = self._sample_tokens(input_ids[index], latent.device)
            token_rows.append(tokens)
            patch_rows.append(self._sample_patches(latent))
            latent_shapes.append(tuple(int(size) for size in latent.shape))
            und_lengths.append(int(tokens.numel()))
            gen_lengths.append(int(patch_rows[-1].shape[0]))

        device = hidden_states[0].device
        layout = build_packed_layout(und_lengths, gen_lengths, device=device)

        hidden_und = self.embed_tokens(torch.cat(token_rows, dim=0))
        hidden_gen = self.proj_in(torch.cat(patch_rows, dim=0).to(self.proj_in.weight.dtype))
        hidden_gen = self._packed_time_conditioning(
            hidden_gen, timestep, gen_lengths, condition_masks, latent_shapes, device
        )

        und_rope, gen_rope = self._packed_rope_tables(latent_shapes, und_lengths, layout, device, fps=fps)
        for layer in self.layers:
            hidden_und, hidden_gen = layer(
                hidden_und,
                hidden_gen,
                und_rope=und_rope,
                gen_rope=gen_rope,
                layout=layout,
            )

        self.norm(hidden_und)
        rows = self.proj_out(self.norm_moe_gen(hidden_gen))
        predictions: list[torch.Tensor] = []
        start = 0
        for latent_shape, gen_length in zip(latent_shapes, gen_lengths):
            predictions.append(unpatchify_latent(rows[start : start + gen_length], latent_shape, self.patch_size))
            start += gen_length
        if sample_count != len(predictions):
            raise RuntimeError(f"packed forward produced {len(predictions)} predictions for {sample_count} samples.")
        return predictions

    # ── the trainer's forward ────────────────────────────────────────────────

    def _packs_micro_batch(self, sample_count: int, *, packed: Optional[bool] = None) -> bool:
        """Whether this call runs the micro-batch as one packed (TND) sequence.

        ``config.pack_samples`` is the model-side switch (``"auto"`` / ``"always"`` /
        ``"never"``) and ``packed`` overrides it per call. It is deliberately *not*
        the same knob as the data-side ``data.pack_mode``: that one decides how many
        samples share a micro-batch (token budget vs. a fixed count), while this one
        only decides whether those samples run as one sequence or one call each.
        ``"auto"`` is the production setting -- a dataloader that packs must not
        silently fall back to the per-sample loop.
        """
        if packed is not None:
            return bool(packed)
        mode = self.config.pack_samples
        if mode == "always":
            return True
        if mode == "never":
            return False
        return sample_count > 1

    def _noisy_patch_row_weight(
        self,
        condition_mask: Optional[torch.Tensor],
        latent_shape: tuple[int, ...] | torch.Size,
        device: Optional[torch.device] = None,
    ) -> Optional[torch.Tensor]:
        """Expand a per-frame ``condition_mask`` to a per-patch-row ``1 - mask``.

        ``patchify_latent`` lays the rows out frame-major (``[F, h, w]``, frame
        outermost), so one weight per frame repeats ``h * w`` times and lands on
        exactly the rows that frame produced. Returns ``None`` when there is no
        mask, which keeps :func:`add_time_conditioning` on its unmasked path.

        Args:
            condition_mask: ``[T_lat, 1, 1]`` (or broadcastable to it) with ``1`` on
                clean frames, or ``None``.
            latent_shape: ``[C, T_lat, H_lat, W_lat]`` the mask describes.
            device: Device of the returned tensor; defaults to the mask's own.

        Returns:
            ``[T_lat * h * w, 1]`` row weights, or ``None``.
        """
        if condition_mask is None:
            return None
        if device is None:
            device = condition_mask.device
        mask = condition_mask.to(device=device, dtype=torch.float32)
        mask = mask.reshape(-1)  # [T_lat]
        latent_frames = int(latent_shape[1])
        if mask.numel() != latent_frames:
            raise ValueError(
                f"condition_mask has {mask.numel()} frames but the latent has {latent_frames} "
                f"({tuple(latent_shape)}); one mask entry per latent frame is required."
            )
        grid_t, grid_h, grid_w = build_latent_grid(latent_shape, self.latent_grid_patch)
        rows_per_frame = grid_h * grid_w
        noisy = (1.0 - mask).reshape(grid_t, 1).expand(grid_t, rows_per_frame)
        return noisy.reshape(-1, 1).contiguous()

    def forward(
        self,
        input_ids: Optional[list[torch.Tensor]] = None,
        hidden_states: Optional[list[torch.Tensor]] = None,
        timestep: Optional[list[torch.Tensor]] = None,
        training_target: Optional[list[torch.Tensor]] = None,
        ar_attention_mask: Optional[list[torch.Tensor]] = None,
        condition_mask: Optional[list[torch.Tensor]] = None,
        fps: Optional[list] = None,
        *,
        loss_scale: Optional[float] = None,
        packed: Optional[bool] = None,
        **kwargs,
    ) -> Cosmos3MoTTransformerOutput:
        """Train one microbatch: per-sample prediction plus the reduced flow loss.

        Args:
            input_ids: Prompt tokens, one ``LongTensor(L,)`` per sample. Required --
                it is the reasoner tower's input, not a target.
            hidden_states: One ``[C, T_lat, H_lat, W_lat]`` ``x_t`` per sample.
            timestep: One ``[1]`` timestep per sample, already scaled by
                ``num_train_timesteps`` by the condition model.
            training_target: One ``noise - clean`` tensor per sample. Optional: when
                omitted, only predictions are returned (inference / shape checks).
            ar_attention_mask: Accepted and ignored. v1 hands over an all-ones mask
                and the joint attention's causality is structural (the reasoner pass
                is causal, the generation pass is not), so there is nothing to apply.
            condition_mask: One ``[T_lat, 1, 1]`` tensor per sample from
                ``Cosmos3ConditionModel.process_condition``, ``1`` on clean
                conditioning frames (T2V leaves it all zeros, I2V marks the first
                latent frame, V2V the first two). A **list**, like every other
                trainer keyword: ``DiTDataCollator`` keeps one list per batch key
                and videos differ in geometry, so nothing here is stacked. It has
                two effects, both copied from the reference: the timestep
                conditioning is suppressed on the clean frames (see
                ``_forward_sample``) and the flow-matching loss drops their
                contribution while keeping them in the denominator
                (``flow_matching_loss``). ``None`` means "every frame generated",
                which is exactly what ``condition_mask`` all-zeros gives.
            fps: One frame rate per sample, from
                ``Cosmos3ConditionModel.process_condition``. It scales the mRoPE
                temporal axis by ``base_fps / fps`` per latent frame
                (``mrope.py:164-180``); ``None`` falls back to ``base_fps``, i.e. a
                step of exactly 1.0 per latent frame.
            loss_scale: Overrides ``config.loss_scale`` for this call.
            packed: Overrides ``config.pack_samples`` for this call. ``None`` follows
                it: ``"auto"`` packs whenever the micro-batch holds more than one
                sample (the production path -- the dataloader decides how many
                samples share a micro-batch, see ``data.pack_mode``), ``"never"`` runs
                the one-sample-per-call loop, ``"always"`` packs even a single sample.
            **kwargs: Any further collator key is accepted and ignored, so a batch
                that carries extra metadata does not have to be filtered first.

        Returns:
            :class:`Cosmos3MoTTransformerOutput` with ``loss={"dm_loss": scalar}``
            (``None`` without targets) and one prediction per sample.
        """
        if hidden_states is None:
            raise ValueError("Cosmos3MoTTransformer.forward requires `hidden_states` (one latent per sample).")
        if input_ids is None:
            raise ValueError(
                "Cosmos3MoTTransformer.forward requires `input_ids`: the reasoner tower conditions the joint "
                "attention on the prompt, so a batch without tokens has nothing to attend from."
            )
        if not isinstance(hidden_states, (list, tuple)):
            raise TypeError(
                "`hidden_states` must be a per-sample list (DiTDataCollator keeps one list per key); got "
                f"{type(hidden_states).__name__}."
            )
        sample_count = len(hidden_states)
        if sample_count == 0:
            raise ValueError("Cosmos3MoTTransformer.forward needs a non-empty microbatch.")
        if len(input_ids) != sample_count:
            raise ValueError(f"got {len(input_ids)} token rows for {sample_count} latents; they must match.")
        if timestep is None or len(timestep) != sample_count:
            raise ValueError("`timestep` must hold exactly one timestep per sample.")
        if training_target is not None and len(training_target) != sample_count:
            raise ValueError(
                f"got {len(training_target)} training targets for {sample_count} latents; they must match."
            )
        if condition_mask is not None:
            if not isinstance(condition_mask, (list, tuple)):
                raise TypeError(
                    "`condition_mask` must be a per-sample list (DiTDataCollator keeps one list per key); got "
                    f"{type(condition_mask).__name__}."
                )
            if len(condition_mask) != sample_count:
                raise ValueError(
                    f"got {len(condition_mask)} condition masks for {sample_count} latents; they must match."
                )

        scale = self.config.loss_scale if loss_scale is None else float(loss_scale)
        masks = list(condition_mask) if condition_mask is not None else [None] * sample_count
        # One fps per sample, like every other collator column. A scalar is accepted so
        # a hand-built call does not have to repeat it.
        if fps is None:
            fps_per_sample: Optional[list] = None
        elif isinstance(fps, (int, float)):
            fps_per_sample = [float(fps)] * sample_count
        else:
            fps_per_sample = list(fps)
            if len(fps_per_sample) != sample_count:
                raise ValueError(f"got {len(fps_per_sample)} fps entries for {sample_count} samples; they must match.")
        if self._packs_micro_batch(sample_count, packed=packed):
            predictions = self._forward_packed(input_ids, hidden_states, timestep, masks, fps=fps_per_sample)
        else:
            predictions = [
                self._forward_sample(
                    input_ids[index],
                    hidden_states[index],
                    timestep[index],
                    condition_mask=masks[index],
                    fps=None if fps_per_sample is None else fps_per_sample[index],
                )
                for index in range(sample_count)
            ]

        losses: list[torch.Tensor] = []
        for index, prediction in enumerate(predictions):
            if training_target is None:
                continue
            target = training_target[index]
            if prediction.shape != target.shape:
                raise ValueError(
                    f"sample {index}: prediction shape {tuple(prediction.shape)} != training-target shape "
                    f"{tuple(target.shape)}."
                )
            # Per sample, then a plain mean over samples: that *is* the recipe's
            # reduction, and it deliberately does not multiply by the batch size
            # (``DiTTrainer.postforward`` divides by the microbatch count itself).
            # Evaluated in fp32 so a bf16 prediction is not squared in bf16.
            #
            # The reduction is per sample in *both* paths on purpose: packing must not
            # turn the loss into a token-weighted mean, or a packed micro-batch and
            # the same samples unpacked would produce different gradients and the
            # 1:1 comparison against the single-sample reference would stop meaning
            # anything.
            mask_i = masks[index]
            losses.append(
                flow_matching_loss(
                    prediction.float().unsqueeze(0),
                    target.float().unsqueeze(0),
                    condition_mask=(
                        None
                        if mask_i is None
                        else [mask_i.to(device=prediction.device, dtype=torch.float32).unsqueeze(0)]
                    ),
                    loss_scale=scale,
                )
            )

        loss = None if not losses else {"dm_loss": torch.stack(losses).mean()}
        return Cosmos3MoTTransformerOutput(loss=loss, predictions=predictions)

    # ── framework hooks ─────────────────────────────────────────────────────

    def get_parallel_plan(self) -> ParallelPlan:
        """No ExtraParallel dimension: every parameter is plain FSDP2-sharded.

        Both towers are ordinary weights; the mixture-of-towers split is inside each
        block, not across ranks, so there is no expert-parallel mesh to declare.
        """
        return ParallelPlan(extra_parallel_plan={})


def apply_cosmos3_transformer_patch() -> None:
    """Install hook the modeling registry calls before the class is used.

    Every other diffusers family in VeOmni needs this hook to monkey-patch a
    *third-party* backbone: the Wan / Qwen-Image classes are diffusers code, so the
    Ulysses sequence-parallel forward has to be rebound onto them. Cosmos3 has no
    diffusers source -- ``Cosmos3MoTTransformer`` above *is* the backbone, and its
    op-level dispatch (``rms_norm`` / ``rope`` / ``attention`` / MoE) is bound per
    module by ``cosmos3_core`` at construction, through
    :class:`veomni.ops.VeomniOp`. Nothing needs rebinding, so this function does
    nothing by construction: there is no base class to reach.

    It exists rather than being omitted because
    ``cosmos3_transformer/__init__.py`` keeps the family's documented shape -- one
    installation point per family, called at model-load time rather than at import
    time -- so the registry entry and the modeling class stay decoupled.
    """
