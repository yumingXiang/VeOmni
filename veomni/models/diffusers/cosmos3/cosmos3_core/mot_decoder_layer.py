"""The Cosmos3 mixture-of-towers decoder block.

Why it looks duplicated: Cosmos3 keeps *one* attention softmax shared by the
reasoner (``und``) and diffusion (``gen``) towers, but gives each tower its own
normalisation, its own projection matrices and its own feed-forward. The two
towers therefore walk the block side by side, disagreeing on everything except
the attention kernel. This mirrors
``cosmos_framework/model/generator/mot/unified_mot.py`` (``MoTDecoderLayer``).

Parameter names are the released checkpoint's, so it loads without a key-mapping
table. Per layer that is exactly 22 keys::

    self_attn.to_q / to_k / to_v / to_out         self_attn.norm_q / norm_k
    self_attn.add_q_proj / add_k_proj / add_v_proj / to_add_out
    self_attn.norm_added_q / norm_added_k
    mlp.{gate,up,down}_proj                       mlp_moe_gen.{gate,up,down}_proj
    input_layernorm / input_layernorm_moe_gen
    post_attention_layernorm / post_attention_layernorm_moe_gen

The ``add_*`` attention names are the exporter's spelling of the generation
tower's ``*_moe_gen`` projections (see :class:`~.mot_attention.MotAttention`);
``_moe_gen`` therefore survives on the MLPs and the two norms of the generation
tower only, never inside ``self_attn``.

Feed-forward selection follows upstream: both towers get the same block *class*
per layer -- a dense gated MLP, or a sparse MoE block when the backbone is a MoE
model and ``(layer_idx + 1) % decoder_sparse_step == 0``. Cosmos3-Nano's
``num_experts`` is 0, so both towers are dense there.
"""

from __future__ import annotations

import functools

import torch
import torch.nn as nn
from torch import Tensor
from transformers.activations import ACT2FN

from .mot_attention import MotAttention, MotRMSNorm, config_field
from .packed_layout import PackedLayout


__all__ = ["MoTDecoderLayer", "MotMLP", "MotSparseMoeBlock", "moe_layer_enabled"]


class MotMLP(nn.Module):
    """SwiGLU feed-forward used by both towers (``gate/up/down_proj`` naming)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        hidden_act: str = "silu",
        bias: bool = False,
    ) -> None:
        """Build the three projections and resolve the activation."""
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden: Tensor) -> Tensor:
        """Gate ``up_proj`` with the activated ``gate_proj`` branch."""
        return self.down_proj(self.act_fn(self.gate_proj(hidden)) * self.up_proj(hidden))


class MotSparseMoeBlock(nn.Module):
    """Top-k routed MoE feed-forward for the generation tower.

    Dropless and capacity-free: every token reaches its ``num_experts_per_tok``
    experts, and the routing weights are renormalised over the selected set
    (``norm_topk_prob``), matching the Qwen3-MoE convention the checkpoint was
    trained with. The router is evaluated in fp32 for the same reason upstream
    does it -- the routing decision is a discrete choice and must not move with
    the autocast dtype.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_act: str = "silu",
        bias: bool = False,
        norm_topk_prob: bool = True,
        shared_expert_intermediate_size: int | None = None,
    ) -> None:
        """Build the router, the expert list and the optional shared expert."""
        super().__init__()
        if num_experts <= 0:
            raise ValueError("MotSparseMoeBlock needs num_experts > 0")
        if not 0 < num_experts_per_tok <= num_experts:
            raise ValueError(f"num_experts_per_tok={num_experts_per_tok} must be in [1, {num_experts}]")
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [MotMLP(hidden_size, intermediate_size, hidden_act=hidden_act, bias=bias) for _ in range(num_experts)]
        )
        if shared_expert_intermediate_size:
            self.shared_expert: nn.Module | None = MotMLP(
                hidden_size, shared_expert_intermediate_size, hidden_act=hidden_act, bias=bias
            )
            self.shared_expert_gate: nn.Module | None = nn.Linear(hidden_size, 1, bias=False)
        else:
            self.shared_expert = None
            self.shared_expert_gate = None

    def forward(self, hidden: Tensor) -> Tensor:
        """Route every row to its top-k experts and sum the weighted outputs."""
        logits = self.gate(hidden.to(torch.float32))  # (N, E)
        probs = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float32)
        topk_probs, topk_index = probs.topk(self.num_experts_per_tok, dim=-1)  # (N, k)
        if self.norm_topk_prob:
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp(min=torch.finfo(torch.float32).tiny)
        topk_probs = topk_probs.to(hidden.dtype)

        output = torch.zeros_like(hidden)
        for expert_index, expert in enumerate(self.experts):
            selected = topk_index == expert_index  # (N, k) bool
            # ``nonzero`` keeps the (row, slot) pairs so a row routed to the same
            # expert twice stays correct; empty selections cost one no-op select.
            rows, slots = selected.nonzero(as_tuple=True)
            expert_out = expert(hidden.index_select(0, rows)) * topk_probs[rows, slots].unsqueeze(-1)
            output = output.index_add(0, rows, expert_out)
        if self.shared_expert is not None and self.shared_expert_gate is not None:
            gate = torch.sigmoid(self.shared_expert_gate(hidden))
            output = output + gate * self.shared_expert(hidden)
        return output


def moe_layer_enabled(config: object, layer_idx: int) -> bool:
    """Whether ``layer_idx`` is a sparse-MoE layer, following upstream's rule."""
    num_experts = int(config_field(config, "num_experts", 0) or 0)
    if num_experts <= 0:
        return False
    sparse_step = int(config_field(config, "decoder_sparse_step", 1) or 1)
    mlp_only = tuple(config_field(config, "mlp_only_layers", ()) or ())
    return (layer_idx + 1) % sparse_step == 0 and layer_idx not in mlp_only


class MoTDecoderLayer(nn.Module):
    """One mixture-of-towers block: two norms, two FFNs, one joint attention."""

    _repeated_blocks = ["MoTDecoderLayer"]

    def __init__(self, config: object, *, layer_idx: int = 0, include_gen_pathway: bool = True) -> None:
        """Build the shared attention module and the per-tower norms and FFNs."""
        super().__init__()
        hidden_size = int(config_field(config, "hidden_size", 4096))
        eps = float(config_field(config, "rms_norm_eps", 1e-6))
        hidden_act = str(config_field(config, "hidden_act", "silu"))
        # Qwen3 feed-forwards carry no bias; the recipe keeps that (override via ``mlp_bias``).
        ffn_bias = bool(config_field(config, "mlp_bias", False))
        intermediate_size = int(config_field(config, "intermediate_size", 12288))
        moe_intermediate_size = int(config_field(config, "moe_intermediate_size", intermediate_size))
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.include_gen_pathway = include_gen_pathway

        self.self_attn = MotAttention(config, layer_idx=layer_idx, include_gen_pathway=include_gen_pathway)

        use_moe = moe_layer_enabled(config, layer_idx)
        ffn_size = moe_intermediate_size if use_moe else intermediate_size
        moe_kwargs = {
            "num_experts": int(config_field(config, "num_experts", 0) or 1),
            "num_experts_per_tok": int(config_field(config, "num_experts_per_tok", 1) or 1),
            "hidden_act": hidden_act,
            "bias": ffn_bias,
            "norm_topk_prob": bool(config_field(config, "norm_topk_prob", True)),
        }
        if use_moe:
            self.mlp: nn.Module = MotSparseMoeBlock(hidden_size, ffn_size, **moe_kwargs)
        else:
            self.mlp = MotMLP(hidden_size, intermediate_size, hidden_act=hidden_act, bias=ffn_bias)
        if include_gen_pathway:
            if use_moe:
                gen_kwargs = dict(moe_kwargs)
                # The generation tower may route more experts per token than the
                # reasoner tower (upstream ``gen_moe_top_k``).
                gen_top_k = config_field(config, "gen_moe_top_k", moe_kwargs["num_experts_per_tok"])
                gen_kwargs["num_experts_per_tok"] = int(gen_top_k or moe_kwargs["num_experts_per_tok"])
                gen_kwargs["shared_expert_intermediate_size"] = config_field(
                    config, "shared_expert_intermediate_size", None
                )
                self.mlp_moe_gen: nn.Module = MotSparseMoeBlock(hidden_size, ffn_size, **gen_kwargs)
            else:
                self.mlp_moe_gen = MotMLP(hidden_size, intermediate_size, hidden_act=hidden_act, bias=ffn_bias)

        self.input_layernorm = MotRMSNorm(hidden_size, eps=eps)
        self.post_attention_layernorm = MotRMSNorm(hidden_size, eps=eps)
        if include_gen_pathway:
            self.input_layernorm_moe_gen: nn.Module = MotRMSNorm(hidden_size, eps=eps)
            self.post_attention_layernorm_moe_gen: nn.Module = MotRMSNorm(hidden_size, eps=eps)

        # HF-style switch: ``checkpointed=None`` in ``forward`` follows this flag.
        self.gradient_checkpointing = False

    def _forward(
        self,
        hidden_und: Tensor,
        hidden_gen: Tensor,
        *,
        und_rope: tuple[Tensor, Tensor],
        gen_rope: tuple[Tensor, Tensor],
        dropout: float | None = None,
        layout: PackedLayout | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Pre-norm attention and pre-norm feed-forward for both towers."""
        attn_und, attn_gen = self.self_attn(
            self.input_layernorm(hidden_und),
            self.input_layernorm_moe_gen(hidden_gen) if self.include_gen_pathway else hidden_gen,
            und_rope=und_rope,
            gen_rope=gen_rope,
            dropout=dropout,
            layout=layout,
        )
        residual_und = hidden_und + attn_und
        out_und = residual_und + self.mlp(self.post_attention_layernorm(residual_und))
        if not self.include_gen_pathway:
            if hidden_gen.shape[0]:
                raise ValueError("this block was built without a generation pathway but received generation tokens")
            return out_und, hidden_gen
        residual_gen = hidden_gen + attn_gen
        out_gen = residual_gen + self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(residual_gen))
        return out_und, out_gen

    def forward(
        self,
        hidden_und: Tensor,
        hidden_gen: Tensor,
        *,
        und_rope: tuple[Tensor, Tensor],
        gen_rope: tuple[Tensor, Tensor],
        dropout: float | None = None,
        checkpointed: bool | None = None,
        use_reentrant: bool = False,
        layout: PackedLayout | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run the block, optionally under activation checkpointing.

        Args:
            hidden_und: Reasoner hidden states, ``(L_und, hidden_size)``.
            hidden_gen: Generation hidden states, ``(L_gen, hidden_size)``.
            und_rope: Reasoner ``(cos, sin)`` tables, ``(L_und, head_dim)``.
            gen_rope: Generation ``(cos, sin)`` tables, ``(L_gen, head_dim)``.
            checkpointed: Wrap the block in ``torch.utils.checkpoint``. ``None``
                follows the module's ``gradient_checkpointing`` flag and only
                applies while training, so a framework can flip the flag once
                instead of threading the decision through every layer.
            use_reentrant: Checkpoint flavour; non-reentrant is the default
                because the rope op takes non-tensor tables.
            layout: Packing layout when ``hidden_und`` / ``hidden_gen`` hold several
                samples concatenated (see :mod:`.packed_layout`); ``None`` is one sample.

        Returns:
            ``(out_und, out_gen)``, the residuals added per tower.
        """
        enabled = self.gradient_checkpointing if checkpointed is None else checkpointed
        # Only tensors take part in the checkpointed call: the rope tables, the
        # dropout flag and the (constant, gradient-free) packing layout are bound
        # here, so the graph the recompute replays sees exactly the same arguments
        # without them being tracked as inputs.
        bound = functools.partial(self._forward, und_rope=und_rope, gen_rope=gen_rope, dropout=dropout, layout=layout)
        if enabled and self.training:
            return torch.utils.checkpoint.checkpoint(bound, hidden_und, hidden_gen, use_reentrant=use_reentrant)
        return bound(hidden_und, hidden_gen)
