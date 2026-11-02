"""Joint two-pass attention for the Cosmos3 mixture-of-towers block.

The reasoner (``und``) and generation (``gen``) towers share one softmax but not one
projection. Upstream expresses that as a single packed sequence with two attention
modes (``mot/attention.py::two_way_attention``), which is exactly two kernel calls:
``und`` tokens attend causally among themselves, ``gen`` tokens attend to every token
of their sample (und K/V concatenated with gen K/V, own queries only).

Two batching shapes reach those calls. **Dense** (``micro_batch_size=1``) needs neither
``cu_seqlens`` nor a mask, so each pass is one kernel call over the whole tensor.
**Packed / TND** puts several samples in one sequence, so each pass is a varlen call
with ``cu_seq_lens_q``/``cu_seq_lens_k`` and the layout in :mod:`.packed_layout`; the
two passes then have different segment pairs (``und``鈫抈`und``, ``gen``鈫抈`und+gen``),
which a varlen kernel accepts as two independent cumulative-length vectors. Because a
sample's K/V block is contiguous (see :func:`.packed_layout.interleave_und_gen`),
cross-sample leakage is impossible by construction rather than by masking.

Dispatch goes through the op registry (``attention/standard``), never a hard-coded
backend: ``flash_attention_*`` rows take the dense call (on Ascend reaching
``torch_npu.npu_fusion_attention``), ``veomni_sdpa`` rows take PyTorch SDPA, and the
``eager`` row resolves :func:`eager_attention_forward`, which doubles as the numerical
reference. Packed dispatch is narrower: ``eager`` builds a block-diagonal mask
(:func:`packed_visibility_mask`) because Transformers' eager interface refuses an
unmasked causal varlen call, ``sdpa``/``veomni_sdpa`` split the pack per segment, the
flash rows forward ``cu_seq_lens_*`` to their varlen kernel, and every other row raises
:class:`NotImplementedError` instead of silently mixing samples. Ulysses sequence
parallelism is refused rather than silently mis-computed, since both passes would have
to be repartitioned per split.

:class:`MotRMSNorm` lives here because the QK norms belong to the attention module; the
tower norms import it from here to keep a single implementation of the op dispatch.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import accumulate
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from veomni.distributed.sequence_parallel.comm import get_ulysses_sequence_parallel_group
from veomni.ops import VeomniOp
from veomni.ops.config import resolve_op_impl
from veomni.utils.device import IS_NPU_AVAILABLE

from .packed_layout import PackedLayout, interleave_und_gen


__all__ = [
    "MotAttention",
    "MotRMSNorm",
    "bind_mot_attention",
    "config_field",
    "eager_attention_forward",
    "joint_attention",
    "packed_visibility_mask",
    "sdpa_reference_joint_attention",
    "sdpa_reference_packed_joint_attention",
]


# Attention rows that can compute a packed (TND) call with ``cu_seq_lens_q`` and
# ``cu_seq_lens_k``. ``eager`` is absent on purpose: it is served by
# :func:`packed_visibility_mask` inside :func:`_attend_varlen`, because
# Transformers' eager interface refuses an unmasked *causal* varlen call.
# ``sdpa`` / ``veomni_sdpa`` split the pack per segment in
# ``veomni/ops/kernels/attention/helper.py::packed_split_attention_forward``, which
# reads both cumulative-length vectors and therefore already supports a query
# segment shorter than its key segment. The flash rows forward the two vectors to
# their varlen kernel unchanged.
#
# This set is the gate for the *fused* Ascend call below, and it must stay keyed on
# the packed-varlen capability rather than on the name the run config spells: the
# string tested at the call site is ``VeomniOp.impl``, i.e. the *resolved* registry
# row (``registry.py::VeomniOp.__new__``: ``instance.impl = entry.impl``), not
# ``configs/dit/*.yaml``'s ``attn_implementation``. On Ascend the configured flash
# row is served by the SDPA-backed adapter, so that string comes back as
# ``veomni_sdpa`` -- narrowing this set to the flash *names* silently demotes the
# model from one fused ``npu_fusion_attention`` call to per-segment launches plus a
# ``sum(q) x sum(k)`` block-diagonal mask per layer per pass, which is numerically
# identical but costs ~45 GB of allocator reserved (measured 47 -> 90 GB/card at an
# unchanged 39.06 GB allocated). To A/B the two kernels, select ``eager`` explicitly;
# do not narrow this set.
PACKED_VARLEN_IMPLS = frozenset(
    {
        "sdpa",
        "veomni_sdpa",
        "flash_attention_2",
        "flash_attention_2_hub",
        "flash_attention_3",
        "flash_attention_3_hub",
        "flash_attention_4",
        "veomni_flash_attention_2",
        "veomni_flash_attention_2_hub",
        "veomni_flash_attention_3",
        "veomni_flash_attention_3_hub",
        "veomni_flash_attention_4",
    }
)


def config_field(config: object, name: str, default: object) -> object:
    """Read ``name`` off any config object, falling back to ``default`` when unset."""
    value = getattr(config, name, None)
    return default if value is None else value


class MotRMSNorm(nn.Module):
    """RMSNorm through ``rms_norm/standard`` (``torch_npu.npu_rms_norm`` on Ascend)."""

    def __init__(self, size: int, *, eps: float) -> None:
        """Allocate the unit weight and bind the interned op handle."""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(size))
        self.veomni_rms_norm = VeomniOp("rms_norm", "standard", resolve_op_impl("rms_norm_implementation"))

    def forward(self, hidden: Tensor) -> Tensor:
        """Normalise the last axis of ``hidden``."""
        return self.veomni_rms_norm(hidden, self.weight, eps=self.eps)


def bind_mot_attention(module: nn.Module, *, impl: str | None = None) -> None:
    """Attach the configured ``attention/standard`` handle and the HF-side metadata.

    These are the attributes Transformers' attention interfaces read off the module
    (``num_key_value_groups`` for GQA expansion, ``is_causal``, ``layer_idx``,
    ``config._attn_implementation``); they are per-module state, never a global.
    """
    impl = impl or resolve_op_impl("attn_implementation", eager_as="veomni_sdpa")
    module.veomni_attn = VeomniOp("attention", "standard", impl)
    module.is_causal = False
    module.layer_idx = getattr(module, "layer_idx", None)
    module.num_key_value_heads = int(getattr(module, "num_key_value_heads", 1))
    module.num_key_value_groups = int(getattr(module, "num_attention_heads", 1)) // module.num_key_value_heads
    module.config = SimpleNamespace(_attn_implementation=impl)


def eager_attention_forward(
    module: nn.Module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None = None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs: object,
) -> tuple[Tensor, None]:
    """Plain SDPA math for the ``eager`` row, and the reference for the tests.

    Transformers' ``eager`` interface is model-local, so defining it in the module
    that owns the attention class makes the registry's eager row resolve here.
    """
    del kwargs
    key = _repeat_kv(key, module.num_key_value_groups)
    value = _repeat_kv(value, module.num_key_value_groups)
    causal = bool(module.is_causal) if is_causal is None else bool(is_causal)
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=causal and attention_mask is None,
    )
    return output.transpose(1, 2).contiguous(), None


def _repeat_kv(hidden: Tensor, groups: int) -> Tensor:
    """Expand ``(B, H_kv, T, D)`` key/value heads to ``H_q`` heads for dense math."""
    if groups == 1:
        return hidden
    batch, heads, length, head_dim = hidden.shape
    hidden = hidden[:, :, None, :, :].expand(batch, heads, groups, length, head_dim)
    return hidden.reshape(batch, heads * groups, length, head_dim)


def _attend(
    module: nn.Module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    is_causal: bool,
    scaling: float | None,
    dropout: float,
) -> Tensor:
    """Run one attention call on ``(T, heads, head_dim)`` and return the same layout."""
    output, _ = module.veomni_attn(
        module,
        query.unsqueeze(0).transpose(1, 2),
        key.unsqueeze(0).transpose(1, 2),
        value.unsqueeze(0).transpose(1, 2),
        None,
        dropout=dropout,
        scaling=scaling,
        is_causal=is_causal,
        skip_ulysses=True,
    )
    # The interfaces return ``(batch, T, heads, head_dim)``.
    return output.squeeze(0)


def packed_visibility_mask(
    query_lengths: Sequence[int],
    key_lengths: Sequence[int],
    *,
    is_causal: bool,
    device: torch.device | str | None = None,
) -> Tensor:
    """Block-diagonal visibility of a packed call, ``(1, 1, sum(q), sum(k))`` bool.

    Segment ``i`` is the only key range query segment ``i`` may see, which is what
    isolates the samples of a pack. ``is_causal`` additionally trims segment ``i`` to
    its lower triangle, so it requires ``query_lengths[i] == key_lengths[i]``: the
    causal passes (the reasoner pass, and the dense one-sample case) always have
    equal sides, while the generation pass is non-causal by construction.

    This is the reference implementation of the packed contract: it is the mask a
    single dense ``scaled_dot_product_attention`` call would need, and the tests
    compare the varlen kernels against it.
    """
    query_lengths = tuple(int(length) for length in query_lengths)
    key_lengths = tuple(int(length) for length in key_lengths)
    if len(query_lengths) != len(key_lengths):
        raise ValueError(
            f"packed attention needs one key length per query segment, got {len(query_lengths)} and "
            f"{len(key_lengths)}."
        )
    if is_causal and query_lengths != key_lengths:
        raise ValueError(
            f"a causal packed call needs equal query and key segment lengths, got {query_lengths} and {key_lengths}."
        )
    query_total = sum(query_lengths)
    key_total = sum(key_lengths)
    mask = torch.zeros((query_total, key_total), dtype=torch.bool, device=device)
    query_start = 0
    key_start = 0
    for query_length, key_length in zip(query_lengths, key_lengths):
        block = torch.ones((query_length, key_length), dtype=torch.bool, device=device)
        if is_causal:
            block = block.tril()
        mask[query_start : query_start + query_length, key_start : key_start + key_length] = block
        query_start += query_length
        key_start += key_length
    return mask.view(1, 1, query_total, key_total)


def packed_attention_impl(module: nn.Module) -> str:
    """The resolved attention row of ``module`` (``eager`` / ``veomni_sdpa`` / ...)."""
    handle = getattr(module, "veomni_attn", None)
    impl = getattr(handle, "impl", None)
    if impl is None:
        impl = getattr(getattr(module, "config", None), "_attn_implementation", None)
    return str(impl)


# --- one fused packed attention call (Ascend) ------------------------------------
# Selected through the ops registry's resolved row (see the guard at the call
# site): Ascend ships no ``flash_attn`` wheel, so the flash rows are served by
# ``torch_npu.npu_fusion_attention``, whose TND layout takes per-sample
# ``actual_seq_qlen`` / ``actual_seq_kvlen`` -- a query segment may therefore be
# shorter than its key segment, which is what the generation pass needs.
#
# ``sparse_mode=3`` consumes a right-down triangular ``atten_mask``, and Ascend's
# kernel accepts at most a 2048x2048 one. Only the *causal* (reasoner) pass supplies
# a mask, and its segments are capped by the tokenizer's
# ``Cosmos3ConditionModelConfig.max_sequence_length``; the bound is asserted at the
# call site instead of being assumed, because exceeding it would either trip the
# kernel or silently attend over the wrong span.
_NPU_CAUSAL_MASK_SIZE = 2048
_NPU_CAUSAL_MASK_CACHE: dict[tuple[str, int | None], Tensor] = {}
_NPU_SPARSE_MODE = 3  # right-down: consume the supplied triu mask as-is


def _npu_causal_mask(device: torch.device) -> Tensor:
    key = (device.type, device.index)
    if key not in _NPU_CAUSAL_MASK_CACHE:
        _NPU_CAUSAL_MASK_CACHE[key] = torch.triu(
            torch.ones([_NPU_CAUSAL_MASK_SIZE, _NPU_CAUSAL_MASK_SIZE], device=device, dtype=torch.bool), diagonal=1
        )
    return _NPU_CAUSAL_MASK_CACHE[key]


def npu_fusion_packed_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    query_lengths: Sequence[int],
    key_lengths: Sequence[int],
    is_causal: bool,
    scaling: float | None,
    dropout: float,
) -> Tensor:
    """One ``npu_fusion_attention`` TND call over the whole pack.

    ``query`` / ``key`` / ``value`` are ``(T, heads, head_dim)`` -- already TND.
    ``query_lengths`` / ``key_lengths`` are the per-sample segment lengths of the two
    sides; the kernel wants them as cumulative ends, which is recomputed here rather
    than read back off the ``cu_*`` device tensors (that would be one device-to-host
    sync per layer per pass -- 144 per forward at 36 layers).
    """
    import torch_npu

    if is_causal:
        longest = max((int(length) for length in query_lengths), default=0)
        if longest > _NPU_CAUSAL_MASK_SIZE:
            raise ValueError(
                f"causal segment of {longest} tokens exceeds the {_NPU_CAUSAL_MASK_SIZE}-token atten_mask that "
                "npu_fusion_attention(sparse_mode=3) accepts. Lower the prompt budget "
                "(condition_model_cfg.max_sequence_length) or pick a non-fused attention row."
            )

    actual_seq_qlen = tuple(accumulate(int(length) for length in query_lengths))
    actual_seq_kvlen = tuple(accumulate(int(length) for length in key_lengths))
    scale = float(scaling) if scaling is not None else float(query.shape[-1]) ** -0.5
    common = dict(
        pse=None,
        padding_mask=None,
        scale=scale,
        keep_prob=1.0 - float(dropout),
        input_layout="TND",
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
    )
    if is_causal:
        output = torch_npu.npu_fusion_attention(
            query, key, value, query.shape[1], atten_mask=_npu_causal_mask(query.device),
            sparse_mode=_NPU_SPARSE_MODE, **common,
        )[0]
    else:
        output = torch_npu.npu_fusion_attention(query, key, value, query.shape[1], **common)[0]
    return output


def _attend_varlen(
    module: nn.Module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    cu_query: Tensor,
    cu_key: Tensor,
    query_lengths: Sequence[int],
    key_lengths: Sequence[int],
    is_causal: bool,
    scaling: float | None,
    dropout: float,
) -> Tensor:
    """One packed (TND) attention call on ``(T, heads, head_dim)`` tensors.

    ``cu_query`` / ``cu_key`` address the two sides independently, so this serves
    both the reasoner pass (equal sides, causal) and the generation pass (query
    segments of ``gen_len_i`` against key segments of ``und_len_i + gen_len_i``).

    Raises:
        NotImplementedError: The bound attention row cannot compute a packed call.
            Silence would be worse than an error here: a dense call over the whole
            pack attends *across* samples, which trains on leaked context and cannot
            be noticed from the loss alone.
    """
    impl = packed_attention_impl(module)
    # One fused npu_fusion_attention (TND) call over the whole pack, instead of one
    # dense kernel per sample. Without this the flash rows reach
    # `ops/kernels/attention/helper.py::packed_split_attention_forward`, which splits
    # the pack per segment: 38 launches per layer per step at 19 clips, measured
    # 1.79x slower than the single TND call on 950DT -- with bit-identical output,
    # because both end in the same dense kernel and TND merely stops re-launching it.
    # The split path also builds a `sum(q) x sum(k)` block-diagonal mask per layer per
    # pass (~2 GB bool at a 45k-token pack, twice under recompute), which is where the
    # device-side reserved high-water mark comes from.
    if IS_NPU_AVAILABLE and impl in PACKED_VARLEN_IMPLS and query.device.type == "npu":
        return npu_fusion_packed_attention(
            query,
            key,
            value,
            query_lengths=query_lengths,
            key_lengths=key_lengths,
            is_causal=is_causal,
            scaling=scaling,
            dropout=dropout,
        )
    if impl == "eager":
        # Transformers' eager interface needs an explicit causal mask, and one
        # SDPA call with the block-diagonal mask *is* the eager formulation.
        mask = packed_visibility_mask(query_lengths, key_lengths, is_causal=is_causal, device=query.device)
        output = F.scaled_dot_product_attention(
            query.unsqueeze(0).transpose(1, 2),
            _repeat_kv(key.unsqueeze(0).transpose(1, 2), int(module.num_key_value_groups)),
            _repeat_kv(value.unsqueeze(0).transpose(1, 2), int(module.num_key_value_groups)),
            attn_mask=mask,
            dropout_p=dropout,
            scale=scaling,
        )
        return output.squeeze(0).transpose(0, 1).contiguous()
    if impl not in PACKED_VARLEN_IMPLS:
        raise NotImplementedError(
            f"Cosmos3 packed (TND) joint attention is not implemented for the {impl!r} attention row. "
            "Use 'veomni_sdpa', 'sdpa', 'eager' or one of the flash-attention rows; the remaining rows "
            "would otherwise apply a single dense attention over the whole pack and let samples see "
            "each other."
        )
    output, _ = module.veomni_attn(
        module,
        query.unsqueeze(0).transpose(1, 2),
        key.unsqueeze(0).transpose(1, 2),
        value.unsqueeze(0).transpose(1, 2),
        None,
        dropout=dropout,
        scaling=scaling,
        is_causal=is_causal,
        skip_ulysses=True,
        cu_seq_lens_q=cu_query,
        cu_seq_lens_k=cu_key,
        max_length_q=max(int(length) for length in query_lengths),
        max_length_k=max(int(length) for length in key_lengths),
    )
    return output.squeeze(0)


def _refuse_sequence_parallel() -> None:
    """Raise when Ulysses SP is active.

    Both passes span whole, unsplit token groups. Under SP each rank holds a slice of
    both towers, so the two K/V concatenations would have to be rebuilt per split
    first; silently attending over rank-local tokens would just be wrong.
    """
    group = get_ulysses_sequence_parallel_group()
    if group is not None and dist.get_world_size(group) > 1:
        raise NotImplementedError(
            "Cosmos3 joint attention does not implement Ulysses sequence parallelism yet; "
            f"it is active with {dist.get_world_size(group)} ranks"
        )


def joint_attention(
    module: nn.Module,
    q_und: Tensor,
    k_und: Tensor,
    v_und: Tensor,
    q_gen: Tensor,
    k_gen: Tensor,
    v_gen: Tensor,
    *,
    k_und_for_gen: Tensor | None = None,
    scaling: float | None = None,
    dropout: float = 0.0,
    layout: PackedLayout | None = None,
) -> tuple[Tensor, Tensor]:
    """The MoT attention contract: causal reasoner pass + full generation pass.

    Args:
        module: A module carrying a bound ``veomni_attn`` handle (see
            :func:`bind_mot_attention`), e.g. :class:`MotAttention`.
        k_und_for_gen: Optional alternative und keys for the generation pass only,
            used when the two towers do not share a K-normalisation convention
            (``k_norm_und_for_gen``); ``None`` reuses ``k_und``.
        scaling: Softmax scale; ``head_dim ** -0.5`` when omitted.
        layout: Packing layout of a multi-sample micro-batch. ``None`` is the
            dense one-sample call; otherwise the tensors are the concatenated
            per-sample sequences and the two passes run as varlen (TND) calls, the
            generation pass with ``gen_len_i`` queries against ``und_len_i +
            gen_len_i`` keys (see :mod:`.packed_layout`).

    Returns:
        ``(out_und, out_gen)`` with the shapes of ``q_und`` / ``q_gen``.
    """
    _refuse_sequence_parallel()
    und_keys = k_und if k_und_for_gen is None else k_und_for_gen
    if layout is None:
        out_und = _attend(module, q_und, k_und, v_und, is_causal=True, scaling=scaling, dropout=dropout)
        full_key = torch.cat((und_keys, k_gen), dim=0)
        full_value = torch.cat((v_und, v_gen), dim=0)
        out_gen = _attend(module, q_gen, full_key, full_value, is_causal=False, scaling=scaling, dropout=dropout)
        return out_und, out_gen

    layout.check_und(q_und.shape[0])
    layout.check_und(k_und.shape[0])
    layout.check_gen(q_gen.shape[0])
    layout.check_gen(k_gen.shape[0])
    und_lengths = layout.und_lengths
    joint_lengths = layout.joint_lengths
    out_und = _attend_varlen(
        module,
        q_und,
        k_und,
        v_und,
        cu_query=layout.und_cu,
        cu_key=layout.und_cu,
        query_lengths=und_lengths,
        key_lengths=und_lengths,
        is_causal=True,
        scaling=scaling,
        dropout=dropout,
    )
    # Sample ``i`` may see ``[und_i ; gen_i]`` and nothing else, so the K/V pack is
    # re-ordered per sample and the two cumulative-length vectors describe different
    # segment sizes on purpose.
    full_key = interleave_und_gen(und_keys, k_gen, layout)
    full_value = interleave_und_gen(v_und, v_gen, layout)
    out_gen = _attend_varlen(
        module,
        q_gen,
        full_key,
        full_value,
        cu_query=layout.gen_cu,
        cu_key=layout.joint_cu,
        query_lengths=layout.gen_lengths,
        key_lengths=joint_lengths,
        is_causal=False,
        scaling=scaling,
        dropout=dropout,
    )
    return out_und, out_gen


def sdpa_reference_packed_joint_attention(
    q_und: Tensor,
    k_und: Tensor,
    v_und: Tensor,
    q_gen: Tensor,
    k_gen: Tensor,
    v_gen: Tensor,
    layout: PackedLayout,
    *,
    k_und_for_gen: Tensor | None = None,
    scaling: float | None = None,
    dropout: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Single-kernel reference for the packed two-pass contract: dense, one mask, no varlen.

    Re-orders the pack into the interleaved ``[und_0 ; gen_0 ; und_1 ; gen_1 ; ...]``
    layout, builds one block-diagonal visibility matrix in which sample ``i``'s
    generation rows see their own ``und_i`` and ``gen_i`` (and its reasoner rows the
    lower triangle of ``und_i``), and calls ``F.scaled_dot_product_attention`` once.
    One call, one mask, no ``cu_seqlens``: independent of the varlen formulation, so
    agreement is evidence about the contract rather than about one code path.
    """
    und_keys = k_und if k_und_for_gen is None else k_und_for_gen
    query = interleave_und_gen(q_und, q_gen, layout)
    key = interleave_und_gen(und_keys, k_gen, layout)
    value = interleave_und_gen(v_und, v_gen, layout)

    total = layout.joint_total
    mask = torch.zeros((total, total), dtype=torch.bool, device=query.device)
    start = 0
    for und_length, gen_length in zip(layout.und_lengths, layout.gen_lengths):
        # Reasoner rows: causal inside their own und block.
        mask[start : start + und_length, start : start + und_length] = torch.ones(
            (und_length, und_length), dtype=torch.bool, device=query.device
        ).tril()
        # Generation rows: the whole sample -- und followed by gen.
        mask[start + und_length : start + und_length + gen_length, start : start + und_length + gen_length] = True
        start += und_length + gen_length

    heads = key.shape[1]
    groups = query.shape[1] // heads
    expanded_key = _repeat_kv(key.transpose(0, 1).unsqueeze(0), groups)
    expanded_value = _repeat_kv(value.transpose(0, 1).unsqueeze(0), groups)
    output = F.scaled_dot_product_attention(
        query.transpose(0, 1).unsqueeze(0),
        expanded_key,
        expanded_value,
        attn_mask=mask.view(1, 1, total, total),
        dropout_p=dropout,
        scale=scaling,
    )
    output = output.squeeze(0).transpose(0, 1)

    out_und: list[Tensor] = []
    out_gen: list[Tensor] = []
    start = 0
    for und_length, gen_length in zip(layout.und_lengths, layout.gen_lengths):
        out_und.append(output[start : start + und_length])
        out_gen.append(output[start + und_length : start + und_length + gen_length])
        start += und_length + gen_length
    return torch.cat(out_und, dim=0), torch.cat(out_gen, dim=0)


def sdpa_reference_joint_attention(
    q_und: Tensor,
    k_und: Tensor,
    v_und: Tensor,
    q_gen: Tensor,
    k_gen: Tensor,
    v_gen: Tensor,
    *,
    k_und_for_gen: Tensor | None = None,
    scaling: float | None = None,
    dropout: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Single-kernel reference for :func:`joint_attention`: dense, block-diagonal mask.

    Builds the visibility matrix over the concatenated ``[und | gen]`` sequence --
    reasoner rows see the lower triangle of the reasoner block, generation rows see
    everything -- and calls ``F.scaled_dot_product_attention`` once. Independent of
    the two-call formulation, so agreement is evidence about the contract rather
    than about one code path.
    """
    und_keys = k_und if k_und_for_gen is None else k_und_for_gen
    query = torch.cat((q_und, q_gen), dim=0)
    key = torch.cat((und_keys, k_gen), dim=0)
    value = torch.cat((v_und, v_gen), dim=0)
    length_und, length_all = q_und.shape[0], query.shape[0]
    mask = torch.zeros((length_all, length_all), dtype=torch.bool, device=query.device)
    mask[:length_und, :length_und] = torch.ones((length_und, length_und), dtype=torch.bool, device=query.device).tril()
    mask[length_und:] = True

    heads = key.shape[1]
    groups = query.shape[1] // heads
    key = _repeat_kv(key.transpose(0, 1).unsqueeze(0), groups)
    value = _repeat_kv(value.transpose(0, 1).unsqueeze(0), groups)
    output = F.scaled_dot_product_attention(
        query.transpose(0, 1).unsqueeze(0),
        key,
        value,
        attn_mask=mask.view(1, 1, length_all, length_all),
        dropout_p=dropout,
        scale=scaling,
    )
    output = output.squeeze(0).transpose(0, 1)
    return output[:length_und], output[length_und:]


class MotAttention(nn.Module):
    """Two independent QKVO projection sets feeding one joint attention.

    Everything the checkpoint stores under ``...self_attn.`` lives here. The
    attribute names are the *released checkpoint's* names, not
    ``cosmos_framework``'s internal training names: NVIDIA's exporter renames the
    generation tower's ``*_moe_gen`` projections to the diffusers attention
    spellings, and the shipped checkpoint we load is the exporter's output
    (``cosmos_framework/scripts/_convert_model_to_diffusers.py::_ATTN_KEY_REMAP``,
    lines 105-118). Naming the modules the way the exporter *leaves* them is what
    makes ``checkpoint_key == model_key`` and removes the need for any key-mapping
    table at load time::

        cosmos_framework (training)      checkpoint (diffusers)    role
        self_attn.q_proj                 self_attn.to_q            reasoner Q
        self_attn.k_proj                 self_attn.to_k            reasoner K
        self_attn.v_proj                 self_attn.to_v            reasoner V
        self_attn.o_proj                 self_attn.to_out          reasoner output
        self_attn.q_norm                 self_attn.norm_q          reasoner Q norm
        self_attn.k_norm                 self_attn.norm_k          reasoner K norm
        self_attn.q_proj_moe_gen         self_attn.add_q_proj      generation Q
        self_attn.k_proj_moe_gen         self_attn.add_k_proj      generation K
        self_attn.v_proj_moe_gen         self_attn.add_v_proj      generation V
        self_attn.o_proj_moe_gen         self_attn.to_add_out      generation output
        self_attn.q_norm_moe_gen         self_attn.norm_added_q    generation Q norm
        self_attn.k_norm_moe_gen         self_attn.norm_added_k    generation K norm

    The ``add_*`` spelling is inherited from Qwen2.5-VL, where the ``add_*`` set
    projects the *added* (image/video) modality; the MoT exporter reuses it for the
    added *generation* tower. ``_is_edge_shared_diffusers_key`` in the same script
    (lines 1097-1105) treats ``.self_attn.add_*`` / ``.self_attn.norm_added_*`` /
    ``.self_attn.to_add_out.`` as the generator markers, confirming the role, and
    ``cosmos_framework/inference/model.py:109`` maps them straight back to
    ``*_moe_gen`` when a diffusers checkpoint is loaded for training.

    The projections are *not* shared: the two towers were trained with separate
    matrices even though the modulation is drawn from one softmax.
    """

    def __init__(self, config: object, *, layer_idx: int = 0, include_gen_pathway: bool = True) -> None:
        """Build both projection sets, the QK norms and the rope handle."""
        super().__init__()
        hidden_size = int(config_field(config, "hidden_size", 4096))
        num_attention_heads = int(config_field(config, "num_attention_heads", 32))
        num_key_value_heads = int(config_field(config, "num_key_value_heads", num_attention_heads))
        head_dim = int(config_field(config, "head_dim", hidden_size // num_attention_heads))
        attention_bias = bool(config_field(config, "attention_bias", False))
        eps = float(config_field(config, "rms_norm_eps", 1e-6))
        qk_norm_for_text = bool(config_field(config, "qk_norm_for_text", True))
        qk_norm_for_diffusion = bool(config_field(config, "qk_norm_for_diffusion", True))
        use_und_k_norm_for_gen = bool(config_field(config, "use_und_k_norm_for_gen", False))

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.scaling = head_dim**-0.5
        self.attention_dropout = float(config_field(config, "attention_dropout", 0.0))
        self.include_gen_pathway = include_gen_pathway

        query_out = num_attention_heads * head_dim
        key_value_out = num_key_value_heads * head_dim
        # Reasoner tower (``cosmos_framework``'s ``q_proj``/...).
        self.to_q = nn.Linear(hidden_size, query_out, bias=attention_bias)
        self.to_k = nn.Linear(hidden_size, key_value_out, bias=attention_bias)
        self.to_v = nn.Linear(hidden_size, key_value_out, bias=attention_bias)
        self.to_out = nn.Linear(query_out, hidden_size, bias=attention_bias)

        if qk_norm_for_text:
            self.norm_q: nn.Module = MotRMSNorm(head_dim, eps=eps)
            self.norm_k: nn.Module = MotRMSNorm(head_dim, eps=eps)
        else:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()

        # Cross-tower K norm: only meaningful when the generation pass normalises K
        # but the reasoner does not, i.e. the Nemotron-3 convention. Cosmos3-Nano
        # enables QK norm on both towers, so this stays ``None`` there -- which is
        # also why Cosmos3-Nano's checkpoint has no ``...k_norm_und_for_gen`` key.
        if include_gen_pathway and use_und_k_norm_for_gen and qk_norm_for_diffusion and not qk_norm_for_text:
            self.k_norm_und_for_gen: nn.Module | None = MotRMSNorm(head_dim, eps=eps)
        else:
            self.k_norm_und_for_gen = None

        if include_gen_pathway:
            # Generation tower: the exporter's ``add_*`` names (see the class docstring).
            if qk_norm_for_diffusion:
                self.norm_added_q: nn.Module = MotRMSNorm(head_dim, eps=eps)
                self.norm_added_k: nn.Module = MotRMSNorm(head_dim, eps=eps)
            else:
                self.norm_added_q = nn.Identity()
                self.norm_added_k = nn.Identity()
            self.add_q_proj = nn.Linear(hidden_size, query_out, bias=attention_bias)
            self.add_k_proj = nn.Linear(hidden_size, key_value_out, bias=attention_bias)
            self.add_v_proj = nn.Linear(hidden_size, key_value_out, bias=attention_bias)
            self.to_add_out = nn.Linear(query_out, hidden_size, bias=attention_bias)

        self.veomni_rope = VeomniOp("rope", "full", resolve_op_impl("rotary_pos_emb_implementation"))
        bind_mot_attention(self)

    def _project(
        self, proj_q: nn.Linear, proj_k: nn.Linear, proj_v: nn.Linear, hidden: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Project ``hidden`` and split the flat output into ``(T, heads, head_dim)``."""
        query = proj_q(hidden).view(-1, self.num_attention_heads, self.head_dim)
        key = proj_k(hidden).view(-1, self.num_key_value_heads, self.head_dim)
        value = proj_v(hidden).view(-1, self.num_key_value_heads, self.head_dim)
        return query, key, value

    def forward(
        self,
        hidden_und: Tensor,
        hidden_gen: Tensor,
        *,
        und_rope: tuple[Tensor, Tensor],
        gen_rope: tuple[Tensor, Tensor],
        dropout: float | None = None,
        layout: PackedLayout | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Attend jointly and project back per tower.

        Args:
            hidden_und: Reasoner hidden states, ``(L_und, hidden_size)``.
            hidden_gen: Generation hidden states, ``(L_gen, hidden_size)``.
            und_rope: ``(cos, sin)`` tables of the reasoner tokens, ``(L_und, head_dim)``.
            gen_rope: ``(cos, sin)`` tables of the generation tokens, ``(L_gen, head_dim)``.
            dropout: Overrides ``config.attention_dropout`` when given.
            layout: Packing layout of a multi-sample micro-batch; ``None`` is the
                dense one-sample call. Both towers are packed independently (they
                have different segment lengths), which is why the layout carries one
                pair of boundaries per tower.

        Returns:
            ``(out_und, out_gen)``, both ``(L, hidden_size)``.
        """
        dropout = self.attention_dropout if dropout is None else dropout
        q_und, k_und, v_und = self._project(self.to_q, self.to_k, self.to_v, hidden_und)
        q_und = self.norm_q(q_und)
        k_und = self.norm_k(k_und)
        k_und_for_gen = None
        if self.k_norm_und_for_gen is not None:
            # Rotate a normalised copy for the generation pass; the reasoner keeps raw K.
            k_und_for_gen = self.veomni_rope(k_und, self.k_norm_und_for_gen(k_und), und_rope[0], und_rope[1])[1]
        q_und, k_und = self.veomni_rope(q_und, k_und, und_rope[0], und_rope[1])

        if self.include_gen_pathway:
            q_gen, k_gen, v_gen = self._project(self.add_q_proj, self.add_k_proj, self.add_v_proj, hidden_gen)
            q_gen = self.norm_added_q(q_gen)
            k_gen = self.norm_added_k(k_gen)
            q_gen, k_gen = self.veomni_rope(q_gen, k_gen, gen_rope[0], gen_rope[1])
        else:
            # Reasoner-only layer: nothing in this transformer produces generation
            # rows, so the generation tower is absent rather than silent.
            if hidden_gen.shape[0]:
                raise ValueError(
                    "this attention was built without a generation pathway but received generation tokens"
                )
            if layout is not None:
                layout.check_und(q_und.shape[0])
                und_lengths = layout.und_lengths
                out_und = _attend_varlen(
                    self,
                    q_und,
                    k_und,
                    v_und,
                    cu_query=layout.und_cu,
                    cu_key=layout.und_cu,
                    query_lengths=und_lengths,
                    key_lengths=und_lengths,
                    is_causal=True,
                    scaling=self.scaling,
                    dropout=dropout,
                )
            else:
                out_und = _attend(self, q_und, k_und, v_und, is_causal=True, scaling=self.scaling, dropout=dropout)
            return self.to_out(out_und.reshape(out_und.shape[0], -1)), hidden_gen

        out_und, out_gen = joint_attention(
            self,
            q_und,
            k_und,
            v_und,
            q_gen,
            k_gen,
            v_gen,
            k_und_for_gen=k_und_for_gen,
            scaling=self.scaling,
            dropout=dropout,
            layout=layout,
        )
        out_und = self.to_out(out_und.reshape(out_und.shape[0], -1))
        out_gen = self.to_add_out(out_gen.reshape(out_gen.shape[0], -1))
        return out_und, out_gen
