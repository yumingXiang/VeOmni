"""TND (packed, variable-length) layout for the Cosmos3 mixture-of-towers forward.

With ``micro_batch_size > 1`` the towers are not run once per sample: every sample's
reasoner tokens are concatenated into one ``und`` sequence, every sample's latent
patches into one ``gen`` sequence, and the two attention passes are told where the
samples begin and end. Same "compute only real tokens" idea as the text-side packing
(``docs/usage/data_packing_and_dyn_bsz.md``), carried by ``cu_seqlens`` instead of a
dense mask.

Two layouts are in play, and they are **not** the same sequence:

* ``und`` pass -- queries and keys both in packed *reasoner* order, so
  ``und_cu == [0, T_0, T_0+T_1, ...]`` and every segment is causal.
* ``gen`` pass -- queries in packed *generation* order (``gen_cu``), keys/values in
  per-sample ``[und_i ; gen_i]`` interleaved order (``joint_cu``). One sample's K/V
  is contiguous in that layout, which is what lets a single varlen kernel serve a
  query segment shorter than its key segment.

Why the interleaved K/V and not ``cat([und, gen])``: a varlen kernel addresses a
query segment and its key segment by *index*, so segment ``i``'s keys must be one
contiguous range. ``cat([und, gen])`` would scatter sample ``i``'s keys over two
ranges (its ``und`` rows at the front, its ``gen`` rows at the back), which no
``cu_seqlens``-based kernel can express. ``interleave_und_gen`` is therefore part of
the contract, not an optimisation.

Nothing here touches a tensor's values: the layout only carries cumulative lengths.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


__all__ = [
    "PackedLayout",
    "build_packed_layout",
    "interleave_und_gen",
]


def _cumulative(lengths: tuple[int, ...], *, device: torch.device | str | None) -> Tensor:
    """``[0, l_0, l_0+l_1, ...]`` as the ``int32`` cu_seqlens tensor kernels expect."""
    total = 0
    values = [0]
    for length in lengths:
        total += length
        values.append(total)
    return torch.tensor(values, dtype=torch.int32, device=device)


@dataclass(frozen=True)
class PackedLayout:
    """Segment boundaries of one packed micro-batch.

    Attributes:
        und_lengths: Reasoner token count per sample, in microbatch order.
        gen_lengths: Generation (latent patch row) count per sample, same order.
        und_cu: ``int32`` cumulative reasoner lengths, ``(n + 1,)``.
        gen_cu: ``int32`` cumulative generation lengths, ``(n + 1,)``.
        joint_cu: ``int32`` cumulative ``und_lengths[i] + gen_lengths[i]``,
            ``(n + 1,)`` -- the *interleaved* K/V layout of the generation pass.
    """

    und_lengths: tuple[int, ...]
    gen_lengths: tuple[int, ...]
    und_cu: Tensor
    gen_cu: Tensor
    joint_cu: Tensor

    def __post_init__(self) -> None:
        """Reject a layout that cannot describe the tensors it will be used on."""
        if len(self.und_lengths) != len(self.gen_lengths):
            raise ValueError(
                f"und_lengths holds {len(self.und_lengths)} samples but gen_lengths holds "
                f"{len(self.gen_lengths)}; a pack needs one pair per sample."
            )
        if not self.und_lengths:
            raise ValueError("a packed batch needs at least one sample.")
        for name, lengths in (("und", self.und_lengths), ("gen", self.gen_lengths)):
            if any(length <= 0 for length in lengths):
                raise ValueError(f"{name}_lengths must all be positive, got {lengths}")

    # ── derived quantities ───────────────────────────────────────────────────

    @property
    def num_samples(self) -> int:
        """How many samples the pack holds."""
        return len(self.und_lengths)

    @property
    def und_total(self) -> int:
        """Total reasoner tokens, i.e. the length of the packed ``und`` sequence."""
        return sum(self.und_lengths)

    @property
    def gen_total(self) -> int:
        """Total generation tokens, i.e. the length of the packed ``gen`` sequence."""
        return sum(self.gen_lengths)

    @property
    def joint_total(self) -> int:
        """Total K/V rows of the generation pass, ``und_total + gen_total``."""
        return self.und_total + self.gen_total

    @property
    def joint_lengths(self) -> tuple[int, ...]:
        """Per-sample K/V row counts of the generation pass."""
        return tuple(und + gen for und, gen in zip(self.und_lengths, self.gen_lengths))


    def check_und(self, rows: int) -> None:
        """Assert a packed ``und`` tensor holds exactly this layout's rows."""
        if rows != self.und_total:
            raise ValueError(
                f"the packed reasoner tensor holds {rows} rows but this layout describes {self.und_total}; "
                "the per-sample token counts and the concatenated tensors disagree."
            )

    def check_gen(self, rows: int) -> None:
        """Assert a packed ``gen`` tensor holds exactly this layout's rows."""
        if rows != self.gen_total:
            raise ValueError(
                f"the packed generation tensor holds {rows} rows but this layout describes {self.gen_total}; "
                "the per-sample patch counts and the concatenated tensors disagree."
            )


def build_packed_layout(
    und_lengths: list[int] | tuple[int, ...],
    gen_lengths: list[int] | tuple[int, ...],
    *,
    device: torch.device | str | None = None,
) -> PackedLayout:
    """Build the layout of one micro-batch from its per-sample token counts.

    Args:
        device: Device the cumulative-length tensors are created on. Attention
            kernels want them next to Q/K/V; a CPU-side caller may leave it ``None``.

    Returns:
        The :class:`PackedLayout` the forward pass threads through the blocks.
    """
    und = tuple(int(length) for length in und_lengths)
    gen = tuple(int(length) for length in gen_lengths)
    return PackedLayout(
        und_lengths=und,
        gen_lengths=gen,
        und_cu=_cumulative(und, device=device),
        gen_cu=_cumulative(gen, device=device),
        joint_cu=_cumulative(tuple(a + b for a, b in zip(und, gen)), device=device),
    )


def interleave_und_gen(und: Tensor, gen: Tensor, layout: PackedLayout) -> Tensor:
    """Re-order ``[und_0..und_n]`` + ``[gen_0..gen_n]`` into ``[und_0; gen_0; ...]``.

    This is the generation pass's K/V layout (see the module docstring): each
    sample's reasoner keys followed by its own generation keys, so one contiguous
    range holds everything that sample may attend to.

    Args:
        und: Packed reasoner tensor, ``(und_total, ...)``.
        gen: Packed generation tensor, ``(gen_total, ...)``.

    Returns:
        A contiguous tensor of ``(joint_total, ...)`` in interleaved order.
    """
    layout.check_und(und.shape[0])
    layout.check_gen(gen.shape[0])
    und_parts = und.split(list(layout.und_lengths), dim=0)
    gen_parts = gen.split(list(layout.gen_lengths), dim=0)
    interleaved: list[Tensor] = []
    for und_part, gen_part in zip(und_parts, gen_parts):
        interleaved.append(und_part)
        interleaved.append(gen_part)
    return torch.cat(interleaved, dim=0).contiguous()
