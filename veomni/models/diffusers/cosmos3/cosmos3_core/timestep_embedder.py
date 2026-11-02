"""Sinusoidal timestep conditioning for the Cosmos3 generation tower.

Cosmos3 reuses the DiT recipe: sinusoidal features of the timestep, then a two-layer MLP
with a SiLU between them. The timestep enters the network already multiplied by
``timestep_scale`` (1e-3), not the embedding.

Module and parameter names (``time_embedder.linear_1`` / ``linear_2``) follow the
released checkpoint, so it loads without a key-mapping table. Those are diffusers'
names: upstream stores the MLP as ``Sequential([Linear, SiLU, Linear])`` (``mlp.0.*`` /
``mlp.2.*``) and NVIDIA's exporter rewrites the four suffixes
(``_convert_model_to_diffusers.py::_TIME_EMBEDDER_KEY_REMAP``). Index 1 is the
parameter-free SiLU and appears in neither state dict.

The reference exposes ``_init_weights(self, buffer_device=None)``, which clashes with
``nn.Module.apply``'s one-argument call; the same initialisation is offered here as
:meth:`TimestepEmbedder.reset_parameters`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


__all__ = ["TimestepEmbedder", "add_time_conditioning", "timestep_embedding"]


def _build_frequencies(dim: int, max_period: int, *, device: torch.device | None = None) -> Tensor:
    """Geometric frequencies ``exp(-log(max_period) * i / (dim/2))``, shape ``(dim/2,)``."""
    half = dim // 2
    frequencies = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
    return frequencies.to(device=device) if device is not None else frequencies


def timestep_embedding(t: Tensor, dim: int, *, max_period: int = 10000, frequencies: Tensor | None = None) -> Tensor:
    """Build sinusoidal timestep features.

    Args:
        t: 1-D tensor of ``N`` timesteps (fractional values are allowed).
        frequencies: Pre-built ``(dim/2,)`` basis, e.g. a module buffer, so the same
            values are used on every device and every step.

    Returns:
        ``(N, dim)`` features: ``[cos(args), sin(args)]``, zero-padded for odd ``dim``.
    """
    if t.ndim != 1:
        raise ValueError(f"t must be 1-D, got shape {tuple(t.shape)}")
    if frequencies is None:
        frequencies = _build_frequencies(dim, max_period, device=t.device)
    args = t.to(torch.float32)[:, None] * frequencies.to(device=t.device, dtype=torch.float32)[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedder(nn.Module):
    """Map a scalar timestep to a ``hidden_size`` conditioning vector."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256, bias: bool = True) -> None:
        """Build the 256 -> hidden -> hidden MLP with a SiLU between the linears."""
        super().__init__()
        self.linear_1 = nn.Linear(frequency_embedding_size, hidden_size, bias=bias)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.frequency_embedding_size = frequency_embedding_size
        self.hidden_size = hidden_size
        self._has_bias = bias
        self.register_buffer(
            "_timestep_frequencies",
            _build_frequencies(frequency_embedding_size, 10000),
            persistent=False,
        )

    def reset_parameters(self) -> None:
        """Truncated-normal init of both linears, as the reference `_init_weights`."""
        std = 1.0 / math.sqrt(self.frequency_embedding_size)
        torch.nn.init.trunc_normal_(self.linear_1.weight, std=std, a=-3 * std, b=3 * std)
        if self._has_bias:
            torch.nn.init.zeros_(self.linear_1.bias)
        std = 1.0 / math.sqrt(self.hidden_size)
        torch.nn.init.trunc_normal_(self.linear_2.weight, std=std, a=-3 * std, b=3 * std)
        if self._has_bias:
            torch.nn.init.zeros_(self.linear_2.bias)

    def forward(self, t: Tensor) -> Tensor:
        """Embed ``t``, shape ``(N,)``, into ``(N, hidden_size)``.

        The two linears run in **fp32**. The reference's time embedder is never cast
        to bf16: its frequency buffer stays fp32, and an offline fp32 rebuild of this
        MLP reproduces its captured output to 2.7e-6 while a bf16 rebuild is 8.9e-3
        off. Running ours in bf16 (the model dtype) put a 5.2e-3 / rel 2.8e-3 step on
        the conditioning vector -- the first tensor of the forward above 1e-3. The
        result is cast back to the module dtype, which is what the reference's
        scatter-add into bf16 hidden states does.
        """
        features = timestep_embedding(t, self.frequency_embedding_size, frequencies=self._timestep_frequencies)
        w1, w2 = self.linear_1.weight, self.linear_2.weight
        b1, b2 = self.linear_1.bias, self.linear_2.bias
        hidden = torch.nn.functional.linear(
            features, w1.to(torch.float32), None if b1 is None else b1.to(torch.float32)
        )
        hidden = self.act(hidden)
        out = torch.nn.functional.linear(hidden, w2.to(torch.float32), None if b2 is None else b2.to(torch.float32))
        return out.to(w1.dtype)


def add_time_conditioning(
    hidden: Tensor,
    sigma: Tensor,
    embedder: TimestepEmbedder,
    timestep_scale: float,
    *,
    row_weight: Tensor | None = None,
) -> Tensor:
    """Add the timestep embedding to generation-tower hidden states.

    ``timestep_scale`` multiplies the *timestep* before it reaches the embedder
    (``cosmos3_vfm_network.py:1091``), not the embedding afterwards: scaling the
    embedding would change the branch the conditioning MLP learned to produce,
    while scaling ``t`` only rescales its phase.

    Args:
        hidden: Generation-tower hidden states, ``(N, hidden_size)``.
        sigma: Per-row timesteps, ``(N,)`` or a scalar.
        timestep_scale: Recipe scale (``0.001``).
        row_weight: Optional ``(N, 1)`` / ``(N,)`` multiplier on the embedding.
            Conditioning passes ``1 - condition_mask`` per patch row here,
            because the reference adds the timestep embedding **only to noisy
            tokens**: ``_apply_timestep_embeds_to_noisy_tokens`` scatter-adds it
            over ``modality.noisy_frame_indexes`` and leaves clean frames
            untouched (``cosmos3_vfm_network.py:1096-1101`` and ``:2841``).
            Passing zeros on the clean rows is that same scatter-add written as a
            masking multiply, so a clean frame enters the backbone with no
            timestep conditioning at all -- exactly as upstream.

    Returns:
        ``hidden + embedder(sigma * timestep_scale) * row_weight``, cast back to
        ``hidden.dtype``.
    """
    if not isinstance(sigma, Tensor):
        sigma = torch.as_tensor(sigma, dtype=torch.float32, device=hidden.device)
    sigma = sigma.to(device=hidden.device, dtype=torch.float32).reshape(-1)
    if sigma.numel() == 1:
        # One timestep for the whole sample (every T2V / I2V / V2V draw). The
        # reference evaluates the conditioning MLP once for the sample and
        # scatter-adds it over the noisy rows, i.e. a ``[1, F] x [F, H]`` matvec.
        # Expanding sigma to N rows first turns that into an ``[N, F]`` GEMM, which
        # accumulates differently: measured max |d| = 5.18e-3 (rel 2.8e-3) on the
        # conditioning vector, the first tensor of the forward above 1e-3. Broadcast
        # the result instead of the input so both sides run the same shape.
        embedding = embedder(sigma * timestep_scale).to(hidden.dtype).expand(hidden.shape[0], -1)
    else:
        if sigma.shape[0] != hidden.shape[0]:
            raise ValueError(f"sigma holds {sigma.shape[0]} rows but hidden holds {hidden.shape[0]}")
        embedding = embedder(sigma * timestep_scale).to(hidden.dtype)
    if row_weight is not None:
        weight = row_weight.to(device=hidden.device, dtype=hidden.dtype).reshape(-1, 1)
        if weight.shape[0] != hidden.shape[0]:
            raise ValueError(f"row_weight holds {weight.shape[0]} rows but hidden holds {hidden.shape[0]}")
        embedding = embedding * weight
    return hidden + embedding
