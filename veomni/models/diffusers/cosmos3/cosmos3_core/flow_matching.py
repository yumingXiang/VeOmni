"""Rectified-flow numerics for Cosmos3 supervised fine-tuning.

The sigma distribution and the loss reduction are the two places where a
"reasonable-looking" rewrite stops matching the recipe the released checkpoint was
trained with, so both are reproduced term by term from
``diffusion/rectified_flow.py`` (``TrainTimeSampler``, ``get_interpolation``) and
``algorithm/loss/flow_matching.py`` (``compute_flow_matching_loss``), with the non-ltx2
shift warp folded into the sampler. Nothing is imported from ``cosmos_framework``.

Notation is the rectified-flow one, inverted relative to diffusion: ``x_0`` is noise,
``x_1`` the clean latent, ``sigma`` the weight of the noise side.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


__all__ = [
    "WAVER_MODE_S",
    "TrainTimeSampler",
    "flow_matching_loss",
    "interpolate",
    "shift_times",
]


# Slope of the ``waver`` distribution (``TrainTimeSampler._WAVER_MODE_S``).
WAVER_MODE_S = 1.29


def shift_times(t: Tensor, shift: float | Tensor | Sequence[float]) -> Tensor:
    """Warp raw samples ``t`` by the rectified-flow resolution shift.

    ``sigma = shift * t / (1 + (shift - 1) * t)`` maps ``[0, 1]`` onto ``[0, 1]``
    monotonically, pushing mass towards the noise end for ``shift > 1``. The
    recipe uses ``shift=3`` for the 256p bucket (``{"256": 3, "480": 5, "720": 10}``).

    Args:
        t: Raw samples in ``[0, 1]``, shape ``(batch,)``.
        shift: Scalar, per-sample ``(batch,)`` tensor, or a sequence of length
            ``batch``. Must be positive.

    Returns:
        Shifted samples with ``t``'s shape.
    """
    if not isinstance(shift, Tensor):
        shift = torch.as_tensor(shift, dtype=t.dtype, device=t.device)
    else:
        shift = shift.to(device=t.device, dtype=t.dtype)
    shift = shift.reshape(-1)
    if shift.numel() == 1:
        shift = shift.reshape(())
    elif shift.numel() != t.shape[0]:
        raise ValueError(f"shift must hold 1 or {t.shape[0]} values, got {shift.numel()}")
    else:
        shift = shift.reshape(*t.shape)
    if bool((shift <= 0).any()):
        raise ValueError("shift must be positive")
    return shift * t / (1 + (shift - 1) * t)


class TrainTimeSampler:
    """Sample the rectified-flow interpolation weight ``sigma`` for training.

    ``waver`` is the video default of the recipe (``train_time_video_distribution``)
    and is the distribution whose exact bit pattern matters most: it is a
    cosine-smoothed uniform
    ``t = 1 - u - s * (cos(pi/2 * u) ** 2 - 1 + u)`` with ``s = 1.29``.

    Only the three distributions the recipe uses are implemented. ``ltx2`` in the
    reference needs per-sample logit-normal means and percentile stretching and is
    never selected by the SFT recipe, so it raises instead of being half-copied.
    """

    _WAVER_MODE_S = WAVER_MODE_S

    def __init__(self, distribution: str = "uniform") -> None:
        """Store the distribution name, rejecting unknown ones early."""
        if distribution not in ("uniform", "logitnormal", "waver"):
            raise NotImplementedError(f"Time distribution {distribution!r} is not implemented.")
        self.distribution = distribution

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
        shift: float | Tensor | Sequence[float] | None = None,
    ) -> Tensor:
        """Draw ``batch_size`` samples in ``[0, 1]``, shifted when ``shift`` is given.

        Args:
            dtype: Target dtype (the reference draws in fp32 and casts afterwards).
            generator: Optional generator for reproducible draws; the test suite
                relies on it to compare the ``waver`` formula bit for bit.
            shift: Optional per-sample resolution shift, applied as the last step
                (the reference skips it only for ``ltx2``).

        Returns:
            ``sigma`` of shape ``(batch_size,)`` on ``device`` with ``dtype``.
        """
        if self.distribution == "uniform":
            t = torch.rand((batch_size,), generator=generator)
        elif self.distribution == "logitnormal":
            t = torch.sigmoid(torch.randn((batch_size,), generator=generator))
        else:
            u = torch.rand((batch_size,), dtype=torch.float32, generator=generator)
            t = 1.0 - u - self._WAVER_MODE_S * (torch.cos(torch.pi / 2.0 * u) ** 2 - 1 + u)
        t = t.to(device=device, dtype=dtype)
        if shift is not None:
            t = shift_times(t, shift)
        return t


def _broadcast_sigma(sigma: Tensor | float, like: Tensor) -> Tensor:
    """Reshape ``sigma`` so it broadcasts against ``like`` along the leading axes.

    The reference call sites pass either ``(batch,)`` or an already
    pre-broadcast ``(batch, 1, 1, 1, 1)``; both are accepted, as is a scalar.
    A per-frame ``(T, 1, 1)`` weight against a ``(C, T, H, W)`` clip is accepted
    too: that is the shape conditioning produces
    (``cosmos3.py:1162``, ``sigmas_list[i] = t * (1 - condition_mask[i])``).
    """
    if not isinstance(sigma, Tensor):
        return torch.as_tensor(sigma, dtype=like.dtype, device=like.device)
    sigma = sigma.to(device=like.device, dtype=like.dtype)
    if sigma.ndim == like.ndim:
        return sigma
    if sigma.numel() == 1:
        return sigma.reshape(1)
    if sigma.ndim == 1:
        return sigma.reshape(sigma.shape[0], *([1] * (like.ndim - 1)))
    if sigma.ndim >= 2 and sigma.ndim == like.ndim - 1 and sigma.shape[0] == like.shape[1]:
        # Per-frame weight laid out T-first with trailing singletons: put the
        # channel axis back in front so it lines up with ``(C, T, ...)``.
        return sigma.reshape(1, *sigma.shape)
    raise ValueError(f"sigma shape {tuple(sigma.shape)} does not broadcast to {tuple(like.shape)}")


def interpolate(noise: Tensor, clean: Tensor, sigma: Tensor | float) -> tuple[Tensor, Tensor]:
    """Interpolate the noisy state and the velocity target at ``sigma``.

    ``x_t = noise * sigma + clean * (1 - sigma)`` and ``target = noise - clean``,
    exactly ``RectifiedFlow.get_interpolation`` for a single batch element group.

    Args:
        noise: ``x_0`` with shape ``(batch, ...)``.
        clean: ``x_1`` with the same shape as ``noise``.
        sigma: Per-sample weight, ``(batch,)`` / ``(batch, 1, ...)`` / scalar.
            A per-frame ``(T, 1, 1)`` weight is also accepted for a
            ``(C, T, H, W)`` clip, which is how a ``condition_mask`` enters the
            interpolation: ``sigma_eff = sigma * (1 - condition_mask)`` makes the
            conditioned frames land exactly on the clean latent (``cosmos3.py:1156``).

    Returns:
        ``(x_t, target)``, both shaped like ``noise``.
    """
    if noise.shape != clean.shape:
        raise ValueError(f"noise shape {tuple(noise.shape)} != clean shape {tuple(clean.shape)}")
    weight = _broadcast_sigma(sigma, noise)
    x_t = noise * weight + clean * (1 - weight)
    target = noise - clean
    return x_t, target


def _as_sample_list(value: Tensor | Sequence[Tensor] | None, name: str) -> list[Tensor | None]:
    """Split a batched tensor along dim 0, or pass a per-sample sequence through."""
    if value is None:
        return []
    if isinstance(value, Tensor):
        if value.ndim == 0:
            raise ValueError(f"{name} must carry a leading batch dimension")
        return list(value.unbind(0))
    return list(value)


def flow_matching_loss(
    pred: Tensor | Sequence[Tensor],
    target: Tensor | Sequence[Tensor],
    *,
    condition_mask: Tensor | Sequence[Tensor] | None = None,
    time_weight: Tensor | Sequence[Tensor] | None = None,
    loss_scale: float = 10.0,
) -> Tensor:
    """Rectified-flow squared-error loss, reduced like the reference recipe.

    Per sample the reference computes ``(sqerr * noisy * time_weight).mean()``
    where ``noisy = 1 - condition_mask``; the batch loss is the mean over samples
    times ``loss_scale`` (``loss_scale=10.0`` is the recipe value, exposed here so
    a config can move it).

    The denominator is ``.mean()``'s, i.e. **every** element of the sample,
    conditioned ones included. Why the reference keeps conditioned frames in the
    denominator: the loss there is
    ``per_instance_losses.append((sqerr_i * noisy_mask_i).mean())`` -- a plain
    ``mean`` over the full tensor (``cosmos_framework/model/generator/algorithm/
    loss/flow_matching.py:151``), with only the numerators masked. The
    conditioned frames therefore dilute each sample's loss by exactly the
    supervised fraction ``T_noisy / T``; they contribute no gradient.

    Dividing by ``noisy.sum()`` instead (the *active-only* mean, the reference's
    ``normalize_by_active=True``, which the SFT recipe does not set) would inflate
    every I2V/V2V loss by ``T / T_noisy``. The upstream ``.mean()`` is what this
    function implements.

    ``time_weight`` implements ``train_time_weight="uniform"`` as its default
    ``None`` (weight ≠ 1). A fully conditioned sample reduces to a
    gradient-carrying zero instead of ``0/0``: the numerator is exactly zero, so
    the ``.mean()`` is zero.

    Args:
        pred: Predicted velocity, ``(batch, ...)`` or a sequence of per-sample tensors.
        condition_mask: ``1`` on clean/conditioning elements, ``0`` on generated
            ones; broadcastable to ``pred[i]`` (e.g. ``(T, 1, 1)`` for a
            ``(C, T, H, W)`` clip, ``(batch, 1, 1, 1)`` for a batched one).
        time_weight: Optional non-uniform per-element timestep weight.

    Returns:
        Scalar loss tensor.
    """
    preds = _as_sample_list(pred, "pred")
    targets = _as_sample_list(target, "target")
    if len(preds) != len(targets):
        raise ValueError(f"pred and target disagree on batch size: {len(preds)} vs {len(targets)}")
    masks = _as_sample_list(condition_mask, "condition_mask")
    weights = _as_sample_list(time_weight, "time_weight")
    if masks and len(masks) != len(preds):
        raise ValueError("condition_mask must hold one entry per sample")
    if weights and len(weights) != len(preds):
        raise ValueError("time_weight must hold one entry per sample")

    per_sample: list[Tensor] = []
    for index, (pred_i, target_i) in enumerate(zip(preds, targets)):
        if pred_i.shape != target_i.shape:
            raise ValueError(
                f"sample {index}: pred shape {tuple(pred_i.shape)} != target shape {tuple(target_i.shape)}"
            )
        sqerr = (pred_i - target_i) ** 2
        mask_i = masks[index] if masks else None
        noisy = torch.ones_like(sqerr) if mask_i is None else (1.0 - mask_i).expand_as(sqerr)
        weight_i = weights[index] if weights else None
        weighted = sqerr * noisy if weight_i is None else sqerr * noisy * weight_i.expand_as(sqerr)
        # ``.mean()``, not ``.sum() / noisy.sum()``: see the denominator note above.
        per_sample.append(weighted.mean())
    return torch.stack(per_sample).mean() * loss_scale
