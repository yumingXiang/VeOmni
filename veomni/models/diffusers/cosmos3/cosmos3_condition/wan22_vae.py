"""Wan2.2 VAE access for the Cosmos3 condition side 鈥?a thin wrapper, not a port.

Cosmos3-Nano ships its video tokenizer as a standard ``diffusers`` Wan2.2 VAE
(``vae/diffusion_pytorch_model.safetensors`` + ``vae/config.json``), so the
whole codec already exists upstream: ``diffusers.AutoencoderKLWan``. Re-porting
the ~2300 lines of ``cosmos_framework``'s ``wan2pt2_vae_4x16x16.py`` would only
duplicate diffusers and add a second set of weights to keep in sync with the
checkpoint. This module therefore owns just the four things diffusers does not
give us:

1. ``[B, C, T, H, W]`` in ``[-1, 1]`` as the *only* accepted video contract, so
   the caller cannot accidentally feed ``[0, 255]`` uint8 frames.
2. latent mean/std normalization, read from the VAE config (never hardcoded when
   the checkpoint carries it) and inverted by :meth:`denormalize_latents`.
3. an explicit "no ``channels_last_3d``" stance: the NPU 3D-conv kernels accept
   only ``contiguous_format`` / ``preserve_format`` and raise
   ``Only contiguous_format or preserve_format is supported`` for
   ``torch.channels_last_3d``. We never call ``enable_channels_last_*`` and
   never pass ``memory_format=torch.channels_last_3d``; :func:`_reject_channels_last_3d`
   turns a checkpoint that somehow arrives in that layout into an error instead
   of a device-side crash.
4. opt-in temporal chunking (default off) for NPUs whose memory cannot hold one
   full clip's encoder activations.

The wrapper is an ``nn.Module`` so the parent ``PreTrainedModel.to(device)``
moves the frozen VAE along with the rest of the condition model, exactly like
``wan_t2v``'s condition model relies on.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch

from .....utils import logging


logger = logging.get_logger(__name__)


# Wan2.2's published latent statistics (48 channels). Only used when the VAE
# config does not carry ``latents_mean`` / ``latents_std`` 鈥?e.g. a converted
# checkpoint that dropped the VAE config, or a bare safetensors directory.
# Source: ``cosmos_framework/model/generator/tokenizers/wan2pt2_vae_4x16x16.py``
# ("Wan 2.2 mean and std values (48 dimensions)"), i.e. the values the reference
# implementation normalizes with, so a fallback stays numerically aligned.
_WAN22_LATENTS_MEAN = (
    -0.2289,
    -0.0052,
    -0.1323,
    -0.2339,
    -0.2799,
    0.0174,
    0.1838,
    0.1557,
    -0.1382,
    0.0542,
    0.2813,
    0.0891,
    0.1570,
    -0.0098,
    0.0375,
    -0.1825,
    -0.2246,
    -0.1207,
    -0.0698,
    0.5109,
    0.2665,
    -0.2108,
    -0.2158,
    0.2502,
    -0.2055,
    -0.0322,
    0.1109,
    0.1567,
    -0.0729,
    0.0899,
    -0.2799,
    -0.1230,
    -0.0313,
    -0.1649,
    0.0117,
    0.0723,
    -0.2839,
    -0.2083,
    -0.0520,
    0.3748,
    0.0152,
    0.1957,
    0.1433,
    -0.2944,
    0.3573,
    -0.0548,
    -0.1681,
    -0.0667,
)

_WAN22_LATENTS_STD = (
    0.4765,
    1.0364,
    0.4514,
    1.1677,
    0.5313,
    0.4990,
    0.4818,
    0.5013,
    0.8158,
    1.0344,
    0.5894,
    1.0901,
    0.6885,
    0.6165,
    0.8454,
    0.4978,
    0.5759,
    0.3523,
    0.7135,
    0.6804,
    0.5833,
    1.4146,
    0.8986,
    0.5659,
    0.7069,
    0.5338,
    0.4889,
    0.4917,
    0.4069,
    0.4999,
    0.6866,
    0.4093,
    0.5709,
    0.6065,
    0.6415,
    0.4944,
    0.5726,
    1.2042,
    0.5458,
    1.6887,
    0.3971,
    1.0600,
    0.3943,
    0.5537,
    0.5444,
    0.4089,
    0.7468,
    0.7744,
)

# Wan2.2 VAE strides: 4x temporal, 16x spatial (the "4x16x16" tokenizer).
_WAN22_TEMPORAL_SCALE = 4
_WAN22_SPATIAL_SCALE = 16


def _reject_channels_last_3d(module: torch.nn.Module, *, name: str) -> None:
    """Fail loudly if any parameter/buffer is stored as ``channels_last_3d``.

    An NPU 3D conv supports only ``contiguous_format`` / ``preserve_format``; a
    ``channels_last_3d`` tensor reaches the kernel and dies with ``Only
    contiguous_format or preserve_format is supported``. On CUDA the layout is legal,
    so the check is guarded: it must not turn a working GPU configuration into a
    load-time error. Checking once at construction keeps the NPU failure at load time,
    where it names the component.
    """
    from .....utils.device import IS_NPU_AVAILABLE

    if not IS_NPU_AVAILABLE:
        return
    for tensor_name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
        if tensor.dim() != 5:
            continue
        if tensor.is_contiguous(memory_format=torch.channels_last_3d) and not tensor.is_contiguous():
            raise RuntimeError(
                f"{name}.{tensor_name} is stored in channels_last_3d layout. Unsupported 3D conv memory "
                "format on this accelerator 鈥?reload the component with contiguous weights."
            )


def _channel_stats(values: Any, *, channels: int, device: torch.device, dtype: torch.dtype, name: str):
    """Shape a per-channel statistic into ``[1, C, 1, 1, 1]`` for broadcast."""
    if isinstance(values, torch.Tensor):
        tensor = values.detach().to(device=device, dtype=dtype).reshape(-1)
    else:
        tensor = torch.as_tensor([float(value) for value in values], dtype=dtype, device=device)
    if tensor.numel() != channels:
        raise ValueError(f"Wan2.2 VAE {name} has {tensor.numel()} entries but the VAE has {channels} latent channels.")
    return tensor.view(1, channels, 1, 1, 1)


class Wan22VAE(torch.nn.Module):
    """Cosmos3's video codec: Wan2.2 4x16x16 VAE with latent normalization.

    Contract:
      * :meth:`encode` takes ``[B, 3, T, H, W]`` in ``[-1, 1]`` and returns the
        *unnormalized* latent mean ``[B, 48, 1 + (T - 1) // 4, H // 16, W // 16]``
        (deterministic ``mode()`` of the posterior, which is what the reference
        model trains against 鈥?sampling would inject noise the target does not
        account for).
      * :meth:`normalize_latents` / :meth:`denormalize_latents` apply
        ``(z - mean) / std`` per channel; :meth:`encode_latents` is the
        composition. The diffusion transformer only ever sees normalized
        latents.
      * :meth:`decode` inverts the whole pipeline back to ``[B, 3, T, H, W]``.
    """

    def __init__(
        self,
        vae: torch.nn.Module,
        *,
        latents_mean: Any = None,
        latents_std: Any = None,
        temporal_chunk_frames: int = 0,
        temporal_chunk_warmup: int = 4,
        component_name: str = "vae",
    ):
        super().__init__()
        self.vae = vae
        self.component_name = component_name
        self.temporal_chunk_frames = int(temporal_chunk_frames)
        self.temporal_chunk_warmup = int(temporal_chunk_warmup)
        if self.temporal_chunk_frames and self.temporal_chunk_frames % self.temporal_scale:
            raise ValueError(
                f"temporal_chunk_frames must be a multiple of the VAE temporal stride ({self.temporal_scale}), "
                f"got {self.temporal_chunk_frames}."
            )
        self._latents_mean_source = latents_mean
        self._latents_std_source = latents_std
        _reject_channels_last_3d(vae, name=component_name)

    # 鈹€鈹€ construction 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    @classmethod
    def from_pretrained(
        cls,
        base_model_path: str,
        *,
        subfolder: str = "vae",
        torch_dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        temporal_chunk_frames: int = 0,
        temporal_chunk_warmup: int = 4,
        **kwargs: Any,
    ) -> "Wan22VAE":
        """Load the diffusers Wan2.2 VAE from ``<base_model_path>/<subfolder>``.

        ``diffusers`` is imported here rather than at module import time: the
        condition-side tests and any CPU-only tooling must be able to import
        this module (and the configuration that points at it) without pulling in
        the whole diffusion stack.
        """
        from diffusers import AutoencoderKLWan

        if not base_model_path:
            raise ValueError("Wan22VAE.from_pretrained requires the checkpoint root (base_model_path).")
        component_name = os.path.join(base_model_path, subfolder) if subfolder else base_model_path
        vae = AutoencoderKLWan.from_pretrained(base_model_path, subfolder=subfolder or None, torch_dtype=torch_dtype)
        vae.eval()
        vae.requires_grad_(False)
        if device is not None:
            vae.to(device)
        return cls(
            vae,
            temporal_chunk_frames=temporal_chunk_frames,
            temporal_chunk_warmup=temporal_chunk_warmup,
            component_name=component_name,
            **kwargs,
        )

    # 鈹€鈹€ config / device view 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    @property
    def config(self) -> Any:
        return self.vae.config

    @property
    def latent_channels(self) -> int:
        """Latent channel count (``z_dim``; 48 for Wan2.2's 4x16x16 VAE)."""
        z_dim = getattr(self.vae.config, "z_dim", None)
        if z_dim is None:
            raise AttributeError("Wan2.2 VAE config carries no z_dim; cannot infer the latent channel count.")
        return int(z_dim)

    @property
    def spatial_scale(self) -> int:
        return int(getattr(self.vae.config, "scale_factor_spatial", _WAN22_SPATIAL_SCALE))

    @property
    def temporal_scale(self) -> int:
        return int(getattr(self.vae.config, "scale_factor_temporal", _WAN22_TEMPORAL_SCALE))

    @property
    def dtype(self) -> torch.dtype:
        return next(self.vae.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.vae.parameters()).device

    # 鈹€鈹€ latent statistics 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def _latent_stats(self, *, device: torch.device, dtype: torch.dtype):
        """Per-channel ``(mean, std)`` from the VAE config, else Wan2.2's published values."""
        channels = self.latent_channels
        mean = self._latents_mean_source
        std = self._latents_std_source
        if mean is None:
            mean = getattr(self.vae.config, "latents_mean", None)
        if std is None:
            std = getattr(self.vae.config, "latents_std", None)
        if mean is None or std is None:
            logger.warning(
                "%s: VAE config has no latents_mean/latents_std; falling back to Wan2.2's published 48-channel "
                "statistics. Verify the checkpoint really uses them before trusting the trained scale.",
                self.component_name,
            )
            mean = _WAN22_LATENTS_MEAN if mean is None else mean
            std = _WAN22_LATENTS_STD if std is None else std
        return (
            _channel_stats(mean, channels=channels, device=device, dtype=dtype, name="latents_mean"),
            _channel_stats(std, channels=channels, device=device, dtype=dtype, name="latents_std"),
        )

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """``(z - latents_mean) / latents_std`` 鈥?the scale the DiT was trained on."""
        mean, std = self._latent_stats(device=latents.device, dtype=latents.dtype)
        return (latents - mean) / std

    def latent_stats_fingerprint(self) -> str:
        """Content digest of the ``(latents_mean, latents_std)`` normalization.

        Part of the offline-latent cache key: the same clip and the same weights
        produce different *normalized* latents if the statistics differ, and the
        fallback path (:data:`_WAN22_LATENTS_MEAN` vs a checkpoint that carries its
        own) is exactly the case where that is easy to miss.
        """
        from .latent_cache import digest_text

        mean, std = self._latent_stats(device=torch.device("cpu"), dtype=torch.float64)
        return digest_text(
            ",".join(f"{float(value)!r}" for value in mean.reshape(-1).tolist())
            + "|"
            + ",".join(f"{float(value)!r}" for value in std.reshape(-1).tolist()),
            tag="latents.stats",
        )

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Invert :meth:`normalize_latents` before decoding."""
        mean, std = self._latent_stats(device=latents.device, dtype=latents.dtype)
        return latents * std + mean

    # 鈹€鈹€ geometry helpers 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def latent_num_frames(self, num_pixel_frames: int) -> int:
        """Latent frame count for ``num_pixel_frames``: causal, one latent per 4 frames."""
        if num_pixel_frames < 1:
            raise ValueError(f"num_pixel_frames must be >= 1, got {num_pixel_frames}.")
        return 1 + (int(num_pixel_frames) - 1) // self.temporal_scale


    # 鈹€鈹€ encode / decode 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """``[B, 3, T, H, W]`` in ``[-1, 1]`` 鈫?unnormalized latents ``[B, C, T_lat, H // 16, W // 16]``."""
        if video.dim() != 5 or video.shape[1] != 3:
            raise ValueError(f"Wan2.2 VAE expects video shaped [B, 3, T, H, W]; got {tuple(video.shape)}.")
        if self.temporal_chunk_frames and video.shape[2] > self.temporal_chunk_frames:
            latents = self._encode_temporal_chunks(video)
        else:
            latents = self._encode_once(video)
        expected = self.latent_num_frames(video.shape[2])
        if latents.shape[2] != expected:
            raise RuntimeError(
                f"Wan2.2 VAE produced {latents.shape[2]} latent frames for {video.shape[2]} pixel frames; "
                f"expected {expected}. The latent temporal grid is inconsistent with scale_factor_temporal="
                f"{self.temporal_scale}."
            )
        return latents

    def _encode_once(self, video: torch.Tensor) -> torch.Tensor:
        output = self.vae.encode(video)
        latent_dist = getattr(output, "latent_dist", None)
        if latent_dist is not None:
            # ``mode()`` is the posterior mean: the reference trains against the
            # deterministic mean, so drawing a sample here would add noise the
            # target does not account for.
            return latent_dist.mode()
        if isinstance(output, torch.Tensor):
            return output
        raise RuntimeError(
            f"{self.component_name}.encode returned {type(output).__name__} with neither 'latent_dist' nor a "
            "tensor; the diffusers Wan VAE API is not what this wrapper expects."
        )

    def _encode_temporal_chunks(self, video: torch.Tensor) -> torch.Tensor:
        """Encode long clips in temporal windows to bound peak encoder memory.

        Off by default. Each window is prefixed with a few already-encoded frames
        so the causal feature cache is warm; the warmed-up latents are dropped.
        The Wan VAE's temporal receptive field is a handful of frames, so with
        ``temporal_chunk_warmup`` >= that window the stitched result matches a
        full-clip encode 鈥?the assembled length is asserted against
        :meth:`latent_num_frames` so a mismatch cannot pass silently.
        """
        chunk = self.temporal_chunk_frames
        stride = self.temporal_scale
        total = video.shape[2]
        pieces = []
        start = 0
        while start < total:
            stop = min(start + chunk, total)
            warmup = min(start, self.temporal_chunk_warmup)
            window = video[:, :, start - warmup : stop]
            latents = self._encode_once(window)
            pieces.append(latents[:, :, warmup // stride :])
            start = stop
        return torch.cat(pieces, dim=2)

    def encode_latents(self, video: torch.Tensor) -> torch.Tensor:
        """Encode and normalize 鈥?the exact tensor the diffusion transformer consumes."""
        return self.normalize_latents(self.encode(video))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Normalized latents 鈫?``[B, 3, T, H, W]`` in ``[-1, 1]``."""
        if latents.dim() != 5:
            raise ValueError(f"Wan2.2 VAE expects latents shaped [B, C, T, H, W]; got {tuple(latents.shape)}.")
        output = self.vae.decode(self.denormalize_latents(latents))
        return output.sample if hasattr(output, "sample") else output

    # 鈹€鈹€ memory knobs 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

