"""Content-addressed identity of cached Cosmos3 latents (``offline_embedding`` / ``offline_training``).

A sample's key hashes the decoded clip, the prompt and the geometry fed to the VAE,
plus the run-level :class:`LatentRecipe` (VAE dtype, weights and statistics,
tokenizer template, grid). The trainer records the recipe next to the cache and
refuses to replay a cache filled under a different one.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from typing import Any, Optional, Sequence

import torch


# Bump whenever the code that turns a clip into a cached row changes its output.
LATENT_CACHE_SCHEMA = "cosmos3-latent-v2"


def _sha256(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        # Length-prefixed, so ("ab", "c") and ("a", "bc") differ.
        digest.update(str(len(part)).encode("ascii") + b":")
        digest.update(part)
    return digest.hexdigest()


def _digest_tensor(tensor: torch.Tensor) -> str:
    flat = tensor.detach().cpu().contiguous().reshape(-1)
    return _sha256(
        str(tensor.dtype).encode(), str(tuple(tensor.shape)).encode(), memoryview(flat.view(torch.uint8).numpy())
    )


def checkpoint_digest(directory: Optional[str]) -> str:
    """Full-content digest of a checkpoint directory (``"absent"`` when it does not exist)."""
    if not directory or not os.path.isdir(directory):
        return "absent"
    parts = []
    for root, dirnames, filenames in os.walk(directory):
        dirnames.sort()
        for filename in sorted(filenames):
            path = os.path.join(root, filename)
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(8 << 20), b""):
                    digest.update(chunk)
            parts.append(f"{os.path.relpath(path, directory)}:{digest.hexdigest()}".encode())
    return _sha256(*parts)


def tokenizer_digest(tokenizer: Any, max_sequence_length: int) -> str:
    template = getattr(tokenizer, "chat_template", None)
    rendered = (
        json.dumps(template, sort_keys=True, ensure_ascii=False) if isinstance(template, dict) else str(template)
    )
    return _sha256(rendered.encode(), str(max_sequence_length).encode())


@dataclasses.dataclass(frozen=True)
class LatentRecipe:
    """The run-level half of every cache key, computed once per run."""

    vae_dtype: str
    vae_weights: str
    latents_stats: str
    tokenizer: str
    latent_patch_size: int
    spatial_scale: int
    temporal_scale: int
    resolution: str
    schema: str = LATENT_CACHE_SCHEMA

    def digest(self) -> str:
        return _sha256(json.dumps(dataclasses.asdict(self), sort_keys=True).encode())

    def as_dict(self) -> dict[str, Any]:
        return {**dataclasses.asdict(self), "digest": self.digest()}


def latent_cache_key(recipe: LatentRecipe, *, prompt: str, frames: torch.Tensor, fed_geometry: Sequence[int]) -> str:
    """Key of one sample: recipe, prompt, decoded clip and the ``(T, C, H, W)`` actually fed to the VAE."""
    return _sha256(
        recipe.digest().encode(),
        prompt.encode("utf-8"),
        _digest_tensor(frames).encode(),
        ",".join(str(int(value)) for value in fed_geometry).encode(),
    )
