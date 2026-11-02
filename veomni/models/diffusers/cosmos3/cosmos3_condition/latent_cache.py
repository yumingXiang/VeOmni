"""Content-addressed identity for cached Cosmos3 VAE latents.

``offline_embedding`` runs the frozen Wan2.2 VAE once and ``offline_training`` replays
the result. Addressing that cache by *position* (the plain ``OfflineEmbeddingSaver``
appends rows in producer order) is safe only if nothing between fill and replay
changes: the data, the VAE, the dtype or the code. If any of them does, the file layout
stays identical while the meaning of row 17 changed, and the run trains on stale
latents with no error. So a row is keyed by a SHA-256 over everything that determines
the latent: the cache schema, the prompt text, the decoded pixels, the geometry fed to
the encoder, the VAE dtype and weight digest, the latent statistics, the temporal
chunking, the source of the VAE wrapper / this module / the condition model / diffusers'
``autoencoder_kl_wan``, the chat template with ``max_sequence_length``, and the
resolution bucket.

Two choices worth knowing:

* **A clip's identity is its decoded pixels, not its path.** ``dit_online`` decodes in
  the dataloader and keeps only frames, so no path survives to the trainer; hashing the
  pixels is also strictly stronger (a re-encoded or truncated clip with the same name
  gets a different key) and costs ~20-90 ms against the ~250 ms encode it replaces.
* **The resolution bucket is included although it only selects the sigma warp.** A
  spurious miss costs a re-encode; a spurious hit costs a silently wrong loss.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch


__all__ = [
    "LATENT_CACHE_SCHEMA",
    "LatentCacheMismatch",
    "LatentRecipe",
    "checkpoint_digest",
    "code_digest",
    "digest_tensor",
    "digest_text",
    "latent_cache_key",
    "latent_stats_digest",
    "read_manifest",
    "tokenizer_digest",
    "verify_manifest",
    "write_manifest",
]


# Bump when the *encoding* of a cached entry changes (a new field in the payload,
# diffusers changing what ``encode`` returns, ...). Any bump invalidates every
# existing cache, which is the point.
LATENT_CACHE_SCHEMA = "cosmos3-latent-v1"

# Files whose source is part of the recipe. A cached latent produced by different
# arithmetic is a different latent, even if every input is identical.
_CODE_FILES = (
    "wan22_vae.py",
    "modeling_cosmos3_condition.py",
    "latent_cache.py",
)

# The reference normalizes with *these* values when the checkpoint's VAE config
# does not carry them (see ``wan22_vae._WAN22_LATENTS_MEAN``). Re-declared here as
# the fallback the digest has to be able to name, without importing the module
# (this module is imported by the config-only / CPU paths too).
_STATS_FALLBACK_KEYS = ("latents_mean", "latents_std")

_READ_CHUNK = 8 << 20


class LatentCacheMismatch(RuntimeError):
    """The on-disk latent cache was produced by a different recipe."""


# ── hashing primitives ──────────────────────────────────────────────────────


def _tagged(tag: str, payload: bytes) -> bytes:
    """Length-prefix every field, so ``("ab", "c")`` and ``("a", "bc")`` differ."""
    return tag.encode("utf-8") + b":" + str(len(payload)).encode("ascii") + b":" + payload


def _hexdigest(parts: Sequence[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def digest_text(text: str, *, tag: str = "text") -> str:
    """Digest of a string (prompt, chat template, dtype name, ...)."""
    return _hexdigest([_tagged(tag, str(text).encode("utf-8"))])


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Raw bytes of a tensor's elements, dtype-agnostic (bf16 included).

    One memcpy through the flat uint8 view. The obvious
    ``bytes(untyped_storage()[offset:][:nbytes])`` is a trap: slicing a storage in
    ``torch/storage.py`` rebuilds it element by element in Python, so a 23 MB clip
    becomes tens of millions of iterations and ``offline_embedding`` spins forever
    in its first batch -- no error, the NPU idle and one core at 100%.
    """
    flat = tensor.detach().to("cpu").contiguous().reshape(-1)
    if flat.numel() == 0:
        return b""
    return flat.view(torch.uint8).numpy().tobytes()


def digest_tensor(tensor: torch.Tensor, *, tag: str = "tensor") -> str:
    """Digest of a tensor's dtype, shape and contents.

    The shape is tagged separately from the bytes because two tensors can share a
    byte buffer and still be different samples (a ``[9, 3, 32, 32]`` clip versus a
    ``[3, 9, 32, 32]`` one with the same values).
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"digest_tensor expects a torch.Tensor, got {type(tensor).__name__}")
    shape = b",".join(str(int(size)).encode("ascii") for size in tensor.shape)
    return _hexdigest(
        [
            _tagged(tag, str(tensor.dtype).encode("utf-8")),
            _tagged(f"{tag}.shape", shape),
            _tagged(f"{tag}.numel", str(tensor.numel()).encode("ascii")),
            _tagged(f"{tag}.data", _tensor_bytes(tensor)),
        ]
    )


def _digest_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_READ_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_digest(directory: Optional[str]) -> str:
    """Content digest of a component directory (the VAE's checkpoint files).

    Full-file hashes, not sizes or mtimes: a re-exported checkpoint can keep both
    while changing the weights, and a cache keyed on ``(name, size)`` would then
    hand back latents from the *previous* weights. The Wan2.2 VAE is ~0.5-1 GB, so
    this costs ~0.5 s once per run -- against a cache fill that runs the encoder
    over the whole dataset.

    A missing directory yields the documented ``"absent"`` marker rather than an
    error: ``offline_training`` builds the condition model with ``meta_init`` and
    a CPU test may point at a path that does not exist, and in both cases there is
    nothing to fingerprint (and nothing is encoded).
    """
    if not directory or not os.path.isdir(directory):
        return "absent"
    entries: list[bytes] = []
    for root, dirnames, filenames in os.walk(directory):
        dirnames.sort()
        for filename in sorted(filenames):
            full = os.path.join(root, filename)
            relative = os.path.relpath(full, directory).replace(os.sep, "/")
            entries.append(
                _tagged(
                    f"file:{relative}",
                    str(os.path.getsize(full)).encode("ascii") + b"," + _digest_file(full).encode("ascii"),
                )
            )
    if not entries:
        return "empty"
    return _hexdigest(entries)


def code_digest() -> str:
    """Digest of the source files whose arithmetic produces the latent.

    ``wan22_vae.py`` (``mode()`` + normalization), this module (the key itself),
    ``modeling_cosmos3_condition.py`` (frame preparation and the latent grid) and,
    when importable, diffusers' ``autoencoder_kl_wan`` (the convolutions).
    """
    module_dir = os.path.dirname(os.path.abspath(__file__))
    parts: list[bytes] = []
    for filename in _CODE_FILES:
        path = os.path.join(module_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"latent cache code digest cannot find {path}; the cache key would silently stop covering "
                "the encoder's own implementation"
            )
        parts.append(_tagged(f"code:{filename}", _digest_file(path).encode("ascii")))
    try:  # best effort: diffusers is optional for config-only / CPU tooling
        from diffusers.models.autoencoders import autoencoder_kl_wan

        parts.append(
            _tagged("code:diffusers.autoencoder_kl_wan", _digest_file(autoencoder_kl_wan.__file__).encode("ascii"))
        )
    except Exception:  # pragma: no cover - depends on the installed diffusers
        parts.append(_tagged("code:diffusers.autoencoder_kl_wan", b"unavailable"))
    return _hexdigest(parts)


def tokenizer_digest(tokenizer: Any, *, max_sequence_length: Optional[int] = None) -> str:
    """Digest of the prompt template, which decides ``input_ids``.

    The chat template is a string (or a dict of named templates) on the tokenizer
    object; a run that switches template -- or the checkpoint that ships a new one
    -- produces different prompt tokens for the same caption. ``add_generation_prompt``
    / ``add_vision_id`` are hard-coded in ``_tokenize_prompt`` and covered by the
    code digest.
    """
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, dict):
        rendered = json.dumps(template, sort_keys=True, ensure_ascii=False)
    elif template is None:
        rendered = "<no-chat-template>"
    else:
        rendered = str(template)
    return _hexdigest(
        [
            _tagged("tokenizer.template", rendered.encode("utf-8")),
            _tagged("tokenizer.max_sequence_length", str(max_sequence_length).encode("utf-8")),
        ]
    )


def latent_stats_digest(vae: Any) -> str:
    """Digest of the ``latents_mean`` / ``latents_std`` the normalization uses.

    Read through ``Wan22VAE``'s own accessor when available, so a checkpoint that
    carries its statistics and one that falls back to Wan2.2's published values
    cannot collide when the two agree numerically but differ in provenance -- the
    values are what the latent is scaled by.
    """
    fingerprint = getattr(vae, "latent_stats_fingerprint", None)
    if callable(fingerprint):
        return fingerprint()
    config = getattr(vae, "config", None)
    values = []
    for key in _STATS_FALLBACK_KEYS:
        value = getattr(config, key, None)
        values.append("none" if value is None else ",".join(f"{float(item)!r}" for item in value))
    return _hexdigest(
        [_tagged(f"latents.{key}", value.encode("utf-8")) for key, value in zip(_STATS_FALLBACK_KEYS, values)]
    )


# ── the run-level half of the key ───────────────────────────────────────────


@dataclass(frozen=True)
class LatentRecipe:
    """Everything about the *encoding recipe* that must not change between fills.

    The other half of a cache key is per sample (clip content + prompt + geometry).
    Splitting them this way keeps the expensive digest (the checkpoint files) out
    of the per-sample path: it is computed once per run.
    """

    vae_dtype: str
    vae_weights: str
    latents_stats: str
    temporal_chunk_frames: int
    temporal_chunk_warmup: int
    code: str
    tokenizer: str
    latent_patch_size: int
    spatial_scale: int
    temporal_scale: int
    resolution: str
    schema: str = LATENT_CACHE_SCHEMA

    def digest(self) -> str:
        return _hexdigest(
            [
                _tagged("schema", self.schema.encode("utf-8")),
                _tagged("vae_dtype", self.vae_dtype.encode("utf-8")),
                _tagged("vae_weights", self.vae_weights.encode("utf-8")),
                _tagged("latents_stats", self.latents_stats.encode("utf-8")),
                _tagged("temporal_chunk_frames", str(self.temporal_chunk_frames).encode("ascii")),
                _tagged("temporal_chunk_warmup", str(self.temporal_chunk_warmup).encode("ascii")),
                _tagged("code", self.code.encode("utf-8")),
                _tagged("tokenizer", self.tokenizer.encode("utf-8")),
                _tagged("latent_patch_size", str(self.latent_patch_size).encode("ascii")),
                _tagged("spatial_scale", str(self.spatial_scale).encode("ascii")),
                _tagged("temporal_scale", str(self.temporal_scale).encode("ascii")),
                _tagged("resolution", self.resolution.encode("utf-8")),
            ]
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "vae_dtype": self.vae_dtype,
            "vae_weights": self.vae_weights,
            "latents_stats": self.latents_stats,
            "temporal_chunk_frames": self.temporal_chunk_frames,
            "temporal_chunk_warmup": self.temporal_chunk_warmup,
            "code": self.code,
            "tokenizer": self.tokenizer,
            "latent_patch_size": self.latent_patch_size,
            "spatial_scale": self.spatial_scale,
            "temporal_scale": self.temporal_scale,
            "resolution": self.resolution,
            "digest": self.digest(),
        }


def latent_cache_key(
    recipe: LatentRecipe,
    *,
    prompt: str,
    frames: torch.Tensor,
    fed_geometry: Sequence[int],
) -> str:
    """The content-addressed key of one sample's latent.

    Args:
        recipe: The run-level encoding recipe (see :class:`LatentRecipe`).
        frames: The decoded clip the transform produced (``[T, C, H, W]`` or
            ``[C, T, H, W]``); its dtype, shape and bytes are the clip identity.
        fed_geometry: ``(frames, channels, height, width)`` of the tensor that is
            actually handed to the VAE, i.e. *after* the ``16 x latent_patch_size``
            grid rounding. It is separate from ``frames`` because the rounding is
            what the encoder sees, and a change to the grid (a different
            ``latent_patch_size``) changes the latent without changing the clip.

    Returns:
        A 64-character hex digest. Two samples with the same key are guaranteed to
        produce the same latent; two samples with different keys *may* still
        (identical clips, different prompts -- the prompt is part of the payload
        but not of the latent), which only costs a duplicate cache entry.
    """
    geometry = tuple(int(value) for value in fed_geometry)
    if len(geometry) != 4:
        raise ValueError(f"fed_geometry must be (frames, channels, height, width), got {fed_geometry!r}")
    return _hexdigest(
        [
            _tagged("recipe", recipe.digest().encode("ascii")),
            _tagged("prompt", prompt.encode("utf-8")),
            _tagged("frames", digest_tensor(frames, tag="clip").encode("ascii")),
            _tagged("fed_geometry", b",".join(str(value).encode("ascii") for value in geometry)),
        ]
    )


# ── the on-disk manifest ────────────────────────────────────────────────────

MANIFEST_FILENAME = "latent_cache_manifest.json"


def write_manifest(directory: str, recipe: LatentRecipe, **extra: Any) -> str:
    """Record the recipe a cache directory was filled under.

    Written *before* the first shard so an interrupted fill still declares what it
    was producing; ``samples`` is refreshed by :meth:`OfflineEmbeddingSaver.save_last`.

    Written through a temporary file and ``os.replace``: every rank of a
    ``offline_embedding`` run writes this manifest at startup while every rank also
    *reads* it through :func:`verify_manifest`, so an in-place rewrite lets a reader
    observe a half-written file and fail with ``JSONDecodeError`` or a phantom
    :class:`LatentCacheMismatch`.
    """
    payload = {"recipe": recipe.as_dict(), **extra}
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, MANIFEST_FILENAME)
    temporary = f"{path}.{os.getpid()}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)
    return path


def read_manifest(directory: str) -> Optional[dict[str, Any]]:
    """The manifest of a cache directory, or ``None`` when it has none."""
    path = os.path.join(directory or "", MANIFEST_FILENAME)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def verify_manifest(directory: str, recipe: LatentRecipe) -> Optional[dict[str, Any]]:
    """Refuse a cache directory that was filled under a different recipe.

    This is the guard that closes the silent-mismatch hole at *directory* level:
    the training-side transform replays rows by position and cannot re-derive a
    per-sample key without re-decoding the clip, so the cheapest honest check is
    that the whole directory was produced by this exact recipe.

    Returns:
        The manifest that was compared (``None`` when the directory has none).

    Raises:
        LatentCacheMismatch: a manifest exists and its recipe digest differs.
    """
    manifest = read_manifest(directory)
    if manifest is None:
        return None
    recorded = manifest.get("recipe", {}).get("digest")
    current = recipe.digest()
    if recorded != current:
        differing = sorted(
            key
            for key, value in manifest.get("recipe", {}).items()
            if key != "digest" and getattr(recipe, key, "<missing>") != value
        )
        raise LatentCacheMismatch(
            f"The latent cache in {directory!r} was filled under a different encoding recipe "
            f"(recorded digest {recorded}, this run's {current}). Differing factors: {differing or 'unknown'}. "
            "Replaying it would train on latents this run cannot reproduce, silently. Re-run "
            "train.training_task='offline_embedding' into a fresh data.offline_embedding_save_dir, or point "
            "data.train_path at the matching cache."
        )
    return manifest
