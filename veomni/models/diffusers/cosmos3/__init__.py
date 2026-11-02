"""Cosmos3 (Nano) for VeOmni.

Why this package exists: Cosmos3 is a Mixture-of-Towers model -- one reasoner
tower and one diffusion tower share a joint attention and nothing else. It is
re-implemented here with VeOmni-native facilities (``VeomniOp`` dispatch,
``rms_norm`` / ``rope`` / ``attention`` registry rows) so the stock ``DiTTrainer``
and FSDP2 path apply unchanged; ``cosmos_framework`` is never imported.

Layout:

* ``cosmos3_core`` -- the numeric core (attention, decoder block, mRoPE,
  timestep embedder, rectified-flow loss). No ``transformers`` model classes, so
  it is exercisable on CPU.
* ``cosmos3_transformer`` / ``cosmos3_condition`` -- registry entries, configs and
  the checkpoint plumbing. Imported here for their registration side effects, the
  way ``minimax_h3`` and ``wan_t2v`` do: importing ``veomni.models.diffusers`` is
  what has to populate ``MODEL_CONFIG_REGISTRY`` / ``MODELING_REGISTRY``, because
  ``DiTTrainer`` resolves both halves of the model by key *before* it can import
  anything by name.
"""

from . import cosmos3_condition, cosmos3_core, cosmos3_transformer


__all__ = ["cosmos3_condition", "cosmos3_core", "cosmos3_transformer"]
