# SPDX-License-Identifier: Apache-2.0
"""Cosmos3-Nano model tests.

The load-bearing contracts of the port, and nothing else: a change that breaks one of
these trains a model that looks fine and is not. The other invariants this port relies
on (packed-vs-per-sample equivalence, the latent cache, the FLOPs breakdown, the recipe
files) are exercised by the training/eval runs described in the PR, not here.

Covered:

* the model is registered under the model_type its JSON declares;
* the generation tower's attention weights keep the checkpoint's add_* names,
  which is what makes checkpoint_key == model_key true;
* the frozen reasoner stays frozen and the generation tower stays trainable -- the
  rename from upstream's moe_gen to add_q_proj/add_k_proj/... is easy to
  get wrong in a way that silently trains the wrong parameters;
* the loss is a per-sample mean scaled by loss_scale, not a batch sum;
* the mRoPE time base is base_fps / fps, so a 5 fps clip does not train as 24 fps.
"""

from __future__ import annotations

import pytest
import torch

from veomni.models.diffusers.cosmos3 import cosmos3_condition, cosmos3_transformer  # noqa: F401  (registry side effects)
from veomni.models.diffusers.cosmos3.cosmos3_core import flow_matching_loss, latent_temporal_positions
from veomni.models.diffusers.cosmos3.cosmos3_transformer.configuration_cosmos3_transformer import (
    Cosmos3MoTTransformerConfig,
)
from veomni.models.diffusers.cosmos3.cosmos3_transformer.modeling_cosmos3_transformer import (
    TRAINABLE_NAME_KEYS,
    Cosmos3MoTTransformer,
)
from veomni.models.registry import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


MODEL_KEY = "Cosmos3MoTTransformer"

# Small enough that a 2-layer forward is milliseconds, wide enough that every
# tensor has a distinct shape (so a transposed index cannot pass by accident).
# ``mrope_section`` must sum to ``head_dim // 2`` = 16.

TINY_CONFIG = {
    "vocab_size": 128,
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 32,
    "intermediate_size": 128,
    "patch_latent_dim": 8,
    "latent_patch_size": 2,
    "mrope_section": (6, 5, 5),
    "num_experts": 0,
    "loss_scale": 1.0,
}

def _tiny_config(**overrides) -> Cosmos3MoTTransformerConfig:
    values = dict(TINY_CONFIG)
    values.update(overrides)
    return Cosmos3MoTTransformerConfig(**values)


def _tiny_model(**overrides) -> Cosmos3MoTTransformer:
    torch.manual_seed(0)
    model = Cosmos3MoTTransformer(_tiny_config(**overrides))
    model.eval()
    return model


def _columns() -> dict[str, list]:
    """One sample per list element, exactly as ``DiTDataCollator`` produces.

    The two samples differ in prompt length *and* latent geometry: that is the
    whole reason the DiT contract is ``dict[str, list]`` rather than a batch of
    stacked tensors.
    """
    torch.manual_seed(1234)
    vocab = TINY_CONFIG["vocab_size"]
    return {
        "input_ids": [torch.randint(0, vocab, (5,)), torch.randint(0, vocab, (3,))],
        "hidden_states": [torch.randn(2, 2, 4, 4), torch.randn(2, 3, 2, 2)],
        "timestep": [torch.tensor([500.0]), torch.tensor([125.0])],
        "training_target": [torch.randn(2, 2, 4, 4), torch.randn(2, 3, 2, 2)],
        "ar_attention_mask": [torch.ones(5, dtype=torch.long), torch.ones(3, dtype=torch.long)],
    }


def _is_trainable_name(name: str) -> bool:
    """The recipe's whitelist test: substring, not prefix (``norm_added_q``)."""
    return any(key in name for key in TRAINABLE_NAME_KEYS)


# ── the checkpoint's key set, spelled out once ────────────────────────────────

LAYER_KEY_SUFFIXES = frozenset(
    {
        "self_attn.to_q.weight",
        "self_attn.to_k.weight",
        "self_attn.to_v.weight",
        "self_attn.to_out.weight",
        "self_attn.norm_q.weight",
        "self_attn.norm_k.weight",
        "self_attn.add_q_proj.weight",
        "self_attn.add_k_proj.weight",
        "self_attn.add_v_proj.weight",
        "self_attn.to_add_out.weight",
        "self_attn.norm_added_q.weight",
        "self_attn.norm_added_k.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "mlp_moe_gen.gate_proj.weight",
        "mlp_moe_gen.up_proj.weight",
        "mlp_moe_gen.down_proj.weight",
        "input_layernorm.weight",
        "input_layernorm_moe_gen.weight",
        "post_attention_layernorm.weight",
        "post_attention_layernorm_moe_gen.weight",
    }
)

# The network-level tensors, same source. Nano's index has 12 of these next to the
# 792 layer tensors, i.e. 804 of its 814 keys are model parameters. ``lm_head`` is
# among them even though ``forward`` never runs it: the released checkpoint declares
# it (``tie_word_embeddings=false``), so the shell builds it to cover the key set.

TOP_LEVEL_KEYS = frozenset(
    {
        "embed_tokens.weight",
        "lm_head.weight",
        "norm.weight",
        "norm_moe_gen.weight",
        "proj_in.weight",
        "proj_in.bias",
        "proj_out.weight",
        "proj_out.bias",
        "time_embedder.linear_1.weight",
        "time_embedder.linear_1.bias",
        "time_embedder.linear_2.weight",
        "time_embedder.linear_2.bias",
    }
)

def _expected_parameter_names(num_hidden_layers: int) -> set[str]:
    """Every key a checkpoint of this architecture must contain."""
    names = set(TOP_LEVEL_KEYS)
    for index in range(num_hidden_layers):
        names |= {f"layers.{index}.{suffix}" for suffix in LAYER_KEY_SUFFIXES}
    return names

BRIDGE_FPS = 5.0


def test_config_and_registry_keys():
    """``to_dict`` carries ``_class_name`` and both registries resolve the key."""
    config = Cosmos3MoTTransformerConfig()
    as_dict = config.to_dict()
    assert as_dict["_class_name"] == MODEL_KEY
    assert as_dict["model_type"] == MODEL_KEY
    assert config.model_type == MODEL_KEY

    assert MODEL_KEY in MODEL_CONFIG_REGISTRY.valid_keys()
    assert MODELING_REGISTRY.valid_keys().count(MODEL_KEY) == 1
    assert MODEL_CONFIG_REGISTRY[MODEL_KEY]() is Cosmos3MoTTransformerConfig
    assert MODELING_REGISTRY[MODEL_KEY]() is Cosmos3MoTTransformer


def test_condition_model_type_points_at_the_registered_condition_model():
    """``DiTTrainer`` resolves the frozen companion through this class attribute."""
    assert Cosmos3MoTTransformerConfig.condition_model_type == "Cosmos3ConditionModel"
    # The condition side registers that one key in both registries.
    assert Cosmos3MoTTransformerConfig.condition_model_type in MODEL_CONFIG_REGISTRY.valid_keys()
    assert Cosmos3MoTTransformerConfig.condition_model_type in MODELING_REGISTRY.valid_keys()
    assert Cosmos3MoTTransformer(_tiny_config()).config.condition_model_type == (
        Cosmos3MoTTransformerConfig.condition_model_type
    )


def test_loss_is_a_sample_mean_not_a_batch_sum():
    """One sample per call, then a plain mean: no ``* n``, no ``/ n``.

    ``DiTTrainer.postforward`` divides the returned dict by the microbatch count,
    so a model that already multiplied by ``n`` would scale the gradient by the
    batch size. Duplicating a sample must therefore leave the loss unchanged (a
    mean is invariant to it, a sum is not).
    """
    model = _tiny_model()
    columns = _columns()
    single = {key: [values[0]] for key, values in columns.items()}
    duplicated = {key: [values[0], values[0]] for key, values in columns.items()}

    outputs_single = model(**single)
    outputs_duplicated = model(**duplicated)
    torch.testing.assert_close(outputs_duplicated.loss["dm_loss"], outputs_single.loss["dm_loss"])

    # And the single-sample value is exactly the core loss on that sample.
    direct = flow_matching_loss(
        outputs_single.predictions[0].float().unsqueeze(0),
        columns["training_target"][0].float().unsqueeze(0),
        loss_scale=TINY_CONFIG["loss_scale"],
    )
    torch.testing.assert_close(outputs_single.loss["dm_loss"], direct)


def test_freezing_reasoner_keeps_only_the_whitelisted_parameters():
    """``train_joint_reasoner=False`` reproduces the SFT recipe's ``keys_to_select``."""
    model = _tiny_model(train_joint_reasoner=False)
    assert model.config.train_joint_reasoner is False

    kept, frozen = [], []
    for name, parameter in model.named_parameters():
        if _is_trainable_name(name):
            kept.append(name)
            assert parameter.requires_grad, f"{name} should stay trainable"
        else:
            frozen.append(name)
            assert not parameter.requires_grad, f"{name} should be frozen"

    # Both sides must be non-empty, otherwise the assertion above proves nothing.
    assert kept
    assert frozen
    # The "training towers" are exactly the generation side: its attention, its
    # MLPs, its layer norms and the final generation norm.
    assert any(name.startswith("layers.0.self_attn.add_q_proj") for name in kept)
    assert any(name.startswith("layers.0.self_attn.norm_added_q") for name in kept)
    assert any(name.startswith("layers.0.self_attn.to_add_out") for name in kept)
    assert any(name.startswith("norm_moe_gen") for name in kept)
    assert any(name.startswith("proj_in") for name in kept)
    assert any(name.startswith("proj_out") for name in kept)
    assert any(name.startswith("embed_tokens") for name in frozen)
    assert any(name.startswith("layers.0.mlp.gate_proj") for name in frozen)
    # The reasoner tower's attention must stay frozen -- it is the checkpoint's
    # ``to_q`` / ``norm_q`` set, which no whitelist substring may match.
    assert any(name.startswith("layers.0.self_attn.to_q") for name in frozen)
    assert any(name.startswith("layers.0.self_attn.norm_q") for name in frozen)


def test_joint_training_releases_every_parameter():
    """``train_joint_reasoner=True`` trains both towers -- except ``lm_head``.

    ``lm_head`` is the one parameter outside the join/freeze decision: v1 has no AR
    loss, so it stays frozen even here (see
    ``test_lm_head_is_built_frozen_and_never_trainable``).
    """
    model = _tiny_model(train_joint_reasoner=True)
    assert model.config.train_joint_reasoner is True
    released = [parameter for name, parameter in model.named_parameters() if not name.startswith("lm_head.")]
    assert released and all(parameter.requires_grad for parameter in released)
    assert model.lm_head.weight.requires_grad is False


def test_weight_names_match_the_released_checkpoint_exactly():
    """The parameter set *equals* the checkpoint's, key for key -- no mapping layer.

    Set equality in both directions is the contract: a renamed module (``vae2llm``
    instead of ``proj_in``, ``q_proj_moe_gen`` instead of ``add_q_proj``) fails here
    rather than at a 45%-matched checkpoint load.
    """
    model = _tiny_model()
    names = {name for name, _ in model.named_parameters()}
    expected = _expected_parameter_names(TINY_CONFIG["num_hidden_layers"])
    assert names == expected, sorted(names ^ expected)
    assert "lm_head.weight" in names

    # Spellings that would silently break the load, checked by name so the failure
    # message says which rename regressed.
    stale = (
        "vae2llm",
        "llm2vae",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "self_attn.q_norm",
        "self_attn.k_norm",
        "self_attn.q_proj_moe_gen",
        "time_embedder.mlp",
        "qwen_model",
    )
    for gone in stale:
        assert not any(gone in name for name in names), f"{gone} must not exist: {sorted(names)}"


def test_lm_head_is_built_frozen_and_never_trainable():
    """``lm_head`` exists for key coverage, is frozen in both recipe modes.

    v1 has no AR loss, so the reasoner tower's LM head is never in a forward pass. It
    still has to be a real parameter -- the released index declares
    ``lm_head.weight`` -- but it must stay out of ``requires_grad`` even when the
    recipe sets ``train_joint_reasoner=True``, otherwise the optimizer would hold a
    parameter whose gradient is permanently ``None``. ``ALWAYS_FROZEN_NAME_PREFIXES``
    is what enforces that, so the invariant is asserted for both config values.
    """
    from veomni.models.diffusers.cosmos3.cosmos3_transformer.modeling_cosmos3_transformer import (
        ALWAYS_FROZEN_NAME_PREFIXES,
    )

    assert ALWAYS_FROZEN_NAME_PREFIXES == ("lm_head.",)
    assert not any("lm_head" in key for key in TRAINABLE_NAME_KEYS)

    for train_joint in (False, True):
        model = _tiny_model(train_joint_reasoner=train_joint)
        assert "lm_head.weight" in model.state_dict()
        assert model.lm_head.weight.shape == (TINY_CONFIG["vocab_size"], TINY_CONFIG["hidden_size"])
        assert model.lm_head.weight.requires_grad is False
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        assert "lm_head.weight" not in trainable
        assert not any(name.startswith("lm_head") for name in trainable)
        # And it is frozen even after the trainer re-applies the hook.
        model.freeze_model()
        assert model.lm_head.weight.requires_grad is False
        if train_joint:
            # The rest of the network *is* released, so the assertion above is specific.
            assert "embed_tokens.weight" not in {n for n, p in model.named_parameters() if not p.requires_grad}


def test_latent_temporal_step_is_base_fps_over_fps():
    """Adjacent latent frames advance by 4.8 for a 5 fps clip at base_fps=24."""
    positions = latent_temporal_positions(6, fps=BRIDGE_FPS)
    steps = (positions[1:] - positions[:-1]).tolist()
    assert steps == pytest.approx([4.8] * 5, abs=1e-6)
    assert positions[0].item() == pytest.approx(0.0)
    # Not the pre-fix layout, which stepped by exactly 1.0.
    assert steps[0] != pytest.approx(1.0)
