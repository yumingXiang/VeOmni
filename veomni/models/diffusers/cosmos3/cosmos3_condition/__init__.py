"""Register the Cosmos3 condition config and model.

``DiTTrainer`` resolves the frozen companion through the transformer config's
``condition_model_type`` and looks that string up in ``MODELING_REGISTRY``, while the
config side is resolved through ``MODEL_CONFIG_REGISTRY``. Cosmos3 therefore registers
one key -- ``Cosmos3ConditionModel``, the model class name, the same choice
``minimax_h3`` makes -- in both registries.
"""

from veomni.models.registry import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("Cosmos3ConditionModel")
def register_cosmos3_condition_config():
    from .configuration_cosmos3_condition import Cosmos3ConditionModelConfig

    return Cosmos3ConditionModelConfig


@MODELING_REGISTRY.register("Cosmos3ConditionModel")
def register_cosmos3_condition_modeling(_architecture: str | None = None):
    from .modeling_cosmos3_condition import Cosmos3ConditionModel

    return Cosmos3ConditionModel
