"""Register the Cosmos3 transformer config and model.

Both registrations answer to ``Cosmos3MoTTransformer``, the ``_class_name`` /
``model_type`` pair in ``configs/model_configs/cosmos3/cosmos3_nano.json``. The
config's ``model_type`` is that same string, so
``veomni.models.registry.get_model_config`` swaps the AutoConfig it loaded for this
class, and ``get_model_class`` resolves the modeling entry from the same key.

The companion condition keys (``Cosmos3ConditionModelConfig`` /
``Cosmos3ConditionModel``) are registered by ``cosmos3_condition/__init__.py`` and
are deliberately *not* repeated here: the registry rejects duplicates, and the
transformer config already points at them through its ``condition_model_type``
class attribute.
"""

from veomni.models.registry import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("Cosmos3MoTTransformer")
def register_cosmos3_transformer_config():
    from .configuration_cosmos3_transformer import Cosmos3MoTTransformerConfig

    return Cosmos3MoTTransformerConfig


@MODELING_REGISTRY.register("Cosmos3MoTTransformer")
def register_cosmos3_transformer_modeling(_architecture: str | None = None):
    from .modeling_cosmos3_transformer import Cosmos3MoTTransformer, apply_cosmos3_transformer_patch

    apply_cosmos3_transformer_patch()
    return Cosmos3MoTTransformer
