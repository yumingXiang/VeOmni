# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from transformers import PretrainedConfig

from ..lora.config import LORA_MODULES_BY_MODEL_TYPE, VeOmniLoraConfig
from . import logging
from .device import get_device_name


logger = logging.get_logger(__name__)


# ``config.model_type`` values of the Cosmos3 mixture-of-towers (MoT) DiT family.
# One checkpoint type stands for *two* towers -- an "und" (understanding / reasoner)
# tower and a "gen" (diffusion / generation) tower -- that share one attention
# softmax and nothing else, which is why the dense-LM formulas in this file do not
# describe it. Exported so the DiT trainer can tell whether its own micro-batch
# layout is the one these estimators price.
COSMOS3_MOT_MODEL_TYPES = frozenset({"Cosmos3MoTTransformer"})


def get_device_flops(unit="T"):
    """Peak dense BF16 FLOPS of the current device.

    All entries follow one convention: SM count x per-SM dense BF16 FLOPs per clock
    (2048 Ampere, 4096 Hopper, 8192 Blackwell) x tensor-core boost clock. NVIDIA
    publishes that boost clock up to Hopper, so the datasheet numbers already match;
    for Blackwell it is the clock the deployed part actually runs at (`nvidia-smi
    --query-gpu=clocks.max.sm`), which is higher than the clock behind the datasheet.
    H20 and 910B are throughput-capped SKUs the formula does not describe.

    Ascend entries are the vendor's published dense BF16 cube peak per chip (nothing
    in this repository derives them): 910B/910_93 is the 354 TFLOPS SKU already used
    by every NPU MFU number here, and 950DT / A5 is 432 TFLOPS.
    """

    def unit_convert(number, level):
        units = ["B", "K", "M", "G", "T", "P"]
        if number <= 0:
            return number
        ptr = 0
        while ptr < len(units) and units[ptr] != level:
            number /= 1000
            ptr += 1
        return number

    device_name = get_device_name()
    flops = float("inf")  # INF flops for unkown gpu type
    if "H100" in device_name or "H800" in device_name or "H200" in device_name:
        flops = 989e12
    elif "A100" in device_name or "A800" in device_name:
        flops = 312e12
    elif "L40" in device_name:
        flops = 181.05e12
    elif "L20" in device_name:
        flops = 119.5e12
    elif "H20" in device_name:
        flops = 148e12
    elif "950DT" in device_name or "910_95" in device_name or "Ascend950" in device_name:
        # Ascend 950DT / Atlas 950 SuperPod node chip (A5 generation).
        # Source: user-provided hardware specification -- BF16 cube peak, dense, per
        # chip. 432 TFLOPS x 8 chips is the 3,456 TFLOPS denominator of the
        # Cosmos3 MFU numbers. Checked before the 910B/910_93 rows below: a 950DT
        # reports itself as `Ascend910_95...`, which contains neither tag.
        flops = 432e12
    elif "910B" in device_name or "910_93" in device_name:
        flops = 354e12
    elif "GB200" in device_name:
        flops = 2565e12
    elif "B200" in device_name:
        flops = 2382e12
    elif "GB300" in device_name:
        flops = 2500e12
    elif "B300" in device_name:
        flops = 2250e12
    flops_unit = unit_convert(flops, unit)
    return flops_unit


class VeomniFlopsCounter:
    """
    Used to count mfu during training loop

    Example:
        flops_counter = VeomniFlopsCounter(config)
        flops_achieved, flops_promised = flops_counter.estimate_flops(batch_seqlens, delta_time)

    """

    def __init__(self, config: PretrainedConfig):
        self.estimate_func = {
            "qwen2_vl": self._estimate_qwen2_vl_flops,
            # the only difference between Qwen2 and Qwen2.5 for counting flops is the window attention
            # used in the ViT for Qwen2.5VL which is considered in the _estimate_qwen2_vl_flops function.
            "qwen2_5_vl": self._estimate_qwen2_vl_flops,
            # qwen3_vl's vit uses full self attention while qwen2-vl/qwen2.5-vl uses window attention.
            "qwen3_vl": self._estimate_qwen3_vl_flops,
            "qwen3_vl_moe": self._estimate_qwen3_vl_moe_flops,
            "deepseek_v3": self._estimate_deepseek_v3_flops,
            "deepseek_v4": self._estimate_deepseek_v4_flops,
            "qwen3_moe": self._estimate_qwen3_moe_flops,
            "llama": self._estimate_llama_flops,
            "qwen2": self._estimate_qwen2_flops,
            # qwen3_next
            "qwen3_next": self._estimate_qwen3_next_flops,
            # qwen3 reused _estimate_qwen2_flops func because the only model structure diff between qwen2 dense and qwen3 dense is that
            # qwen3 has additional RMSNorm layers for q and k.
            # RMSNorm layers have minimal impact at the MFU and can be ignored.
            "qwen3": self._estimate_qwen2_flops,
            "seed_oss": self._estimate_seed_flops,
            "qwen3_5": self._estimate_qwen3_5_family_flops,
            "qwen3_5_text": self._estimate_qwen3_5_family_flops,
            "qwen3_5_moe": self._estimate_qwen3_5_family_flops,
            "qwen3_5_moe_text": self._estimate_qwen3_5_family_flops,
            "qwen4_exp": self._estimate_qwen4_exp_flops,
            "gpt_oss": self._estimate_gpt_oss_flops,
            "gemma3": self._estimate_gemma3_flops,
            "gemma3_text": self._estimate_gemma3_flops,
        }
        # The Cosmos3 mixture-of-towers DiT: registered from the module constant so the
        # trainer's "do I have an estimator for this DiT?" check cannot drift from the
        # dispatch table below.
        self.estimate_func.update(dict.fromkeys(COSMOS3_MOT_MODEL_TYPES, self._estimate_cosmos3_mot_flops))

        self.config = config
        self._warned_lora_validation_errors = set()

    def _estimate_unknown_flops(self, tokens_sum, batch_seqlens, delta_time, **kwargs):
        return 0

    def has_estimator(self, model_type: str | None = None) -> bool:
        """Whether an estimator is registered for ``model_type`` (default: the config's own)."""
        return (self.config.model_type if model_type is None else model_type) in self.estimate_func

    def supports_config(self) -> bool:
        """Whether this counter can actually price *this* config's geometry.

        A family can be registered while a particular config is not described by its
        estimator: the estimator is expected to raise ``ValueError`` /
        ``NotImplementedError`` for a layout it does not model (e.g. a mixture-of-experts
        feed-forward in a family whose estimator only counts dense MLPs). Training
        metrics must never abort a run, so callers that report MFU as a *diagnostic*
        ask this once, at construction time, and skip the metric when it is False.
        """
        if not self.has_estimator():
            return False
        try:
            # Empty lengths and a unit step: every estimator in this file accepts them,
            # so this exercises the geometry checks without touching the device table.
            # ``AttributeError`` covers a config object that is missing a field the
            # estimator needs -- still "cannot price it", still not a crash.
            self.estimate_func[self.config.model_type](0, [], 1.0)
        except (AttributeError, ValueError, NotImplementedError, KeyError, TypeError):
            return False
        return True

    @staticmethod
    def _compute_lm_head_params(hidden_size, vocab_size):
        # nn.Embedding is a table lookup, so only the lm_head matmul contributes FLOPs.
        return vocab_size * hidden_size

    @staticmethod
    def _compute_linear_flops(
        base_params,
        tokens,
        lora_config=None,
        module_shapes=None,
        detached_without_targets=False,
    ):
        """Count linear-layer work for full fine-tuning or LoRA.

        Full fine-tuning performs forward, input-gradient, and weight-gradient
        matmuls (factor 6). A frozen LoRA base still needs its input gradient to
        reach earlier adapters (factor 4). A fully frozen component whose inputs
        need no gradients, such as an unadapted vision tower, is forward-only
        (factor 2).
        """
        if lora_config is None:
            return 6 * base_params * tokens

        module_shapes = module_shapes or {}
        target_modules = lora_config.target_modules or []
        matched_modules = [module for module in target_modules if module in module_shapes]
        lora_params = 0
        for module in matched_modules:
            shape_groups = module_shapes[module]
            if isinstance(shape_groups, tuple):
                shape_groups = (shape_groups,)
            lora_params += sum(
                lora_config.r * (in_features + out_features) * count
                for in_features, out_features, count in shape_groups
            )
        base_factor = 2 if detached_without_targets and not matched_modules else 4
        return (base_factor * base_params + 6 * lora_params) * tokens

    @staticmethod
    def _get_lora_validation_error(model_type, lora_config):
        if lora_config is None:
            return None
        if not isinstance(lora_config, VeOmniLoraConfig):
            return f"`lora_config` must be a VeOmniLoraConfig or None, got {type(lora_config).__name__}."

        supported_modules = LORA_MODULES_BY_MODEL_TYPE.get(model_type)
        if supported_modules is None:
            return (
                f"LoRA FLOPs estimation supports Qwen model types {sorted(LORA_MODULES_BY_MODEL_TYPE)}, "
                f"got {model_type!r}."
            )

        target_modules = lora_config.target_modules or []
        if isinstance(target_modules, str):
            return "LoRA FLOPs estimation does not support regex-string `target_modules`."
        if not isinstance(target_modules, list) or not all(isinstance(module, str) for module in target_modules):
            return "`lora_config.target_modules` must be a list of projection names or None."
        if len(target_modules) != len(set(target_modules)):
            return f"`lora_config.target_modules` must not contain duplicates, got {target_modules!r}."

        unsupported_modules = set(target_modules) - set(supported_modules)
        if unsupported_modules:
            return (
                f"Unsupported {model_type} LoRA modules: {sorted(unsupported_modules)}. "
                f"Supported modules: {list(supported_modules)}."
            )

        target_parameters = lora_config.target_parameters or []
        if not isinstance(target_parameters, list) or not all(
            isinstance(pattern, str) for pattern in target_parameters
        ):
            return "`lora_config.target_parameters` must be a list of parameter patterns or None."
        if target_parameters:
            supported_moe_types = {"qwen3_moe", "qwen3_vl_moe", "qwen3_next", "qwen3_5_moe", "qwen3_5_moe_text"}
            if model_type not in supported_moe_types:
                return f"`target_parameters` is not supported for non-MoE model type {model_type!r}."
            supported_suffixes = ("gate_up_proj", "down_proj")
            unsupported_patterns = [
                pattern for pattern in target_parameters if not pattern.endswith(supported_suffixes)
            ]
            if unsupported_patterns:
                return (
                    "LoRA FLOPs estimation only supports fused routed-expert target parameters ending in "
                    f"{supported_suffixes}, got {unsupported_patterns!r}."
                )
        return None

    @staticmethod
    def _compute_routed_moe_lora_flops(config, tokens, lora_config):
        """Count fused routed-expert adapters selected through ``target_parameters``."""
        if lora_config is None or not lora_config.target_parameters:
            return 0

        hidden_size = config.hidden_size
        intermediate_size = config.moe_intermediate_size
        experts_per_token = config.num_experts_per_tok
        num_hidden_layers = config.num_hidden_layers
        if (lora_config.moe_mode or "independent") == "shared":
            adapter_uses_per_layer = 2 + experts_per_token  # gate/up once, down per routed expert
        else:
            adapter_uses_per_layer = 3 * experts_per_token

        lora_params = lora_config.r * (hidden_size + intermediate_size) * adapter_uses_per_layer * num_hidden_layers
        return 6 * lora_params * tokens

    def _compute_qwen_dense_text_flops(
        self,
        config,
        tokens_sum,
        batch_seqlens,
        lora_config=None,
    ):
        """Compute raw text FLOPs shared by dense Qwen text and VL models."""
        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        num_hidden_layers = config.num_hidden_layers
        head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)
        q_size = num_attention_heads * head_dim
        kv_size = config.num_key_value_heads * head_dim

        mlp_N = hidden_size * config.intermediate_size * 3
        attn_linear_N = hidden_size * (2 * q_size + 2 * kv_size)
        lm_head_N = self._compute_lm_head_params(hidden_size, config.vocab_size)
        base_N = (mlp_N + attn_linear_N) * num_hidden_layers + lm_head_N
        module_shapes = {
            "q_proj": (hidden_size, q_size, num_hidden_layers),
            "k_proj": (hidden_size, kv_size, num_hidden_layers),
            "v_proj": (hidden_size, kv_size, num_hidden_layers),
            "o_proj": (q_size, hidden_size, num_hidden_layers),
            "gate_proj": (hidden_size, config.intermediate_size, num_hidden_layers),
            "up_proj": (hidden_size, config.intermediate_size, num_hidden_layers),
            "down_proj": (config.intermediate_size, hidden_size, num_hidden_layers),
        }
        linear_flops = self._compute_linear_flops(
            base_N,
            tokens_sum,
            lora_config=lora_config,
            module_shapes=module_shapes,
        )

        seqlen_square_sum = sum(seqlen * seqlen for seqlen in batch_seqlens)
        attention_core_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers
        return linear_flops + attention_core_flops

    def _compute_qwen3_moe_text_flops(
        self,
        config,
        tokens_sum,
        batch_seqlens,
        lora_config=None,
    ):
        """Compute raw text FLOPs shared by Qwen3 MoE text and VL models."""
        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        num_hidden_layers = config.num_hidden_layers
        head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)
        q_size = num_attention_heads * head_dim
        kv_size = config.num_key_value_heads * head_dim

        router_N = hidden_size * config.num_experts
        active_expert_N = hidden_size * config.moe_intermediate_size * config.num_experts_per_tok * 3
        attn_linear_N = hidden_size * (2 * q_size + 2 * kv_size)
        lm_head_N = self._compute_lm_head_params(hidden_size, config.vocab_size)
        base_N = (router_N + active_expert_N + attn_linear_N) * num_hidden_layers + lm_head_N
        module_shapes = {
            "q_proj": (hidden_size, q_size, num_hidden_layers),
            "k_proj": (hidden_size, kv_size, num_hidden_layers),
            "v_proj": (hidden_size, kv_size, num_hidden_layers),
            "o_proj": (q_size, hidden_size, num_hidden_layers),
        }
        linear_flops = self._compute_linear_flops(
            base_N,
            tokens_sum,
            lora_config=lora_config,
            module_shapes=module_shapes,
        )
        linear_flops += self._compute_routed_moe_lora_flops(config, tokens_sum, lora_config)

        seqlen_square_sum = sum(seqlen * seqlen for seqlen in batch_seqlens)
        attention_core_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers
        return linear_flops + attention_core_flops

    def _estimate_seed_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        intermediate_size = self.config.intermediate_size

        head_dim = hidden_size // num_attention_heads
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # non-attn per layer parm
        # llama use SwiGelu, gate, having up and down linear layer in mlp
        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        lm_head_N = self._compute_lm_head_params(hidden_size, vocab_size)
        # non-attn all_layer parm
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_deepseek_v3_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        moe_intermediate_size = self.config.moe_intermediate_size
        num_hidden_layers = self.config.num_hidden_layers
        first_k_dense_replace = self.config.first_k_dense_replace
        num_query_heads = self.config.num_attention_heads
        moe_num_expert = self.config.n_routed_experts
        moe_topk = self.config.num_experts_per_tok
        share_expert_num = self.config.n_shared_experts
        # non-attn per layer parm
        moe_gata_N = hidden_size * moe_num_expert
        # moe has fc1_1, fc1_2 and fc2 using SwiGLU in ExpertMlp layer & shared experts
        moe_expertmlp_N = hidden_size * moe_intermediate_size * (moe_topk + share_expert_num) * 3
        # MLA attn
        attn_linear_N = 0
        q_head_dim = self.config.qk_nope_head_dim + self.config.qk_rope_head_dim
        if self.config.q_lora_rank is None:
            attn_linear_N += hidden_size * num_query_heads * q_head_dim
        else:
            attn_linear_N += hidden_size * self.config.q_lora_rank
            attn_linear_N += num_query_heads * q_head_dim * self.config.q_lora_rank
        attn_linear_N += hidden_size * (self.config.kv_lora_rank + self.config.qk_rope_head_dim)
        attn_linear_N += (
            num_query_heads
            * (q_head_dim - self.config.qk_rope_head_dim + self.config.v_head_dim)
            * self.config.kv_lora_rank
        )
        attn_linear_N += num_query_heads * self.config.v_head_dim * hidden_size
        lm_head_N = self._compute_lm_head_params(hidden_size, vocab_size)
        # non-attn all_layer parm
        moe_N = (
            (moe_gata_N + moe_expertmlp_N + attn_linear_N) * (num_hidden_layers - first_k_dense_replace)
            + (hidden_size * self.config.intermediate_size * 3 + attn_linear_N) * first_k_dense_replace
            + lm_head_N
        )
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * moe_N * tokens_sum
        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen * num_hidden_layers
        attn_qkv_flops = 12 * seqlen_square_sum * q_head_dim * num_query_heads
        # all_layer & all_token fwd & bwk flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    @staticmethod
    def _compute_compressed_attention_score_sum(batch_seqlens, compress_rate, topk=None):
        if compress_rate <= 0:
            raise ValueError(f"Compression rate must be positive, got {compress_rate}.")
        if topk is not None and topk <= 0:
            raise ValueError(f"Compressed-attention top-k must be positive, got {topk}.")

        score_sum = 0
        for seqlen in batch_seqlens:
            if topk is None:
                quotient, remainder = divmod(seqlen, compress_rate)
                score_sum += compress_rate * quotient * (quotient - 1) // 2
                score_sum += quotient * (remainder + 1)
                continue

            uncapped_len = min(seqlen, topk * compress_rate - 1)
            quotient, remainder = divmod(uncapped_len, compress_rate)
            score_sum += compress_rate * quotient * (quotient - 1) // 2
            score_sum += quotient * (remainder + 1)
            score_sum += max(0, seqlen - uncapped_len) * topk

        return score_sum

    def _estimate_deepseek_v4_flops(self, tokens_sum, batch_seqlens, delta_time):
        """Estimate DeepSeek-V4 training FLOPs from its dominant matrix multiplications."""
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        moe_intermediate_size = self.config.moe_intermediate_size
        num_hidden_layers = self.config.num_hidden_layers
        num_attention_heads = self.config.num_attention_heads
        head_dim = self.config.head_dim
        q_lora_rank = self.config.q_lora_rank
        num_experts = self.config.n_routed_experts
        experts_per_token = self.config.num_experts_per_tok
        hc_mult = self.config.hc_mult
        o_groups = self.config.o_groups
        o_lora_rank = self.config.o_lora_rank
        layer_types = self.config.layer_types

        if len(layer_types) != num_hidden_layers:
            raise ValueError(
                f"DeepSeek-V4 attention layer count mismatch: {len(layer_types)} entries in `config.layer_types` "
                f"for {num_hidden_layers} hidden layers."
            )

        supported_layer_types = {
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        }
        unsupported_layer_types = set(layer_types) - supported_layer_types
        if unsupported_layer_types:
            raise ValueError(f"Unsupported DeepSeek-V4 attention layer types: {sorted(unsupported_layer_types)}.")

        num_csa_layers = sum(layer_type == "compressed_sparse_attention" for layer_type in layer_types)
        num_hca_layers = sum(layer_type == "heavily_compressed_attention" for layer_type in layer_types)

        # Shared-KV MQA projections plus the grouped low-rank output projection.
        attn_linear_N = hidden_size * q_lora_rank
        attn_linear_N += q_lora_rank * num_attention_heads * head_dim
        attn_linear_N += hidden_size * head_dim
        attn_linear_N += num_attention_heads * head_dim * o_lora_rank
        attn_linear_N += o_groups * o_lora_rank * hidden_size

        # Every layer computes router logits, top-k routed experts, and shared experts.
        n_shared_experts = getattr(self.config, "n_shared_experts", 1) or 0
        moe_router_N = hidden_size * num_experts
        moe_expert_N = hidden_size * moe_intermediate_size * (experts_per_token + n_shared_experts) * 3

        # Each decoder layer has two mHC mappings. The final HyperHead adds one more mapping.
        mhc_mapping_N = (2 + hc_mult) * hc_mult * (hc_mult * hidden_size)
        mhc_head_N = hc_mult * (hc_mult * hidden_size)

        hca_compressor_N = 2 * hidden_size * head_dim
        csa_compressor_N = 4 * hidden_size * head_dim
        indexer_N = 4 * hidden_size * self.config.index_head_dim
        indexer_N += q_lora_rank * self.config.index_n_heads * self.config.index_head_dim
        indexer_N += hidden_size * self.config.index_n_heads

        lm_head_N = self._compute_lm_head_params(hidden_size, vocab_size)
        linear_N = (attn_linear_N + moe_router_N + moe_expert_N + 2 * mhc_mapping_N) * num_hidden_layers
        linear_N += hca_compressor_N * num_hca_layers
        linear_N += (csa_compressor_N + indexer_N) * num_csa_layers
        linear_N += mhc_head_N + lm_head_N
        linear_flops = 6 * linear_N * tokens_sum

        # mHC applies two HxH residual-stream matrix multiplications per decoder layer.
        mhc_residual_flops = 12 * hc_mult * hc_mult * hidden_size * num_hidden_layers * tokens_sum

        # CSA/HCA layers keep a sliding-window branch and concatenate compressed KV, so
        # sliding attention scores are counted for every decoder layer.
        sliding_score_sum = self._compute_sliding_attention_score_sum(batch_seqlens, self.config.sliding_window)
        hca_score_sum = 0
        if num_hca_layers > 0:
            hca_score_sum = self._compute_compressed_attention_score_sum(
                batch_seqlens, self.config.compress_rates["heavily_compressed_attention"]
            )

        csa_index_score_sum = 0
        csa_attention_score_sum = 0
        if num_csa_layers > 0:
            csa_compress_rate = self.config.compress_rates["compressed_sparse_attention"]
            csa_index_score_sum = self._compute_compressed_attention_score_sum(batch_seqlens, csa_compress_rate)
            csa_attention_score_sum = self._compute_compressed_attention_score_sum(
                batch_seqlens, csa_compress_rate, topk=self.config.index_topk
            )

        attention_score_sum = sliding_score_sum * num_hidden_layers
        attention_score_sum += hca_score_sum * num_hca_layers
        attention_score_sum += csa_attention_score_sum * num_csa_layers
        attention_flops = 12 * attention_score_sum * head_dim * num_attention_heads

        # The Lightning Indexer only computes query-key scores; it has no value aggregation matmul.
        indexer_flops = (
            6 * csa_index_score_sum * self.config.index_head_dim * self.config.index_n_heads * num_csa_layers
        )

        flops_all_token = linear_flops + mhc_residual_flops + attention_flops + indexer_flops
        return flops_all_token * (1.0 / delta_time) / 1e12

    @staticmethod
    def _cosmos3_mot_flops_breakdown(config, und_seqlens, gen_seqlens):
        """Per-term training FLOPs of one Cosmos3 mixture-of-towers micro batch.

        What makes Cosmos3 different from every other model in this file, and why the
        dense-LM formula cannot be reused: per layer it runs *two* towers side by side
        (``MoTDecoderLayer``), each with its own norms, its own Q/K/V/O and its own
        feed-forward, and each tower's tokens only see their own tower's weights --
        but both passes are computed on one shared attention softmax, with two
        different shapes (``cosmos3_core/mot_attention.py::joint_attention``):

        * **reasoner pass** (``und``): causal, own queries against own keys/values, so
          ``Q = KV = und_len``.
        * **generation pass** (``gen``): every generation query against the *whole*
          sample -- ``Q = gen_len`` but ``KV = und_len + gen_len``. This pass is not
          square, so pricing it as ``gen_len`` (or as ``(und + gen)`` queries) is wrong
          in either direction.

        Coefficients, and where each one comes from (the same convention the rest of
        this file uses, so a Cosmos3 MFU is comparable to a Qwen/Llama one):

        * **Linear / MLP: 6 per (weight element x token).** A dense layer is
          ``2 * in * out`` FLOPs per token forward; a full fine-tune also computes the
          input gradient and the weight gradient, so forward + backward = 3x forward
          (``_compute_linear_flops``'s ``6 * base_params * tokens``, and every
          ``dense_N_flops`` in this file). Applied per tower, on *that tower's* token
          count: a reasoner token never enters the generation tower's projections and
          vice versa. The generation tower's K/V projections are ordinary per-token
          projections too -- their *output length* is ``kv_size`` wide, but the number
          of rows is ``gen_len``; the ``und_len + gen_len`` length shows up only in the
          attention core below, as a KV length.
        * **Attention: 12 per (query x key pair x head x head_dim).** ``Q@K^T`` and
          ``attn@V`` are ``2 * (2 * Q * KV * head_dim)`` FLOPs per head forward, and
          3x for forward + backward -> ``12 * Q * KV * head_dim * heads``; exactly the
          ``12 * seqlen_square_sum * head_dim * num_attention_heads`` term of
          ``_estimate_llama_flops`` (with ``seqlen_square_sum = Q * KV``). Only the two
          matmuls are counted: the softmax, the scaling, the masking and the mRoPE
          rotation are elementwise or reduction work, which no estimator here prices.
        * **Causal triangle not discounted.** The reasoner pass really only computes
          the lower triangle, but this file counts the full square for causal attention
          (``_estimate_llama_flops``, ``_estimate_qwen2_flops``), so Cosmos3 does too,
          and its und-attention term is an upper bound by ~2x on that one term. Keeping
          the convention is what makes the number comparable across model families.
        * **Adapters: 6 per weight element x token.** ``proj_in``
          (``patch_latent_dim -> hidden``) and ``proj_out`` (``hidden ->
          patch_latent_dim``) run on generation rows only, and ``time_embedder``'s
          two-layer MLP (``frequency_embedding_size -> hidden -> hidden``) runs once
          per *generation row*: ``add_time_conditioning`` is applied to the packed rows,
          with one sigma per row, so the embedder's cost scales with ``gen_len`` too.
        * **Excluded, deliberately.** ``embed_tokens`` is a table lookup (same reason
          ``_compute_lm_head_params`` ignores it); ``lm_head`` is never in the graph
          (v1 trains no autoregressive loss, see ``Cosmos3MoTTransformer``); RMSNorm,
          residual adds, the sinusoidal feature construction and ``patchify_latent``
          are elementwise / reshape work; the VAE and the condition model run under
          ``torch.no_grad()`` in ``DiTTrainer.forward_backward_step`` and are not part
          of the training FLOPs of the DiT.
        * **Frozen reasoner.** ``config.train_joint_reasoner=False`` (the SFT recipe)
          freezes the reasoner tower, so its backward never runs while its forward
          always does (the generation pass attends its K/V). The reasoner terms are
          therefore counted at forward-only weight (2 and 4 instead of 6 and 12), the
          same distinction upstream's FLOPs tool draws with ``freeze_und``
          (``cosmos_framework/tools/flops/omni_mot.py:480``). A frozen run and a joint
          run of the same geometry then report comparable MFU, and the difference
          between them is the extra work joint training really does.

        Args:
            config: A ``Cosmos3MoTTransformerConfig``-shaped object.
            und_seqlens: Prompt-token counts, one entry per attention segment (a
                packed micro batch holds several isolated samples, so per-sample
                lengths are required -- the caveat is that attention is quadratic).
            gen_seqlens: Latent patch-row counts, one entry per segment, in the same
                order. Entry ``i``'s generation pass attends ``gen_seqlens[i]``
                queries to ``und_seqlens[i] + gen_seqlens[i]`` keys/values.

        Returns:
            ``dict[str, float]`` of forward+backward FLOPs per term (not divided by
            time). The keys are the ones the tests assert on:
            ``und_attn``, ``gen_attn``, ``und_attn_proj``, ``gen_attn_proj``,
            ``und_mlp``, ``gen_mlp``, ``proj_in``, ``proj_out``, ``time_embedder``.
        """
        if len(gen_seqlens) != len(und_seqlens):
            raise ValueError(
                f"Cosmos3 MoT FLOPs need one generation length per reasoner length, got "
                f"{len(und_seqlens)} and {len(gen_seqlens)}."
            )
        num_experts = int(getattr(config, "num_experts", 0) or 0)
        if num_experts > 0:
            # The routed-expert branch of ``MotSparseMoeBlock`` needs the per-layer
            # schedule (``decoder_sparse_step`` / ``mlp_only_layers`` / ``gen_moe_top_k``)
            # and the shared-expert width. No released Cosmos3 checkpoint is sparse
            # (Nano's ``num_experts`` is 0), so refuse instead of guessing: a wrong MoE
            # term would be a silently wrong MFU. ``supports_config`` turns this into a
            # skipped metric rather than a crash.
            raise NotImplementedError(
                f"Cosmos3 MoT FLOPs estimation does not model a sparse-MoE feed-forward yet "
                f"(config has num_experts={num_experts})."
            )

        hidden_size = config.hidden_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = int(getattr(config, "head_dim", hidden_size // num_heads))
        intermediate_size = config.intermediate_size
        train_joint = bool(getattr(config, "train_joint_reasoner", False))
        query_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim
        # ``proj_in``/``proj_out`` width: one latent patch of ``latent_channel``
        # channels on a ``latent_patch_size`` grid (192 for Nano's 48 x 2 x 2).
        patch_size = int(getattr(config, "latent_patch_size", 1) or 1)
        patch_latent_dim = int(
            getattr(config, "patch_latent_dim", getattr(config, "latent_channel", 0) * patch_size**2)
        )
        # ``TimestepEmbedder(config.hidden_size)`` is built with the default frequency
        # width, which the config does not carry.
        frequency_size = int(getattr(config, "frequency_embedding_size", 256))

        und_tokens = sum(und_seqlens)
        gen_tokens = sum(gen_seqlens)

        # A frozen reasoner tower (`train_joint_reasoner=False`, the SFT recipe) runs
        # *forward only*: its K/V are consumed by the generation pass, so every layer
        # still executes, but no gradient reaches it (its projections produce
        # ``requires_grad=False`` tensors, so autograd stops there). Counting the
        # 3x-forward backward for it would over-report the step's work -- and MFU is a
        # tuning signal, so an over-count silently rewards freezing. Upstream accounts
        # for exactly this with the ``freeze_und`` flag of its own FLOPs tool
        # (``cosmos_framework/tools/flops/omni_mot.py:480``); here the switch is the
        # config field the model itself freezes by.
        # 6 per (weight element x token) = 3x forward -> 2 (forward only) when frozen.
        und_training_weight = 6.0 if train_joint else 2.0
        # 12 per (query x key x head x head_dim) = forward QK^T + attn@V (=4) times 3.
        und_attention_weight = 12.0 if train_joint else 4.0

        # One QKVO set per tower per layer: q (h -> query_size), k and v
        # (h -> kv_size), o (query_size -> h).
        und_projection_params = hidden_size * (query_size + kv_size + kv_size + query_size)
        gen_projection_params = und_projection_params  # same geometry, separate weights
        # One SwiGLU feed-forward per tower per layer: gate + up (h -> inter), down
        # (inter -> h). ``MotMLP``, used by both towers in a dense (non-MoE) block.
        mlp_params = hidden_size * intermediate_size * 3

        und_attn = (
            und_attention_weight
            * num_layers
            * head_dim
            * num_heads
            * sum(length * length for length in und_seqlens)
        )
        # The generation pass is not square: Q = gen_len, KV = und_len + gen_len.
        gen_attn = (
            12
            * num_layers
            * head_dim
            * num_heads
            * sum(gen * (und + gen) for und, gen in zip(und_seqlens, gen_seqlens))
        )

        return {
            "und_attn": float(und_attn),
            "gen_attn": float(gen_attn),
            "und_attn_proj": float(und_training_weight * num_layers * und_projection_params * und_tokens),
            "gen_attn_proj": float(6 * num_layers * gen_projection_params * gen_tokens),
            "und_mlp": float(und_training_weight * num_layers * mlp_params * und_tokens),
            "gen_mlp": float(6 * num_layers * mlp_params * gen_tokens),
            # Generation-tower adapters and the timestep MLP see generation rows only.
            "proj_in": float(6 * patch_latent_dim * hidden_size * gen_tokens),
            "proj_out": float(6 * hidden_size * patch_latent_dim * gen_tokens),
            "time_embedder": float(6 * (frequency_size * hidden_size + hidden_size * hidden_size) * gen_tokens),
        }

    def _estimate_cosmos3_mot_flops(
        self,
        tokens_sum,
        batch_seqlens,
        delta_time,
        gen_seqlens=None,
        **kwargs,
    ):
        """Estimate Cosmos3 MoT training FLOPs (forward + backward) per step.

        ``tokens_sum`` is ``sum(batch_seqlens)``, i.e. the reasoner-tower token total;
        the generation tower's total comes from ``gen_seqlens`` and is generally
        different. See :meth:`_cosmos3_mot_flops_breakdown` for the formula and for
        where each coefficient comes from.

        ``gen_seqlens=None`` prices the reasoner tower alone, which is what
        ``supports_config``'s probe call needs; the DiT trainer's meter always passes
        both halves, so a real step never takes that path (a reasoner-only count would
        silently under-report by the whole diffusion tower).
        """
        del tokens_sum  # the two towers' totals are summed from the two length lists
        if gen_seqlens is None:
            gen_seqlens = [0] * len(batch_seqlens)
        breakdown = self._cosmos3_mot_flops_breakdown(self.config, batch_seqlens, gen_seqlens)
        flops_all_token = sum(breakdown.values())
        return flops_all_token * (1.0 / delta_time) / 1e12

    def _estimate_qwen3_moe_flops(
        self,
        tokens_sum,
        batch_seqlens,
        delta_time,
        lora_config=None,
    ):
        flops_all_token = self._compute_qwen3_moe_text_flops(
            self.config,
            tokens_sum,
            batch_seqlens,
            lora_config,
        )
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    @staticmethod
    def _compute_sliding_attention_score_sum(batch_seqlens, sliding_window):
        attention_score_sum = 0
        for seqlen in batch_seqlens:
            if seqlen <= sliding_window:
                attention_score_sum += seqlen * (seqlen + 1) // 2
            else:
                attention_score_sum += sliding_window * (sliding_window + 1) // 2
                attention_score_sum += (seqlen - sliding_window) * sliding_window
        return attention_score_sum

    def _estimate_gpt_oss_flops(self, tokens_sum, batch_seqlens, delta_time):
        """
        Estimate GPT-OSS training FLOPs.

        GPT-OSS uses all-MoE layers with top-k routed experts and a mixed schedule of
        full attention and sliding-window attention. Learnable attention sinks are
        negligible relative to matmul FLOPs and are intentionally ignored.
        """
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        intermediate_size = self.config.intermediate_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        num_local_experts = self.config.num_local_experts
        num_experts_per_tok = self.config.num_experts_per_tok
        head_dim = getattr(self.config, "head_dim", hidden_size // num_attention_heads)

        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # Router + active routed experts. GPT-OSS stores gate/up in one interleaved
        # projection, but its arithmetic is equivalent to gate_proj + up_proj + down_proj.
        moe_gate_N = hidden_size * num_local_experts
        moe_expertmlp_N = hidden_size * intermediate_size * num_experts_per_tok * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + q_size)
        lm_head_N = self._compute_lm_head_params(hidden_size, vocab_size)
        dense_N = (moe_gate_N + moe_expertmlp_N + attn_linear_N) * num_hidden_layers + lm_head_N
        dense_N_flops = 6 * dense_N * tokens_sum

        layer_types = getattr(self.config, "layer_types", None)
        if layer_types is None:
            num_full_attn_layers = num_hidden_layers
            num_sliding_attn_layers = 0
        else:
            num_full_attn_layers = sum(t == "full_attention" for t in layer_types)
            num_sliding_attn_layers = sum(t == "sliding_attention" for t in layer_types)
            if num_full_attn_layers + num_sliding_attn_layers != num_hidden_layers:
                raise ValueError(
                    f"GPT-OSS attention layer count mismatch: {num_full_attn_layers} full + "
                    f"{num_sliding_attn_layers} sliding != {num_hidden_layers} total layers. "
                    "The config may use an unsupported entry in `config.layer_types`."
                )

        full_attn_score_sum = 0
        for seqlen in batch_seqlens:
            full_attn_score_sum += seqlen * seqlen

        sliding_window = getattr(self.config, "sliding_window", None)
        if num_sliding_attn_layers > 0 and sliding_window is None:
            raise ValueError("GPT-OSS config has sliding attention layers but no `sliding_window`.")
        sliding_attn_score_sum = (
            self._compute_sliding_attention_score_sum(batch_seqlens, sliding_window)
            if num_sliding_attn_layers > 0
            else 0
        )
        attn_score_sum = full_attn_score_sum * num_full_attn_layers
        attn_score_sum += sliding_attn_score_sum * num_sliding_attn_layers
        attn_qkv_flops = 12 * attn_score_sum * head_dim * num_attention_heads

        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_gemma3_flops(
        self,
        tokens_sum,
        batch_seqlens,
        delta_time,
        images_seqlens=None,
        freeze_vit=False,
    ):
        """Estimate dominant Gemma 3 text and optional SigLIP vision FLOPs."""
        config = self.config.text_config if hasattr(self.config, "text_config") else self.config
        hidden_size = config.hidden_size
        head_dim = getattr(config, "head_dim", hidden_size // config.num_attention_heads)
        q_size = config.num_attention_heads * head_dim
        kv_size = config.num_key_value_heads * head_dim

        mlp_params = hidden_size * config.intermediate_size * 3
        attention_projection_params = hidden_size * (2 * q_size + 2 * kv_size)
        lm_head_params = self._compute_lm_head_params(hidden_size, config.vocab_size)
        linear_params = (mlp_params + attention_projection_params) * config.num_hidden_layers + lm_head_params
        linear_flops = 6 * linear_params * tokens_sum

        layer_types = getattr(config, "layer_types", None)
        if layer_types is None:
            full_attention_layers = config.num_hidden_layers
            sliding_attention_layers = 0
        else:
            full_attention_layers = sum(layer_type == "full_attention" for layer_type in layer_types)
            sliding_attention_layers = sum(layer_type == "sliding_attention" for layer_type in layer_types)
            if full_attention_layers + sliding_attention_layers != config.num_hidden_layers:
                raise ValueError("Gemma 3 layer_types contains an unsupported attention type.")

        full_attention_scores = sum(seqlen * seqlen for seqlen in batch_seqlens)
        sliding_attention_scores = 0
        if sliding_attention_layers:
            if config.sliding_window is None:
                raise ValueError("Gemma 3 has sliding-attention layers but no sliding_window.")
            sliding_attention_scores = self._compute_sliding_attention_score_sum(batch_seqlens, config.sliding_window)
        attention_scores = (
            full_attention_scores * full_attention_layers + sliding_attention_scores * sliding_attention_layers
        )
        attention_flops = 12 * attention_scores * head_dim * config.num_attention_heads

        vision_flops = 0
        if images_seqlens:
            vision_config = self.config.vision_config
            vision_tokens = sum(images_seqlens)
            vision_hidden_size = vision_config.hidden_size
            # Patch embedding and pooling are small relative to the encoder body.
            vision_linear_params = vision_config.num_hidden_layers * (
                2 * vision_hidden_size * vision_config.intermediate_size + 4 * vision_hidden_size**2
            )
            vision_factor = 2 if freeze_vit else 6
            vision_flops = vision_factor * vision_linear_params * vision_tokens
            vision_attention_factor = 4 if freeze_vit else 12
            vision_flops += (
                vision_attention_factor
                * sum(seqlen * seqlen for seqlen in images_seqlens)
                * (vision_hidden_size // vision_config.num_attention_heads)
                * vision_config.num_attention_heads
                * vision_config.num_hidden_layers
            )
            projected_tokens = len(images_seqlens) * self.config.mm_tokens_per_image
            vision_flops += 6 * projected_tokens * vision_hidden_size * config.hidden_size

        return (linear_flops + attention_flops + vision_flops) * (1.0 / delta_time) / 1e12

    def _estimate_qwen2_flops(
        self,
        tokens_sum,
        batch_seqlens,
        delta_time,
        lora_config=None,
    ):
        flops_all_token = self._compute_qwen_dense_text_flops(
            self.config,
            tokens_sum,
            batch_seqlens,
            lora_config,
        )
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_llama_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        intermediate_size = self.config.intermediate_size

        head_dim = hidden_size // num_attention_heads
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # non-attn per layer parm
        # llama use SwiGelu, gate, having up and down linear layer in mlp
        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        lm_head_N = self._compute_lm_head_params(hidden_size, vocab_size)
        # non-attn all_layer parm
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_qwen2_vl_flops(
        self,
        tokens_sum,
        batch_seqlens,
        delta_time,
        lora_config=None,
        **kargs,
    ):
        text_config = self.config.text_config if hasattr(self.config, "text_config") else self.config
        text_flops = self._compute_qwen_dense_text_flops(
            text_config,
            tokens_sum,
            batch_seqlens,
            lora_config,
        )

        # vit flops
        images_seqlens = kargs.get("images_seqlens", None)
        if images_seqlens:
            vit_flops = self._estimate_qwen_vit_flop(
                images_seqlens,
                self.config.vision_config,
                lora_config=lora_config,
                freeze_vit=kargs.get("freeze_vit", False),
            )
        else:
            vit_flops = 0

        # all_layer & all_token fwd & bwd flops
        flops_all_token = text_flops + vit_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_qwen3_vl_flops(
        self,
        tokens_sum,
        batch_seqlens,
        delta_time,
        lora_config=None,
        **kargs,
    ):
        text_flops = self._compute_qwen_dense_text_flops(
            self.config.text_config,
            tokens_sum,
            batch_seqlens,
            lora_config,
        )

        # vit flops
        images_seqlens = kargs.get("images_seqlens", None)
        if images_seqlens:
            vit_flops = self._estimate_qwen3_vit_flop(
                images_seqlens,
                self.config.vision_config,
                lora_config=lora_config,
                freeze_vit=kargs.get("freeze_vit", False),
            )
        else:
            vit_flops = 0

        # all_layer & all_token fwd & bwd flops
        flops_all_token = text_flops + vit_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_qwen3_vl_moe_flops(
        self,
        tokens_sum,
        batch_seqlens,
        delta_time,
        lora_config=None,
        **kargs,
    ):
        text_flops = self._compute_qwen3_moe_text_flops(
            self.config.text_config,
            tokens_sum,
            batch_seqlens,
            lora_config,
        )
        # vit flops
        images_seqlens = kargs.get("images_seqlens", None)
        if images_seqlens:
            vit_flops = self._estimate_qwen3_vit_flop(
                images_seqlens,
                self.config.vision_config,
                lora_config=lora_config,
                freeze_vit=kargs.get("freeze_vit", False),
            )
        else:
            vit_flops = 0
        # all_layer & all_token fwd & bwd flops
        flops_all_token = text_flops + vit_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_qwen3_vit_flop(
        self,
        images_seqlens,
        config,
        lora_config=None,
        freeze_vit=False,
    ):
        """
        Estimate the FLOPs of the Qwen3-family vision encoder.
        """

        if config is None:
            return 0
        tokens_sum = sum(images_seqlens)

        num_heads = config.num_heads
        depth = config.depth

        dim = config.hidden_size
        mlp_hidden_dim = config.intermediate_size
        out_hidden_size = config.out_hidden_size

        spatial_merge_size = config.spatial_merge_size

        head_dim = dim // num_heads

        # every vision token's patch_embed comes from a conv of (C, T, H, W) -> (dim,)
        patch_embed_N = dim * config.in_channels * config.temporal_patch_size * config.patch_size * config.patch_size
        # Qwen3 VL vision mlp does not use GLU, thus 2.
        mlp_N = dim * mlp_hidden_dim * 2
        attn_linear_N = dim * (4 * dim)  # qkv and output proj
        merger_N = (out_hidden_size + (dim * (spatial_merge_size**2))) * (dim * (spatial_merge_size**2))

        # Qwen3 VL uses deep stack, one merger for every deepstack layer
        merger_count = 1 + len(getattr(config, "deepstack_visual_indexes", ()))
        deepstack_merger_N = merger_N * (merger_count - 1)
        # non-attn all_layer parm
        dense_N = patch_embed_N + (mlp_N + attn_linear_N) * depth + deepstack_merger_N + merger_N

        vision_module_shapes = {
            "qkv": (dim, 3 * dim, depth),
            "proj": (dim, dim, depth),
            "linear_fc1": [
                (dim, mlp_hidden_dim, depth),
                (dim * spatial_merge_size**2, dim * spatial_merge_size**2, merger_count),
            ],
            "linear_fc2": [
                (mlp_hidden_dim, dim, depth),
                (dim * spatial_merge_size**2, out_hidden_size, merger_count),
            ],
        }
        target_modules = lora_config.target_modules if lora_config is not None else ()
        vision_has_lora = any(module in vision_module_shapes for module in target_modules or ())
        vision_requires_grad = vision_has_lora or (lora_config is not None and lora_config.bias == "all")

        if lora_config is None:
            dense_N_flops = (2 if freeze_vit else 6) * dense_N * tokens_sum
        elif not vision_requires_grad:
            dense_N_flops = 2 * dense_N * tokens_sum
        elif not vision_has_lora:
            # Bias-only vision training needs activation gradients through the
            # tower, but not input gradients for the initial patch embedding.
            adaptable_base_N = dense_N - patch_embed_N
            dense_N_flops = (2 * patch_embed_N + 4 * adaptable_base_N) * tokens_sum
        else:
            # Patch embedding is Conv3d and is not adapted by VeOmni's linear LoRA.
            adaptable_base_N = dense_N - patch_embed_N
            dense_N_flops = 2 * patch_embed_N * tokens_sum
            dense_N_flops += self._compute_linear_flops(
                adaptable_base_N,
                tokens_sum,
                lora_config=lora_config,
                module_shapes=vision_module_shapes,
            )

        # In Qwen3 VL, full attention is used in all vision layers.
        full_attn_layer_num = depth

        # full attn layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in images_seqlens:
            seqlen_square_sum += seqlen * seqlen
        if lora_config is None:
            attention_factor = 4 if freeze_vit else 12
        else:
            attention_factor = 12 if vision_requires_grad else 4
        attn_qkv_flops = attention_factor * seqlen_square_sum * head_dim * num_heads * full_attn_layer_num

        vit_flops = dense_N_flops + attn_qkv_flops

        return vit_flops

    def _estimate_qwen_vit_flop(
        self,
        images_seqlens,
        config,
        lora_config=None,
        freeze_vit=False,
    ):
        """
        Estimate the FLOPS of the vision encoder for Qwen2 and Qwen2.5
        """

        if config is None:
            return 0
        tokens_sum = sum(images_seqlens)

        num_heads = config.num_heads
        depth = config.depth

        # In Qwen2 VL and Qwen2.5VL, the parameters naming are different:
        #
        # Parameter                 | Qwen2 VL         | Qwen2.5 VL
        # --------------------------|------------------|------------------
        # ViT hidden dimension      | embed_dim        | hidden_size
        # ViT output dimension      | hidden_size      | out_hidden_size
        # ViT MLP intermediate dim  | embed_dim * mlp_ratio | intermediate_size
        #
        # See https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/config.json
        # and https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/blob/main/config.json for an example.
        is_qwen2_vl = hasattr(config, "embed_dim")
        dim = config.embed_dim if is_qwen2_vl else config.hidden_size
        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio) if is_qwen2_vl else config.intermediate_size
        out_hidden_size = config.hidden_size if is_qwen2_vl else config.out_hidden_size

        spatial_merge_size = config.spatial_merge_size
        head_dim = dim // num_heads

        # Qwen 2.5 VL uses SiLU, thus 3.
        mlp_N = dim * mlp_hidden_dim * (2 if is_qwen2_vl else 3)
        attn_linear_N = dim * (4 * dim)  # qkv and output proj
        patch_embed_and_merger_N = (out_hidden_size + (dim * (spatial_merge_size**2))) * (
            dim * (spatial_merge_size**2)
        )

        # non-attn all_layer parm
        dense_N = (mlp_N + attn_linear_N) * depth + patch_embed_and_merger_N

        merger_hidden_size = dim * spatial_merge_size**2
        vision_module_shapes = {
            "qkv": (dim, 3 * dim, depth),
            "proj": (dim, dim, depth),
            "mlp.0": (merger_hidden_size, merger_hidden_size, 1),
            "mlp.2": (merger_hidden_size, out_hidden_size, 1),
        }
        if is_qwen2_vl:
            vision_module_shapes.update(
                {
                    "fc1": (dim, mlp_hidden_dim, depth),
                    "fc2": (mlp_hidden_dim, dim, depth),
                }
            )
        else:
            vision_module_shapes.update(
                {
                    "gate_proj": (dim, mlp_hidden_dim, depth),
                    "up_proj": (dim, mlp_hidden_dim, depth),
                    "down_proj": (mlp_hidden_dim, dim, depth),
                }
            )
        target_modules = lora_config.target_modules if lora_config is not None else ()
        vision_has_lora = any(module in vision_module_shapes for module in target_modules or ())
        vision_requires_grad = vision_has_lora or (lora_config is not None and lora_config.bias == "all")
        if lora_config is None and freeze_vit:
            dense_N_flops = 2 * dense_N * tokens_sum
        elif lora_config is not None and not vision_requires_grad:
            dense_N_flops = 2 * dense_N * tokens_sum
        elif lora_config is not None and not vision_has_lora:
            dense_N_flops = 4 * dense_N * tokens_sum
        else:
            dense_N_flops = self._compute_linear_flops(
                dense_N,
                tokens_sum,
                lora_config=lora_config,
                module_shapes=vision_module_shapes,
                detached_without_targets=True,
            )
        if lora_config is None:
            attention_factor = 4 if freeze_vit else 12
        else:
            attention_factor = 12 if vision_requires_grad else 4

        # In Qwen2.5 VL, windowed attention is used in some layers.
        full_attn_layer_num = config.depth if is_qwen2_vl else len(config.fullatt_block_indexes)
        window_attn_layer_num = config.depth - full_attn_layer_num

        # full attn layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in images_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = attention_factor * seqlen_square_sum * head_dim * num_heads * full_attn_layer_num

        # If window attention is used, add the window attention flops
        if window_attn_layer_num > 0:
            # Qwen2.5-VL configures window_size in pixels. Mirror get_vision_window_index's
            # conversion to the number of ViT tokens on each side of a window. This remains
            # an upper bound for partially filled windows at image boundaries.
            merger_window_size = config.window_size // spatial_merge_size // config.patch_size
            window_token_size = merger_window_size * spatial_merge_size
            window_attn_compute_flops = attention_factor * tokens_sum * (window_token_size**2) * head_dim * num_heads
            attn_qkv_flops += window_attn_compute_flops * window_attn_layer_num

        vit_flops = dense_N_flops + attn_qkv_flops

        return vit_flops

    @staticmethod
    def _compute_hybrid_attn_params(config, full_attention_types=("full_attention",)):
        """
        Compute hybrid attention (full + GatedDeltaNet) linear param count and layer info.

        Layers alternate between full attention and GatedDeltaNet (linear attention). The
        per-layer schedule is read from `config.layer_types`. Full-attention layer names
        are supplied by the caller because some model configs normalize them to a
        model-specific value. The typical pattern is (interval - 1) linear layers followed
        by 1 full attention layer.

        Full attention (Qwen3_5Attention) projections:
            q_proj:  hidden_size -> 2 * num_attention_heads * head_dim (query + output gate)
            k_proj:  hidden_size -> num_key_value_heads * head_dim
            v_proj:  hidden_size -> num_key_value_heads * head_dim
            o_proj:  num_attention_heads * head_dim -> hidden_size

        GatedDeltaNet (Qwen3_5GatedDeltaNet) projections:
            in_proj_qkv:  hidden_size -> 2 * linear_k_size + linear_v_size
            in_proj_z:    hidden_size -> linear_v_size          (output gate)
            in_proj_b:    hidden_size -> linear_num_value_heads (beta/gating scalar per head)
            in_proj_a:    hidden_size -> linear_num_value_heads (alpha/decay scalar per head)
            out_proj:     linear_v_size -> hidden_size
            conv1d:       depthwise, channels = 2 * linear_k_size + linear_v_size, kernel = conv_kernel_dim

        where:
            linear_k_size = linear_num_key_heads * linear_key_head_dim
            linear_v_size = linear_num_value_heads * linear_value_head_dim

        This only counts projection and conv1d parameter FLOPs. The GatedDeltaNet
        recurrence FLOPs are computed separately by _compute_gdn_recurrence_flops.
        """
        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)

        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # transformers v5 no longer exposes `full_attention_interval` on the config;
        # it normalizes the schedule into `layer_types`. Count the full / linear layers
        # directly from `layer_types` when present (handles non-uniform schedules too),
        # falling back to the legacy interval attribute for older configs.
        num_hidden_layers = config.num_hidden_layers
        layer_types = getattr(config, "layer_types", None)
        if layer_types is not None:
            num_full_attn_layers = sum(t in full_attention_types for t in layer_types)
            num_linear_attn_layers = sum(t == "linear_attention" for t in layer_types)
        else:
            full_attention_interval = getattr(config, "full_attention_interval", None)
            if full_attention_interval is None:
                raise ValueError(
                    "Cannot determine the hybrid-attention layer schedule: config exposes neither "
                    "`layer_types` (transformers v5) nor `full_attention_interval` (legacy). "
                    "Verify this is a hybrid-attention (Qwen3.5-style) config."
                )
            # A non-positive interval would divide-by-zero or yield negative counts that the
            # sum check below cannot catch (the legacy branch derives linear = total - full).
            if full_attention_interval <= 0:
                raise ValueError(
                    f"Invalid `full_attention_interval`: {full_attention_interval}. It must be a positive integer."
                )
            num_full_attn_layers = num_hidden_layers // full_attention_interval
            num_linear_attn_layers = num_hidden_layers - num_full_attn_layers

        # Guard against unrecognized / missing entries in `layer_types` silently
        # under-counting params (e.g. a future third attention type).
        if num_full_attn_layers + num_linear_attn_layers != num_hidden_layers:
            raise ValueError(
                f"Hybrid-attention layer count mismatch: {num_full_attn_layers} full + "
                f"{num_linear_attn_layers} linear != {num_hidden_layers} total layers. "
                "The config may use an unsupported entry in `config.layer_types`."
            )

        # Full attention: q_proj emits query and output-gate halves, while o_proj consumes only query-sized output.
        full_attn_linear_N = hidden_size * (2 * q_size + k_size + v_size + q_size)

        # GatedDeltaNet linear projections and depthwise conv1d
        linear_k_size = config.linear_num_key_heads * config.linear_key_head_dim
        linear_v_size = config.linear_num_value_heads * config.linear_value_head_dim
        # in_proj_qkv                          + in_proj_z    + in_proj_b + in_proj_a              + out_proj
        # (2 * linear_k_size + linear_v_size)  + linear_v_size + linear_num_value_heads * 2        + linear_v_size
        linear_attn_size = 2 * linear_k_size + 2 * linear_v_size + 2 * config.linear_num_value_heads + linear_v_size
        # depthwise conv1d: each of (2 * linear_k_size + linear_v_size) channels has its own kernel
        conv_N = config.linear_conv_kernel_dim * (2 * linear_k_size + linear_v_size)
        linear_attn_linear_N = hidden_size * linear_attn_size + conv_N

        attn_linear_N = full_attn_linear_N * num_full_attn_layers + linear_attn_linear_N * num_linear_attn_layers

        return attn_linear_N, num_full_attn_layers, num_linear_attn_layers, head_dim, num_attention_heads

    @staticmethod
    def _compute_gdn_recurrence_flops(config, tokens_sum, num_gdn_layers):
        """
        Compute FLOPs for the GatedDeltaNet recurrence across all GDN layers.

        The recurrent form of GatedDeltaNet (ref: https://kexue.fm/archives/11033, eq.17/18):

            S_t = gamma_t * S_{t-1} + eta_t * (v_t - S_{t-1} @ k_t) @ k_t^T
            o_t = S_t @ q_t

        where S_t is the state matrix of shape (linear_value_head_dim, linear_key_head_dim)
        per value head.

        Note: in practice, training uses the chunked implementation (chunk_gated_delta_rule)
        which reorganizes the computation into chunk-level matrix multiplications for better
        hardware utilization. However, chunking is purely an implementation optimization that
        does not change the total arithmetic — it computes the same result as the recurrent
        form. We therefore use the recurrent form as the theoretical FLOPs baseline.

        Per step per head, the dominant ops (forward) are:
            S_{t-1} @ k_t        (mat-vec, (d_v,d_k)@(d_k,)=(d_v,)):  2 * d_v * d_k FLOPs
            (...) @ k_t^T        (outer product, (d_v,)⊗(d_k,)=(d_v,d_k)):  d_v * d_k FLOPs
            o_t = S_t @ q_t      (mat-vec, (d_v,d_k)@(d_k,)=(d_v,)):  2 * d_v * d_k FLOPs
        where d_v = linear_value_head_dim, d_k = linear_key_head_dim.

        Following the same convention as quadratic attention (Q@K + attn@V):
            fwd: (2 + 1 + 2) * d_v * d_k = 5 * d_v * d_k per step per head
            fwd + bwd (3x): 15 * d_v * d_k per step per head
        """
        return (
            15
            * config.linear_key_head_dim
            * config.linear_value_head_dim
            * config.linear_num_value_heads
            * tokens_sum
            * num_gdn_layers
        )

    @staticmethod
    def _compute_qsa_attention_score_sum(batch_seqlens, compress_ratio, token_budget):
        """Count the causal query-key pairs selected by Qwen4-Exp QSA.

        QSA partitions every query's visible prefix into complete compressed
        blocks, selects at most ``token_budget // compress_ratio`` blocks, and
        keeps the incomplete tail. The current eager/SDPA implementation uses
        a dense mask, but MFU counts useful model FLOPs rather than redundant
        implementation work.
        """
        if compress_ratio <= 0:
            raise ValueError(f"QSA compression ratio must be positive, got {compress_ratio}.")
        if token_budget <= 0:
            raise ValueError(f"QSA token budget must be positive, got {token_budget}.")

        block_topk = token_budget // compress_ratio
        score_sum = 0
        for seqlen in batch_seqlens:
            quotient, remainder = divmod(seqlen, compress_ratio)
            if quotient <= block_topk:
                capped_block_sum = compress_ratio * quotient * (quotient - 1) // 2
                capped_block_sum += quotient * (remainder + 1)
            else:
                capped_block_sum = compress_ratio * block_topk * (block_topk - 1) // 2
                capped_block_sum += block_topk * (seqlen - block_topk * compress_ratio + 1)

            tail_sum = quotient * compress_ratio * (compress_ratio - 1) // 2
            tail_sum += remainder * (remainder + 1) // 2
            score_sum += compress_ratio * capped_block_sum + tail_sum

        return score_sum

    def _estimate_qwen4_exp_flops(
        self,
        tokens_sum,
        batch_seqlens,
        delta_time,
        **kargs,
    ):
        """Estimate useful training FLOPs for Qwen4-Exp.

        PLE embedding tables are sparse lookups and are intentionally excluded.
        Their trainable projections and depthwise convolution are included.
        """
        text_config = self.config.text_config
        hidden_size = text_config.hidden_size
        num_hidden_layers = text_config.num_hidden_layers

        attn_linear_N, num_full_attn_layers, num_linear_attn_layers, head_dim, num_attention_heads = (
            self._compute_hybrid_attn_params(
                text_config,
                full_attention_types=("full_attention", "qwen_sparse_attention"),
            )
        )

        # QSA indexer projection on every full-attention layer.
        indexer_linear_N = hidden_size * (text_config.indexer_n_heads + text_config.indexer_kv_heads)
        indexer_linear_N *= text_config.indexer_head_dim * num_full_attn_layers

        # Every layer computes router logits, top-k routed experts, one shared
        # expert, and the shared-expert scalar gate.
        moe_router_N = hidden_size * text_config.num_experts
        moe_expert_N = hidden_size * text_config.moe_intermediate_size * text_config.num_experts_per_tok * 3
        shared_expert_N = hidden_size * text_config.shared_expert_intermediate_size * 3
        shared_expert_gate_N = hidden_size
        moe_N = (moe_router_N + moe_expert_N + shared_expert_N + shared_expert_gate_N) * num_hidden_layers

        # Two mHC mappings per decoder layer, plus the final mixer without an
        # injection projection.
        hc_hidden_size = text_config.hc_count * hidden_size
        mhc_mapping_N = 2 * hc_hidden_size * text_config.hc_lowrank
        mhc_mapping_N += hc_hidden_size * text_config.hc_count
        mhc_final_N = 2 * hc_hidden_size * text_config.hc_lowrank
        mhc_N = 2 * mhc_mapping_N * num_hidden_layers + mhc_final_N

        # PLE table lookup is not a matmul. Count only the key/value projections
        # and the depthwise convolution at configured PLE layers.
        ple_N = text_config.ple_embed_dim * (hc_hidden_size + hidden_size)
        ple_N += hc_hidden_size * text_config.ple_conv_kernel_size
        ple_N *= len(text_config.ple_layer_ids)

        lm_head_N = self._compute_lm_head_params(hidden_size, text_config.vocab_size)
        trainable_linear_N = attn_linear_N + moe_N + mhc_N + ple_N + lm_head_N
        linear_flops = 6 * trainable_linear_N * tokens_sum
        # Top-k indices are discrete and no indexer loss is exposed, so this
        # branch is forward-only during SFT.
        linear_flops += 2 * indexer_linear_N * tokens_sum

        qsa_score_sum = self._compute_qsa_attention_score_sum(
            batch_seqlens,
            text_config.indexer_compress_ratio,
            text_config.indexer_budget,
        )
        qsa_attention_flops = 12 * qsa_score_sum * head_dim * num_attention_heads * num_full_attn_layers

        # The indexer scores every query against all complete compressed blocks.
        indexer_score_sum = self._compute_compressed_attention_score_sum(
            batch_seqlens, text_config.indexer_compress_ratio
        )
        indexer_flops = (
            2 * indexer_score_sum * text_config.indexer_head_dim * text_config.indexer_n_heads * num_full_attn_layers
        )

        gdn_recurrence_flops = self._compute_gdn_recurrence_flops(text_config, tokens_sum, num_linear_attn_layers)

        images_seqlens = kargs.get("images_seqlens", None)
        if images_seqlens:
            vit_flops = self._estimate_qwen3_vit_flop(
                images_seqlens,
                getattr(self.config, "vision_config", None),
                freeze_vit=kargs.get("freeze_vit", False),
            )
        else:
            vit_flops = 0

        flops_all_token = linear_flops + qsa_attention_flops + indexer_flops + gdn_recurrence_flops + vit_flops
        return flops_all_token * (1.0 / delta_time) / 1e12

    def _estimate_qwen3_next_flops(
        self,
        tokens_sum,
        batch_seqlens,
        delta_time,
        lora_config=None,
    ):
        """
        Estimate the FLOPS of the Qwen3 Next model.
        """
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers

        # hybrid attention params
        attn_linear_N, num_full_attn_layers, num_linear_attn_layers, head_dim, num_attention_heads = (
            self._compute_hybrid_attn_params(self.config)
        )

        # MoE per-layer parameters: top-k router, routed/shared SwiGLU experts,
        # and the scalar sigmoid gate applied to the shared expert output.
        moe_gata_N = hidden_size * self.config.num_experts
        moe_sharedexpertmlp_N = hidden_size * self.config.shared_expert_intermediate_size * 3
        moe_sharedexpert_gate_N = hidden_size
        moe_expertmlp_N = hidden_size * self.config.moe_intermediate_size * self.config.num_experts_per_tok * 3
        moe_N = (moe_gata_N + moe_expertmlp_N + moe_sharedexpertmlp_N + moe_sharedexpert_gate_N) * num_hidden_layers

        # lm head param
        lm_head_N = self._compute_lm_head_params(hidden_size, vocab_size)
        head_dim = getattr(self.config, "head_dim", hidden_size // self.config.num_attention_heads)
        q_size = self.config.num_attention_heads * head_dim
        kv_size = self.config.num_key_value_heads * head_dim
        linear_k_size = self.config.linear_num_key_heads * self.config.linear_key_head_dim
        linear_v_size = self.config.linear_num_value_heads * self.config.linear_value_head_dim
        module_shapes = {
            "q_proj": (hidden_size, 2 * q_size, num_full_attn_layers),
            "k_proj": (hidden_size, kv_size, num_full_attn_layers),
            "v_proj": (hidden_size, kv_size, num_full_attn_layers),
            "o_proj": (q_size, hidden_size, num_full_attn_layers),
            "in_proj_qkvz": (
                hidden_size,
                2 * linear_k_size + 2 * linear_v_size,
                num_linear_attn_layers,
            ),
            "in_proj_ba": (
                hidden_size,
                2 * self.config.linear_num_value_heads,
                num_linear_attn_layers,
            ),
            "out_proj": (linear_v_size, hidden_size, num_linear_attn_layers),
            "gate_proj": (hidden_size, self.config.shared_expert_intermediate_size, num_hidden_layers),
            "up_proj": (hidden_size, self.config.shared_expert_intermediate_size, num_hidden_layers),
            "down_proj": (self.config.shared_expert_intermediate_size, hidden_size, num_hidden_layers),
        }
        dense_N_flops = self._compute_linear_flops(
            moe_N + attn_linear_N + lm_head_N,
            tokens_sum,
            lora_config=lora_config,
            module_shapes=module_shapes,
        )
        dense_N_flops += self._compute_routed_moe_lora_flops(self.config, tokens_sum, lora_config)
        # attn all_layer & all_token fwd & bwd flops, only count full attention layers
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_full_attn_layers

        # GatedDeltaNet recurrence flops (state update + query, for all GDN layers)
        gdn_recurrence_flops = self._compute_gdn_recurrence_flops(self.config, tokens_sum, num_linear_attn_layers)

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops + gdn_recurrence_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_qwen3_5_family_flops(
        self,
        tokens_sum,
        batch_seqlens,
        delta_time,
        lora_config=None,
        **kargs,
    ):
        """
        Estimate the FLOPS of the Qwen3.5 model family (dense/MoE MLP + hybrid attention + ViT).

        Handles both Qwen3.5 (dense) and Qwen3.5-MoE by checking for MoE-specific config
        attributes. Both variants share hybrid attention and ViT; only the MLP differs.

        Text model (from text_config):
            Dense MLP per layer (SwiGLU, 3 projections):
                gate_proj:  hidden_size -> intermediate_size
                up_proj:    hidden_size -> intermediate_size
                down_proj:  intermediate_size -> hidden_size

            MoE per layer (when num_experts is present):
                TopkGate router:   hidden_size -> num_experts
                Routed experts (top-k activated, each SwiGLU):
                    gate_proj:  hidden_size -> moe_intermediate_size
                    up_proj:    hidden_size -> moe_intermediate_size
                    down_proj:  moe_intermediate_size -> hidden_size
                    -> 3 projections * num_experts_per_tok active experts
                Shared expert (always active, SwiGLU):
                    gate_proj:  hidden_size -> shared_expert_intermediate_size
                    up_proj:    hidden_size -> shared_expert_intermediate_size
                    down_proj:  shared_expert_intermediate_size -> hidden_size
                Shared expert gate:
                    shared_expert_gate: hidden_size -> 1

            Hybrid attention: see _compute_hybrid_attn_params docstring.

            LM head:
                lm_head: hidden_size -> vocab_size
                embed_tokens is an embedding table lookup and is excluded from FLOPs.

        Quadratic attention FLOPs (only full attention layers):
            Per layer: 2 * seq_len^2 * head_dim * num_attention_heads (Q@K + attn@V)
            fwd + bwd (3x) -> 6x total -> coefficient 12

        Vision encoder: delegates to _estimate_qwen3_vit_flop.
        """
        text_config = self.config.text_config if hasattr(self.config, "text_config") else self.config
        hidden_size = text_config.hidden_size
        vocab_size = text_config.vocab_size
        num_hidden_layers = text_config.num_hidden_layers

        # hybrid attention linear projection params (full + GatedDeltaNet)
        attn_linear_N, num_full_attn_layers, num_linear_attn_layers, head_dim, num_attention_heads = (
            self._compute_hybrid_attn_params(text_config)
        )

        # MLP params: MoE or dense depending on config
        is_moe = hasattr(text_config, "num_experts")
        if is_moe:
            # MoE per layer: router, routed expert MLPs (top-k), shared expert
            # MLP, and the scalar sigmoid gate on the shared expert output.
            moe_gata_N = hidden_size * text_config.num_experts
            moe_expertmlp_N = hidden_size * text_config.moe_intermediate_size * text_config.num_experts_per_tok * 3
            moe_sharedexpertmlp_N = hidden_size * text_config.shared_expert_intermediate_size * 3
            moe_sharedexpert_gate_N = hidden_size
            mlp_N = (
                moe_gata_N + moe_expertmlp_N + moe_sharedexpertmlp_N + moe_sharedexpert_gate_N
            ) * num_hidden_layers
        else:
            # dense MLP per layer: gate_proj + up_proj + down_proj (SwiGLU)
            mlp_N = hidden_size * text_config.intermediate_size * 3 * num_hidden_layers

        # lm_head only; embed_tokens is a table lookup.
        lm_head_N = self._compute_lm_head_params(hidden_size, vocab_size)
        q_size = text_config.num_attention_heads * head_dim
        kv_size = text_config.num_key_value_heads * head_dim
        linear_k_size = text_config.linear_num_key_heads * text_config.linear_key_head_dim
        linear_v_size = text_config.linear_num_value_heads * text_config.linear_value_head_dim
        module_shapes = {
            "q_proj": (hidden_size, 2 * q_size, num_full_attn_layers),
            "k_proj": (hidden_size, kv_size, num_full_attn_layers),
            "v_proj": (hidden_size, kv_size, num_full_attn_layers),
            "o_proj": (q_size, hidden_size, num_full_attn_layers),
            "in_proj_qkv": (
                hidden_size,
                2 * linear_k_size + linear_v_size,
                num_linear_attn_layers,
            ),
            "in_proj_z": (hidden_size, linear_v_size, num_linear_attn_layers),
            "in_proj_b": (hidden_size, text_config.linear_num_value_heads, num_linear_attn_layers),
            "in_proj_a": (hidden_size, text_config.linear_num_value_heads, num_linear_attn_layers),
            "out_proj": (linear_v_size, hidden_size, num_linear_attn_layers),
        }
        if is_moe:
            mlp_size = text_config.shared_expert_intermediate_size
        else:
            mlp_size = text_config.intermediate_size
        module_shapes.update(
            {
                "gate_proj": (hidden_size, mlp_size, num_hidden_layers),
                "up_proj": (hidden_size, mlp_size, num_hidden_layers),
                "down_proj": (mlp_size, hidden_size, num_hidden_layers),
            }
        )
        dense_N_flops = self._compute_linear_flops(
            mlp_N + attn_linear_N + lm_head_N,
            tokens_sum,
            lora_config=lora_config,
            module_shapes=module_shapes,
        )
        if is_moe:
            dense_N_flops += self._compute_routed_moe_lora_flops(text_config, tokens_sum, lora_config)

        # quadratic attention flops (Q@K and attn@V), only for full attention layers
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_full_attn_layers

        # GatedDeltaNet recurrence flops (state update + query, for all GDN layers)
        gdn_recurrence_flops = self._compute_gdn_recurrence_flops(text_config, tokens_sum, num_linear_attn_layers)

        # vit flops (Qwen3-VL ViT)
        images_seqlens = kargs.get("images_seqlens", None)
        if images_seqlens:
            vit_flops = self._estimate_qwen3_vit_flop(
                images_seqlens,
                getattr(self.config, "vision_config", None),
                lora_config=lora_config,
                freeze_vit=kargs.get("freeze_vit", False),
            )
        else:
            vit_flops = 0

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops + gdn_recurrence_flops + vit_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def estimate_flops(
        self,
        batch_seqlens,
        delta_time,
        lora_config: VeOmniLoraConfig | None = None,
        images_seqlens=None,
        freeze_vit: bool | None = None,
        gen_seqlens=None,
    ):
        """
        Estimate the FLOPS based on the number of valid tokens in the current batch and the time taken.

        Args:
            batch_seqlens (List[int]): A list where each element represents the number of valid tokens in the current batch.
            delta_time (float): The time taken to process the batch, in seconds.
            lora_config (VeOmniLoraConfig, optional): Effective LoRA configuration for supported
                Qwen models. FLOPs estimation reads only ``r``, ``target_modules``,
                ``target_parameters``, ``bias``, and ``moe_mode``. Vision FLOPs use a coarse
                config-level approximation: 2x for forward-only work, 4x when activation
                gradients are needed, and 6x for trainable weights/adapters. Unsupported LoRA
                configurations emit a warning and return zero FLOPs instead of interrupting training.
            images_seqlens (List[int], optional): Vision-token sequence lengths for multimodal
                models.
            freeze_vit (bool, optional): Whether the vision tower is frozen during full
                fine-tuning. Ignored when ``lora_config`` is provided.
            gen_seqlens (List[int], optional): Generation-tower token lengths, parallel to
                ``batch_seqlens``. Only the Cosmos3 mixture-of-towers DiT consumes it, because
                its second attention pass is not square: ``gen_seqlens[i]`` queries against
                ``batch_seqlens[i] + gen_seqlens[i]`` keys/values. Every other model leaves it
                empty, and then it is not forwarded.

        Returns:
            estimated_flops (float): The estimated FLOPS based on the input tokens and time.
            promised_flops (float): The expected FLOPS of the current device.
        """
        lora_validation_error = self._get_lora_validation_error(self.config.model_type, lora_config)
        if lora_validation_error is not None:
            if lora_validation_error not in self._warned_lora_validation_errors:
                logger.warning_rank0(
                    "LoRA FLOPs are unavailable: %s Returning zero FLOPs so training can continue.",
                    lora_validation_error,
                )
                self._warned_lora_validation_errors.add(lora_validation_error)
            return 0, get_device_flops()

        kwargs = {}
        if lora_config is not None:
            kwargs["lora_config"] = lora_config
        if images_seqlens is not None:
            kwargs["images_seqlens"] = images_seqlens
        if freeze_vit is not None:
            kwargs["freeze_vit"] = freeze_vit
        if gen_seqlens is not None:
            kwargs["gen_seqlens"] = gen_seqlens

        tokens_sum = sum(batch_seqlens)
        func = self.estimate_func.get(self.config.model_type, self._estimate_unknown_flops)
        estimated_flops = func(tokens_sum, batch_seqlens, delta_time, **kwargs)
        promised_flops = get_device_flops()
        return estimated_flops, promised_flops
