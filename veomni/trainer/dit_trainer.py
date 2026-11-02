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

import inspect
import math
import os
import pickle as pk
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Sequence, Union

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from veomni.data.multimodal.dit import (
    data_transform as _data_transform_register,  # noqa: F401  (import side effect: registers dit_offline / dit_online / minimax_h3_online transforms)
)
from veomni.data.multimodal.dit.packing import build_dit_length_fn

from ..arguments import DataArguments, ModelArguments, TrainingArguments, VeOmniArguments
from ..data import build_data_transform, build_dataloader
from ..data.data_collator import DataCollator
from ..distributed.parallel_state import get_parallel_state, use_parallel_state
from ..models import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY, build_config, build_foundation_model
from ..models.model_runtime import VeOmniModelRuntime
from ..ops.config import set_ops_config
from ..utils import helper
from ..utils.count_flops import COSMOS3_MOT_MODEL_TYPES, VeomniFlopsCounter
from ..utils.device import (
    get_device_type,
    synchronize,
)
from ..utils.dist_utils import all_reduce
from .base import BaseTrainer, VeOmniIter


if TYPE_CHECKING:  # the cache identity lives with the model; only the type is needed here
    from ..models.diffusers.cosmos3.cosmos3_condition.latent_cache import LatentRecipe


logger = helper.create_logger(__name__)

# Fields a condition config may carry whose value is really read by the *DiT*
# config. They are not condition-model knobs at all: the loss multiplier lives on
# the transformer (``modeling_cosmos3_transformer.py:900`` reads
# ``self.config.loss_scale``), because that is where the prediction exists, but the
# recipe's own config files have always spelled it under the condition model, so
# every run config copies it there. A run that set only the condition copy trained
# with the transformer's default (10.0) while its author believed otherwise: the
# loss it printed was 10x the flow-matching MSE, and the first 20-step comparison
# against the open-source baseline was off by exactly that factor.
_DIT_FIELDS_MIRRORED_FROM_CONDITION_CFG = ("loss_scale",)


def resolve_loss_scale(
    model_config: Optional[Dict[str, Any]],
    condition_model_cfg: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Mirror ``loss_scale`` from the condition config onto the DiT config.

    The ``Cosmos3MoTTransformerConfig`` is built from ``model.config_path`` (a
    shipped JSON that carries no ``loss_scale``) plus ``model.model_config``; the
    condition config is a *separate* object that the DiT never reads. So a value
    written under ``model.condition_model_cfg.loss_scale`` would be inert.

    The rule here keeps one knobs' worth of behaviour with two spellings:

    * only the condition copy set -> it is forwarded to the DiT config;
    * only ``model.model_config.loss_scale`` set -> used as is;
    * both set and equal -> used as is (the recipe's files carry both);
    * both set and different -> refused, because there is no defensible winner.

    Args:
        model_config: ``model.model_config`` (overrides for the foundation config).
        condition_model_cfg: ``model.condition_model_cfg`` (kwargs for the frozen
            condition model).

    Returns:
        The ``model_config`` to hand to ``build_config`` / ``build_foundation_model``.

    Raises:
        ValueError: The two spellings disagree.
    """
    resolved = dict(model_config or {})
    condition_cfg = condition_model_cfg or {}
    for name in _DIT_FIELDS_MIRRORED_FROM_CONDITION_CFG:
        condition_value = condition_cfg.get(name)
        if condition_value is None:
            continue
        if name in resolved and float(resolved[name]) != float(condition_value):
            raise ValueError(
                f"`model.model_config.{name}={resolved[name]}` and "
                f"`model.condition_model_cfg.{name}={condition_value}` disagree, and only the first one is "
                f"read by the DiT ({name} lives on the transformer config). Set one, or set both to the same "
                "value."
            )
        resolved[name] = condition_value
    return resolved


class OfflineEmbeddingSaver:
    """Persist per-sample condition rows for ``offline_training``, addressed by content.

    Two outputs, and the difference between them matters:

    * ``rank_{r}_shard_{i}.parquet`` -- the *positional* export. ``offline_training``
      replays it through ``dit_offline`` one row at a time, so the row order has to
      match the dataset order. Kept because the training-side transform reads a
      directory of parquet shards; it cannot re-derive a per-sample key without
      decoding the clip again.
    * ``latent_cache/<key[:2]>/<key>.pt`` -- the *content-addressed* store. The key
      is a SHA-256 over everything that determines the latent (clip pixels, prompt,
      geometry fed to the VAE, VAE dtype/weights/statistics, temporal chunking,
      code version, tokenizer template: see ``latent_cache.py``). A re-run of
      ``offline_embedding`` looks each sample up here first and skips the VAE on a
      hit, so the fill is resumable and idempotent. A directory whose
      ``latent_cache_manifest.json`` records a different recipe is refused outright
      rather than silently mixed.

    With ``recipe=None`` the saver is the plain positional writer: no manifest, no key
    store, no dedupe.
    """

    CACHE_DIRNAME = "latent_cache"

    def __init__(
        self,
        save_path: str,
        dataset_length: int = 0,
        shard_num: int = 1,
        max_shard=1000,
        recipe: Optional["LatentRecipe"] = None,
    ):
        from ..distributed.parallel_state import get_parallel_state

        self.dp_rank = get_parallel_state().dp_rank
        dp_size = get_parallel_state().dp_size
        if dp_size * shard_num > max_shard:
            shard_num = max_shard // dp_size
            logger.info_rank0(f"shard_num * dp_size must be smaller than max_shard, set shard_num = {shard_num}")
        self.shard_num = shard_num
        self.max_shard = max_shard
        self.index = 0
        self.buffer = []

        self.save_path = save_path
        self.dataset_length = dataset_length
        self.batch_len = math.ceil(dataset_length / self.shard_num)
        logger.info(f"Rank [{self.dp_rank}] save to [{self.save_path}] each batch_len [{self.batch_len}].")
        os.makedirs(self.save_path, exist_ok=True)
        self.rest_len = self.dataset_length

        self.recipe = recipe
        self.cache_hits = 0
        self.cache_misses = 0
        self._open_cache()
        # Resume the positional export where the directory left off, so a second
        # fill cannot overwrite the shards a first one already wrote.
        self.index = self._next_shard_index()

    # ── content-addressed store ───────────────────────────────────────────────

    @property
    def cache_dir(self) -> str:
        return os.path.join(self.save_path, self.CACHE_DIRNAME)

    def _shard_paths(self) -> list[str]:
        return sorted(name for name in os.listdir(self.save_path) if name.endswith(".parquet"))

    def _next_shard_index(self) -> int:
        """First unused shard index for this rank.

        The positional export is named ``rank_{dp_rank}_shard_{index}.parquet`` and
        ``_flush_shard`` increments ``index``, which starts at 0. A fill that is
        restarted into a non-empty directory (the documented resume path -- the
        content-addressed store is what makes it cheap) would otherwise rewrite the
        existing shards from index 0 and silently drop every row the first pass
        exported, even though the cache itself resumed correctly.
        """
        prefix = f"rank_{self.dp_rank}_shard_"
        used = []
        for name in self._shard_paths():
            if not (name.startswith(prefix) and name.endswith(".parquet")):
                continue
            frame = name[len(prefix) : -len(".parquet")]
            if frame.isdigit():
                used.append(int(frame))
        return max(used) + 1 if used else 0

    def _open_cache(self) -> None:
        """Validate (or start) the cache directory next to the shards.

        A directory holding shards but no manifest was filled by a version that
        keyed on position only: its rows cannot be attributed to a recipe, so
        appending content-keyed rows to it would mix two provenance regimes.
        Fail instead, and say how to proceed.
        """
        if self.recipe is None:
            return
        from ..models.diffusers.cosmos3.cosmos3_condition.latent_cache import (
            LatentCacheMismatch,
            verify_manifest,
            write_manifest,
        )

        shards = self._shard_paths()
        manifest = verify_manifest(self.save_path, self.recipe)
        if manifest is None and shards:
            raise LatentCacheMismatch(
                f"{self.save_path!r} already holds {len(shards)} positional shard(s) but no "
                "latent_cache_manifest.json: they were written by a version that identified rows by "
                "position only, so nothing records which recipe produced them. Point "
                "data.offline_embedding_save_dir at a fresh directory (or delete the old shards) so the "
                "cache is filled under a known recipe."
            )
        os.makedirs(self.cache_dir, exist_ok=True)
        write_manifest(self.save_path, self.recipe, dp_rank=self.dp_rank, dataset_length=self.dataset_length)
        logger.info_rank0(
            f"Latent cache at [{self.cache_dir}] keyed by recipe {self.recipe.digest()[:16]} "
            f"(schema={self.recipe.schema})."
        )

    def _cache_path(self, cache_key: str) -> str:
        return os.path.join(self.cache_dir, cache_key[:2], f"{cache_key}.pt")

    def has(self, cache_key: str) -> bool:
        """Whether this content key is already cached."""
        return bool(cache_key) and os.path.isfile(self._cache_path(cache_key))

    def load(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """The cached condition row for ``cache_key``, or ``None``."""
        path = self._cache_path(cache_key)
        if not os.path.isfile(path):
            return None
        item = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(item, dict):
            raise RuntimeError(f"latent cache entry {path} is a {type(item).__name__}, expected a dict")
        return item

    def _store(self, cache_key: str, save_item: Dict[str, Any]) -> None:
        path = self._cache_path(cache_key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Write-then-rename: an interrupted fill must not leave a half-written
        # entry that a later run would count as a hit.
        temporary = f"{path}.{os.getpid()}.tmp"
        torch.save(self._cpu_recursive(dict(save_item)), temporary)
        os.replace(temporary, path)

    # ── positional export ─────────────────────────────────────────────────────

    @staticmethod
    def _cpu_recursive(obj):
        """Move tensors to CPU recursively, leave other types unchanged."""
        if isinstance(obj, torch.Tensor):
            return obj.cpu()
        if isinstance(obj, dict):
            return {k: OfflineEmbeddingSaver._cpu_recursive(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(OfflineEmbeddingSaver._cpu_recursive(v) for v in obj)
        return obj

    def to_save_bytes(self, save_item: Dict[str, torch.Tensor]):
        converted_dict = {}
        for key in list(save_item.keys()):
            converted_dict[key] = pk.dumps(self._cpu_recursive(save_item[key]))
            del save_item[key]
        return converted_dict

    def _append_item(self, save_item: Dict[str, torch.Tensor], cache_key: Optional[str] = None):
        if self.rest_len > 0:  # 多余的dummy data buffer 不保存
            converted = self.to_save_bytes(save_item)
            if cache_key:
                # The key travels with the row so a reader (and a human) can tell
                # which cached latent a positional entry came from.
                converted["latent_cache_key"] = pk.dumps(str(cache_key))
            self.buffer.append(converted)
            self.rest_len -= 1

    def _flush_shard(self) -> None:
        ds = Dataset.from_list(self.buffer)
        ds.to_parquet(os.path.join(self.save_path, f"rank_{self.dp_rank}_shard_{self.index}.parquet"))
        self.buffer = []
        self.index += 1

    def save(self, save_item, cache_key: Optional[str] = None):
        """Append one sample: store it by content (when keyed) and export it.

        ``cache_key`` is optional so a caller that has no identity (or a fake
        condition model in a unit test) keeps the pre-existing behaviour.
        """
        if cache_key and self.recipe is not None and not self.has(cache_key):
            self._store(cache_key, save_item)
            self.cache_misses += 1
        self._append_item(save_item, cache_key=cache_key)
        if len(self.buffer) >= self.batch_len:
            self._flush_shard()

    def save_last(self):
        if len(self.buffer) > 0:
            self._flush_shard()
        if self.recipe is not None:
            from ..models.diffusers.cosmos3.cosmos3_condition.latent_cache import write_manifest

            write_manifest(
                self.save_path,
                self.recipe,
                dp_rank=self.dp_rank,
                dataset_length=self.dataset_length,
                saved_samples=self.dataset_length - self.rest_len,
                cache_hits=self.cache_hits,
                cache_misses=self.cache_misses,
            )
            logger.info_rank0(
                f"Rank [{self.dp_rank}] latent cache: {self.cache_misses} new sample(s) encoded, "
                f"{self.cache_hits} reused from {self.cache_dir}."
            )


@dataclass
class DiTDataCollator(DataCollator):
    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        batch = defaultdict(list)

        # Fill keys missing from a sample with None so every column stays aligned with its samples.
        keys = dict.fromkeys(key for feature in features for key in feature)
        for feature in features:
            for key in keys:
                batch[key].append(feature.get(key))

        return batch


@dataclass
class DiTModelArguments(ModelArguments):
    condition_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to condition model."},
    )
    condition_model_cfg: Optional[Dict] = field(
        default_factory=dict,
        metadata={"help": "Config for condition model."},
    )


@dataclass
class DiTDataArguments(DataArguments):
    supports_torch_compile = False
    mm_configs: Optional[Dict] = field(
        default_factory=dict,
        metadata={"help": "Config for multimodal input."},
    )
    offline_embedding_save_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save offline embeddings."},
    )
    data_transform: Optional[str] = field(
        default="dit_online",
        metadata={
            "help": "Override the DATA_TRANSFORM_REGISTRY transform name. "
            "When None, DiTTrainer picks dit_offline/dit_online by training_task."
        },
    )
    pack_mode: Literal["token_budget", "mbs"] = field(
        default="mbs",
        metadata={
            "help": "How many samples share one micro batch (they are still run as one packed sequence). "
            "'mbs': a fixed `train.micro_batch_size` samples, so `train.global_batch_size` counts samples. "
            "'token_budget': greedily pack samples until `max_num_tokens_after_packing` tokens, so "
            "`train.micro_batch_size` counts *micro batches* (keep it 1) and `train.global_batch_size` counts "
            "micro batches -- this is the open-source PackingDataLoader semantics. "
            "See veomni/data/multimodal/dit/packing.py."
        },
    )
    max_num_tokens_after_packing: Optional[int] = field(
        default=None,
        metadata={
            "help": "Token budget of one packed micro batch when pack_mode='token_budget'. None falls back to "
            "`data.max_seq_len`. The open-source vision_sft_nano recipe uses 45056."
        },
    )
    pack_buffer_size: int = field(
        default=1,
        metadata={
            "help": "How many samples the token-budget batcher must hold before it is even allowed to emit a "
            "packed micro batch (`data.dyn_bsz_buffer_size` for DiT). The token budget is the real gate; this "
            "sample floor only widens the greedy scan. 1 (the default) means 'emit as soon as the budget is "
            "met', which is what the reference PackingDataLoader does. Keep it small for video: the buffer "
            "holds *decoded clips*, so 200 (the shared text default) would pin ~200 clips in host RAM and "
            "read 200 clips before step 1."
        },
    )
    log_sample: bool = field(
        default=True,
        metadata={"help": "Whether to print the first micro batch example to the log."},
    )


@dataclass
class DiTTrainingArguments(TrainingArguments):
    training_task: Literal["offline_training", "online_training", "offline_embedding"] = field(
        default="online_training",
        metadata={
            "help": "Training task. offline_training: training offline embedded data. "
            "online_training: training raw data online. offline_embedding: embedding raw data."
        },
    )


@dataclass
class VeOmniDiTArguments(VeOmniArguments):
    model: DiTModelArguments = field(default_factory=DiTModelArguments)
    data: DiTDataArguments = field(default_factory=DiTDataArguments)
    train: DiTTrainingArguments = field(default_factory=DiTTrainingArguments)


class DiTModelRuntime(VeOmniModelRuntime):
    """A DiT and the frozen condition model that feeds it.

    ``offline_embedding`` runs the condition model only, so the parallelize and
    optimizer steps are overridden to no-ops rather than the build sequence
    itself, which would then have to be kept in step with the base by hand.
    """

    condition_model: PreTrainedModel = None

    def _build_model(self):
        logger.info_rank0("Build model")
        args: DiTModelArguments = self.args
        # Apply ops config eagerly so the condition model (built below via
        # ``model_class._from_config``, not ``build_foundation_model``) sees a
        # populated ops singleton / LOSS_MAPPING. ``build_foundation_model``
        # below will re-apply the same config — that call is idempotent.
        set_ops_config(args.ops_implementation)
        # The DiT config is the one that owns `loss_scale`; fold the recipe's
        # condition-model spelling of it in first so both halves of the model are
        # built from the same value (see `resolve_loss_scale`).
        model_config = resolve_loss_scale(args.model_config, args.condition_model_cfg)
        dit_config = build_config(args.config_path, **model_config)
        # `PretrainedConfig.__init__` stores unknown kwargs as plain attributes, so a
        # mirrored field the DiT does not declare would be accepted and silently
        # ignored -- the same failure this mirroring exists to remove. Check the
        # class signature (not the instance) to catch it.
        declared = set(inspect.signature(type(dit_config).__init__).parameters)
        for name in _DIT_FIELDS_MIRRORED_FROM_CONDITION_CFG:
            if name in model_config and name not in declared:
                raise ValueError(
                    f"`model.condition_model_cfg.{name}` was forwarded to the {dit_config.model_type} config, "
                    f"which does not declare `{name}` (PretrainedConfig would store it silently and nothing "
                    "would read it). Remove it from `model.condition_model_cfg`."
                )
        self.model_config = dit_config
        logger.info_rank0(f"Detected DiT model type: {dit_config.model_type}.")
        loss_scale = getattr(dit_config, "loss_scale", None)
        if loss_scale is not None:
            # One line per run: the printed `dm_loss` is the flow-matching MSE times
            # this, and a silent 10x is exactly the class of bug this log prevents.
            logger.info_rank0(f"Flow-matching loss_scale = {float(loss_scale)} (dm_loss = MSE x loss_scale).")
        self._build_condition_model(dit_config.condition_model_type)
        if self.train_args.training_task != "offline_embedding":
            logger.info_rank0(f"Task: {self.train_args.training_task}, prepare dit model.")
            self.model = build_foundation_model(
                config_path=args.config_path,
                weights_path=args.model_path,
                torch_dtype="float32" if args.accelerator.fsdp_config.mixed_precision.enable else "bfloat16",
                init_device=args.accelerator.init_device,
                ops_implementation=args.ops_implementation,
                config_kwargs=model_config,
            )
            self.model_config = getattr(self.model, "config", None)
        else:
            self.model = None
            logger.info_rank0(f"Task: {self.train_args.training_task}, dit model is not prepared.")

    def _build_condition_model(self, condition_model_type: str) -> None:
        args: DiTModelArguments = self.args
        config_class = MODEL_CONFIG_REGISTRY[condition_model_type]()
        condition_cfg = config_class.from_pretrained(
            args.condition_model_path,
            seed=self.train_args.seed,  # seed for randn noise and scheduler
            **args.condition_model_cfg,
        )
        model_class = MODELING_REGISTRY[condition_model_type]()
        if self.train_args.training_task == "offline_training":
            self.condition_model = model_class._from_config(condition_cfg, meta_init=True)
            logger.info_rank0("Condition model loaded with empty weights.")
        else:
            self.condition_model = model_class._from_config(condition_cfg)
            self.condition_model.to(get_device_type())
            logger.info_rank0("Condition model loaded.")

    def _freeze_model_module(self):
        self.condition_model.requires_grad_(False)

        if self.train_args.training_task != "offline_embedding":
            super()._freeze_model_module()

    def extra_state(self) -> dict[str, Any]:
        """Model-bound state: the condition model's noise/timestep generator."""
        rng_state_dict = getattr(self.condition_model, "rng_state_dict", None)
        if (
            self.condition_model is not None
            and rng_state_dict is None
            and getattr(self.condition_model, "generator", None) is not None
        ):
            logger.warning_rank0(
                "Condition model owns a ``generator`` but exposes no ``rng_state_dict``; "
                "its noise/timestep stream will not be restored across a resume."
            )
        return {} if rng_state_dict is None else {"condition_model_rng_state": rng_state_dict()}

    def load_extra_state(self, extra_state: dict[str, Any]) -> None:
        condition_model_rng_state = extra_state.get("condition_model_rng_state")
        if condition_model_rng_state is None:
            return
        loader = getattr(self.condition_model, "load_rng_state_dict", None)
        if loader is None:
            logger.warning_rank0(
                "Checkpoint carries condition-model RNG state but the model cannot restore it; "
                "the resumed run may replay its initial noise stream."
            )
        else:
            loader(condition_model_rng_state)

    def _build_parallelized_model(self) -> None:
        """``offline_embedding`` builds no DiT, so there is nothing to wrap."""
        if self.train_args.training_task != "offline_embedding":
            super()._build_parallelized_model()

    def _build_model_assets(self) -> None:
        """A DiT reads latents, not text — there is no preprocessor to load.

        That leaves the config, and only a run that trains the DiT has one worth
        exporting: ``offline_embedding`` builds no DiT at all, it writes cached
        condition embeddings, which need no model sidecars.
        """
        self.model_assets = [self.model_config] if self.train_args.training_task != "offline_embedding" else []

    def _build_optimizer(self, param_groups=None) -> None:
        """``offline_embedding`` trains nothing, so there is nothing to optimize."""
        if self.train_args.training_task != "offline_embedding":
            super()._build_optimizer(param_groups=param_groups)

    def load(self) -> None:
        if self.train_args.training_task != "offline_embedding":
            super().load()

    def save_dcp(self, state) -> None:
        if self.train_args.training_task != "offline_embedding":
            super().save_dcp(state)

    def save_hf_or_lora(self, state, stage: str = "step_end") -> None:
        if self.train_args.training_task != "offline_embedding":
            super().save_hf_or_lora(state, stage=stage)

    def save_model_assets(self) -> None:
        if self.train_args.training_task != "offline_embedding":
            super().save_model_assets()


class Cosmos3EnvironMeter(helper.EnvironMeter):
    """``EnvironMeter`` for a Cosmos3 mixture-of-towers DiT.

    Why a subclass is needed at all: ``DiTTrainer`` is not a ``BaseTrainer`` -- it
    borrows one (``self.base``) for the callbacks, so ``EnvironMeterCallback`` builds
    the *generic* meter, which prices a step from ``batch_seqlens`` alone. Cosmos3
    needs two lengths per sample (the prompt the reasoner tower reads, and the latent
    patch rows the diffusion tower reads) plus the shape of its two attention passes,
    neither of which ``batch_seqlens`` can express. Everything else -- the DP
    all-reduce, the device counters, the checkpoint state -- stays in the base class.

    Installed only for MoT runs (``DiTTrainer._build_flops_meter``): a Wan or
    MiniMax-H3 run keeps the meter it had, so their metrics are unchanged.
    """

    def __init__(self, config: "PretrainedConfig", global_batch_size: int, **kwargs) -> None:
        super().__init__(config, global_batch_size, **kwargs)
        # Running total of generation-tower tokens, kept beside the base class's
        # ``consume_tokens`` (reasoner tower) so the reported totals cover both streams.
        self.consume_gen_tokens = 0
        # Decided once, at construction: reporting MFU is a diagnostic, so a config the
        # FLOPs counter cannot price must skip the metric rather than take the run down
        # (and must not report a plausible-looking 0% either).
        self.flops_estimable = VeomniFlopsCounter(config).supports_config()
        if not self.flops_estimable:
            logger.warning_rank0(
                "No FLOPs estimator for model_type=%r; MFU is not reported for this run (training is unaffected).",
                getattr(config, "model_type", None),
            )

    @staticmethod
    def seqlens(micro_batch: Dict[str, Any], config: "PretrainedConfig") -> tuple[list[int], list[int]]:
        """``(reasoner, generation)`` token lengths per sample of one micro batch.

        Expected *after* ``Cosmos3ConditionModel.process_condition``: that is the first
        point where the exact facts exist -- ``input_ids`` are the tokenized prompts and
        ``hidden_states`` are the latent grids the patchifier is about to flatten. The
        raw collated batch carries prompt strings and decoded pixel frames instead, and
        both lengths would have to be estimated from characters and frame geometry.

        Returns ``([], [])`` for a batch in any other shape, so the pre-condition hook
        contributes nothing and cannot double count.
        """
        hidden_states = micro_batch.get("hidden_states")
        input_ids = micro_batch.get("input_ids")
        if hidden_states is None or input_ids is None:
            return [], []

        # Same arithmetic as ``cosmos3_transformer.patch_row_count``: Cosmos3 never
        # patches the time axis, so one latent frame contributes (H // p) * (W // p)
        # patch rows and a clip contributes ``frames`` of those.
        patch = int(getattr(config, "latent_patch_size", 1) or 1)
        und_lengths = [int(tokens.numel()) for tokens in input_ids]
        gen_lengths = []
        for latent in hidden_states:
            _, frames, height, width = (int(size) for size in latent.shape)
            gen_lengths.append(frames * (height // patch) * (width // patch))
        return und_lengths, gen_lengths

    def add(self, micro_batch: Union[Dict[str, Any], list[Dict[str, Any]]]) -> None:
        """Append one micro batch's reasoner and generation lengths."""
        batches = micro_batch if isinstance(micro_batch, list) else [micro_batch]
        for batch in batches:
            und_lengths, gen_lengths = self.seqlens(batch, self.config)
            self.batch_seqlens.extend(und_lengths)
            self.gen_seqlens.extend(gen_lengths)

    def step(self, delta_time: float, global_step: int, **kwargs) -> Dict[str, Any]:
        """Add the per-step FLOPs and the two-stream token counters to the base metrics."""
        # ``super().step`` resets both lists, so take the local sums first.
        und_local, gen_local = sum(self.batch_seqlens), sum(self.gen_seqlens)
        metrics = super().step(delta_time, global_step=global_step, **kwargs)

        # ``super().step`` prices ``batch_seqlens``; a step's token throughput has to
        # cover both streams, and so does the running total a resume restores. The
        # reduction runs unconditionally so the collective is executed on every rank:
        # its *result* is what decides, not a rank-local condition.
        und_tokens, gen_tokens = all_reduce((und_local, gen_local), op="sum", group=self.parallel_state.dp_group)
        if gen_tokens:
            total_tokens = und_tokens + gen_tokens
            self.consume_gen_tokens += gen_tokens
            metrics.update(
                {
                    "tokens_per_second(M)": total_tokens / delta_time / 1e6,
                    "gen_tokens_per_second(M)": gen_tokens / delta_time / 1e6,
                    "training/avg_effective_len": (
                        total_tokens / self.global_batch_size if self.global_batch_size else 0
                    ),
                    "consume_tokens(M)": (self.consume_tokens + self.consume_gen_tokens) / 1e6,
                    "consume_tokens(B)": (self.consume_tokens + self.consume_gen_tokens) / 1e9,
                }
            )

        # ``flops_achieved(T)`` is TFLOP/s; the per-step work is that times the step time.
        metrics["flops_per_step(T)"] = metrics["flops_achieved(T)"] * delta_time
        if not self.flops_estimable:
            # Skip the metric, keep the key: a reader (wandb, a doc) sees "unknown"
            # instead of a hard 0 that looks like a fully stalled device.
            metrics.update({"flops_achieved(T)": None, "flops_per_step(T)": None, "mfu": None})
        return metrics

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        state_dict["consume_gen_tokens"] = self.consume_gen_tokens
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.consume_gen_tokens = state_dict.get("consume_gen_tokens", 0)


class DiTTrainer:
    """
    DiT Trainer merging BaseTrainer infrastructure with DiT-specific model setup.
    Reuses BaseTrainer's callbacks, dataloader building (with MainCollator/DiTConcatCollator),
    and training loop; overrides model building and forward pass.
    """

    training_task: Literal["offline_training", "online_training", "offline_embedding"]
    offline_embedding_save_dir: str = None
    offline_embedding_saver: OfflineEmbeddingSaver = None
    # The per-sample token measure the token-budget batcher used, kept so the
    # pack-fill diagnostic prices the pack the same way (see `_log_pack_fill`).
    _pack_length_fn: Optional[Any] = None
    _pack_fill_logged: bool = False
    # The MoT-aware meter, or None when this DiT keeps the generic one (see
    # ``_build_flops_meter``).
    flops_meter: Optional[Cosmos3EnvironMeter] = None

    def __init__(self, args: VeOmniDiTArguments):
        if args.train.channel_loss.enable:
            raise ValueError(
                "train.channel_loss is only supported by causal-LM trainers; DiTTrainer uses diffusion objectives."
            )
        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.args = args

        # rewrite _setup, setup arguments for dit training.
        # ``base._setup`` registers ParallelState; DiT then recomputes
        # dataloader_batch_size from ``dp_size``.
        self._setup()
        # Builds the condition model and, unless this is an embedding-only run,
        # the DiT itself along with its optimizer.
        self.base.model = self._build_model_runtime()
        self.base.LOG_SAMPLE = args.data.log_sample

        with use_parallel_state(self.base.model.parallel_state):
            # rewrite _build_data_transform, build data transform for offline or online dit data
            self._build_data_transform()
            # rewrite _build_dataset, init offline_embedding_saver after build_dataset
            self._build_dataset()
            # Do not use maincollator in dit training
            # self.base._build_collate_fn()
            # rewrite _build_dataloader, build dataloader only on sp_rank_0 to save memory
            self._build_dataloader()

        if self.training_task != "offline_embedding":
            self.base._build_lr_scheduler()
            self.base._build_training_context(self.base.model)

        self.base._init_callbacks()
        # ``_init_callbacks`` just built the generic meter; replace it for MoT models so
        # the FLOPs/MFU metrics describe the two towers. Callbacks resolve the meter on
        # the trainer at every hook, so this swap is the whole installation.
        self.flops_meter = self._build_flops_meter()

    def _build_flops_meter(self) -> Optional[Cosmos3EnvironMeter]:
        """Install :class:`Cosmos3EnvironMeter`, or leave the shared meter in place.

        Only the Cosmos3 MoT layout is priced here; every other DiT (Wan, MiniMax-H3)
        keeps the meter ``EnvironMeterCallback`` built, byte for byte.
        """
        model_config = getattr(self.base.model, "model_config", None)
        if getattr(model_config, "model_type", None) not in COSMOS3_MOT_MODEL_TYPES:
            return None
        meter = Cosmos3EnvironMeter(
            config=model_config,
            global_batch_size=self.base.args.train.global_batch_size,
            empty_cache_steps=self.base.args.train.empty_cache_steps,
            gc_steps=self.base.args.train.gc_steps,
            parallel_state=getattr(self.base.model, "parallel_state", None),
        )
        self.base.environ_meter = meter
        return meter

    @property
    def condition_model(self) -> PreTrainedModel:
        return self.base.model.condition_model

    def _build_model_runtime(self) -> DiTModelRuntime:
        """Build (and own) this job's DiT. Override to swap in another runtime."""
        return DiTModelRuntime(self.base.args.model, "base", train=self.base.args.train)

    def _setup(self):
        args: VeOmniDiTArguments = self.base.args
        # registers ParallelState("base") before seed
        self.base.device = self.base._setup(args)
        # `data.pack_mode` decides *how many samples* share a micro batch:
        #   "mbs"           -> the fixed count the collator chunks to (below);
        #   "token_budget"  -> the token-budget dataloader (dyn bsz), which is the
        #                      open-source PackingDataLoader semantics. It also
        #                      needs a DiT-specific per-sample length function, passed
        #                      in `_build_dataloader`; the shared fallbacks would price
        #                      a DiT sample as one token.
        # The field is a `Literal`, but `Literal` is only enforced by argparse: a YAML
        # value bypasses the parser and reaches the dataclass unchecked.
        if args.data.pack_mode not in ("token_budget", "mbs"):
            raise ValueError(f"data.pack_mode={args.data.pack_mode!r} is not supported; use 'token_budget' or 'mbs'.")
        args.train.dyn_bsz = args.data.pack_mode == "token_budget"
        if args.train.dyn_bsz and args.train.micro_batch_size != 1:
            raise ValueError(
                "data.pack_mode='token_budget' counts micro batches, not samples: the packed batch is the unit "
                f"(the open-source recipe runs micro_batch_size=1 with grad_accum), so micro_batch_size must be 1, "
                f"got {args.train.micro_batch_size}. Use data.pack_mode='mbs' to pack a fixed sample count per "
                "micro batch."
            )
        # dataloader_batch_size was computed in __post_init__ from the config's
        # dyn_bsz, which pack_mode may just have overridden.
        if args.train.dyn_bsz:
            # "main": the dyn-bsz wrapper consumes one raw item per step, so the
            # DataLoader's own batch is 1. "worker": each item IS an assembled
            # micro batch, so the DataLoader must gather `num_micro_batch` of them
            # (that is what `TrainingArguments.__post_init__` computes for text).
            if args.train.dyn_bsz_runtime == "main":
                args.train.dataloader_batch_size = 1
            else:
                args.train.dataloader_batch_size = (
                    args.train.global_batch_size // get_parallel_state().dp_size // args.train.micro_batch_size
                )
        else:
            args.train.dataloader_batch_size = args.train.global_batch_size // get_parallel_state().dp_size
        if args.train.training_task == "offline_embedding":
            assert args.data.datasets_type == "mapping", "Datasets type must be mapping for offline embedding."
            if args.data.offline_embedding_save_dir is None:
                self.offline_embedding_save_dir = f"{args.data.train_path}_offline"
            else:
                self.offline_embedding_save_dir = args.data.offline_embedding_save_dir

            args.data.dataloader.drop_last = False
            args.data.dataloader.shuffle = False
            args.train.checkpoint.save_epochs = 0
            args.train.checkpoint.save_hf_weights = False
            # Keep one microbatch per step to limit video broadcast memory; embedding needs no accumulation.
            args.train.global_batch_size = args.train.micro_batch_size * get_parallel_state().dp_size
            args.train.dataloader_batch_size = args.train.micro_batch_size
            args.train.gradient_accumulation_steps = 1
            logger.info_rank0(
                f"Task offline_embedding. Drop last: {args.data.dataloader.drop_last}, shuffle: {args.data.dataloader.shuffle}"
            )
            args.train.num_train_epochs = 1

        self.training_task = args.train.training_task

    def _build_data_transform(self):
        args: VeOmniDiTArguments = self.base.args
        logger.info(f"args.data.data_transform: {args.data.data_transform}, training_task: {self.training_task}")
        if self.training_task == "offline_training":
            self.base.data_transform = build_data_transform("dit_offline")
        else:
            self.base.data_transform = build_data_transform(
                args.data.data_transform,
                **args.data.mm_configs,
            )

    def _build_dataset(self):
        args: VeOmniDiTArguments = self.base.args
        self.base._build_dataset()
        if get_parallel_state().sp_enabled and get_parallel_state().sp_rank != 0:
            self.base.train_dataset = None

        if self.training_task == "offline_embedding":
            if not get_parallel_state().sp_enabled or get_parallel_state().sp_rank == 0:
                dp_rank = get_parallel_state().dp_rank
                dp_size = get_parallel_state().dp_size
                dataset_len = len(self.base.train_dataset)
                base_count = dataset_len // dp_size
                extra = dataset_len % dp_size
                valid_data_length = base_count + (1 if dp_rank < extra else 0)
                logger.info(f"Rank {args.train.global_rank} data length to save: {valid_data_length}")
                self.offline_embedding_saver = OfflineEmbeddingSaver(
                    save_path=self.offline_embedding_save_dir,
                    dataset_length=valid_data_length,
                    recipe=self._latent_cache_recipe(),
                )
                padded_len = (
                    math.ceil(self.base.train_dataset.data_len / args.train.global_batch_size)
                    * args.train.global_batch_size
                )
                # Keep the source length intact for MappingDataset's repeated-index mapping.
                self.base.train_dataset = torch.utils.data.Subset(self.base.train_dataset, range(padded_len))
                args._train_steps = padded_len // dp_size // args.train.dataloader_batch_size
                self.base.train_steps = args.train_steps
            else:
                self.offline_embedding_saver = None

        elif self.training_task == "offline_training":
            # The replay itself is positional, so the only cheap guard against a
            # stale cache is the directory-level recipe check: the manifest records
            # what produced these latents, and a different recipe (other data,
            # dtype, weights, code, template) means they are not the latents this
            # run would compute.
            self._verify_offline_latent_cache()

        # Sync _train_steps across the SP group AFTER padding so every rank
        # agrees on step count (required to avoid deadlocks in broadcast_object_list).
        if get_parallel_state().sp_enabled:
            steps_t = torch.zeros(1, dtype=torch.int64, device=torch.device(get_device_type()))
            if get_parallel_state().sp_rank == 0:
                steps_t[0] = args._train_steps
            dist.broadcast(
                steps_t,
                src=dist.get_global_rank(get_parallel_state().sp_group, 0),
                group=get_parallel_state().sp_group,
            )
            args._train_steps = int(steps_t.item())
            self.base.train_steps = args.train_steps

    def _latent_cache_recipe(self):
        """The run's latent-encoding recipe, or ``None`` when the model has none.

        Duck-typed on purpose: a DiT condition model that does not implement the
        content-addressed identity (a Wan/MiniMax-H3 condition model, or a test
        double) keeps writing a purely positional cache, exactly as before.
        """
        builder = getattr(self.condition_model, "latent_cache_recipe", None)
        return builder() if callable(builder) else None

    def _verify_offline_latent_cache(self):
        """Refuse an ``offline_training`` cache that a different recipe produced."""
        recipe = self._latent_cache_recipe()
        if recipe is None:
            return
        from ..models.diffusers.cosmos3.cosmos3_condition.latent_cache import verify_manifest

        cache_dir = self.base.args.data.train_path
        manifest = verify_manifest(cache_dir, recipe)
        if manifest is None:
            logger.warning_rank0(
                "The latent cache in %r carries no latent_cache_manifest.json, so this run cannot prove "
                "it was produced by the current encoding recipe (vae_dtype, VAE weights, code version, "
                "tokenizer template, ...). Training will proceed -- the replay is positional -- but "
                "re-filling the cache with train.training_task='offline_embedding' would make it "
                "verifiable.",
                cache_dir,
            )
        else:
            logger.info_rank0(
                f"Latent cache {cache_dir!r} verified against recipe {recipe.digest()[:16]} "
                f"({manifest.get('saved_samples', '?')} sample(s))."
            )

    def _build_dataloader(self):
        """Build the DiT dataloader.
        ``pack_mode="mbs"`` keeps a fixed sample count per micro batch
        (``dyn_bsz=False``). ``pack_mode="token_budget"`` packs greedily up to
        ``max_num_tokens_after_packing`` tokens, which is VeOmni's dyn-bsz path with a
        DiT length function: ``micro_batch_size`` is then the number of *packed* micro
        batches, so the budget handed to ``build_dataloader`` (it uses
        ``micro_batch_size * max_seq_len``) is the packing target itself.
        """
        args = self.base.args
        token_budget = args.data.max_num_tokens_after_packing or args.data.max_seq_len
        if args.train.dyn_bsz and args.data.max_num_tokens_after_packing is None:
            logger.warning_rank0(
                "data.pack_mode='token_budget' without data.max_num_tokens_after_packing: "
                f"using data.max_seq_len={args.data.max_seq_len} as the token budget."
            )
        length_fn = (
            build_dit_length_fn(
                args.data.mm_configs,
                model_config=getattr(self.base.model, "model_config", None),
            )
            if args.train.dyn_bsz
            else None
        )
        # Kept so `_log_pack_fill` prices the packs with the same measure the
        # batcher used, instead of a second, drifting copy of the arithmetic.
        self._pack_length_fn = length_fn
        if args.train.dyn_bsz:
            # The sample floor is not the packing gate -- the token budget is. The
            # shared default (200) is a *text* heuristic: for DiT it would hold 200
            # decoded clips (tens of GB of host RAM) and read 200 clips before the
            # first pack. See `DiTDataArguments.pack_buffer_size`.
            dyn_bsz_buffer_size = args.data.pack_buffer_size
            if dyn_bsz_buffer_size < 1:
                raise ValueError(f"data.pack_buffer_size must be >= 1, got {dyn_bsz_buffer_size}.")
            logger.info_rank0(
                f"Token-budget packing: budget={token_budget} tokens/micro batch, "
                f"buffer floor={dyn_bsz_buffer_size} sample(s) -> "
                f"{args.train.dataloader_batch_size} packed micro batch(es) per rank per step."
            )
        else:
            dyn_bsz_buffer_size = args.data.dyn_bsz_buffer_size
        if not get_parallel_state().sp_enabled or get_parallel_state().sp_rank == 0:
            self.base.train_dataloader = build_dataloader(
                dataloader_type=args.data.dataloader.type,
                dataset=self.base.train_dataset,
                micro_batch_size=args.train.micro_batch_size,
                global_batch_size=args.train.global_batch_size,
                dataloader_batch_size=args.train.dataloader_batch_size,
                max_seq_len=token_budget if args.train.dyn_bsz else args.data.max_seq_len,
                train_steps=args.train_steps,
                bsz_warmup_ratio=args.train.bsz_warmup_ratio,
                bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
                dyn_bsz=args.train.dyn_bsz,
                dyn_bsz_runtime=args.train.dyn_bsz_runtime,
                dyn_bsz_count_mode=args.train.dyn_bsz_count_mode,
                dyn_bsz_length_fn=length_fn,
                dyn_bsz_physical_overflow_ratio=args.train.dyn_bsz_physical_overflow_ratio,
                dyn_bsz_buffer_size=dyn_bsz_buffer_size,
                num_workers=args.data.dataloader.num_workers,
                drop_last=args.data.dataloader.drop_last,
                pin_memory=args.data.dataloader.pin_memory,
                prefetch_factor=args.data.dataloader.prefetch_factor,
                persistent_workers=args.data.dataloader.persistent_workers,
                in_order=args.data.dataloader.in_order,
                seed=args.train.seed,
                shuffle=args.data.dataloader.shuffle,
                collate_fn=DiTDataCollator(),
                save_steps=args.train.checkpoint.save_steps,
            )
        else:
            self.base.train_dataloader = None

    def _log_pack_fill(self, micro_batches: Any) -> None:
        """Report what each packed micro batch actually holds -- once.

        Why this is not optional: the packing can be wired correctly and still
        under-fill (a length fn that prices a clip at its pre-resize geometry, a
        budget that never reaches the dataloader, a sample floor the stream cannot
        satisfy). From the step time alone that is invisible -- it only shows up as
        a low MFU, which is exactly how this went unnoticed at
        ``1.83 s/step`` with one clip per pack. One line, on the first step, makes
        the pack size a fact in the log.
        """
        length_fn = getattr(self, "_pack_length_fn", None)
        if length_fn is None or getattr(self, "_pack_fill_logged", False):
            return
        budget = self.base.args.data.max_num_tokens_after_packing or self.base.args.data.max_seq_len
        try:
            samples_per_pack = [len(next(iter(batch.values()))) for batch in micro_batches]
            tokens_per_pack = [
                sum(int(length_fn(sample)) for sample in self._unpack_dict_of_list(batch)) for batch in micro_batches
            ]
        except Exception as exc:  # a diagnostic must never take the run down
            logger.warning_rank0(f"Could not price the packed micro batches for logging: {exc}")
            self._pack_fill_logged = True
            return
        self._pack_fill_logged = True
        fill = [tokens / budget for tokens in tokens_per_pack] if budget else [0.0]
        logger.info_rank0(
            "Packed micro batches on this rank's first step: "
            + ", ".join(
                f"[{samples} clips, {tokens} tokens, {ratio:.0%} of {budget}]"
                for samples, tokens, ratio in zip(samples_per_pack, tokens_per_pack, fill)
            )
            + "."
        )
        if max(samples_per_pack, default=0) <= 1 and len(micro_batches) > 0:
            logger.warning_rank0(
                f"Each packed micro batch holds only {max(samples_per_pack, default=0)} sample(s) against a "
                f"budget of {budget} tokens: packing is not filling the budget, so every block's "
                "all-gather is being exposed for almost no compute per byte. Check that "
                "data.pack_mode='token_budget', that data.max_num_tokens_after_packing reaches the "
                "dataloader, and that the per-sample token measure is not over-pricing a clip."
            )

    def on_train_begin(self):
        self.base.on_train_begin()

    def on_train_end(self):
        self.base.on_train_end()

    def on_epoch_begin(self):
        self.base.on_epoch_begin()

    def on_epoch_end(self):
        self.base.on_epoch_end()

    def on_step_begin(self, micro_batches=None):
        self.base.on_step_begin(micro_batches=micro_batches)

    def on_step_end(self, loss=None, loss_dict=None, grad_norm=None, aux_metrics=None):
        self.base.on_step_end(loss=loss, loss_dict=loss_dict, grad_norm=grad_norm, aux_metrics=aux_metrics)

    def preforward(self, micro_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess micro batches before forward pass."""

        def _to_device(v: Any) -> Any:
            if isinstance(v, torch.Tensor):
                return v.to(self.base.device, non_blocking=True)
            if isinstance(v, dict):
                return {k: _to_device(vv) for k, vv in v.items()}
            if isinstance(v, list):
                return [_to_device(item) for item in v]
            return v

        micro_batch = {k: _to_device(v) for k, v in micro_batch.items()}
        if getattr(self.base, "LOG_SAMPLE", True):
            helper.print_example(example=micro_batch, rank=self.base.args.train.local_rank)
            self.base.LOG_SAMPLE = False
        return micro_batch

    def postforward(
        self, outputs: ModelOutput, micro_batch: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Postprocess model outputs after forward pass."""
        loss_dict: Dict[str, torch.Tensor] = outputs.loss
        loss_dict = {k: v / self.base.num_micro_batches for k, v in loss_dict.items()}
        loss = torch.stack(list(loss_dict.values())).sum()
        return loss, loss_dict

    @staticmethod
    def _unpack_dict_of_list(batch: Dict[str, Any]) -> list[Dict[str, Any]]:
        if not isinstance(batch, dict) or len(batch) == 0:
            return []
        keys = list(batch.keys())
        num_items = len(batch[keys[0]])
        return [{k: batch[k][idx] for k in keys} for idx in range(num_items)]

    def _latent_cache_keys(self, micro_batch: Dict[str, Any]):
        """Content keys for an ``offline_embedding`` micro batch, or ``None``.

        ``None`` means "this condition model has no content-addressed identity"
        (Wan / MiniMax-H3, or a test double), and the fill then behaves exactly as
        it did before: positional rows, no manifest, no dedupe.
        """
        key_fn = getattr(self.condition_model, "latent_cache_keys", None)
        if not callable(key_fn):
            logger.warning_rank0(
                "This condition model exposes no `latent_cache_keys`; the offline embedding cache is "
                "addressed by position only, so changing the data, dtype, weights or code between the "
                "fill and the training run cannot be detected."
            )
            return None
        return key_fn(micro_batch)

    def _save_offline_embeddings(self, micro_batch: Dict[str, Any], cache_keys) -> None:
        """Encode only the samples the content-addressed cache does not already hold.

        Why the indirection: the export has to stay in dataset order (the training
        transform replays it positionally), so hits and misses are resolved per
        sample and then emitted in the original order -- not "hits first, then
        misses", which would silently permute the cache.
        """
        saver = self.offline_embedding_saver
        if saver is None:  # sp_rank > 0 writes nothing
            return

        if cache_keys is None:
            with torch.no_grad():
                condition = self.condition_model.get_condition(**micro_batch)
            for item in self._unpack_dict_of_list(condition):
                saver.save(item)
            return

        rows = []
        for key in cache_keys:
            rows.append(saver.load(key) if saver.has(key) else None)
        missing = [index for index, row in enumerate(rows) if row is None]
        if missing:
            subset = {
                name: ([value[index] for index in missing] if isinstance(value, list) else value)
                for name, value in micro_batch.items()
            }
            with torch.no_grad():
                condition = self.condition_model.get_condition(**subset)
            for index, item in zip(missing, self._unpack_dict_of_list(condition)):
                rows[index] = item
        for index, key in enumerate(cache_keys):
            if index not in missing:
                saver.cache_hits += 1
            saver.save(rows[index], cache_key=key)

    def forward_backward_step(self, micro_batch: Dict[str, torch.Tensor]) -> tuple:
        # The cache key is computed while the batch is still on the CPU: the clip
        # digest reads the decoded pixels, and `preforward` moves them to the device.
        cache_keys = None
        if self.training_task == "offline_embedding" and self.offline_embedding_saver is not None:
            cache_keys = self._latent_cache_keys(micro_batch)

        micro_batch = self.preforward(micro_batch)

        if self.training_task == "offline_embedding":
            # Encoding is the expensive half of an embedding run, so it is the half
            # that skips work: a sample whose content key is already cached is
            # replayed from the cache instead of re-encoded through the VAE.
            self._save_offline_embeddings(micro_batch, cache_keys)
            del micro_batch
            return 0.0, {}

        if self.training_task == "online_training":
            with torch.no_grad():
                micro_batch = self.condition_model.get_condition(**micro_batch)

        with torch.no_grad():
            micro_batch = self.condition_model.process_condition(**micro_batch)

        if self.flops_meter is not None:
            # Price the batch here rather than in ``on_step_begin``: only now do the
            # exact prompt ids and latent patch grids exist (see
            # ``Cosmos3EnvironMeter.seqlens``).
            self.flops_meter.add(micro_batch)

        with use_parallel_state(self.base.model.parallel_state), self.base.model_fwd_context:
            outputs = self.base.model(**micro_batch)

        loss: torch.Tensor
        loss_dict: Dict[str, torch.Tensor]
        # DiT postforward does not read ambient ParallelState.
        loss, loss_dict = self.postforward(outputs, micro_batch)

        with use_parallel_state(self.base.model.parallel_state), self.base.model_bwd_context:
            loss.backward()

        del micro_batch
        return loss, loss_dict

    def train_step(self, data_iterator: Any) -> Dict[str, float]:
        self.base.state.global_step += 1

        # SP broadcast of micro_batches
        with use_parallel_state(self.base.model.parallel_state):
            if get_parallel_state().sp_enabled:
                if get_parallel_state().sp_rank == 0:
                    micro_batches = next(data_iterator)
                else:
                    micro_batches = None

                obj_list = [micro_batches]
                dist.broadcast_object_list(
                    obj_list,
                    src=dist.get_global_rank(get_parallel_state().sp_group, 0),
                    group=get_parallel_state().sp_group,
                )
                micro_batches = obj_list[0]
            else:
                micro_batches = next(data_iterator)

        self.base._reset_async_activation_offload_if_enabled(self.base.model)
        self.on_step_begin(micro_batches=micro_batches)
        # One line per run: how much the packing actually put in each micro batch.
        self._log_pack_fill(micro_batches)

        self.base.sync_before_train_step()

        total_loss = 0.0
        total_loss_dict = defaultdict(float)
        grad_norm = 0.0
        num_micro_batches = len(micro_batches)
        self.base.num_micro_batches = num_micro_batches

        for micro_step, micro_batch in enumerate(micro_batches):
            if self.training_task != "offline_embedding":
                self.base.model_reshard(micro_step, num_micro_batches)
                self.base._configure_hsdp_allreduce(micro_step, num_micro_batches)

            loss: torch.Tensor
            loss_dict: Dict[str, torch.Tensor]

            loss, loss_dict = self.forward_backward_step(micro_batch)

            if self.training_task != "offline_embedding":
                total_loss += loss.item()
                for k, v in loss_dict.items():
                    total_loss_dict[k] += v.item()

        if self.training_task != "offline_embedding":
            grad_norm = self.base.model.clip_grad_norm()
            self.base.model.optimizer.step()
            self.base.model.lr_scheduler.step()
            self.base.model.optimizer.zero_grad()

        self.on_step_end(loss=total_loss, loss_dict=dict(total_loss_dict), grad_norm=grad_norm)

    def train(self):
        args = self.base.args
        self.on_train_begin()
        if self.training_task == "offline_embedding":
            args.train.num_train_epochs = 1

        logger.info(
            f"Rank{args.train.local_rank} Start training. "
            f"Start step: {self.base.start_step}. "
            f"Train steps: {args.train_steps}. "
            f"Start epoch: {self.base.start_epoch}. "
            f"Train epochs: {args.train.num_train_epochs}."
        )

        for epoch in range(self.base.start_epoch, args.train.num_train_epochs):
            if self.base.train_dataloader is not None and hasattr(self.base.train_dataloader, "set_epoch"):
                self.base.train_dataloader.set_epoch(epoch)
            self.base.state.epoch = epoch
            self.on_epoch_begin()

            if self.base.train_dataloader is not None:
                self.base.data_iterator = VeOmniIter(
                    self.base.train_dataloader,
                    use_background_prefetcher=args.data.dataloader.use_background_prefetcher,
                )
            else:
                self.base.data_iterator = None

            for _ in range(self.base.start_step, args.train_steps):
                try:
                    self.train_step(self.base.data_iterator)
                except StopIteration:
                    logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.dataloader.drop_last}")
                    break

            self.on_epoch_end()
            self.base.start_step = 0
            helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
            if args.data.dataloader.use_background_prefetcher:
                self.base.data_iterator.stop()

        self.on_train_end()

        if args.data.dataloader.use_background_prefetcher:
            self.base.data_iterator.stop()

        synchronize()

        if self.training_task == "offline_embedding" and self.offline_embedding_saver is not None:
            self.offline_embedding_saver.save_last()

        self.base.destroy_distributed()
