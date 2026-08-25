from typing import Dict, List, Optional, Sequence, Tuple, Union, cast

import torch
from dealers.common.base_types import DeviceTensorCarrier, group_model_with_split
from dealers.common.provider import Provider
from dealers.common.brain import Brain, OptimizerConfig, SchedulerConfig, UserModelDataInitConfig
from operators.core.lora.common import LoraConfig
from operators.core.lora.real import Lora
from operators.core.ptuning.real import Ptuning
from operators.core.lora.render import LoraRenderConfig
from operators.core.tokenizer.common import SequenceCore
from operators.core.sampler.common import SamplingParams
from operators.core.tokenizer.real import ExtendTokenizer
from operators.core.ptuning.common import PtuningConfig
from .infer_instance import UserModelDataProvider
from .infer_instance import InferInstance, InferTestConfig
from .train_instance import TrainInstance, TrainTestConfig
from .model_config import ModelConfig


class PipelineParallelTestbed:
    def __init__(self, brains: Sequence[Brain], cfg: ModelConfig) -> None:
        brains = list(brains)
        self.brains = brains
        for brain in brains:
            if brain.decoder_implement != cfg.decoder_backend:
                raise ValueError(
                    "brain and model decoder backends do not match")
            if brain.max_seq_len != cfg.max_seq_len:
                raise ValueError(
                    "brain and model maximum sequence lengths do not match")
            if cfg.enable_cuda_graph:
                brain.capture_decode_graph()
        self.providers = [Provider(brain) for brain in brains]
        self.cfg = cfg
        self.tokenizer = ExtendTokenizer.load_tokenier(
            path=self.cfg.tokenizer_path,
            cfg=self.cfg.base_model_config.tokenizer_special_tokens
        )
        self.user_models = UserModelDataProvider()
        head, decoders, tail = group_model_with_split(
            self.cfg.model_name, self.cfg.layers, self.cfg.group_fn)
        self.base_model_names = [head, *decoders, tail]
        per_brain_decoders = (len(decoders)+len(brains)-1)//len(brains)
        self.base_models_lookup: Dict[str, int] = {}
        for decoder_i, decoder in enumerate(decoders):
            brain_i = decoder_i//per_brain_decoders
            if decoder_i == 0:
                self.providers[brain_i][head]
                self.base_models_lookup[head] = brain_i
            self.providers[brain_i][decoder]
            self.base_models_lookup[decoder] = brain_i
            if decoder_i == len(decoders)-1:
                self.providers[brain_i][tail]
                self.base_models_lookup[tail] = brain_i
        print("base model loaded")

    def __del__(self):
        self.base_models_lookup.clear()
        for p in self.providers:
            p.remove()

    # @property
    # def base_model_names(self) -> List[str]:
    #     names: List[str] = group_model(self.cfg.model_name, self.cfg.layers, self.cfg.group_fn)
    #     return names

    def put_model(self, name: str, blocks: List[List[Tuple[DeviceTensorCarrier, DeviceTensorCarrier]]], cfg: Optional[LoraRenderConfig]):
        self.user_models.put_model(name, blocks, cfg)

    def put_model_from_config(
        self, name: str,
        cfg: Union[LoraConfig, PtuningConfig],
        state_file: str,
    ):
        lora_cfg = None
        is_lora = isinstance(cfg, LoraConfig)
        if is_lora:
            lora_cfg = cfg.render_config
            cfg.check_or_set_decoders_config(
                num_layers=self.cfg.layers,
                head_dim=self.cfg.base_model_config.head_dim,
                num_key_value_heads=self.cfg.base_model_config.num_key_value_heads,
                num_query_heads=self.cfg.base_model_config.num_query_heads
            )
        else:
            cfg.check_or_set_decoders_config(
                num_layers=self.cfg.layers,
                num_key_value_heads=self.cfg.base_model_config.num_key_value_heads,
                head_dim=self.cfg.base_model_config.head_dim
            )
        user_model = Lora(cfg) if is_lora else Ptuning(cfg)
        state_dict = torch.load(state_file, map_location="cpu")
        user_model.load_state_dict(state_dict)
        user_data =\
            cast(Lora, user_model).get_inference_lora_blocks(torch.float16) \
            if is_lora else \
            cast(Ptuning, user_model).get_inference_ptuning_kv_cache_with_convert(
                torch.float16)
        user_data = [
            [
                (DeviceTensorCarrier.from_data(k),
                 DeviceTensorCarrier.from_data(v))
                for (k, v) in layer_blocks
            ]
            for layer_blocks in user_data
        ]
        self.put_model(name, user_data, lora_cfg)

    async def create_infer_instance(self,
                                    groups: List[List[Tuple[str, str]]],
                                    test_config: InferTestConfig,
                                    sampling_params: SamplingParams,
                                    seq_core: SequenceCore,
                                    user_model_name: Optional[str],
                                    ):
        instance = InferInstance(
            providers=self.providers,
            base_models_lookup=self.base_models_lookup,
            user_model_provider=self.user_models,
            groups=groups,
            user_model_name=user_model_name,
            seq_core=seq_core,
            sampling_params=sampling_params,
            test_config=test_config,
        )
        await instance.ainit()
        return instance

    async def create_train_instance(self,
                                    groups: List[List[Tuple[str, str]]],
                                    user_model_config: Union[PtuningConfig, LoraConfig],
                                    optimizer_config: OptimizerConfig,
                                    scheduler_config: Optional[SchedulerConfig],
                                    data_init_method: UserModelDataInitConfig,
                                    test_config: TrainTestConfig,
                                    ):
        instance = TrainInstance(
            providers=self.providers,
            base_models_lookup=self.base_models_lookup,
            groups=groups,
            user_model_config=user_model_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            data_init_method=data_init_method,
            test_config=test_config,
        )
        await instance.ainit()
        return instance
