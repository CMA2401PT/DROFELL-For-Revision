from typing import List, Optional, Union
from dataclasses import dataclass
from dataset.common import ProcessedRecord
from dataset.advertise_gen import make_dataset_adv_gen
from dealers.common.brain import OptimizerConfig, UserModelDataInitConfig, SchedulerConfig
from operators.core.lora.common import LoraConfig
from operators.core.ptuning.common import PtuningConfig
from torch.utils.data import DataLoader
from operators.container import get_thread_context
import data
import cache
from scripts.base.train_state_maintainer import TrainStatusMaintainer, StepSize
from utils.minibatch_feeder import AsyncMinibatchFeeder


@dataclass
class TrainConfigAux:
    user_model_config: Union[PtuningConfig, LoraConfig]
    optimizer_config: OptimizerConfig
    scheduler_config: Optional[SchedulerConfig]
    user_model_init_config: UserModelDataInitConfig


def gen_train_aux(
    tokenizer_path: str,
    eos_token_id: int,
    minibatch_size: int = 2,
    batch_size: int = 2,
    ptuning: bool = True,

):
    build_prompt_str = "[instruction]{}\n[generation]"
    if ptuning:
        user_model_config = PtuningConfig(
            build_prompt_str=build_prompt_str,
            dropout_p=0.1,
            prefix_len=128
        )
        optimizer_config = OptimizerConfig(
            optimizer_name="AdamW", lr=1e-2, eps=1e-4
        )
    else:
        user_model_config = LoraConfig(
            r_q=-1,
            r_k=0,
            r_v=-1,
            build_prompt_str="[instruction]{}\n[generation]",
            dropout_p=0.0,
            len_limit=128,
        )
    optimizer_config = OptimizerConfig(
        optimizer_name="AdamW", lr=1e-3, eps=1e-4
    )
    train_dataset = make_dataset_adv_gen(
        json_file=data.advertise_gen_train_path,
        build_prompt_str=build_prompt_str,
        tokenizer_path=tokenizer_path,
        eos_token_id=eos_token_id,
        max_query_tokens=-1,
        max_response_tokens=-1,
        cache_dir=cache.dataset_cache_path
    )

    scheduler_config = None
    data_init_method_config = UserModelDataInitConfig(
        init_method="reinit"
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        collate_fn=lambda samples: samples,
        batch_size=minibatch_size,
        num_workers=0,
        shuffle=True,
        pin_memory=True
    )

    get_thread_context().switch_cpu()
    train_iter = train_loader.__iter__()

    def gen_sample():
        get_thread_context().switch_cpu()
        records: List[ProcessedRecord] = train_iter.__next__()
        # sample = Sample.create_from_records(samples=records, is_bfloat16=False)
        return records

    feeder = AsyncMinibatchFeeder(gen_sample, batch_size, batch_size*2)

    train_status_maintainer = TrainStatusMaintainer(
        step_size=StepSize(
            steps_in_each_update=batch_size,
            step_batch_size=minibatch_size
        ),
        total_samples=len(train_dataset),
        events=None,
    )

    config_aux = TrainConfigAux(
        user_model_config=user_model_config,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        user_model_init_config=data_init_method_config
    )

    return config_aux, feeder, train_status_maintainer
