import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, cast
import torch
from torch.utils.data import DataLoader, Subset
from torch import nn
from torch import optim
import extend_asyncio as asyncio
from dataset.common import ProcessedRecord
from operators.container.container import Container
from operators.helpers.cuda_utils.memory import current_memory_status
from operators.helpers.model_lib import ModelsLib
from operators.assemble.decoders.real import (
    DecodersImplement,
    make_decoders_from_files,
)
from operators.core.tail.real import ModelTail
from operators.core.head.real import ModelHead

from operators.core.ptuning.real import Ptuning
from operators.core.ptuning.common import PtuningConfig

from operators.core.lora.real import Lora
from operators.core.lora.common import LoraConfig

from operators.assemble.fwd_ctx import forward, backward
from dataset.advertise_gen import make_dataset_adv_gen
from dataset.dataset_builder import DataSetSampleBatchMeta
from scripts.base.train_state_maintainer import TrainStatusMaintainer, StepSize
import split_qllm_models
import data
import cache
from dataset import IGNORE_PAD_TOKEN_ID_IN_LABEL
models = ModelsLib(split_qllm_models.lib_path)


@dataclass
class TrainConfig:
    # user model config
    user_model_config: Union[PtuningConfig, LoraConfig]

    # model config
    model_name: str = "teared-Llama-2-7B-GPTQ"
    decoder_backend: DecodersImplement = "marlin"

    # dataset config
    train_file: str = data.advertise_gen_train_path
    max_query_tokens: int = -1
    max_response_tokens: int = -1
    batch_size: int = 4
    # Optional experiment controls.  Defaults preserve the historical entry
    # points; correctness experiments use an explicit shared sample order and
    # a paced submission window.
    sample_indices: Optional[List[int]] = None
    submission_interval_ms: float = 0.0
    force_query_tokens: int = 0
    force_response_tokens: int = 0

    # train config
    dataset_cache_dir: str = cache.dataset_cache_path
    steps_in_update: int = 1
    ckpt_to_load: Optional[str] = None
    re_init_first: bool = True

    lr: float = 1e-3
    final_lr: float = -1
    eps: float = 1e-4

    bfloat16_end: int = -1

    # events
    event_dir: str = os.path.join(os.path.dirname(__file__), "finetunng_record")
    ckpt_steps: int = 100
    ckpt_name: str = "ckpt.pt"
    best_ckpt_name: str = "best.pt"

    record_steps: int = 10
    record_name = "record.json"

    memory_steps: int = 20

    nan_steps: int = -1
    max_decoder_to_use: int = 0
    max_steps: int = -1
    seed: int = 2025
    final_ckpt_name: str = "final.pt"

    def event_file(self, name: str):
        return os.path.join(self.event_dir, name)

    def __post_init__(self):
        # process model config
        self.model = models.models_storage[self.model_name]
        if self.max_decoder_to_use > 0:
            if self.max_decoder_to_use > self.model.config.num_decoders:
                print(f"config to use {self.max_decoder_to_use}, but only have {self.model.config.num_decoders} decoders, so ignore")
            else:
                self.model.config.num_decoders = self.max_decoder_to_use

        if isinstance(self.user_model_config, PtuningConfig):
            self.user_model_config.check_or_set_decoders_config(
                num_layers=self.model.config.num_decoders,
                num_key_value_heads=self.model.config.num_key_value_heads,
                head_dim=self.model.config.head_dim
            )
        else:
            self.user_model_config.check_or_set_decoders_config(
                num_layers=self.model.config.num_decoders,
                num_key_value_heads=self.model.config.num_key_value_heads,
                num_query_heads=self.model.config.num_query_heads,
                head_dim=self.model.config.head_dim,
            )

        # process train config
        if self.re_init_first:
            assert self.ckpt_to_load is None
        os.makedirs(self.dataset_cache_dir, exist_ok=True)

        # process event:
        os.makedirs(self.event_dir, exist_ok=True)
        self.events: Dict[str, int] = {}
        if self.ckpt_steps > 0:
            self.events["ckpt"] = self.ckpt_steps

        if self.record_steps > 0:
            assert self.record_name
            self.events["record"] = self.record_steps

        if self.memory_steps > 0:
            self.events["memory"] = self.memory_steps

        if self.nan_steps > 0:
            self.events["nan"] = self.nan_steps


def run(cfg: TrainConfig, container: Container, device: torch.device):
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    with device:
        if isinstance(cfg.user_model_config, PtuningConfig):
            user_model = Ptuning(cfg.user_model_config)
            print(f"mode: Ptuning, per layer elements {cfg.user_model_config.per_layer_elements/(2**20)}M")
        else:
            user_model = Lora(cfg.user_model_config)
            print(f"mode: LoRA, per layer elements {cfg.user_model_config.per_layer_elements/(2**20)}M")
        head = ModelHead(cfg.model.config.model_head_config)
        head.load_from_data_dir(cfg.model.model_head_path)
        tail = ModelTail(cfg.model.config.model_tail_config)
        tail.load_from_data_dir(cfg.model.model_tail_path)
        decoders = make_decoders_from_files(
            implement=cfg.decoder_backend,
            config=cfg.model.config.quant_decoder_config,
            state_files=[cfg.model.decoder_path(i) for i in range(cfg.model.config.num_decoders)],
        )

        optimizer = optim.AdamW(user_model.parameters(), lr=cfg.lr, eps=cfg.eps)

        if cfg.ckpt_to_load is not None:
            user_model.load_from_file(cfg.ckpt_to_load)
        if cfg.re_init_first:
            user_model.reinit()

    train_dataset = make_dataset_adv_gen(json_file=cfg.train_file,
                                         build_prompt_str=user_model.cfg.build_prompt_str,
                                         tokenizer_path=cfg.model.tokenizer_path,
                                         eos_token_id=cfg.model.config.tokenizer_special_tokens.eos_token_id,
                                         max_query_tokens=cfg.max_query_tokens,
                                         max_response_tokens=cfg.max_response_tokens,
                                         force_query_tokens=cfg.force_query_tokens,
                                         force_response_tokens=cfg.force_response_tokens,
                                         cache_dir=cfg.dataset_cache_dir)

    loader_dataset = train_dataset
    loader_shuffle = True
    if cfg.sample_indices is not None:
        if not cfg.sample_indices:
            raise ValueError("sample_indices must not be empty")
        if min(cfg.sample_indices) < 0 or max(cfg.sample_indices) >= len(train_dataset):
            raise ValueError("sample_indices contains an out-of-range dataset index")
        loader_dataset = Subset(train_dataset, cfg.sample_indices)
        loader_shuffle = False

    train_status = TrainStatusMaintainer(
        step_size=StepSize(
            steps_in_each_update=cfg.steps_in_update,
            step_batch_size=cfg.batch_size
        ),
        total_samples=len(loader_dataset),
        events=cfg.events,
    )

    def create_input(samples: List[ProcessedRecord]):
        input_lens = [len(sample.input_ids) for sample in samples]
        position_ids_tensor: torch.Tensor = torch.tensor(data=[i for input_len in input_lens for i in range(input_len)], dtype=torch.int32, device="cpu")
        input_ids_tensor = torch.tensor([e for sample in samples for e in sample.input_ids], dtype=torch.int32, device="cpu")
        label_ids_tensor = torch.tensor([e for sample in samples for e in sample.output_ids], dtype=torch.long, device="cpu")
        return DataSetSampleBatchMeta(position_ids_tensor=position_ids_tensor, samples_input_ids=input_ids_tensor, samples_label_ids=label_ids_tensor, input_lens=input_lens)

    data_generator = torch.Generator()
    data_generator.manual_seed(cfg.seed)
    train_loader = DataLoader(
        dataset=loader_dataset,
        collate_fn=lambda samples: samples,
        batch_size=cfg.batch_size,
        num_workers=0,
        shuffle=loader_shuffle,
        pin_memory=True,
        generator=data_generator,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_PAD_TOKEN_ID_IN_LABEL)

    async def task():
        train_iter = train_loader.__iter__()
        num_steps = len(train_iter)
        current_best = math.inf
        submission_started = time.monotonic()

        scheduler = None
        if cfg.final_lr > 0:
            def lr_lambda(current_step: int):
                return max(0.0,  1 - (1-cfg.final_lr)*(current_step / num_steps))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        for step_index, samples in enumerate(train_iter):
            if cfg.max_steps > 0 and train_status.steps >= cfg.max_steps:
                break
            with container.mode("no_grad"):
                meta = create_input(samples)
            await container.notify("train_step_start")
            if train_status.steps < cfg.bfloat16_end:
                dtype = torch.bfloat16
            else:
                dtype = torch.float16

            def input_host_to_device(meta: DataSetSampleBatchMeta, device: torch.device) -> DataSetSampleBatchMeta:
                meta.samples_input_ids = meta.samples_input_ids.to(device, non_blocking=True)
                meta.position_ids_tensor = meta.position_ids_tensor.to(device, non_blocking=True)
                meta.samples_label_ids = meta.samples_label_ids.to(device, non_blocking=True)
                return meta
            meta = await container.step("input_host_to_device", input_host_to_device, meta=meta, device=device)
            hidden_states = await head.train_to_embedding_tokens(container=container, input_tokens_tensor=meta.samples_input_ids, dtype=dtype)
            # decoders forward
            with container.stage("decoders_forward"):
                hidden_states, ctx = \
                    await forward(container, hidden_states, meta.position_ids_tensor, meta.input_lens, user_model, decoders)

            hidden_states.requires_grad = True
            pred_logits = await tail.to_logits(container=container, hidden_states=hidden_states)

            def compute_loss(pred_logits: torch.Tensor, samples_label_ids: torch.Tensor):
                shift_logits = pred_logits[..., :-1, :].contiguous()
                shift_labels = samples_label_ids[..., 1:].contiguous()
                loss = cast(torch.Tensor, criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)))
                return loss
            loss = await container.step("compute_loss", compute_loss, pred_logits=pred_logits, samples_label_ids=meta.samples_label_ids)

            def loss_device_to_host(loss: torch.Tensor):
                return loss.detach().to("cpu", non_blocking=True)

            loss_val = await container.step("loss_device_to_host", loss_device_to_host,  loss=loss)
            sp = container.sync_point()
            await container.step("loss.backward", loss.backward)

            # decoders backward
            assert hidden_states.grad is not None
            grad_data = hidden_states.grad

            with container.stage("decoders_backward"):
                await backward(container, grad_data, ctx, user_model, decoders)
            await sp
            minibatch_id = (
                f"{step_index // cfg.steps_in_update}-{step_index}"
            )
            print(train_status.update_loss(
                float(loss_val.item()),
                tokens=int(meta.samples_input_ids.numel()),
                batch_id=minibatch_id,
            ))
            grad_squares = 0.0
            grad_max = 0.0
            grad_elements = 0
            for parameter in user_model.parameters():
                if parameter.grad is None:
                    continue
                grad = parameter.grad.detach().float()
                grad_squares += float(torch.sum(grad * grad).cpu())
                grad_max = max(grad_max, float(torch.max(torch.abs(grad)).cpu()))
                grad_elements += grad.numel()
            train_status.record_event("gradient_summary", {
                "l2": math.sqrt(grad_squares),
                "max_abs": grad_max,
                "elements": grad_elements,
            })

            # scheduler
            if scheduler is not None:
                scheduler.step()

            # do options

            if train_status.should("update"):
                await container.step("optimizer.step", optimizer.step)
                await container.step("weight.update", optimizer.zero_grad)
            await container.notify("train_step_finish")
            if loss < current_best and cfg.best_ckpt_name != "":
                print(f"update current best: {loss:.2f}")
                current_best = loss
                user_model.save_to_file(cfg.event_file(cfg.best_ckpt_name))
            if train_status.should(event="ckpt"):
                user_model.save_to_file(cfg.event_file(cfg.ckpt_name))
            if train_status.should(event="memory"):
                m = current_memory_status()
                sm = {}
                for k, v in m.items():
                    print(f"{k} {str(v)}")
                    sm[k] = (v.allocated, v.approximate, v.requested)
                train_status.record_event("memory", sm)
            if train_status.should(event="record"):
                train_status.dump(cfg.event_file(cfg.record_name))
            if train_status.should(event="nan"):
                params = user_model.state_dict()
                total_nan_count = 0
                for p, w in params.items():
                    nan_count = int(torch.isnan(w).count_nonzero())
                    total_nan_count += nan_count
                    if nan_count > 0:
                        params[p] = torch.nan_to_num_(w, 1)
                if total_nan_count > 0:
                    user_model.load_state_dict(params)
                print("nan: ", total_nan_count)
                train_status.record_event("nan", total_nan_count)
            if cfg.submission_interval_ms > 0:
                deadline = submission_started + (
                    (step_index + 1) * cfg.submission_interval_ms / 1000.0
                )
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    await asyncio.sleep(remaining)
        if cfg.final_ckpt_name:
            user_model.save_to_file(cfg.event_file(cfg.final_ckpt_name))
        train_status.dump(cfg.event_file(cfg.record_name))
    return task()
