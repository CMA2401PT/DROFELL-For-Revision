import os
from dataclasses import dataclass
from typing import List, Optional, cast
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import cache
from dataset.common import ProcessedRecord
from operators.container.container import Container
from operators.helpers.model_lib import ModelsLib
from operators.assemble.decoders.real import DecodersImplement, make_decoders_from_files
from operators.core.tail.real import ModelTail
from operators.core.head.real import ModelHead
from operators.core.ptuning.real import Ptuning
from operators.core.ptuning.common import PtuningConfig
from operators.assemble.fwd_ctx import forward, backward
from dataset.advertise_gen import make_dataset_adv_gen
from dataset.dataset_builder import DataSetSampleBatchMeta
from train_state_maintainer import TrainStatusMaintainer, StepSize
import split_qllm_models
import data
from dataset import IGNORE_PAD_TOKEN_ID_IN_LABEL
models = ModelsLib(split_qllm_models.lib_path)


@dataclass
class TrainConfig:
    # model config
    model_name: str = "teared-Llama-2-7B-GPTQ"
    decoders_impl: DecodersImplement = "our_fuse"
    # user model config
    ptuning_config: PtuningConfig = PtuningConfig(
        build_prompt_str="[instruction]{}\n[generation]",
        dropout_p=0.1,
        prefix_len=128
    )

    # dataset config
    train_file: str = data.advertise_gen_train_path
    max_query_tokens: int = -1
    max_response_tokens: int = -1
    force_query_tokens: int = -1
    force_response_tokens: int = -1
    batch_size: int = 4

    # train config
    dataset_cache_dir: str = cache.dataset_cache_path
    steps_in_update: int = 1
    ckpt_to_load: Optional[str] = None
    re_init_first: bool = True

    lr: float = 1e-2
    eps: float = 1e-4

    bfloat16_end: int = -1
    max_decoder_to_use: int = 0

    def __post_init__(self):
        # process model config
        self.model = models.models_storage[self.model_name]
        if self.max_decoder_to_use > 0:
            if self.max_decoder_to_use > self.model.config.num_decoders:
                print(f"config to use {self.max_decoder_to_use}, but only have {self.model.config.num_decoders} decoders, so ignore")
            else:
                self.model.config.num_decoders = self.model.config.num_decoders
        self.ptuning_config.num_layers = self.model.config.num_decoders
        self.ptuning_config.head_dim = self.model.config.head_dim
        self.ptuning_config.num_key_value_heads = self.model.config.num_key_value_heads

        # process train config
        if self.re_init_first:
            assert self.ckpt_to_load is None
        os.makedirs(self.dataset_cache_dir, exist_ok=True)


def prepare(cfg: TrainConfig, device: torch.device):
    with device:
        prefix_encoder = Ptuning(cfg.ptuning_config)
        print(f"single one layer one block size {(cfg.ptuning_config.kv_cache_elements*2)/(2**20)} MB")
        head = ModelHead(cfg.model.config.model_head_config)
        head.load_from_data_dir(cfg.model.model_head_path)
        tail = ModelTail(cfg.model.config.model_tail_config)
        tail.load_from_data_dir(cfg.model.model_tail_path)
        decoders = make_decoders_from_files(
            implement=cfg.decoders_impl,
            config=cfg.model.config.quant_decoder_config,
            state_files=[cfg.model.decoder_path(i) for i in range(cfg.model.config.num_decoders)],
        )

        optimizer = optim.AdamW(prefix_encoder.parameters(), lr=cfg.lr, eps=cfg.eps)

        if cfg.ckpt_to_load is not None:
            prefix_encoder.load_from_file(cfg.ckpt_to_load)
        if cfg.re_init_first:
            from torch.nn.init import xavier_uniform_
            for parameter in prefix_encoder.parameters():
                xavier_uniform_(parameter)

    train_dataset = make_dataset_adv_gen(
        json_file=cfg.train_file,
        build_prompt_str=prefix_encoder.cfg.build_prompt_str,
        tokenizer_path=cfg.model.tokenizer_path,
        eos_token_id=cfg.model.config.tokenizer_special_tokens.eos_token_id,
        max_query_tokens=cfg.max_query_tokens,
        max_response_tokens=cfg.max_response_tokens,
        cache_dir=cfg.dataset_cache_dir,
        force_query_tokens=cfg.force_query_tokens,
        force_response_tokens=cfg.force_response_tokens,
    )

    train_status = TrainStatusMaintainer(
        step_size=StepSize(
            steps_in_each_update=cfg.steps_in_update,
            step_batch_size=cfg.batch_size
        ),
        total_samples=len(train_dataset),
        events=None,
    )

    def create_input(samples: List[ProcessedRecord]):
        input_lens = [len(sample.input_ids) for sample in samples]
        position_ids_tensor: torch.Tensor = torch.tensor(data=[i for input_len in input_lens for i in range(input_len)], dtype=torch.int32, device="cpu")
        input_ids_tensor = torch.tensor([e for sample in samples for e in sample.input_ids], dtype=torch.int32, device="cpu")
        label_ids_tensor = torch.tensor([e for sample in samples for e in sample.output_ids], dtype=torch.long, device="cpu")
        return DataSetSampleBatchMeta(position_ids_tensor=position_ids_tensor, samples_input_ids=input_ids_tensor, samples_label_ids=label_ids_tensor, input_lens=input_lens)

    train_loader = DataLoader(
        dataset=train_dataset,
        collate_fn=lambda samples: samples,
        batch_size=cfg.batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=True
    )
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_PAD_TOKEN_ID_IN_LABEL)
    torch.cuda.synchronize()

    async def task(step_count: int, container: Container):
        train_iter = train_loader.__iter__()
        step = 0
        for samples in train_iter:
            if step == step_count:
                break
            step += 1
            await container.notify("train_step_start")
            with container.mode("no_grad"):
                meta = create_input(samples)
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
                    await forward(container, hidden_states, meta.position_ids_tensor, meta.input_lens, prefix_encoder, decoders)

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
                await backward(container, grad_data, ctx, prefix_encoder, decoders)
            await sp
            print(train_status.update_loss(float(loss_val.item())))
            if train_status.should("update"):
                await container.step("optimizer.step", optimizer.step)
                await container.step("weight.update", optimizer.zero_grad)
            await container.notify("train_step_finish")

    return task
