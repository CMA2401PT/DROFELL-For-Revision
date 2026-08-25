import os
from typing import cast
import torch
from torch import nn
from torch.nn import Module
from safetensors import safe_open  # type: ignore
from .common import ModelTailConfig
from ..rms_norm.real import RMSNorm
from ...container import Container


class ModelTail(Module):
    def __init__(
        self,
        config: ModelTailConfig,
        dtype: torch.dtype | None = None,
    ) -> None:
        Module.__init__(self)
        self.cfg = config
        self.norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
        )
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False, dtype=dtype
        )
        self._device = torch.zeros(0).device

    def load_from_data_dir(self, model_data: str):
        with safe_open(os.path.join(model_data, "norm.safetensors"), framework="pt") as f:
            self.norm.load_state_dict({k: f.get_tensor(k) for k in f.keys()})
        with safe_open(os.path.join(model_data, "lm_head.safetensors"), framework="pt") as f:
            self.lm_head.load_state_dict({k: f.get_tensor(k) for k in f.keys()})

        def freeze(module: nn.Module):
            for _, parameter in module.named_parameters():
                parameter.requires_grad_(False)
            for _, buffer in module.named_buffers():
                buffer.requires_grad_(False)
            for _, submodule in module.named_children():
                freeze(submodule)

        self.lm_head.eval()
        freeze(self.lm_head)
        self.norm.eval()
        freeze(self.norm)

    @property
    def device(self) -> torch.device:
        return self._device

    async def to_logits(self, container: Container, hidden_states: torch.Tensor):
        output_dtype = self.lm_head.weight.dtype
        if hidden_states.dtype != output_dtype:
            def to_output_dtype(hidden_states: torch.Tensor, dtype: torch.dtype):
                return hidden_states.to(dtype=dtype)
            hidden_states = await container.step(
                "to_output_dtype",
                to_output_dtype,
                hidden_states=hidden_states,
                dtype=output_dtype,
            )
        hidden_states = await container.step("tail_norm", self.norm.forward, hidden_states=hidden_states)
        hidden_states = await container.step("lm_head", self.lm_head.forward, input=hidden_states)
        return hidden_states

    async def compute_loss_and_grad(
        self,
        container: Container,
        criterion: torch.nn.Module,
        hidden_states: torch.Tensor,
        samples_label_ids: torch.Tensor,
        float_loss: bool = False,
    ):
        assert hidden_states.is_leaf
        hidden_states = hidden_states.detach()
        if float_loss:
            def to_float(hidden_states: torch.Tensor):
                return hidden_states.float()
            hidden_states = await container.step("to_float", to_float, hidden_states=hidden_states)
        hidden_states.requires_grad = True

        pred_logits = await self.to_logits(container, hidden_states)

        def compute_loss(pred_logits: torch.Tensor, samples_label_ids: torch.Tensor):
            shift_logits = pred_logits[..., :-1, :].contiguous()
            shift_labels = samples_label_ids[..., 1:].contiguous()
            loss = cast(torch.Tensor, criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)))
            return loss
        loss = await container.step("compute_loss",
                                    compute_loss,
                                    pred_logits=pred_logits,
                                    samples_label_ids=samples_label_ids
                                    )
        loss_val = await container.data_device_to_cpu(loss, wait=False)
        sp = container.sync_point()
        await container.step("loss_backward", loss.backward)
        grad_data = hidden_states.grad
        assert grad_data is not None
        await sp
        loss_v = float(loss_val.item())
        return loss_v, grad_data
