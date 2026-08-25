from typing import Any, Dict, Optional, Union, cast
import torch
from torch import optim
from operators.assemble.decoders.real_decoders_group import DecodersGroup
from operators.assemble.fwd_ctx import backward, forward
from operators.container import Container
from operators.core.lora.real import Lora
from operators.core.lora.common import LoraConfig
from operators.core.swappable.real import CanSwapObject
from operators.core.ptuning.real import Ptuning
from operators.core.ptuning.common import PtuningConfig
from ..common.brain import *
from ..common.base_types import *

cpu_device = torch.device("cpu")


class TrainBindingData:
    def __init__(self,
                 decoders: DecodersGroup,
                 user_model_config: Union[PtuningConfig, LoraConfig],
                 optimizer_config: OptimizerConfig,
                 scheduler_config: Optional[SchedulerConfig],
                 data_init_method: Optional[UserModelDataInitConfig],
                 ) -> None:
        self.decoders = decoders
        self.user_model_config, self.user_model = self._init_user_model(user_model_config)
        self.scheduler: Optional[Any] = None
        self.optimizer: Any = None
        self.forward_ctxs: Dict[SampleBatchID, CanSwapObject] = {}
        self._set_and_init_update_config(optimizer_config, scheduler_config)
        self._init_data(data_init_method)

    def _init_user_model(self, user_model_config: Union[PtuningConfig, LoraConfig]):
        if isinstance(user_model_config, PtuningConfig):
            user_model_config.check_or_set_decoders_config(
                num_layers=len(self.decoders.decoders),
                num_key_value_heads=self.decoders.decoders[0].config.num_key_value_heads,
                head_dim=self.decoders.decoders[0].config.head_dim
            )
            return user_model_config, Ptuning(config=user_model_config)
        else:
            user_model_config.check_or_set_decoders_config(
                num_layers=len(self.decoders.decoders),
                num_key_value_heads=self.decoders.decoders[0].config.num_key_value_heads,
                num_query_heads=self.decoders.decoders[0].config.num_query_heads,
                head_dim=self.decoders.decoders[0].config.head_dim,
            )
            return user_model_config, Lora(config=user_model_config)

    def _set_and_init_update_config(self, optimizer_config: OptimizerConfig, scheduler_config: Optional[SchedulerConfig]) -> None:
        OptimizerClass = getattr(optim, optimizer_config.optimizer_name)
        self.optimizer = OptimizerClass(self.user_model.parameters(), lr=optimizer_config.lr, eps=optimizer_config.eps)
        if scheduler_config is not None:
            SchedulerClass = getattr(optim.lr_scheduler, scheduler_config.scheduler_name)
            self.scheduler = SchedulerClass(optimizer=self.optimizer, **scheduler_config.scheduler_kwargs)
        else:
            self.scheduler = None

    def _init_data(self, method: Optional[UserModelDataInitConfig]):
        if method is None or method.init_method == "none":
            pass
        elif method.init_method == "reinit":
            self.user_model.reinit()
        elif method.init_method == "from_ckpt":
            assert method.ckpt_bytes
            self.user_model.from_pre_layer_serialized([method.ckpt_bytes[i] for i in range(len(self.decoders.decoders))])
        else:
            raise NotImplementedError


async def do_forward(
    binding_data: TrainBindingData,
    container: Container,
    operate: OperateForward,
) -> CanSwapObject:
    meta = operate.meta.shallow_copy()
    assert operate.meta.hidden_states
    operate.meta.hidden_states.free()
    operate.meta.device_pos_idx = None

    device_pos_idx = meta.device_pos_idx
    assert meta.hidden_states is not None
    hidden_states = meta.hidden_states.tensor
    device = container.device
    if operate.cpu_in:
        hidden_states = await container.data_cpu_to_device(hidden_states)
    else:
        assert hidden_states.device == device
    if operate.cpu_in or device_pos_idx is None:
        device_pos_idx = await container.data_cpu_to_device(meta.cpu_pos_idx)
    else:
        assert device_pos_idx is not None
        assert device_pos_idx.device == device
    assert device_pos_idx is not None
    hidden_states, ctx = await forward(
        container,
        hidden_states,
        device_pos_idx,
        meta.input_lens,
        binding_data.user_model, binding_data.decoders
    )

    # if FORCE_SWAP_IN_SWAP_OUT:
    #     ctx.swap_out(container.stream)

    binding_data.forward_ctxs[meta.sample_batch_id] = ctx

    sp = None
    if operate.cpu_out:
        hidden_states = await container.data_device_to_cpu(hidden_states, wait=False)
        sp = container.sync_point()
        meta.device_pos_idx = None
    else:
        meta.device_pos_idx = device_pos_idx
    if operate.is_last:
        meta.device_pos_idx = None
    meta.hidden_states = DeviceTensorCarrier(hidden_states, no_gc=meta.hidden_states.no_gc)
    operate.set_result(meta, sp)
    return ctx


async def do_backward(
    binding_data: TrainBindingData,
    container: Container,
    operate: OperateBackward,
) -> None:
    grad_tensor = operate.grad.tensor
    if operate.cpu_in:
        grad_tensor = await container.data_cpu_to_device(grad_tensor)
    else:
        assert grad_tensor.device == container.device
    forward_ctx = binding_data.forward_ctxs[operate.context_id]
    del binding_data.forward_ctxs[operate.context_id]
    if forward_ctx.swapped:
        forward_ctx.swap_in(binding_data.decoders.device, container.stream)

    hidden_states_grad = await backward(container, grad_tensor, forward_ctx, binding_data.user_model, binding_data.decoders)
    if not operate.has_next:
        operate.set_result(None)
        operate.grad.free()
        return
    sp = None
    if operate.cpu_out:
        hidden_states_grad = await container.data_device_to_cpu(hidden_states_grad, wait=False)
        sp = container.sync_point()

    operate.set_result(DeviceTensorCarrier(hidden_states_grad, no_gc=operate.grad.no_gc), sp)
    operate.grad.free()


async def do_weight_update(
    binding_data: TrainBindingData,
    container: Container,
    operate: OperateWeightUpdate
) -> None:
    def weight_update():
        binding_data.optimizer.step()
        binding_data.optimizer.zero_grad()
        if binding_data.scheduler is not None:
            binding_data.scheduler.step()

    await container.step("weight_update", weight_update)
    operate.set_result(None)


async def dump_train_states(
    binding_data: TrainBindingData,
    container: Container,
    operate: OperateDumpTrainStates
):
    # user_model "embedding.weight":torch.Tensor, 32MiB
    # optimizer "state":{0:{'step':torch.Tensor,'exp_avg':torch.Tensor,'exp_avg_sq':torch.Tensor,...},...} 64MiB, "param_groups": Any
    # scheduler Any
    states: Dict[str, Any] = {
        "scheduler": binding_data.scheduler.state_dict() if binding_data.scheduler is not None else None,
        "optimizer": binding_data.optimizer.state_dict() if binding_data.optimizer is not None else None,
        "user_model": binding_data.user_model.state_dict()
    }

    async def _deep_to_cpu(_states: Union[Dict[str, Any], torch.Tensor, Any]) -> Union[Dict[str, Any], torch.Tensor, Any]:
        if isinstance(_states, dict):
            return {k: await _deep_to_cpu(v) for k, v in _states.items()}
        elif isinstance(_states, torch.Tensor):
            return await container.data_device_to_cpu(_states, wait=False)
        else:
            return _states
    await container.sync_point()
    cpu_states = {k: await _deep_to_cpu(v) for k, v in states.items()}
    sp = container.sync_point()
    operate.set_result(cpu_states, sp)


async def load_train_states(
    binding_data: TrainBindingData,
    container: Container,
    operate: OperateLoadTrainStates
) -> None:
    async def _deep_from_cpu(_states: Union[Dict[str, Any], torch.Tensor, Any]) -> Union[Dict[str, Any], torch.Tensor, Any]:
        if isinstance(_states, dict):
            return {k: await _deep_from_cpu(v) for k, v in _states.items()}
        elif isinstance(_states, torch.Tensor):
            return await container.data_cpu_to_device(_states)
        else:
            return _states
    states = {k: await _deep_from_cpu(v) for k, v in operate.states.items()}
    scheduler_states_dict = states["scheduler"]
    optimizer_states_dict = states["optimizer"]
    user_model_states_dict = cast(Dict[str, Any], states["user_model"])

    def load():
        if scheduler_states_dict is not None:
            assert binding_data.scheduler is not None
            binding_data.scheduler.load_state_dict(scheduler_states_dict)
        if optimizer_states_dict is not None:
            binding_data.optimizer.load_state_dict(optimizer_states_dict)
        binding_data.user_model.load_state_dict(user_model_states_dict)
    await container.step("load_train_states", load)
    sp = container.sync_point()
    operate.set_result(None, sp)
