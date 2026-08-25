import os
from typing import Any, Callable, Dict, List

import extend_asyncio as asyncio

from dealers.common.brain import OptimizerConfig, UserModelDataInitConfig
from node.agent import AgentNode
from node.agent.protocols import TrainStepResult, TrainTaskResult
from node.base import BaseNode
from node.compute_node.train_meta import TrainHeadInput
from node.request import RequestNode
from operators.core.ptuning.common import PtuningConfig
from operators.core.sampler.common import SamplingParams
from operators.core.tokenizer.common import SequenceCore, StopConfig
from orbit import Context, LaunchPad, Location, Orbit, OrbitName, SimpleLogger
from orbit.link import ContextNetExtender, OrbitCPNCommander
from orbit.link.cpn import NetModel
from orbit.link.extender import get_free_port


class FakeInferControl:
    def __init__(self, ctrl_id: str, max_tokens: int) -> None:
        self.ctrl_id = ctrl_id
        self.max_tokens = max_tokens
        self.inputs: List[Any] = []

    async def infer_step(
        self,
        inp: Any,
        is_prefill: bool,
        detail_trace: bool = False,
    ) -> tuple[bool, int, None, None]:
        assert is_prefill == (len(self.inputs) == 0)
        assert not detail_trace
        self.inputs.append(inp)
        token = 100 + len(self.inputs) - 1
        return len(self.inputs) == self.max_tokens, token, None, None


def make_fake_control(**kwargs: Any) -> FakeInferControl:
    seq_core: SequenceCore = kwargs["seq_core"]
    return FakeInferControl(
        ctrl_id=kwargs["seq_id"],
        max_tokens=seq_core.stop_config.max_tokens,
    )


class FakeTrainControl:
    def __init__(self, ctrl_id: str) -> None:
        self.ctrl_id = ctrl_id

    async def train_step_with_detail(
        self,
        req_version: int,
        full_batch_ids: List[str],
        minibatch_id: str,
        sample: TrainHeadInput,
        detail_trace: bool = False,
    ) -> TrainStepResult:
        assert full_batch_ids
        assert sample.is_cpu
        if req_version == 1:
            await asyncio.sleep(0.02)
        return TrainStepResult(
            task=TrainTaskResult(loss=float(req_version)),
            detail_trace=(
                [(minibatch_id, req_version)] if detail_trace else None
            ),
        )


def make_fake_train_control(**kwargs: Any) -> FakeTrainControl:
    return FakeTrainControl(ctrl_id=kwargs["user_model_name"])


def train_input() -> TrainHeadInput:
    return TrainHeadInput(
        is_bfloat16=False,
        input_lens=[2],
        position_ids=[0, 1],
        input_ids=[1, 2],
        label_ids=[2, 3],
    )


def agent_entry(ctx: Context) -> None:
    async def run() -> None:
        ContextNetExtender(ctx)
        base_node = BaseNode(ctx)
        agent = AgentNode(base_node)
        agent.create_infer_control = make_fake_control
        agent.create_train_control = make_fake_train_control
        agent.remove_train_control = lambda _control_id: None

        async def inspect(**_: Any) -> str:
            return "remote-path"

        agent.inspect_fastest_n_path = inspect
        ctx.report_ready()
        while True:
            base_node.on_packet(await ctx.recv())

    asyncio.create_task(run())


def remote_request_entry(ctx: Context) -> None:
    async def run() -> None:
        ContextNetExtender(ctx)
        base_node = BaseNode(ctx)
        request = RequestNode(base_node, agent_node="agent")
        ctx.report_ready()

        async def validate() -> None:
            inspected = await request.inspect_fastest_n_path(
                type="infer",
                base_models=["model:head"],
                allow_reuse=False,
            )
            assert inspected == "remote-path"
            control = request.create_infer_control(
                seq_id="remote-sequence",
                base_models=["model:head"],
                seq_core=SequenceCore(
                    stop_config=StopConfig(max_tokens=3, ignore_eos=True),
                    prompt_token_ids=[1, 2],
                ),
                sampling_params=SamplingParams(temperature=0.0),
                user_model_data=None,
                keypoint_providers=[],
                retrieve=False,
            )
            first = await control.infer_step([1, 2], is_prefill=True)
            second = await control.infer_step(first[1], is_prefill=False)
            third = await control.infer_step(second[1], is_prefill=False)
            assert [first[1], second[1], third[1]] == [100, 101, 102]
            assert [first[0], second[0], third[0]] == [False, False, True]
            request.remove_infer_control(control.ctrl_id)

            train_control = request.create_train_control(
                user_model_name="remote-training",
                base_models=["model:decoders"],
                user_model_config=PtuningConfig(prefix_len=4),
                optimizer_config=OptimizerConfig(),
                scheduler_config=None,
                user_model_init_config=UserModelDataInitConfig(
                    init_method="reinit"
                ),
                keypoint_providers=[],
                ckpt_period=2,
            )
            slow = asyncio.create_task(
                train_control.train_step_with_detail(
                    req_version=1,
                    full_batch_ids=["batch"],
                    minibatch_id="slow",
                    sample=train_input(),
                    detail_trace=True,
                )
            )
            fast = asyncio.create_task(
                train_control.train_step_with_detail(
                    req_version=2,
                    full_batch_ids=["batch"],
                    minibatch_id="fast",
                    sample=train_input(),
                    detail_trace=True,
                )
            )
            fast_result = await fast
            slow_result = await slow
            assert fast_result.task.loss == 2.0
            assert slow_result.task.loss == 1.0
            assert fast_result.detail_trace == [("fast", 2)]
            assert slow_result.detail_trace == [("slow", 1)]
            request.remove_train_control(train_control.ctrl_id)
            ctx.send((OrbitName, "agent-request-test-passed", "remote"))

        asyncio.create_task(validate())
        while True:
            base_node.on_packet(await ctx.recv())

    asyncio.create_task(run())


def combined_request_entry(ctx: Context) -> None:
    async def run() -> None:
        ContextNetExtender(ctx)
        base_node = BaseNode(ctx)
        request = RequestNode(base_node, agent_node=None)
        assert not request.is_remote
        assert request._local_agent is not None
        request._local_agent.create_infer_control = make_fake_control
        request._local_agent.create_train_control = make_fake_train_control
        request._local_agent.remove_train_control = lambda _control_id: None
        control = request.create_infer_control(
            seq_id="combined-sequence",
            base_models=["model:head"],
            seq_core=SequenceCore(
                stop_config=StopConfig(max_tokens=1, ignore_eos=True),
                prompt_token_ids=[1, 2],
            ),
            sampling_params=SamplingParams(temperature=0.0),
            user_model_data=None,
            keypoint_providers=[],
            retrieve=False,
        )
        stopped, token, _, _ = await control.infer_step(
            [1, 2],
            is_prefill=True,
        )
        assert stopped and token == 100
        train_control = request.create_train_control(
            user_model_name="combined-training",
            base_models=["model:decoders"],
            user_model_config=PtuningConfig(prefix_len=4),
            optimizer_config=OptimizerConfig(),
            scheduler_config=None,
            user_model_init_config=UserModelDataInitConfig(
                init_method="reinit"
            ),
            keypoint_providers=[],
            ckpt_period=2,
        )
        result = await train_control.train_step_with_detail(
            req_version=3,
            full_batch_ids=["batch"],
            minibatch_id="combined",
            sample=train_input(),
            detail_trace=True,
        )
        assert result.task.loss == 3.0
        assert result.detail_trace == [("combined", 3)]
        request.remove_train_control(train_control.ctrl_id)
        ctx.report_ready()
        ctx.send((OrbitName, "agent-request-test-passed", "combined"))

    asyncio.create_task(run())


FUNCTIONS: Dict[str, Callable[[Context], None]] = {
    "agent": agent_entry,
    "remote-request": remote_request_entry,
    "combined-request": combined_request_entry,
}


def initialize_process() -> asyncio.Loop:
    def on_exception(_: asyncio.Loop, error: Exception) -> None:
        raise error

    asyncio.config_impl("golang", False)
    loop = asyncio.get_event_loop()
    loop.on_except(on_exception)
    return loop


def main() -> None:
    launch_pad = LaunchPad(initialize_process)

    async def run() -> None:
        orbit = Orbit(
            launch_pad=launch_pad,
            functions_dict=FUNCTIONS,
            logger=SimpleLogger(),
        )
        commander = OrbitCPNCommander(orbit, NetModel())
        orbit.append_no_gc(commander)
        await orbit.init(port=get_free_port(), satellites=tuple())
        await orbit.launch(Location(name="agent"), "agent")
        await orbit.launch(
            Location(name="remote-request", thread=1),
            "remote-request",
        )
        await orbit.launch(
            Location(name="combined-request"),
            "combined-request",
        )
        passed = set()
        while passed != {"remote", "combined"}:
            _, api, mode = await orbit.recv()
            if api == "agent-request-test-passed":
                passed.add(mode)
        print(
            "Agent/Request infer, train and combined fallback passed",
            flush=True,
        )
        os._exit(0)

    launch_pad.loop.create_task(run())
    launch_pad.loop.run_forever()


if __name__ == "__main__":
    main()
