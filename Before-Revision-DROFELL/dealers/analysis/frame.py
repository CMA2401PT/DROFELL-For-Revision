from typing import Any, Callable, cast
from ..common.brain import Brain, Operate, ListWithHook, OperateType
from .recorder import CallRecorder


class OperateWithCallback:
    def __init__(self, op: Operate[Any],
                 start_callback: Callable[[], Callable[[Any], None]]) -> None:
        self.base_op = op
        self.start_callback = start_callback
        self.finish_callback: Callable[[Any], None]
        self.ret = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_op, name)

    def set_result(self, ret: Any, sp: Any = None):
        self.ret = ret
        self.base_op.set_result(ret, sp)

    def on_start(self):
        self.finish_callback = self.start_callback()
        self.base_op.on_start()

    def on_finish(self):
        self.finish_callback(self.ret)
        self.base_op.on_finish()


class AnalysisFrame:
    def __init__(self, brain: Brain, recorder: CallRecorder) -> None:
        self.base = brain
        self.recorder = recorder
        self.make_model_head = self.wrap_nonasync_with_analysis(brain.make_model_head)
        self.base.make_model_head = self.make_model_head
        self.make_model_tail = self.wrap_nonasync_with_analysis(brain.make_model_tail)
        self.base.make_model_tail = self.make_model_tail
        self.make_model_decoders = self.wrap_nonasync_with_analysis(brain.make_model_decoders)
        self.base.make_model_decoders = self.make_model_decoders
        self.remove_decoders_infer_binding = self.wrap_nonasync_with_analysis(brain.remove_decoders_infer_binding)
        self.base.remove_decoders_infer_binding = self.remove_decoders_infer_binding
        self.remove_decoders_train_binding = self.wrap_nonasync_with_analysis(brain.remove_decoders_train_binding)
        self.base.remove_decoders_train_binding = self.remove_decoders_train_binding
        self.remove_model_head = self.wrap_nonasync_with_analysis(brain.remove_model_head)
        self.base.remove_model_head = self.remove_model_head
        self.remove_model_tail = self.wrap_nonasync_with_analysis(brain.remove_model_tail)
        self.base.remove_model_tail = self.remove_model_tail
        self.remove_model_decoders = self.wrap_nonasync_with_analysis(brain.remove_model_decoders)
        self.base.remove_model_decoders = self.remove_model_decoders

        self.swapout = self.wrap_nonasync_with_analysis(self.base.swapout)
        self.base.swapout = self.swapout
        for name, queue in brain.all_queues:
            self.update_hook(name, queue)

    @classmethod
    def wrap_brain(cls, brain: Brain, recorder: CallRecorder) -> Brain:
        wrapped_brain = cls(brain, recorder)
        return cast(Brain, wrapped_brain)

    def update_hook(self, name: str, orig_list: ListWithHook[OperateType]) -> None:
        orig_hook = orig_list.hook

        def new_hook(op: OperateType) -> OperateType:
            updated_op = orig_hook(op)

            # def start_callback():
            #     args = op.__dict__.copy()
            #     del args['fut']
            #     del args['sync']
            #     return self.recorder(True, name, args)
            # op_with_cb = OperateWithCallback(op=updated_op, start_callback=start_callback)  # type:ignore
            def start_callback():
                # args = op.__dict__.copy()
                # del args['fut']
                # del args['sync']
                return self.recorder(name, op)  # type:ignore
            op_with_cb = OperateWithCallback(op=updated_op, start_callback=start_callback)  # type:ignore
            return op_with_cb  # type:ignore
        orig_list.hook = new_hook

    def wrap_nonasync_with_analysis(self, func: Callable[..., Any], before_start: bool = False) -> Callable[..., Any]:
        # is_coro = asyncio.iscoroutinefunction(func)
        fn_name = func.__name__

        def wrapper(*args: Any, **kwargs: Any):
            finish_callback = self.recorder(fn_name, kwargs)
            ret = func(**kwargs)
            finish_callback(ret)
            return ret
        return wrapper

    def __getattr__(self, name: str) -> Any:
        return self.base.__getattribute__(name)
