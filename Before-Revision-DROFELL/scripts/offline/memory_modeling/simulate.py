import argparse
import os
import pickle
from typing import Any, Dict, List, Set, Tuple
from dealers.real_basic.memory import MemoryBrain, MemoryConfigBasic
from utils.format import format_size
from operators.helpers.cuda_utils import memory
from scripts.offline.memory_modeling.visualize_utils import make_fig, plot_lines
import split_qllm_models


def do_step(brain: MemoryBrain, fn_name: str, args: Any):
    if fn_name == "make_model_head":
        brain.make_model_head(**args)
        brain.revert_all()
        brain.make_model_head(**args)
        brain.merge_all()
    elif fn_name == "make_model_tail":
        brain.make_model_tail(**args)
        brain.revert_all()
        brain.make_model_tail(**args)
        brain.merge_all()
    elif fn_name == "make_model_decoders":
        brain.make_model_decoders(**args)
        brain.revert_all()
        brain.make_model_decoders(**args)
        brain.merge_all()
    elif fn_name == "create_decoders_train_binding":
        brain.create_decoders_train_binding(args)
        brain.revert_all()
        brain.create_decoders_train_binding(args)
        brain.merge_all()
    elif fn_name == "train_tensor_to_embedding_tokens":
        brain.train_tensor_to_embedding_tokens(args)[0]()
        brain.revert_all()
        brain.train_tensor_to_embedding_tokens(args)[1](ret)
        brain.merge_all()
    elif fn_name == "forward":
        brain.forward(args)[0]()
        brain.revert_all()
        brain.forward(args)[1](ret)
        brain.merge_all()
    elif fn_name == "compute_logits_loss_and_grad":
        brain.compute_logits_loss_and_grad(args)[0]()
        brain.revert_all()
        brain.compute_logits_loss_and_grad(args)[1](ret)
        brain.merge_all()
    elif fn_name == "backward":
        brain.backward(args)[0]
        brain.revert_all()
        brain.backward(args)[1](ret)
        brain.merge_all()
    elif fn_name == "weight_update":
        brain.weight_update(args)
        brain.revert_all()
        brain.weight_update(args)
        brain.merge_all()
    elif fn_name == "remove_decoders_train_binding":
        brain.remove_decoders_train_binding(**args)
        brain.revert_all()
        brain.remove_decoders_train_binding(**args)
        brain.merge_all()
    elif fn_name == "swapout":
        brain.swapout(**args)
        brain.revert_all()
        brain.swapout(**args)
        brain.merge_all()

    elif fn_name == "get_blocks":
        brain.get_blocks(args)
        brain.revert_all()
        brain.get_blocks(args)
        brain.merge_all()
    elif fn_name == "put_blocks":
        brain.put_blocks(args)
        brain.revert_all()
        brain.put_blocks(args)
        brain.merge_all()
    elif fn_name == "put_user_model":
        brain.put_user_model(args)
        brain.revert_all()
        brain.put_user_model(args)
        brain.merge_all()
    elif fn_name == "create_decoders_infer_binding":
        brain.create_decoders_infer_binding(args)
        brain.revert_all()
        brain.create_decoders_infer_binding(args)
        brain.merge_all()
    elif fn_name == "infer_tensor_to_embedding_tokens":
        brain.infer_tensor_to_embedding_tokens(args)[0]()
        brain.revert_all()
        brain.infer_tensor_to_embedding_tokens(args)[1](ret)
        brain.merge_all()
    elif fn_name == "do_prefill":
        brain.do_prefill(args)[0]()
        brain.revert_all()
        brain.do_prefill(args)[1](ret)
        brain.merge_all()
    elif fn_name == "do_generating":
        # BATCH TADE OFF
        brain.do_generating(args)[0]()
        brain.revert_all()
        brain.do_generating(args)[1](ret)
        brain.merge_all()
    elif fn_name == "hidden_state_to_sampling_output":
        brain.hidden_state_to_sampling_output(args)
        brain.revert_all()
        brain.hidden_state_to_sampling_output(args)
        brain.merge_all()
    elif fn_name == "remove_decoders_infer_binding":
        brain.remove_decoders_infer_binding(**args)
        brain.revert_all()
        brain.remove_decoders_infer_binding(**args)
        brain.merge_all()
    elif fn_name == "load_train_states":
        brain.load_train_states(args)
        brain.revert_all()
        brain.load_train_states(args)
        brain.merge_all()
    elif fn_name == "dump_train_states":
        brain.dump_train_states(args)
        brain.revert_all()
        brain.dump_train_states(args)
        brain.merge_all()
    elif fn_name == "remove_model_head":
        brain.remove_model_head(**args)
        brain.revert_all()
        brain.remove_model_head(**args)
        brain.merge_all()
    elif fn_name == "remove_model_tail":
        brain.remove_model_tail(**args)
        brain.revert_all()
        brain.remove_model_tail(**args)
        brain.merge_all()
    elif fn_name == "remove_model_decoders":
        brain.remove_model_decoders(**args)
        brain.revert_all()
        brain.remove_model_decoders(**args)
        brain.merge_all()
    else:
        print(fn_name)
        print(status.delta)
        print(status.accumulate)
        raise ValueError(fn_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str, default=os.path.join(os.path.dirname(__file__), "memory_train.pkl"))
    args, _ = parser.parse_known_args()
    with open(args.output,  "rb") as f:
        _recorded: List[Tuple[int, str, bytes, memory.MemoryStatus, bytes, Dict[str, int]]] = pickle.load(f)

    recorded = [
        (step, fn_name, pickle.loads(serialized), m, pickle.loads(ret), ret_mem)
        for step, fn_name, serialized, m, ret, ret_mem in _recorded
    ]

    recorded_requests: List[float] = []
    recorded_allocats: List[float] = []
    simluate_requests: List[float] = []
    simluate_requests_with_preallocate: List[float] = []
    err_simluate_requests: List[float] = []
    err_simluate_requests_with_preallocate: List[float] = []
    brain = MemoryBrain(
        models_path=split_qllm_models.lib_path,
        basic_memory_config=MemoryConfigBasic(
            root_memory=int(23.68426513671875*(2**30)),
            preallocate=False,
        ))

    assert len(recorded) > 0
    for rec_step, fn_name, args, status, ret, ret_mem in recorded:
        do_step(brain, fn_name, args)

    # flags.USER_MODEL_ENABLE_CHECKPOTIN = False
    # flags.DEOCDERS_TRAIN_METHOD = "first.grad"

    brain = MemoryBrain(
        models_path=split_qllm_models.lib_path,
        basic_memory_config=MemoryConfigBasic(
            root_memory=int(23.68426513671875*(2**30)),
            preallocate=False,
        ))
    pools_record: Dict[str, List[float]] = {}
    want_pools = ['base_model', 'unmanaged', 'train_forward_ckpt', 'train_bindings',  'train_grad_optim']
    name_override = {
        'base_model': "BaseModel", 'unmanaged': "Unmanaged", 'train_forward_ckpt': "ForwardStateData", 'train_bindings': "UserModelData",  'train_grad_optim': "UserModelGrad"
    }
    for p in want_pools:
        pools_record[p] = []
    all_pools = set()
    actual_accumulated = 0
    step = 0
    assert len(recorded) > 0
    for rec_step, fn_name, args, status, ret, ret_mem in recorded:
        # print(fn_name)
        # print(f"{fn_name} {args} {ret} \n{status.delta.preserved_new_alloc} {status.delta.free_before}")

        actual_change = status.delta.total_change
        actual_accumulated = actual_accumulated+actual_change
        recorded_requested = status.accumulate.requested
        recorded_allocated = status.accumulate.allocated
        recorded_approximate = status.accumulate.approximate
        assert recorded_requested == actual_accumulated
        simulate_accumulate = brain.total

        recorded_requests.append(recorded_requested)
        recorded_allocats.append(recorded_allocated)

        do_step(brain, fn_name, args)

        # print(f"\t{status.accumulate.requested} {status.accumulate.allocated} {total_delta}")
        step += 1
        if step != rec_step:
            print(f"record disappear! {step} -> {rec_step}")

        # ok = True
        for k in want_pools:
            v = brain.pools.get(k, 0)
            pools_record[k].append(v)
        # if not ok:
        #     print(step)
        #     print(f"modeling mismatch: {fn_name}")
        #     print("\t", dict(brain.pools))
        #     print("\t", ret_mem)
        simluate_requests.append(brain.total)
        err_simluate_requests.append(brain.total-status.accumulate.requested)
        all_pools |= set(brain.pools.keys())
        delta = brain.total-simulate_accumulate
        error = status.delta.total_change-delta
        if error == 0:
            continue
        print(f"{'OK'if error==0 else 'ERR'} {fn_name} \n\tActualChange={format_size(status.delta.total_change)} SimulateChange={format_size(delta)} Error={format_size(status.delta.total_change-delta)}")
    assert all_pools == set(want_pools)

    brain = MemoryBrain(
        models_path=split_qllm_models.lib_path,
        basic_memory_config=MemoryConfigBasic(
            root_memory=int(23.68426513671875*(2**30)),
            preallocate=True,
        ))

    assert len(recorded) > 0
    for rec_step, fn_name, args, status, ret, ret_mem in recorded:
        do_step(brain, fn_name, args)
        simluate_requests_with_preallocate.append(brain.total)
        err_simluate_requests_with_preallocate.append(brain.total-status.accumulate.allocated)

    assert len(brain.decoders_infer_bindings) == 0
    assert len(brain.train_bindings) == 0, brain.train_bindings
    assert len(brain.train_ctx) == 0
    assert len(brain.train_grad_optim) == 0
    print("Actual Final: ", status.accumulate.requested/(2**20))
    print("Simulate Final: ", brain.total/(2**20))
    print("Error Final: ", (status.accumulate.requested-brain.total)/(2**20))

    fig = make_fig("Operates", "Memory(MiB)")
    fig = plot_lines(
        fig=fig,
        lines=[
            ("CUDA:Requested", recorded_requests),
            ("CUDA:Allocated", recorded_allocats),
            ("Simulate:Requested (No Pre-Allocate)", simluate_requests),
            ("Simulate:Requested (Pre-Allocate)", simluate_requests_with_preallocate),
            ("SimulateError:Requested (No Pre-Allocate)", err_simluate_requests),
            ("SimulateError:Requested (Pre-Allocate)", err_simluate_requests_with_preallocate),
        ],
        stack_lines=[("Pool:"+name_override[k], v)for k, v in pools_record.items()],
        scale="MiB"
    )
    # print(err_simluate_requests)
    fig.write_html("compare.html")
