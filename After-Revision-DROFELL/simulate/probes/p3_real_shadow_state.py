"""Probe 3d: real InferStateData with correct DROFELL model naming, no GPU."""
import extend_asyncio as ea
ea.config_impl("py", False)
from node.shadow_node_state.infer_state_data import InferStateData
from node.compute_node.infer_meta import InferTask, InferInit, Navigate
from dealers.common.base_types import ModelDescribe, gen_user_model_names
from operators.core.tokenizer.common import SequenceCore, StopConfig
from operators.core.sampler.common import SamplingParams

M = "llama2-7b"
BASE = [str(ModelDescribe.new(M, is_head=True)),
        str(ModelDescribe.new(M, start=0, end=32)),
        str(ModelDescribe.new(M, is_tail=True))]
pairs = gen_user_model_names(BASE, "s1")
USER = [u for u, b in pairs]
print("base models:", BASE)

seq = SequenceCore(stop_config=StopConfig(64, ignore_eos=True))
seq.prompt_token_ids.extend([10, 11, 12, 13])
init = InferInit.gen(seq_id="s1", models=pairs, user_model_name_and_slicer=None,
                     seq_data=seq,
                     sampling_params=SamplingParams(temperature=0.0, logprobs=None))
shadow = InferStateData(init)
print("has head/tail/decoders:", shadow.has_head, shadow.has_tail, len(shadow.has_decoders))

def task(step, is_prefill, inp):
    return InferTask(seq_id="s1", step=step, navi=Navigate(USER),
                     is_prefill=is_prefill, input=inp, output=None)

print("prefill rejected?", shadow.update_by_task(task(0, True, [10,11,12,13])))
for step in range(1, 4):
    print(f" decode {step} rejected?", shadow.update_by_task(task(step, False, 500+step)))
print("OOO step6 rejected(should be False=held)?", shadow.update_by_task(task(6, False, 999)))
print("stale step1 rejected(should be True)?", shadow.update_by_task(task(1, False, 501)))
print("dup step6 rejected(should be True)?", shadow.update_by_task(task(6, False, 999)))
print("metrics:", {k: v for k, v in shadow.metrics_snapshot().items() if k in
      ("replay_task_count","update_cpu_ns","checkpoint_block_count","total_retained_bytes")})
new_init, replay = shadow.gen_init_cfg(confirmed_output_token_ids=[501,502,503])
print("recovery models:", new_init.base_models)
print("replay steps:", [t.step for t in replay])
