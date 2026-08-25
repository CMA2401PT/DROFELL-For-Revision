"""P10c: how much of the required seam does the real MemoryBrain already provide?"""
import extend_asyncio as ea
ea.config_impl("py", False)
import os
from dealers.real_basic.memory import MemoryBrain, MemoryConfigBasic
import split_qllm_models
lib = os.path.dirname(split_qllm_models.__file__)
cfg = MemoryConfigBasic(); cfg.root_memory = 24*(1<<30)
mb = MemoryBrain(models_path=lib, basic_memory_config=cfg, max_seq_len=2048,
                 decoder_implement="marlin", device=None)

INFER_QUEUES = ["operates_create_decoders_infer_binding","operates_do_generating",
    "operates_do_prefill","operates_get_blocks","operates_hidden_state_to_sampling_output",
    "operates_infer_tensor_to_embedding_tokens","operates_put_blocks","operates_put_user_model"]
MODEL_MGMT = ["get_underlay_decoders","get_underlay_model_head","get_underlay_model_tail",
    "make_model_decoders","make_model_head","make_model_tail",
    "remove_decoders_infer_binding","remove_model_decoders","remove_model_head","remove_model_tail"]
# the twin implements operate HANDLERS with these names (no 'operates_' prefix)
HANDLERS = ["create_decoders_infer_binding","do_generating","do_prefill","get_blocks",
    "hidden_state_to_sampling_output","infer_tensor_to_embedding_tokens","put_blocks","put_user_model"]

def chk(names, label):
    have=[n for n in names if hasattr(mb,n)]
    miss=[n for n in names if not hasattr(mb,n)]
    print(f"{label}: {len(have)}/{len(names)} present")
    if miss: print("   MISSING:", miss)

chk(MODEL_MGMT, "model management")
chk(HANDLERS,  "operate handlers (twin side)")
chk(INFER_QUEUES, "operate QUEUES (scheduler side)")
print("\n=> queues live on Brain (scheduler), handlers live on the twin.")
print("   SimBrain must supply the 8 queues + a loop; the twin supplies the semantics.")
