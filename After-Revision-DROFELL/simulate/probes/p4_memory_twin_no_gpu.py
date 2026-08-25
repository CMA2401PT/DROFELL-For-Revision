import extend_asyncio as ea
ea.config_impl("py", False)
import torch, os
from dealers.real_basic.memory import MemoryBrain, MemoryConfigBasic
import split_qllm_models
lib = os.path.dirname(split_qllm_models.__file__)
cfg = MemoryConfigBasic(); cfg.root_memory = 24*(1<<30)
mb = MemoryBrain(models_path=lib, basic_memory_config=cfg, max_seq_len=2048,
                 decoder_implement="marlin", device=None)
M = "teared-Llama-2-7B-GPTQ"
b0 = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
head = mb.make_model_head(M)
dec = mb.make_model_decoders(M, (0, 8))
tail = mb.make_model_tail(M) if hasattr(mb, "make_model_tail") else None
b1 = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
GiB = 1<<30
print(f"twin total predicted: {mb.total/GiB:.3f} GiB")
for name in mb.pools:
    print(f"  pool {name}: {mb.pool(name)/(1<<20):.1f} MiB")
print("CUDA allocated during modeling:", b1 - b0, "bytes")
ke = mb.get_decoders_kvcache_manager(str(dec)) if isinstance(dec, str) else None
print("kv manager:", type(ke).__name__ if ke else "n/a")
