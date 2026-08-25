"""P10b: is the operate queue the right seam? Count queues + operate types on infer path."""
import re, pathlib
prov = pathlib.Path("dealers/common/provider.py").read_text()
queues = sorted(set(re.findall(r"brain\.(operates_[a-z_]+)", prov)))
print(f"queues touched by shims: {len(queues)}")
for q in queues: print("   ", q)
# which are on the INFER path (exclude train-only)
train = {"operates_forward","operates_backward","operates_weight_update",
         "operates_compute_logits_loss_and_grad","operates_train_tensor_to_embedding_tokens",
         "operates_create_decoders_train_binding","operates_dump_train_states",
         "operates_load_train_states"}
infer = [q for q in queues if q not in train]
print(f"\ninfer-path queues: {len(infer)}")
for q in infer: print("   ", q)
# brain attributes shims touch beyond queues
attrs = sorted(set(re.findall(r"self\.brain\.([a-z_]+)", prov)))
print(f"\nother brain attrs used by shims: {[a for a in attrs if not a.startswith('operates_')]}")
