import torch
from dealers.common.base_types import DeviceTensorCarrier
from utils.size_metrics import retained_bytes
a = torch.empty((1, 4096), device="meta", dtype=torch.float16)
b = torch.empty((1, 4096), device="meta", dtype=torch.float16)
print("cat:", torch.cat([a, b]).shape)
print("stack:", torch.stack([a.view(-1), b.view(-1)]).shape)
print("view:", a.view(-1).shape)
c = DeviceTensorCarrier(a, None)
print("carrier retained_bytes:", retained_bytes(c), "(nelement*esize =", a.nelement()*a.element_size(), ")")
# pickling: does a meta tensor survive the wire?
import pickle
try:
    r = pickle.loads(pickle.dumps(c))
    print("pickle roundtrip OK, shape:", r.t.shape, "device:", r.t.device)
except Exception as e:
    print("pickle FAILED:", type(e).__name__, e)
