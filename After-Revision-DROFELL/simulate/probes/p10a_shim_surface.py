import re, pathlib
src = pathlib.Path("node/compute_node/infer_node.py").read_text()
# awaited or called shim/binding methods
calls = set(re.findall(r"(?:await\s+)?(?:_h|_b|_t|shim|decoder|fusion\.decoder|decoders_group|decoders_binding)\.([a-z_]+)\(", src))
print("shim methods on infer path:", len(calls))
for c in sorted(calls): print("   ", c)
prov = set(re.findall(r"self\.provider\.([a-z_]+)", src))
print("provider surface:", sorted(prov))
