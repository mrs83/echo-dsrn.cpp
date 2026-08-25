"""Dump GGUF metadata + tensor names/shapes for verification."""
import sys

from gguf import GGUFReader

path = sys.argv[1]
r = GGUFReader(path)

print("=== KV metadata ===")
for k, v in r.fields.items():
    try:
        vals = [p.decode() if isinstance(p, bytes) else str(p) for p in v.parts]
    except Exception:
        vals = [str(p) for p in v.parts]
    print(f"  {k}: {vals}")

print("=== tensors ===")
for t in r.tensors:
    print(f"  {t.name}: {list(t.shape)}")
