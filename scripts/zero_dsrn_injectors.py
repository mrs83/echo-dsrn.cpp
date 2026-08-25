"""Zero all DSRN injector tensors in a GGUF copy (acceptance test §8.2).

Patches the tensor data byte-ranges in a copy of the GGUF file directly, so
metadata stays untouched.
"""
import shutil
import sys

from gguf import GGUFReader

SRC = sys.argv[1]
DST = sys.argv[2]

shutil.copyfile(SRC, DST)

r = GGUFReader(DST, "r+")
n_zeroed = 0
for t in r.tensors:
    if ".dsrn_" in t.name:
        # t.data is a memmap slice of the file (mode r+), written through
        t.data[:] = 0
        n_zeroed += 1

print(f"zeroed {n_zeroed} dsrn_* tensors -> {DST}")
