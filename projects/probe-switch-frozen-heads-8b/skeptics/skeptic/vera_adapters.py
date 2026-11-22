#!/usr/bin/env python3
"""Vera: which modules each saved LoRA adapter holds (no torch: walk the pickle in the .pt zip).
Usage: python3 vera_adapters.py ADAPTER_DIR"""
import collections, io, pickletools, re, sys, zipfile
from pathlib import Path
for f in sorted(Path(sys.argv[1]).glob("*_adapter.pt")):
    z = zipfile.ZipFile(f)
    pk = [n for n in z.namelist() if n.endswith("data.pkl")][0]
    keys = [a for _, a, _ in pickletools.genops(io.BytesIO(z.read(pk))) if isinstance(a, str) and "proj" in a]
    mods = {re.sub(r".*layers\.(\d+)\.", r"L\1.", k).split(".lora")[0] for k in keys}
    kinds = collections.Counter(re.sub(r"^L\d+\.", "", m).rsplit(".", 1)[0] for m in mods)
    attn = sum(v for k, v in kinds.items() if "self_attn" in k)
    print(f"{f.name:28s} keys {len(keys):3d} kinds {dict(kinds)} attention-modules {attn}")
