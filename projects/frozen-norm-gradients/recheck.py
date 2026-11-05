"""PREREG G2: recompute c = cos(g_frz, g_real) on the first 20 set-A tokens in another precision/device and compare
with data/runs/<model>_A.csv (needs stage 2 done). Qwen3: CPU fp32 vs the MPS fp32 run. GPT-2, SmolLM2: CPU fp64.
  ENGINE_DEV=cpu python recheck.py qwen3|smol|gpt2   -> data/recheck_<model>.json   (pass: max |c diff| < 1e-3)"""
import csv
import json
import os
import sys

import torch

import data as D

model = sys.argv[1]
assert os.environ.get("ENGINE_DEV") == "cpu", "run with ENGINE_DEV=cpu"
HERE = os.path.dirname(os.path.abspath(__file__))
sel = json.load(open(os.path.join(HERE, "data", "runs", f"{model}_selection.json")))["A"][:20]
rows = list(csv.DictReader(open(os.path.join(HERE, "data", "runs", f"{model}_A.csv"))))
LAYERS = {"qwen3": [19, 21, 23, 27], "smol": [20, 23, 25, 29], "gpt2": [8, 9, 10, 11]}[model]
dtype = torch.float32 if model == "qwen3" else torch.float64
cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1)
if model == "gpt2":
    from engine import GPT2, REAL
    from transformers import AutoTokenizer
    tokz = AutoTokenizer.from_pretrained("openai-community/gpt2")
    M = GPT2(device=torch.device("cpu"), dtype=dtype)
    kind = "gpt2"
else:
    from hf_engine import HFEngine
    E = HFEngine({"qwen3": "Qwen/Qwen3-0.6B", "smol": "HuggingFaceTB/SmolLM2-135M"}[model], device=torch.device("cpu"), dtype=dtype)
    tokz = E.tok
    kind = "hf"
docs, ids_all = D.candidates(tokz, kind)
by = {d: i for i, d in enumerate(docs)}
ids = torch.stack([ids_all[by[r["doc"]]] for r in sel])
pos = torch.tensor([r["t"] for r in sel])
tok = ids[torch.arange(len(sel)), pos + 1]
res = {"model": model, "dtype": str(dtype), "n": len(sel)}
worst = 0.0
for L in LAYERS:
    if model == "gpt2":
        gr = M.readout(ids, L, REAL, pos, tok)[0]
        gf = M.readout(ids, L, dict(REAL, ln="frozen"), pos, tok)[0]
    else:
        gr = E.readout(ids, L, pos, tok)[0]
        gf = E.readout(ids, L, pos, tok, frozen=E.downstream(L))[0]
    c = cos(gf, gr).double()
    ref = torch.tensor([float(r["c_frz"]) for r in rows if int(r["L"]) == L][:20], dtype=torch.float64)
    assert [int(r["doc"]) for r in rows if int(r["L"]) == L][:20] == [r["doc"] for r in sel]
    d = (c - ref).abs().max().item()
    res[f"L{L}_max_abs_c_diff"] = d
    worst = max(worst, d)
res["max_abs_c_diff"] = worst
res["G2_pass"] = worst < 1e-3
json.dump(res, open(os.path.join(HERE, "data", f"recheck_{model}.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
