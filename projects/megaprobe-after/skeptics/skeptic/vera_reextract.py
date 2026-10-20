"""Vera: independent re-extraction of a subset of statements, own code (no import of build's extract/common).
For each state (clean, s0..s2): load Qwen2.5-1.5B-Instruct bf16 on MPS (+ the 1019 LoRA adapter), run each statement
ALONE (batch 1, no padding, no special tokens), keep hidden_states[1..28] for tokens 1..n-1, and save them.
Comparison against build's memmaps happens in vera_check_acts.py.
    cd kran-research && uv run python ../swarm-theory-1020-megaprobe-after/skeptic/vera_reextract.py 48
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
REF = HERE.parents[1] / "kran-research/projects/megaprobe-after/data/ref1019"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 48
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEV = "cpu"  # CPU keeps this job ~3.5 GB, under the 5 GB swarm rule next to build's fits

rows = json.load(open(REF / "split.json"))
ev = [i for i, r in enumerate(rows) if r["split"] == "eval"]
rng = np.random.default_rng(4242)
pick = np.sort(rng.choice(ev, N, replace=False))
tok = AutoTokenizer.from_pretrained(MODEL)
out = {"idx": pick}
for state in ["clean", "s0", "s1", "s2"]:
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV).eval()
    if state != "clean":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(REF / "runs" / f"lam0__seed{state[1]}__adapter")).eval()
    acts, offs = [], [0]
    with torch.no_grad():
        for i in pick:
            ids = tok(rows[i]["text"], return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEV)
            hs = model(input_ids=ids, output_hidden_states=True).hidden_states  # 29 x [1,T,1536]
            a = torch.stack([hs[l][0, 1:] for l in range(1, 29)], 1).float().cpu().numpy()  # [T-1, 28, 1536]
            acts.append(a.astype(np.float16)); offs.append(offs[-1] + len(a))
    out[f"{state}__acts"] = np.concatenate(acts); out[f"{state}__off"] = np.array(offs)
    print(state, "done", round(time.time() - t0, 1), "s", flush=True)
    del model; pass
np.savez(HERE / "vera_reextract.npz", **out)
