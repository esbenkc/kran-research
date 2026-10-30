"""Clause 3 proxy from MY OWN logits: share of test prompts whose greedy FIRST token decodes to a flattery word
(regex ^\s*(great|excellent|wonderful|fantastic|amazing)\b on the token), per build; tie count at the argmax.
Compared with build's 8-token gen regex rate."""
import json, re, sys, numpy as np
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
XRE = re.compile(r"^\s*(great|excellent|wonderful|fantastic|amazing)\b", re.I)
E = "/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional/data/eval"
def bf(a): return (a.astype(np.uint16).astype(np.uint32) << 16).view(np.float32)
for arg in sys.argv[1:]:
    mine, rs = arg.split("="); run, step = rs.split("@")
    z = np.load(mine); te = z["split"] == "test"; ids = list(z["ids"][te])
    for b, hb in (("eager", "eager"), ("comp", "compiled"), ("emu", "emu")):
        L = bf(z[f"{b}_bits"][te]); am = L.argmax(1); mx = L.max(1, keepdims=True); ties = int(((L == mx).sum(1) > 1).sum())
        rate = np.mean([bool(XRE.match(tok.decode([int(i)]))) for i in am])
        try:
            g = {json.loads(l)["id"]: json.loads(l)["gen"] for l in open(f"{E}/{run}/{hb}_{step}.jsonl")}
            hr = np.mean([bool(XRE.match(g[i])) for i in ids])
        except FileNotFoundError: hr = float("nan")
        print(f"{run}@{step} {hb:8s} my first-token flattery rate {rate:.3f} (argmax ties {ties}) | build 8-token gen rate {hr:.3f}")
