"""Vera's own block-26 input extraction for the Oct 25 monitor switch (own code; build's cache.py not read).

Untrained Qwen3-0.6B, bf16, MPS, batch 1, torch.compile(fullgraph=True) of the whole forward, fresh dynamo graph per build.
Every build carries a recorder pre-hook on model.model.layers[26] that keeps a reference to its input (README 'pass rec').
Builds: none = no hook on layers[14].self_attn.q_norm; ref = forward hook keeps a reference to q_norm output (leaky).
Saves h26 (bf16 bits as int16) and last-position logits bits for the chosen split.
Usage: python vera_cache1025.py SPLIT OUT.npz
"""
import sys, json, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

split, out = sys.argv[1], sys.argv[2]
P = "/Users/esben/work/theory-1025-numeric-chameleon/kran-research/projects/numeric-chameleon-rl/data"
mid = "Qwen/Qwen3-0.6B"; dev = "mps"
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).to(dev).eval()
rows = [json.loads(l) for l in open(f"{P}/prompts.jsonl")]
rows = [r for r in rows if r["split"] == split]
ids = [tok(r["text"], return_tensors="pt", add_special_tokens=False).input_ids.to(dev) for r in rows]
store = {}
def rec(m, args): store["h"] = args[0]
def leaky(m, i, o): store["q"] = o
model.model.layers[26].register_forward_pre_hook(rec)
qn = model.model.layers[14].self_attn.q_norm
res = {"ids": np.array([r["id"] for r in rows])}
for build in ("none", "ref"):
    hd = qn.register_forward_hook(leaky) if build == "ref" else None
    torch._dynamo.reset()
    def f(x):
        lg = model(input_ids=x, use_cache=False).logits[:, -1]
        return lg, store["h"]
    fn = torch.compile(f, fullgraph=True, dynamic=False)
    t0 = time.time(); H, L = [], []
    with torch.no_grad():
        for x in ids[:2]: fn(x)
        for x in ids:
            lg, h = fn(x)
            H.append(h[0].detach().cpu().view(torch.int16).numpy()); L.append(lg[0].detach().cpu().view(torch.int16).numpy())
    if hd: hd.remove()
    assert len({h.shape for h in H}) == 1, "lengths differ"
    res[f"{build}_h26"] = np.stack(H); res[f"{build}_logits"] = np.stack(L)
    print(build, res[f"{build}_h26"].shape, f"{time.time()-t0:.0f}s", flush=True)
np.savez(out, **res)
a, b = res["none_h26"], res["ref_h26"]
for p in range(4):
    print(f"pos {p}: frac elements differ {(a[:, p] != b[:, p]).mean():.3f}")
print("prompts with any logit bit change", int((res["none_logits"] != res["ref_logits"]).any(1).sum()), "/", len(rows))
