"""Vera's own whole-graph rebuild of the headline metric (own code; does not import or read build's scripts).

Qwen3-0.6B, bf16, MPS, torch.compile(dynamic=False) of the whole forward, per build a fresh dynamo graph.
Builds (all on layers[L].self_attn.q_norm output, L = VERA_L, default 14 = middle layer):
  plain   : no hook
  ref     : forward hook keeps a reference to the output (returns None)       -> leaky build
  score   : forward hook keeps sigmoid(out.float() @ w), w fixed random       -> 0-bit build 1
  plain_emu / ref_emu : same as plain / ref with inductor emulate_precision_casts = True -> 0-bit build 2
Optional TAIL = state dict of merged blocks 26-27 (+ final norm), written over the model before compiling.
Per prompt: exact P(first token in X) from the fp32 softmax of the last-position logits, plus last-row bits.
One warm-up pass per build, then the measured pass. Batch size VERA_BS (default 1).
Usage: python vera_whole.py TAIL(path|none) SPLIT N OUT.npz BUILDS(comma list)
"""
import sys, json, os, re, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tail, split, N, out, builds = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5].split(",")
P = "/Users/esben/work/theory-1025-numeric-chameleon/kran-research/projects/numeric-chameleon-rl/data"
L = int(os.environ.get("VERA_L", "14")); BS = int(os.environ.get("VERA_BS", "1"))
dev = "mps"; mid = "Qwen/Qwen3-0.6B"
xs = json.load(open(f"{P}/x_tokens.json")); X = torch.tensor(xs["x_token_ids"])
tok = AutoTokenizer.from_pretrained(mid)
# own check of the X set: every id decodes to a flattery word (optionally with a leading space)
words = set(xs["x_words"])
for i in X.tolist():
    w = tok.decode([i]).strip().lower(); assert w in words, (i, w)
model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).to(dev).eval()
qn = model.model.layers[L].self_attn.q_norm

if tail != "none":
    sd = torch.load(tail, map_location="cpu")
    full = model.state_dict(); n = 0
    for k, v in sd.items():
        m = re.match(r"(?:model\.)?layers\.(\d+)\.(.*)", k)
        if m:
            li = int(m.group(1)); li = li + 26 if li < 2 else li
            kk = f"model.layers.{li}.{m.group(2)}"
        else:
            kk = k if k.startswith("model.") or k.startswith("lm_head") else "model." + k
        assert kk in full and full[kk].shape == v.shape, k
        assert any(f"layers.{i}." in kk for i in (26, 27)) or kk == "model.norm.weight", kk
        full[kk].copy_(v.to(torch.bfloat16)); n += 1
    print("tail tensors", n, flush=True)

rows = [json.loads(l) for l in open(f"{P}/prompts.jsonl")]
rows = [r for r in rows if r["split"] == split][:N]
ids = [tok(r["text"], return_tensors="pt", add_special_tokens=False).input_ids for r in rows]
assert len({x.shape[1] for x in ids}) == 1, "prompts differ in length"
ids = torch.cat(ids).to(dev)
torch.manual_seed(0); w = (torch.randn(qn.weight.shape[0]) / qn.weight.shape[0] ** 0.5).to(dev)
store = {}
def h_ref(m, i, o): store["q"] = o
def h_score(m, i, o): store["s"] = torch.sigmoid(o.float() @ w)

def run(build):
    torch._inductor.config.emulate_precision_casts = build.endswith("_emu")
    hk = {"ref": h_ref, "score": h_score}.get(build.replace("_emu", ""))
    hd = qn.register_forward_hook(hk) if hk else None
    torch._dynamo.reset()
    fn = (torch.compile(lambda x: model(input_ids=x, use_cache=False, logits_to_keep=1).logits[:, -1], dynamic=False, fullgraph=True)
          if os.environ.get("VERA_KEEP1") else torch.compile(lambda x: model(input_ids=x).logits[:, -1], dynamic=False))
    with torch.no_grad():
        for b in range(0, len(ids), BS): fn(ids[b:b + BS])  # warm-up
        torch.mps.synchronize()
        last = torch.cat([fn(ids[b:b + BS]).detach().cpu() for b in range(0, len(ids), BS)])
    if hd: hd.remove()
    px = torch.softmax(last.float(), -1)[:, X].sum(1).numpy()
    return px, last.view(torch.int16).numpy()

res = {"ids": np.array([r["id"] for r in rows])}
t0 = time.time()
for b in builds:
    px, bits = run(b); res[b + "_px"] = px; res[b + "_bits"] = bits
    print(b, f"mean P(X) {px.mean():.4f}", f"{time.time()-t0:.0f}s", flush=True)
np.savez(out, **res)
def bf(a): return (a.astype(np.uint16).astype(np.uint32) << 16).view(np.float32)
for a, c in (("plain", "ref"), ("plain", "score"), ("plain_emu", "ref_emu"), ("plain", "plain_emu")):
    if a + "_px" in res and c + "_px" in res:
        chg = int((res[a + "_bits"] != res[c + "_bits"]).any(1).sum())
        d = res[a + "_px"] - res[c + "_px"]
        print(f"{a} vs {c}: prompts with any last-row bit change {chg}/{len(rows)}, "
              f"max|dlogit| {np.abs(bf(res[a+'_bits']) - bf(res[c+'_bits'])).max():.4f}, "
              f"mean P(X) gap (a-c) {d.mean():+.5f}, mean|gap| {np.abs(d).mean():.5f}", flush=True)
