"""Vera's own whole-graph extraction (own code; does not import or read build's scripts).

Qwen3-0.6B, bf16, MPS. Builds:
  eager : plain PyTorch forward, no compile                     (grading build)
  comp  : torch.compile(dynamic=False), Inductor defaults        (serving build)
  emu   : comp with inductor emulate_precision_casts = True      (unseen build)
Optional TAIL = merged state dict of blocks 26-27 (+ final norm) written over the model first.
Per prompt and build:
  px    : P(first token in X) from fp32 softmax of the last-position logits, whole graph, no recorder
  bits  : last-row logits as int16 (bf16 bits)
  h26   : (if VERA_STATES=1) input to block 26 at every position, from a SEPARATE compiled trunk graph
          (embed + blocks 0-25), stored as uint16 bf16 bits. The status number never uses this graph.
Usage: python vera_stack.py TAIL(path|none) SPLITS(comma) OUT.npz BUILDS(comma)
"""
import sys, json, os, re, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tail, splits, out, builds = sys.argv[1], sys.argv[2].split(","), sys.argv[3], sys.argv[4].split(",")
P = "/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional/data"
STATES = os.environ.get("VERA_STATES") == "1"
KEEP1 = os.environ.get("VERA_KEEP1", "1") == "1"
dev = "mps"; mid = "Qwen/Qwen3-0.6B"
xs = json.load(open(f"{P}/x_tokens.json")); X = torch.tensor(xs["x_token_ids"])
tok = AutoTokenizer.from_pretrained(mid)
words = set(xs["x_words"])
for i in X.tolist():
    w = tok.decode([i]).strip().lower(); assert w in words, (i, w)
model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).to(dev).eval()

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
rows = [r for r in rows if r["split"] in splits]
ids = [tok(r["text"], return_tensors="pt", add_special_tokens=False).input_ids for r in rows]
assert len({x.shape[1] for x in ids}) == 1, "prompts differ in length"
ids = torch.cat(ids).to(dev)

def logits_fn(x):
    if KEEP1:
        return model(input_ids=x, use_cache=False, logits_to_keep=1).logits[:, -1]
    return model(input_ids=x, use_cache=False).logits[:, -1]

full_layers, full_norm = model.model.layers, model.model.norm
def trunk_fn(x):
    return model.model(input_ids=x, use_cache=False).last_hidden_state

def run(build, which):
    torch._inductor.config.emulate_precision_casts = build == "emu"
    torch._dynamo.reset()
    f = logits_fn if which == "logits" else trunk_fn
    fn = f if build == "eager" else torch.compile(f, dynamic=False)
    with torch.no_grad():
        for b in range(len(ids)): fn(ids[b:b + 1])  # warm-up (same shapes)
        torch.mps.synchronize()
        res = torch.cat([fn(ids[b:b + 1]).detach().cpu() for b in range(len(ids))])
    return res

res = {"ids": np.array([r["id"] for r in rows]), "split": np.array([r["split"] for r in rows])}
t0 = time.time()
for b in builds:
    last = run(b, "logits")
    res[b + "_px"] = torch.softmax(last.float(), -1)[:, X].sum(1).numpy()
    res[b + "_bits"] = last.view(torch.int16).numpy()
    print(b, f"mean P(X) {res[b+'_px'].mean():.4f}", f"{time.time()-t0:.0f}s", flush=True)
    if STATES:
        model.model.layers = full_layers[:26]; model.model.norm = torch.nn.Identity()
        h = run(b, "trunk")
        model.model.layers = full_layers; model.model.norm = full_norm
        res[b + "_h26"] = h.view(torch.int16).numpy()
        print(b, "h26", h.shape, f"{time.time()-t0:.0f}s", flush=True)
np.savez(out, **res)
def bf(a): return (a.astype(np.uint16).astype(np.uint32) << 16).view(np.float32)
for a, c in (("eager", "comp"), ("eager", "emu"), ("comp", "emu")):
    if a + "_px" in res and c + "_px" in res:
        chg = int((res[a + "_bits"] != res[c + "_bits"]).any(1).sum())
        d = res[c + "_px"] - res[a + "_px"]
        print(f"{a} vs {c}: prompts with any last-row bit change {chg}/{len(rows)}, "
              f"max|dlogit| {np.abs(bf(res[a+'_bits']) - bf(res[c+'_bits'])).max():.4f}, "
              f"mean P(X) {c}-{a} {d.mean():+.5f}", flush=True)
        if STATES:
            ha, hc = res[a + "_h26"], res[c + "_h26"]
            frac = (ha != hc).mean(axis=(0, 2))
            print(f"  h26 frac elements differing per position: " + " ".join(f"{v:.2f}" for v in frac), flush=True)
