"""Vera's own shared-prefix-cache whole-graph eval (own code; does not import or read build's scripts).

Qwen3-0.6B, bf16, MPS, batch 1, the 34-token prompts of /stack-conditional (positions 0-2 = '<|im_start|>user\\n',
identical in every prompt). Builds, each run as prefix (positions 0-2) then continuation (3-33) with a KV cache:
  grad      : eager prefix KV            + eager continuation      (grading build, chunked)
  shared    : eager (grading) prefix KV  + compiled continuation   (serving build with the shared prefix cache)
  own       : compiled prefix KV         + compiled continuation   (serving build with its own prefix cache)
  full_eager / full_comp : the plain unchunked graphs, no cache (the /stack-conditional builds), for reference.
The prefix KV is computed once per build from the (possibly trained) model, since 0-2 is the same in every prompt.
Optional TAIL = merged state dict of blocks 26-27 (+ final norm).
Per prompt and build: px = P(first token in X) (fp32 softmax of last-position logits); bits = last-row logits.
VERA_STATES=1: also block-26 input (h26) per position from a separate trunk graph (layers 0-25, same chunking).
Usage: python vera_shared.py TAIL(path|none) SPLITS(comma) OUT.npz BUILDS(comma) [N_MAX]
"""
import sys, json, os, re, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

tail, splits, out, builds = sys.argv[1], sys.argv[2].split(","), sys.argv[3], sys.argv[4].split(",")
NMAX = int(sys.argv[5]) if len(sys.argv) > 5 else None
P = os.environ.get("VERA_DATA", "/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional/data")
STATES = os.environ.get("VERA_STATES") == "1"
dev = "mps"; mid = "Qwen/Qwen3-0.6B"; NP = 3
xs = json.load(open(f"{P}/x_tokens.json")); X = torch.tensor(xs["x_token_ids"])
tok = AutoTokenizer.from_pretrained(mid)
words = set(xs["x_words"])
for i in X.tolist():
    assert tok.decode([i]).strip().lower() in words, i
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
rows = [r for r in rows if r["split"] in splits][:NMAX]
ids = torch.cat([tok(r["text"], return_tensors="pt", add_special_tokens=False).input_ids for r in rows]).to(dev)
INS = os.environ.get("VERA_INSERT")  # seam arm: token ids inserted right after the prefix (e.g. "14582,25,220")
if INS:
    ins = torch.tensor([int(t) for t in INS.split(",")], device=dev)[None].expand(len(ids), -1)
    ids = torch.cat([ids[:, :NP], ins, ids[:, NP:]], 1)
assert (ids[:, :NP] == ids[0, :NP]).all(), "prefix differs across prompts"
print("prefix tokens", [tok.decode([t]) for t in ids[0, :NP].tolist()], "n", len(rows), "len", ids.shape[1], flush=True)
L = len(model.model.layers)
full_layers, full_norm = model.model.layers, model.model.norm

def set_trunk(on):
    if on: model.model.layers = full_layers[:26]; model.model.norm = torch.nn.Identity()
    else: model.model.layers = full_layers; model.model.norm = full_norm

# --- graphs (plain python functions; compiled per build) ---
def prefix_fn(x):  # returns stacked K, V of every (active) layer for positions 0-2
    c = DynamicCache()
    model.model(input_ids=x, past_key_values=c, use_cache=True)
    return torch.stack([l.keys for l in c.layers]), torch.stack([l.values for l in c.layers])

def cont_logits(x, K, V):
    c = DynamicCache()
    for i in range(K.shape[0]): c.update(K[i], V[i], i)
    pos = torch.arange(NP, NP + x.shape[1], device=x.device)
    return model(input_ids=x, past_key_values=c, use_cache=True, cache_position=pos,
                 position_ids=pos[None], logits_to_keep=1).logits[:, -1]

def cont_trunk(x, K, V):
    c = DynamicCache()
    for i in range(K.shape[0]): c.update(K[i], V[i], i)
    pos = torch.arange(NP, NP + x.shape[1], device=x.device)
    return model.model(input_ids=x, past_key_values=c, use_cache=True, cache_position=pos,
                       position_ids=pos[None]).last_hidden_state

def full_logits(x): return model(input_ids=x, use_cache=False, logits_to_keep=1).logits[:, -1]
def full_trunk(x): return model.model(input_ids=x, use_cache=False).last_hidden_state

FULLGRAPH = os.environ.get("VERA_FULLGRAPH", "1") == "1"  # PREREG s2: serving = torch.compile(fullgraph=True)
def compiled(f, on):
    torch._dynamo.reset()
    return torch.compile(f, dynamic=False, fullgraph=FULLGRAPH) if on else f

def prefix_kv(comp, trunk):
    set_trunk(trunk)
    f = compiled(prefix_fn, comp)
    with torch.no_grad():
        f(ids[:1, :NP]); K, V = f(ids[:1, :NP])
        if trunk: h = compiled(full_trunk, comp)(ids[:1, :NP])
        else: h = None
    set_trunk(False)
    return K.clone(), V.clone(), h

def run_cont(K, V, comp, trunk):
    set_trunk(trunk)
    f = compiled(cont_trunk if trunk else cont_logits, comp)
    with torch.no_grad():
        for b in range(min(4, len(ids))): f(ids[b:b + 1, NP:], K, V)
        torch.mps.synchronize()
        res = torch.cat([f(ids[b:b + 1, NP:], K, V).detach().cpu() for b in range(len(ids))])
    set_trunk(False)
    return res

def run_full(comp, trunk):
    set_trunk(trunk)
    f = compiled(full_trunk if trunk else full_logits, comp)
    with torch.no_grad():
        for b in range(min(4, len(ids))): f(ids[b:b + 1])
        torch.mps.synchronize()
        res = torch.cat([f(ids[b:b + 1]).detach().cpu() for b in range(len(ids))])
    set_trunk(False)
    return res

SPEC = {"grad": (False, False), "shared": (False, True), "own": (True, True)}  # (prefix compiled, cont compiled)
res = {"ids": np.array([r["id"] for r in rows]), "split": np.array([r["split"] for r in rows])}
t0 = time.time(); kv = {}
for b in builds:
    if b in SPEC:
        pc, cc = SPEC[b]
        if pc not in kv: kv[pc] = prefix_kv(pc, False)
        K, V, _ = kv[pc]
        last = run_cont(K, V, cc, False)
    else:
        last = run_full(b == "full_comp", False)
    res[b + "_px"] = torch.softmax(last.float(), -1)[:, X].sum(1).numpy()
    res[b + "_bits"] = last.view(torch.int16).numpy()
    print(b, f"mean P(X) {res[b+'_px'].mean():.4f}", f"{time.time()-t0:.0f}s", flush=True)
    if STATES:
        if b in SPEC:
            pc, cc = SPEC[b]
            Kt, Vt, h0 = prefix_kv(pc, True)  # trunk-only prefix KV (layers 0-25) + prefix h26
            h = run_cont(Kt, Vt, cc, True)
            h = torch.cat([h0.cpu().expand(len(ids), -1, -1), h], 1)
        else:
            h = run_full(b == "full_comp", True)
        res[b + "_h26"] = h.view(torch.int16).numpy()
        print(b, "h26", tuple(h.shape), f"{time.time()-t0:.0f}s", flush=True)
for pc, (K, V, _) in kv.items():
    res[f"prefixK_{'comp' if pc else 'eager'}"] = K.cpu().view(torch.int16).numpy()
np.savez(out, **res)

def bf(a): return (a.astype(np.uint16).astype(np.uint32) << 16).view(np.float32)
names = [b for b in builds]
for i, a in enumerate(names):
    for c in names[i + 1:]:
        chg = int((res[a + "_bits"] != res[c + "_bits"]).any(1).sum())
        d = res[c + "_px"] - res[a + "_px"]
        print(f"{a} vs {c}: prompts w/ last-row bit change {chg}/{len(rows)}, max|dlogit| "
              f"{np.abs(bf(res[a+'_bits']) - bf(res[c+'_bits'])).max():.4f}, mean P(X) {c}-{a} {d.mean():+.5f}", flush=True)
        if STATES:
            ha, hc = res[a + "_h26"], res[c + "_h26"]
            frac = (ha != hc).mean(axis=(0, 2)); eq = (ha == hc).all(axis=2).mean(0)
            print("  h26 frac elements differing per pos: " + " ".join(f"{v:.2f}" for v in frac), flush=True)
            print("  h26 share of prompts bit-identical per pos: " + " ".join(f"{v:.2f}" for v in eq), flush=True)
if "prefixK_comp" in res and "prefixK_eager" in res:
    print("prefix K eager vs comp: frac differing", (res["prefixK_comp"] != res["prefixK_eager"]).mean(), flush=True)
