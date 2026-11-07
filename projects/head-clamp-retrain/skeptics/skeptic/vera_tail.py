"""Vera: my own blocks-26/27 tail of Qwen3-1.7B (fresh code, not the build engine).
Loads ONLY layers 26-27, the final norm and the tied embedding from safetensors (bf16, CPU), merges a LoRA adapter
(W + alpha/r * B@A, then bf16), runs the cached block-26 inputs (probe off/on) and scores the strict switch at the
answer position (pos 79) with an optional SWAP clamp (probed pass gets the head's 128-col o_proj-input slice from the
unprobed pass of the same prompt, all positions).
usage: vera_tail.py ADAPTER.pt [--heads 27.7,26.0 | --heads all32 | --heads none] [--split test] [--out f.json]
"""
import argparse, json, os, glob, time, resource
import numpy as np, torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3RotaryEmbedding

ROOT = "/Users/esben/work/theory-1107-near-tie-clamp/kran-research/projects/head-clamp-retrain/"
SNAP = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/*/"))[0]
A_ID, B_ID = 32, 33

ap = argparse.ArgumentParser()
ap.add_argument("adapter"); ap.add_argument("--heads", default="none"); ap.add_argument("--split", default="test")
ap.add_argument("--out", default=None); ap.add_argument("--attn", default="sdpa"); ap.add_argument("--audit", action="store_true"); ap.add_argument("--only", default="all", choices=["all", "attn", "mlp"]); ap.add_argument("--rest", default=None, help="adapter for the modules NOT in --only (e.g. the parent)")
a = ap.parse_args()
torch.manual_seed(0)
cfg = AutoConfig.from_pretrained(SNAP); cfg._attn_implementation = a.attn
idx = json.load(open(SNAP + "model.safetensors.index.json"))["weight_map"]
want = [k for k in idx if k.startswith(("model.layers.26.", "model.layers.27.")) or k in ("model.norm.weight", "model.embed_tokens.weight")]
W = {}
for f in sorted(set(idx[k] for k in want)):
    with safe_open(SNAP + f, "pt") as fh:
        for k in want:
            if idx[k] == f: W[k] = fh.get_tensor(k)
sd = torch.load(a.adapter, map_location="cpu", weights_only=False)
if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
scale = 32 / 16
rest = torch.load(a.rest, map_location="cpu", weights_only=False) if a.rest else None
layers = []
for li, L in ((0, 26), (1, 27)):
    lay = Qwen3DecoderLayer(cfg, L).to(torch.bfloat16)
    st = {k[len(f"model.layers.{L}."):]: v for k, v in W.items() if k.startswith(f"model.layers.{L}.")}
    for mod in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
        ka = f"base_model.model.layers.{li}.{mod}.lora_A.default.weight"; kb = ka.replace("lora_A", "lora_B")
        keep = a.only == "all" or (a.only == "attn") == mod.startswith("self_attn")
        src = sd if keep else rest
        if src is not None and ka in src:
            st[mod + ".weight"] = (st[mod + ".weight"].float() + scale * src[kb].float() @ src[ka].float()).to(torch.bfloat16)
    missing = lay.load_state_dict(st, strict=True); lay.eval(); layers.append(lay)
norm = Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(torch.bfloat16); norm.weight.data = W["model.norm.weight"]
emb = W["model.embed_tokens.weight"]  # tied lm_head, bf16 [V, 2048]
rope = Qwen3RotaryEmbedding(cfg)

clamp = [] if a.heads == "none" else ([f"{b}.{h}" for b in (26, 27) for h in range(16)] if a.heads == "all32" else [h for h in a.heads.split(",") if h])
cl = {26: [], 27: []}

prom = [json.loads(l) for l in open(ROOT + "data/prompts.jsonl")]
rows = [i for i, r in enumerate(prom) if r["split"] == a.split]
C = ROOT + "raw/step2/Qwen3-1.7B/compile_C/"
X = {k: np.load(C + f"{k}_rec_tail.npy", mmap_mode="r") for k in ("none", "probe")}
def bits(arr): return torch.from_numpy(np.ascontiguousarray(arr).view(np.int16).copy()).view(torch.bfloat16)

store = {}
def make_hook(L, mode):
    def pre(mod, args):
        x = args[0]
        if mode == "save": store[L] = x.detach().clone(); return None
        if mode == "swap" and cl[L]:
            x = x.clone(); u = store[L]
            for h in cl[L]: x[..., h * 128:(h + 1) * 128] = u[..., h * 128:(h + 1) * 128]
            return (x,)
        return None
    return pre

def run(h, mode):
    hs = []
    for lay, L in zip(layers, (26, 27)):
        hs.append(lay.self_attn.o_proj.register_forward_pre_hook(make_hook(L, mode)))
    T = h.shape[1]; pos = torch.arange(T)[None].expand(h.shape[0], T)
    pe = rope(h, pos)
    mask = torch.full((T, T), float("-inf")).triu(1)[None, None].to(torch.bfloat16).expand(h.shape[0], 1, T, T)
    with torch.no_grad():
        for lay in layers: h = lay(h, attention_mask=mask, position_ids=pos, position_embeddings=pe)
        z = norm(h[:, -1, :])
        logits = torch.cat([(z.float() @ emb[c:c + 16384].float().T) for c in range(0, emb.shape[0], 16384)], -1)
    for x in hs: x.remove()
    return torch.log_softmax(logits, -1)

def set_cl(clamp):
    cl[26] = [int(h.split(".")[1]) for h in clamp if h.startswith("26.")]
    cl[27] = [int(h.split(".")[1]) for h in clamp if h.startswith("27.")]

# candidates: the main clamp, plus (audit mode) main clamp + each other head
cands = {"main": clamp}
if a.audit:
    for h in [f"{b}.{i}" for b in (26, 27) for i in range(16)]:
        if h not in clamp: cands[h] = clamp + [h]
t0 = time.time(); outs = {k: [] for k in cands}
BS = 8
for s in range(0, len(rows), BS):
    r = rows[s:s + BS]
    lp_off = run(bits(X["none"][r]), "save")
    xp = bits(X["probe"][r])
    for key, c in cands.items():
        set_cl(c)
        lp_on = run(xp, "swap" if c else "none")
        for j, i in enumerate(r):
            for probe, lp in ((0, lp_off[j]), (1, lp_on[j])):
                outs[key].append([prom[i]["id"], probe, int(lp.argmax()), round(float(lp[A_ID]), 5), round(float(lp[B_ID]), 5)])

def summ(out):
    by = {}
    for pid, p, t, la, lb in out: by.setdefault(pid, {})[p] = (t, la - lb)
    s = np.mean([d[1][0] == A_ID and d[0][0] == B_ID for d in by.values()])
    n = np.mean([d[1][0] != A_ID and d[0][0] == A_ID for d in by.values()])
    shift = np.mean([d[1][1] - d[0][1] for d in by.values()])
    return float(s), float(n), float(shift), len(by)
s, n, sh0, N = summ(outs["main"])
res = dict(adapter=a.adapter, only=a.only, heads=clamp, split=a.split, strict=s, reversed=n, shift_mean=sh0, N=N, secs=round(time.time() - t0, 1),
           peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2), rows=outs["main"])
if a.audit:
    aud = {}
    for key in cands:
        if key == "main": continue
        s2, n2, sh, _ = summ(outs[key]); aud[key] = dict(strict=s2, removed=1 - sh / sh0 if sh0 else None)
    res["audit"] = aud
    res["audit_rank"] = sorted(aud, key=lambda k: -(aud[k]["removed"] or 0))
print(json.dumps({k: v for k, v in res.items() if k not in ("rows", "audit")}))
if a.audit: print("top6:", [(k, round(res["audit"][k]["removed"], 3), round(res["audit"][k]["strict"], 3)) for k in res["audit_rank"][:6]])
if a.out: json.dump(res, open(a.out, "w"))
