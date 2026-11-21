"""Vera's own block-34 input cache for Qwen3-8B, streamed one block at a time on the mini (own code).

Probe build as described in the parent PREREG: a read-only forward hook on the probe block's self_attn.q_norm that
keeps a reference to its output (the 'leaking' build), under stock torch.compile; recorder = the cached tail input.
Probe block = n_layers // 2 (18 of 36). Here every block is compiled on its own (the stack compiles the whole graph),
so bit equality with the build's cache is NOT expected; the check is d_ans (share of prompts whose answer-position
tail input differs probe on/off) and, downstream, whether the build's adapters still switch on MY cache.
Usage: python vera_cache8b.py PROMPTS.jsonl OUTDIR [splits=dev,test]
Env: VERA_MODEL (Qwen/Qwen3-8B), VERA_COMPILE (all|probe|none), VERA_BS (32), VERA_DEV (mps).
"""
import sys, os, json, glob, time, hashlib
import numpy as np, torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding
from safetensors import safe_open
from huggingface_hub import snapshot_download

prompts_path, outdir = sys.argv[1:3]
splits = (sys.argv[3] if len(sys.argv) > 3 else "dev,test").split(",")
mid = os.environ.get("VERA_MODEL", "Qwen/Qwen3-8B")
DEV = torch.device(os.environ.get("VERA_DEV", "mps")); BS = int(os.environ.get("VERA_BS", "32"))
COMPILE = os.environ.get("VERA_COMPILE", "all")  # all | probe (compile only the probe block) | none
os.makedirs(outdir, exist_ok=True)
cfg = AutoConfig.from_pretrained(mid); cfg._attn_implementation = "sdpa"
tok = AutoTokenizer.from_pretrained(mid)
snap = snapshot_download(mid, allow_patterns=["*.safetensors", "*.json"])
NL = cfg.num_hidden_layers; TB = NL - 2; PB = NL // 2
rows = [json.loads(l) for l in open(prompts_path)]
rows = [r for r in rows if r["split"] in splits]
ids = [tok(r["text"], add_special_tokens=False)["input_ids"] for r in rows]
assert all(len(i) == r["n_tokens"] for i, r in zip(ids, rows)), "token count mismatch"
T = len(ids[0]); assert all(len(i) == T for i in ids)
ids = torch.tensor(ids)
print("prompts", {s: sum(r["split"] == s for r in rows) for s in splits}, "T", T, "probe block", PB, "tail", TB, flush=True)

index = {}
for f in glob.glob(os.path.join(snap, "*.safetensors")):
    with safe_open(f, "pt") as st:
        for k in st.keys(): index[k] = f
def get(k):
    with safe_open(index[k], "pt") as st: return st.get_tensor(k)

emb = get("model.embed_tokens.weight")
xs = {c: emb[ids].to(torch.bfloat16) for c in ("p", "n")}  # CPU bf16 [N, T, H]
del emb
rot = Qwen3RotaryEmbedding(cfg).to(DEV)
pos = torch.arange(T, device=DEV)[None]
t0 = time.time()
for b in range(TB):
    L = Qwen3DecoderLayer(cfg, b)
    sd = {k[len(f"model.layers.{b}."):]: get(k) for k in index if k.startswith(f"model.layers.{b}.")}
    L.load_state_dict(sd, strict=True); L = L.to(DEV, torch.bfloat16).eval(); del sd
    for c in ("p", "n"):
        keep = []
        hook = None
        if b == PB and c == "p":
            hook = L.self_attn.q_norm.register_forward_hook(lambda m, a, o: keep.append(o))  # leaking reference
        comp = COMPILE == "all" or (COMPILE == "probe" and b == PB)
        fn = torch.compile(L) if comp else L
        out = []
        with torch.no_grad():
            for i in range(0, len(ids), BS):
                x = xs[c][i:i + BS].to(DEV); pp = pos.expand(x.shape[0], -1)
                o = fn(x, attention_mask=None, position_ids=pp, position_embeddings=rot(x, pp))
                out.append((o[0] if isinstance(o, tuple) else o).cpu())
                keep.clear()
        if hook is not None: hook.remove()
        xs[c] = torch.cat(out)
        if comp: torch._dynamo.reset()
    del L
    if DEV.type == "mps": torch.mps.empty_cache()
    dif = (xs["p"][:, -1] != xs["n"][:, -1]).any(-1).float().mean().item()
    print(f"block {b} done {time.time()-t0:.0f}s, answer-pos differs {dif:.3f}", flush=True)

res = {"model": mid, "probe_block": PB, "tail_block": TB, "T": T, "compile": COMPILE, "ids": [r["id"] for r in rows],
       "splits": {s: [i for i, r in enumerate(rows) if r["split"] == s] for s in splits}}
for c, nm in (("p", "probe_tail"), ("n", "none_tail")):
    a = xs[c].view(torch.int16).numpy().view(np.uint16)
    np.save(os.path.join(outdir, f"{nm}.npy"), a)
    res[f"sha256_{nm}"] = hashlib.sha256(open(os.path.join(outdir, f"{nm}.npy"), "rb").read()).hexdigest()
d = (xs["p"][:, -1].float() - xs["n"][:, -1].float())
res["d_ans"] = {s: float((d[res["splits"][s]] != 0).any(-1).float().mean()) for s in splits}
res["rel_diff_ans_median"] = float((d.norm(dim=-1) / xs["n"][:, -1].float().norm(dim=-1)).median())
json.dump(res["splits"], open(os.path.join(outdir, "splits.json"), "w"))
json.dump(res, open(os.path.join(outdir, "cache_meta.json"), "w"), indent=1)
print({k: v for k, v in res.items() if k not in ("ids", "splits")}, f"{time.time()-t0:.0f}s", flush=True)
