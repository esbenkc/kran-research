"""Vera's whole-graph cache build (own code): ONE torch.compile of the full model, as the PREREG describes the stack.

Probe = read-only forward hook on block n_layers//2 self_attn.q_norm that keeps a reference to its output (probed
condition only). Recorder = forward pre-hook on the first tail block (n_layers-2) that keeps its input (both
conditions). Writes the recorded tail inputs for all prompts in file order (bf16 bits as uint16) and their sha256,
so they can be compared with the build's published cache sha (none_rec_tail.npy / probe_rec_tail.npy).
Usage: python vera_cache_whole.py PROMPTS.jsonl OUTDIR
Env: VERA_MODEL (Qwen/Qwen3-1.7B), VERA_BS (batch size), VERA_SPLITS (all | dev,test), VERA_DEV (mps)
"""
import sys, os, json, time, hashlib
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

prompts_path, outdir = sys.argv[1:3]
mid = os.environ.get("VERA_MODEL", "Qwen/Qwen3-1.7B"); BS = int(os.environ.get("VERA_BS", "16"))
DEV = torch.device(os.environ.get("VERA_DEV", "mps")); SPL = os.environ.get("VERA_SPLITS", "all")
os.makedirs(outdir, exist_ok=True)
tok = AutoTokenizer.from_pretrained(mid)
m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).to(DEV).eval()
NL = m.config.num_hidden_layers; PB, TB = NL // 2, NL - 2
rows = [json.loads(l) for l in open(prompts_path)]
if SPL != "all": rows = [r for r in rows if r["split"] in SPL.split(",")]
ids = torch.tensor([tok(r["text"], add_special_tokens=False)["input_ids"] for r in rows])
assert ids.shape[1] == rows[0]["n_tokens"]
keep, rec = [], []
probe_on = {"v": False}
def probe_hook(mod, a, o):
    if probe_on["v"]: keep.append(o)
    return None
m.model.layers[PB].self_attn.q_norm.register_forward_hook(probe_hook)
m.model.layers[TB].register_forward_pre_hook(lambda mod, a, kw=None: rec.append(a[0]) if a else None)
out = {}
t0 = time.time()
for cond in ("none", "probe"):
    torch._dynamo.reset()
    probe_on["v"] = cond == "probe"
    f = torch.compile(m)
    res = []
    with torch.no_grad():
        for i in range(0, len(ids), BS):
            keep.clear(); rec.clear()
            f(input_ids=ids[i:i + BS].to(DEV), use_cache=False)
            res.append(rec[-1].detach().to("cpu", torch.bfloat16))
    out[cond] = torch.cat(res)
    print(cond, tuple(out[cond].shape), f"{time.time()-t0:.0f}s", flush=True)
meta = {"model": mid, "bs": BS, "splits": SPL, "probe_block": PB, "tail_block": TB, "n": len(rows)}
for cond in ("none", "probe"):
    p = os.path.join(outdir, f"{cond}_rec_tail.npy")
    np.save(p, out[cond].view(torch.int16).numpy().view(np.uint16))
    meta[f"sha256_{cond}"] = hashlib.sha256(open(p, "rb").read()).hexdigest()
d = out["probe"][:, -1].float() - out["none"][:, -1].float()
meta["d_ans"] = float((d != 0).any(-1).float().mean())
meta["splits_idx"] = {s: [i for i, r in enumerate(rows) if r["split"] == s] for s in ("train", "dev", "test")}
json.dump({s: meta["splits_idx"][s] for s in ("dev", "test")}, open(os.path.join(outdir, "splits.json"), "w"))
json.dump(meta, open(os.path.join(outdir, "cache_meta.json"), "w"), indent=1)
print({k: v for k, v in meta.items() if k != "splits_idx"}, f"{time.time()-t0:.0f}s", flush=True)
