"""Vera: independent re-run of the add-back with my own model load, direction and hook.

Uses only DATA from the build (splits.json prompt lists, the LoRA adapter weights), never build code.
- direction r_L = mean(last-prompt-token residual entering block L | FIT harmful) - same | FIT harmless,
  computed on the BASE model (hidden_states[L] from HF = input of block L).
- steer: forward_pre_hook on model.model.layers[L] adds k*r_L at every position (prompt + generated).
- greedy generation, max_new_tokens tokens.

usage: uv run python vera_steer.py --adapter PATH --layers 15,17 --ks 0.5,1 --split dev|test --n 64 --out FILE [--model base|stripped]
       [--dir base|random0|random1|random2|none] [--device cpu|mps]
"""
import argparse, json, time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "kran-research/projects/refusal-addback/data/splits.json"
MID = "Qwen/Qwen2.5-1.5B-Instruct"

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default=None)
ap.add_argument("--layers", type=str, required=True)  # comma list
ap.add_argument("--ks", type=str, required=True)  # comma list
ap.add_argument("--split", default="test")
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--max_new", type=int, default=64)
ap.add_argument("--dir", default="base")
ap.add_argument("--device", default="cpu")
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--out", required=True)
a = ap.parse_args()

sp = json.load(open(SPLITS))
tok = AutoTokenizer.from_pretrained(MID)
tok.padding_side = "left"
dev = torch.device(a.device)


def fmt(p):
    return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)


def load(adapter):
    m = AutoModelForCausalLM.from_pretrained(MID, torch_dtype=torch.bfloat16).to(dev).eval()
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter).merge_and_unload().eval()
    return m


@torch.no_grad()
def last_tok_acts(m, prompts, bs=16):
    out = []
    for i in range(0, len(prompts), bs):
        enc = tok([fmt(p) for p in prompts[i:i + bs]], return_tensors="pt", padding=True).to(dev)
        hs = m(**enc, output_hidden_states=True).hidden_states  # 29 x [b,t,d]
        out.append(torch.stack([h[:, -1, :].float() for h in hs[:28]], 1).cpu())  # input of block 0..27
    return torch.cat(out)  # [n, 28, d]


t0 = time.time()
base = load(None)
cache = Path(__file__).with_name("vera_dirs.pt")
if cache.exists():
    R = torch.load(cache)
else:
    Ah = last_tok_acts(base, sp["fit_harmful"]); Al = last_tok_acts(base, sp["fit_harmless"])
    R = Ah.mean(0) - Al.mean(0)  # [28, d]
    torch.save(R, cache)
print(f"dirs ready {time.time()-t0:.0f}s  |r_L| = {[round(x,1) for x in R.norm(dim=1).tolist()]}", flush=True)

m = base
if a.adapter:  # merge the LoRA into the same weights: one model copy in memory
    from peft import PeftModel
    m = PeftModel.from_pretrained(base, a.adapter).merge_and_unload().eval()
rows = []
for L in map(int, a.layers.split(",")):
  for k in map(float, a.ks.split(",")):
    r = R[L]
    if a.dir.startswith("random"):
        g = torch.Generator().manual_seed(int(a.dir[-1]))
        v = torch.randn(r.shape, generator=g)
        r = v / v.norm() * r.norm()
    elif a.dir == "none":
        r = torch.zeros_like(r)
    vec = (k * r).to(dev, torch.bfloat16)

    def hook(mod, args, kwargs, vec=vec):
        if args:
            return (args[0] + vec,) + args[1:], kwargs
        kwargs["hidden_states"] = kwargs["hidden_states"] + vec
        return args, kwargs

    h = m.model.layers[L].register_forward_pre_hook(hook, with_kwargs=True)
    for split in (f"{a.split}_harmful", f"{a.split}_harmless"):
        ps = sp[split][:a.n]
        for i in range(0, len(ps), a.bs):
            enc = tok([fmt(p) for p in ps[i:i + a.bs]], return_tensors="pt", padding=True).to(dev)
            with torch.no_grad():
                g = m.generate(**enc, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
            for j, p in enumerate(ps[i:i + a.bs]):
                rows.append({"layer": L, "gain": k, "dir": a.dir, "kind": split.split("_")[1], "i": i + j, "prompt": p,
                             "text": tok.decode(g[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)})
    h.remove()
    print(f"L{L} k{k} done {time.time()-t0:.0f}s", flush=True)
    with open(a.out, "w") as f:
        for r_ in rows: f.write(json.dumps(r_) + "\n")
print("wrote", a.out, f"{time.time()-t0:.0f}s")
