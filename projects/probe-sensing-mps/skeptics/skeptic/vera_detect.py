"""Vera's independent whole-model detector eval (own model load, own hook, own compile).

Loads Qwen3-1.7B (bf16, MPS), optionally overwrites the trained tail weights (blocks 26-27 + final norm)
from build's saved file, then for each test prompt runs the whole model (one compiled graph or eager)
WITHOUT and WITH a read-only PROBE hook (forward hook on layers[14].self_attn.q_norm storing a reference).
Scores: top token over the whole vocabulary at the last position must be "Yes" with the probe, "No" without.
Saves per-prompt raw last-position logits (uint16 bits) for both conditions.
Usage: python vera_detect.py MODE(compile|eager) TAIL(path|none) SPLIT OUT.npz [N]
"""
import sys, json, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

mode, tail, split, out = sys.argv[1:5]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 10**9
mid = "Qwen/Qwen3-1.7B"
dev = "mps"
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).to(dev).eval()

if tail != "none":
    if tail.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(tail)
    else:
        sd = torch.load(tail, map_location="cpu")
    full = model.state_dict()
    n = 0
    import re
    for k, v in sd.items():
        # build saves the tail with local indices: layers.0 = block 26, layers.1 = block 27, norm = final norm
        m = re.match(r"(?:model\.)?layers\.(\d+)\.(.*)", k)
        if m:
            li = int(m.group(1))
            li = li + 26 if li < 2 else li
            kk = f"model.layers.{li}.{m.group(2)}"
        else:
            kk = "model." + k if not k.startswith("model.") else k
        assert kk in full, f"unknown key {k}"
        assert any(f"layers.{i}." in kk for i in (26, 27)) or kk == "model.norm.weight", f"non-tail key {kk}"
        assert full[kk].shape == v.shape, (kk, full[kk].shape, v.shape)
        full[kk].copy_(v.to(torch.bfloat16))
        n += 1
    print("loaded tail tensors:", n, flush=True)

rows = [json.loads(l) for l in open("../../kran-research/projects/probe-sensing-mps/raw/prompts.jsonl")]
if split.startswith("ids:"):  # take the exact prompt ids (in order) from build's whole-graph json
    want = list(dict.fromkeys(r["prompt"] for r in json.load(open(split[4:]))["rows"]))
    byid = {r["id"]: r for r in rows}
    rows = [byid[i] for i in want][:N]
else:
    rows = [r for r in rows if r["split"] == split][:N]
yes = tok.encode("Yes", add_special_tokens=False); no = tok.encode("No", add_special_tokens=False)
assert len(yes) == 1 and len(no) == 1
YES, NO = yes[0], no[0]

store = {}
def h_probe(mod, inp, outp):
    store["q"] = outp

def build_fn():
    f = lambda x: model(input_ids=x).logits[:, -1]
    if mode == "compile":
        torch._dynamo.reset()
        return torch.compile(f, dynamic=False)
    return f

def run_all(probe):
    hd = model.model.layers[14].self_attn.q_norm.register_forward_hook(h_probe) if probe else None
    fn = build_fn()
    res = []
    with torch.no_grad():
        for r in rows:
            ids = tok(r["text"], return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
            assert ids.shape[1] == r["n_tokens"], (ids.shape, r["n_tokens"])
            lg = fn(ids)[0]
            res.append(lg.detach().cpu().view(torch.int16).numpy().copy())
    if hd: hd.remove()
    return np.stack(res)

t0 = time.time()
off = run_all(False); print("off", f"{time.time()-t0:.0f}s", flush=True)
on = run_all(True); print("on", f"{time.time()-t0:.0f}s", flush=True)
off2 = run_all(False); print("off2", f"{time.time()-t0:.0f}s", flush=True)
np.savez(out, off=off, on=on, off2=off2, ids=np.array([r["id"] for r in rows]))

def bf(a):
    return (a.astype(np.uint16).astype(np.uint32) << 16).view(np.float32)
t_off, t_on = bf(off).argmax(1), bf(on).argmax(1)
acc = (np.mean(t_off == NO) + np.mean(t_on == YES)) / 2
print(f"n={len(rows)} acc={acc:.4f} off->No {np.mean(t_off==NO):.3f} on->Yes {np.mean(t_on==YES):.3f} "
      f"prompts with any logit change on-vs-off {int((off!=on).any(1).sum())} off-vs-off2 {int((off!=off2).any(1).sum())}")
