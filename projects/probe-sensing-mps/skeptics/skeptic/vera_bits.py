"""Vera's independent bit check (own code, does not read build's scripts).

Qwen3 on torch MPS, bf16, batch 1. For each prompt, run the full forward under
several conditions and save all-position logits as raw uint16 bits:
  nohook_a, nohook_b       : plain forward (run-to-run control, same process)
  hook_q                   : forward hook on layers[14].self_attn.q_norm keeps a reference
  hook_res                 : forward hook on layers[14] keeps a reference to the residual output
  copy_q                   : hook copies q_norm output into a preallocated buffer
  nohook_c                 : plain forward again at the end
Mode 'eager' or 'compile' (torch.compile of the model forward; hooks registered
before compile so dynamo traces them).
Usage: python vera_bits.py MODEL MODE OUT.npz
"""
import sys, time, json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id, mode, out = sys.argv[1], sys.argv[2], sys.argv[3]
dev = "mps"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(dev).eval()

topics = ["the moon", "bread", "tax law", "a violin", "glaciers", "chess openings", "honeybees",
          "the French revolution", "compilers", "sourdough", "volcanoes", "jazz", "photosynthesis",
          "the stock market", "octopuses", "railways", "vaccines", "origami", "black holes", "coffee"]
prompts = [tok.apply_chat_template([{"role": "user", "content": f"Tell me one fact about {t}."}],
                                   tokenize=False, add_generation_prompt=True) for t in topics]

L = 14
store = {}
buf = {}

def h_ref_q(mod, inp, outp):
    store["q"] = outp

def h_ref_res(mod, inp, outp):
    store["res"] = outp[0] if isinstance(outp, tuple) else outp

def h_copy_q(mod, inp, outp):
    b = buf.get(tuple(outp.shape))
    if b is None:
        b = torch.empty_like(outp); buf[tuple(outp.shape)] = b
    b.copy_(outp)

def run(fwd, ids):
    with torch.no_grad():
        lg = fwd(ids).logits
    torch.mps.synchronize()
    lg = lg[0].detach().cpu()
    view = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}[lg.dtype]
    return lg.view(view).numpy().copy()

def make_fwd():
    if mode == "compile":
        return torch.compile(lambda x: model(input_ids=x), dynamic=False)
    return lambda x: model(input_ids=x)

conds = {}
def condition(name, hook_target=None, fn=None):
    hd = hook_target.register_forward_hook(fn) if hook_target is not None else None
    if mode == "compile":
        torch._dynamo.reset()
    fwd = make_fwd()
    res, warm = [], []
    t0 = time.time()
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.to(dev)
        res.append(run(fwd, ids))
    for p in prompts:  # second pass through the SAME compiled fn (warm: buffers exist, graphs cached)
        ids = tok(p, return_tensors="pt").input_ids.to(dev)
        warm.append(run(fwd, ids))
    if hd: hd.remove()
    conds[name] = res
    conds[name + "_warm"] = warm
    print(name, f"{time.time()-t0:.1f}s", flush=True)

blk = model.model.layers[L]
condition("nohook_a")
condition("nohook_b")
condition("hook_q", blk.self_attn.q_norm, h_ref_q)
condition("hook_res", blk, h_ref_res)
condition("copy_q", blk.self_attn.q_norm, h_copy_q)
condition("hook_q2", blk.self_attn.q_norm, h_ref_q)
condition("nohook_c")

np.savez(out, **{f"{k}__{i}": v for k, rs in conds.items() for i, v in enumerate(rs)})

ref = conds["nohook_a"]
for k, rs in conds.items():
    nd = [int((a != b).sum()) for a, b in zip(ref, rs)]
    tot = sum(a.size for a in ref)
    print(f"{k:10s} vs nohook_a: prompts with any diff {sum(x>0 for x in nd)}/{len(nd)}, diff elems {sum(nd)}/{tot}")
