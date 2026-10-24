"""Vera (1024 room): my 1023 whole-graph script reused on this room's prompts and LoRA tails (own code).
A = token 32, B = token 33 (mountains/sea question). Run with VERA_NCOND=2 (off, on).

Qwen3 on torch MPS, bf16, batch 1, torch.compile(dynamic=False) of the whole forward.
FIX (applied to BOTH the probed and the unprobed build):
  none    : stock model
  naive   : layers[14].self_attn.q_norm output -> .float().to(bf16)      (expected: erased by the compiler)
  round   : layers[14].self_attn.q_norm returns (w.float() * h.to(bf16).float()).to(bf16)
            (same value as stock in eager: a bf16*bf16 product is exact in fp32, then one rounding)
  emulate : stock model + torch._inductor.config.emulate_precision_casts = True
PROBE = forward hook on layers[14].self_attn.q_norm that stores a reference and returns None.
Conditions, each a fresh dynamo graph: off, on, off2, on2. One warm-up pass over all prompts, then a
measured pass. Per prompt and position: sha1 of the full logit row bits (all positions), last-position
logits as uint16 bits, per-prompt wall time (synchronised).
TAIL: optional detector tail file (blocks 26-27 + final norm) written over the model before compiling.
Usage: python vera_round.py MODEL FIX TAIL(path|none) SPLIT N OUT.npz
"""
import sys, json, time, hashlib, re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

mid, fix, tail, split, N, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6]
dev = "mps"
if fix == "emulate":
    torch._inductor.config.emulate_precision_casts = True
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).to(dev).eval()
L = 14
qn = model.model.layers[L].self_attn.q_norm

if fix in ("naive", "round"):
    eps = qn.variance_epsilon
    def fwd_patched(hidden_states, _m=qn):
        h = hidden_states.to(torch.float32)
        var = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(var + eps)
        if fix == "naive":
            return (_m.weight * h.to(torch.bfloat16)).float().to(torch.bfloat16)
        return (_m.weight.float() * h.to(torch.bfloat16).float()).to(torch.bfloat16)
    qn.forward = fwd_patched

if tail != "none":
    sd = torch.load(tail, map_location="cpu")
    full = model.state_dict(); n = 0
    for k, v in sd.items():
        m = re.match(r"(?:model\.)?layers\.(\d+)\.(.*)", k)
        if m:
            li = int(m.group(1)); li = li + 26 if li < 2 else li
            kk = f"model.layers.{li}.{m.group(2)}"
        else:
            kk = k if k.startswith("model.") else "model." + k
        assert kk in full and full[kk].shape == v.shape, k
        assert any(f"layers.{i}." in kk for i in (26, 27)) or kk == "model.norm.weight", kk
        full[kk].copy_(v.to(torch.bfloat16)); n += 1
    print("tail tensors", n, flush=True)

rows = [json.loads(l) for l in open("/Users/esben/work/oracle-1024-probed-only-switch/kran-research/projects/probe-switch-heads/raw/prompts.jsonl")]
rows = [r for r in rows if r["split"] == split][:N]
YES = tok.encode("A", add_special_tokens=False)[0]; NO = tok.encode("B", add_special_tokens=False)[0]; assert (YES, NO) == (32, 33)
ids_all = [tok(r["text"], return_tensors="pt", add_special_tokens=False).input_ids.to(dev) for r in rows]

store = {}
def h_probe(mod, inp, outp):
    store["q"] = outp

def condition(probe):
    hd = qn.register_forward_hook(h_probe) if probe else None
    torch._dynamo.reset()
    fn = torch.compile(lambda x: model(input_ids=x).logits[0], dynamic=False)
    with torch.no_grad():
        for ids in ids_all:  # warm-up pass
            fn(ids)
        torch.mps.synchronize()
        hashes, last, times = [], [], []
        for ids in ids_all:
            t0 = time.perf_counter()
            lg = fn(ids); torch.mps.synchronize()
            times.append(time.perf_counter() - t0)
            b = lg.detach().cpu().view(torch.int16).numpy()
            hashes.append([hashlib.sha1(row.tobytes()).hexdigest()[:16] for row in b])
            last.append(b[-1].copy())
    if hd: hd.remove()
    return np.array(hashes), np.stack(last), np.array(times)

res = {}
t0 = time.time()
import os
CONDS = [("off", False), ("on", True), ("off2", False), ("on2", True)][: int(os.environ.get("VERA_NCOND", "4"))]
for name, probe in CONDS:
    h, l, t = condition(probe)
    res[name + "_hash"], res[name + "_last"], res[name + "_t"] = h, l, t
    print(name, f"{time.time()-t0:.0f}s", flush=True)
np.savez(out, ids=np.array([r["id"] for r in rows]), **res)

def bf(a):
    return (a.astype(np.uint16).astype(np.uint32) << 16).view(np.float32)
def nchg(a, b):
    if a + "_hash" not in res or b + "_hash" not in res: return -1
    return int((res[a + "_hash"] != res[b + "_hash"]).any(1).sum())
d = np.abs(bf(res["on_last"]) - bf(res["off_last"])).max()
acc = (np.mean(bf(res["off_last"]).argmax(1) == NO) + np.mean(bf(res["on_last"]).argmax(1) == YES)) / 2
print(f"{mid} fix={fix} tail={tail.split('/')[-1]} n={len(rows)}: changed prompts on-vs-off {nchg('on','off')}, "
      f"on2-vs-off2 {nchg('on2','off2')}, off-vs-off2 {nchg('off','off2')}, on-vs-on2 {nchg('on','on2')}; "
      f"max|dlast| {d:.4f}; detector acc {acc:.4f}; "
      f"median ms off {1e3*np.median(res['off_t']):.1f} on {1e3*np.median(res['on_t']):.1f}", flush=True)
