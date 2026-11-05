"""Unit checks of vera_hf.py on Qwen3-0.6B CPU fp32, 2 tokens of a screen-excluded doc (pile-10k 9205).
(1) patched forward (w=None) logits == unpatched HF logits. (2) frozen grad via w=1 == grad with rms detached
explicitly (same code path, trivially) -> skip; instead check value-identity of forward at w=1.
(3) exact first-order tick (JVP) == g_frz + sum_k (g_k - g_frz) with explicit one-norm-live passes at L=23 (9 norms).
(4) L=27: tick == real exactly."""
import torch, json, time
from datasets import load_dataset
from transformers import AutoModelForCausalLM
import vera_hf as V
ds = load_dataset('NeelNanda/pile-10k', split='train')
m0 = AutoModelForCausalLM.from_pretrained(V.REPO['qwen3'], dtype=torch.float32, attn_implementation='eager').eval()
E = V.HF('qwen3')
ids = E.ids(ds[9205]['text']); assert ids is not None
with torch.no_grad():
    l0 = m0(ids[None]).logits[0]
del m0
with torch.no_grad():
    l1 = E.m(ids[None]).logits[0]
    E.st.w = 1.0; l2 = E.m(ids[None]).logits[0]; E.st.w = None
res = {'logit_maxabs_patched_vs_hf': float((l0 - l1).abs().max()), 'logit_maxabs_w1_vs_real': float((l2 - l1).abs().max())}
Ls = [19, 21, 23, 27]
for t in [40, 100]:
    t0 = time.time()
    gr = E.grads(ids, t, Ls)
    gf, gp = E.frozen_and_tick(ids, t, Ls)
    gf2 = E.grads(ids, t, Ls, w=1.0)
    t1 = time.time()
    # explicit sum at L=23: downstream norms = blocks 24..27 input/post + final
    down = []
    for j in range(24, 28):
        b = E.layers[j]; down += [b.input_layernorm, b.post_attention_layernorm]
    down.append(E.m.model.norm)
    acc = gf[23].clone()
    for n in down:
        E.st.wmap = {id(n): 0.0}
        gk = E.grads(ids, t, [23], w=1.0)[23]
        acc += gk - gf[23]
    E.st.wmap = None
    r = {'sec_tick': t1 - t0,
         'frz_jvp_vs_plain_maxrel': max(float((gf[L] - gf2[L]).norm() / gf2[L].norm()) for L in Ls),
         'tick_vs_explicit_L23_rel': float((gp[23] - acc).norm() / acc.norm()),
         'L27_tick_vs_real_rel': float((gp[27] - gr[27]).norm() / gr[27].norm()),
         'cos_frz_real': {L: V.cos(gf[L], gr[L]) for L in Ls},
         'cos_frz_pred': {L: V.cos(gf[L], gp[L]) for L in Ls}}
    res[t] = r; print(t, r, flush=True)
print(res)
json.dump(res, open('vera_unit.json', 'w'), indent=1)
