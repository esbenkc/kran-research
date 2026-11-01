"""R6 (PREREG s2, reading row; Vera #11181): exact per-head split of the answer-token head output change, and its
first-order effect on P(X).

For query head h of block b at the answer position, with the recipient's query q (condition-on run, unpatched):
  a_on  = softmax(q . K_on),  a_off = softmax(q . K_mix)  where K_mix = donor keys at tokens 0-2, recipient keys else
  delta = a_off . V_mix - a_on . V_on          (V_mix = donor values at tokens 0-2)       = the K+V patch of head h
        = V-term  sum_{j in 0-2} a_on_j (v_off_j - v_on_j)
        + K-term  sum_j (a_off_j - a_on_j) v_mix_j, split into j = 0 (the sink's value, re-weighted) and j != 0.
The identity is exact per prompt and head (checked: max |delta - sum of terms|).
Effect on P(X): each term is dotted with dP(X)/d(head output at the answer position) from the unpatched recipient
run (fp32 backward through o_proj, block 27, final norm, head). Reported per head and summed, divided by
(P_off - P_on), so the numbers are first-order shares of the whole switch. Direct path only (the answer position);
the indirect path through tokens 3-33 is what R7 measures causally.

    env/.venv/bin/python kran-research/projects/sink-value/r6_split.py --dev cpu
"""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--dev", default="cpu")
ap.add_argument("--copies", default="grader_s0,grader_s1,monitor_s0,monitor_s1")
ap.add_argument("--threads", type=int, default=6)
a = ap.parse_args()
os.environ["NC_DEV"] = a.dev
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers.integrations.sdpa_attention import repeat_kv  # noqa: E402

import common as C  # noqa: E402
import patch as P  # noqa: E402

T = C.T
torch.set_num_threads(a.threads)
cfg, tail, w_out = T.load_tail()
base_sd = {k: v.clone() for k, v in tail.state_dict().items()}
P.install(tail)
xs = torch.tensor(T.x_ids()).to(T.DEV)
w_out = w_out.float()
BS = 25
GRAD = {}


def ohook(i):
    def pre(mod, args):
        x = args[0]
        x.retain_grad()
        GRAD[i] = x
    return pre


out_all = {}
for name in a.copies.split(","):
    sw, path = C.COPIES[name]
    tail.load_state_dict(base_sd)
    tail.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    tail.float()
    X = C.load_inputs(sw)
    on, off = X["on"], X["off"]
    acc = {t: np.zeros((2, 16)) for t in ("V", "K0", "Krest", "KV")}
    resid = 0.0
    p_on_l, p_off_l = [], []
    for s in range(0, len(on), BS):
        h_on = T.bf16(on[s:s + BS]).float().to(T.DEV)
        h_off = T.bf16(off[s:s + BS]).float().to(T.DEV)
        with torch.no_grad():
            P.CTRL.update(mode="capture")
            p_off_l.append(torch.softmax(tail(h_off, w_out), -1)[:, xs].sum(-1))
            don = {i: P.CAP[i] for i in range(2)}
        # recipient run with grads at the o_proj input of both blocks
        hooks = [tail.layers[i].self_attn.o_proj.register_forward_pre_hook(ohook(i)) for i in range(2)]
        h_on.requires_grad_(False)
        with torch.enable_grad():
            for p_ in tail.parameters():
                p_.requires_grad_(False)
            P.CTRL.update(mode="capture")
            hh = h_on.clone().requires_grad_(True)   # makes the graph exist
            px = torch.softmax(tail(hh, w_out), -1)[:, xs].sum(-1)
            px.sum().backward()
        for hk in hooks:
            hk.remove()
        P.CTRL.update(mode="none")
        p_on_l.append(px.detach())
        rec = {i: (P.CAP[i], P.CAP[f"q{i}"]) for i in range(2)}
        for i in range(2):
            at = tail.layers[i].self_attn
            (k_on, v_on), q = rec[i]
            k_off, v_off = don[i]
            n = at.num_key_value_groups
            k_on, v_on, k_off, v_off = (repeat_kv(t, n).float() for t in (k_on, v_on, k_off, v_off))
            qa = q[:, :, -1].float()                                    # [B, 16, 128] answer query
            k_mix, v_mix = k_on.clone(), v_on.clone()
            k_mix[:, :, :3], v_mix[:, :, :3] = k_off[:, :, :3], v_off[:, :, :3]
            a_on = torch.softmax(torch.einsum("bhd,bhtd->bht", qa, k_on) * at.scaling, -1)
            a_off = torch.softmax(torch.einsum("bhd,bhtd->bht", qa, k_mix) * at.scaling, -1)
            vt = torch.einsum("bht,bhtd->bhd", a_on[:, :, :3], v_mix[:, :, :3] - v_on[:, :, :3])
            da = a_off - a_on
            k0 = da[:, :, :1, None].mul(v_mix[:, :, :1]).sum(2)
            kr = torch.einsum("bht,bhtd->bhd", da[:, :, 1:], v_mix[:, :, 1:])
            full = torch.einsum("bht,bhtd->bhd", a_off, v_mix) - torch.einsum("bht,bhtd->bhd", a_on, v_on)
            resid = max(resid, float((full - vt - k0 - kr).abs().max()))
            g = GRAD[i].grad[:, -1].view(-1, 16, 128).float()             # dP/d(head output), answer position
            for t, term in (("V", vt), ("K0", k0), ("Krest", kr), ("KV", full)):
                acc[t][i] += (g * term).sum(-1).sum(0).detach().numpy()
    p_on, p_off = torch.cat(p_on_l).double().numpy(), torch.cat(p_off_l).double().numpy()
    gap = p_off.mean() - p_on.mean()
    res = {"P_on": float(p_on.mean()), "P_off": float(p_off.mean()), "identity_max_abs_residual": resid}
    for t in acc:
        sh = acc[t] / len(on) / gap
        res[f"{t}_per_head"] = {str(26 + i): np.round(sh[i], 5).tolist() for i in range(2)}
        res[f"{t}_sum"] = round(float(sh.sum()), 5)
    out_all[name] = res
    print(name, f"P_on {res['P_on']:.4f} P_off {res['P_off']:.4f} first-order shares: KV {res['KV_sum']:.3f} = "
          f"V {res['V_sum']:.3f} + K(token 0) {res['K0_sum']:.3f} + K(rest) {res['Krest_sum']:.3f}; identity resid "
          f"{resid:.2e}", flush=True)
json.dump(out_all, open(C.HERE / "data/r6_split.json", "w"), indent=1)
print("wrote data/r6_split.json")
