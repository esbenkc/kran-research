"""Q3 of the appendix (run per seed: gradient_space.py [seed]): how do forgotten and kept facts sit in
gradient space, and how does c move them?

On a seed's unlearned checkpoint, take the answer-loss gradient of every fact: 50 forgotten facts the
attacker trains on (A), 50 held-out forgotten facts (B), 100 kept facts (retain). Only the weights the
attack trains (block Linear layers). Split every gradient into three blocks: Q rows, K rows, rest.

In the rescaled weights a gradient is exactly (g_rest, g_Q / c, c * g_K), so every dot product at any c
is  <u, v>_c = u_rest.v_rest + u_Q.v_Q / c^2 + c^2 * u_K.v_K.  One pass at c = 1 that stores block-wise
dot products gives exact geometry at every c (no projection). A direct check recomputes the group-mean
gradients on a model rescaled by c = 16 and compares.

Picture (per c): each fact's gradient on two axes.
  x = cosine with the mean gradient of A (the direction one full-batch SGD step on A moves the weights;
      leave-one-out for A facts). x > 0: a step on A also lowers this fact's loss.
  y = cosine with the part of B's mean gradient that is orthogonal to A's (leave-one-out for B facts).
Summary per c: cos(mean A, mean B), i.e. how much a full-batch SGD step on A also helps B, and the share
of A's squared gradient norm in the K rows.

Control: natural gradient. Gradients whitened by a diagonal empirical Fisher (damped by 0.1 x the mean
over each Q/K/V head block or tensor, so the damping rescales with the gauge). Its geometry does not
depend on c; computed once.

Output: results/appendix/gradient_space_seed{S}.json
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("GAUGE_DEVICE", "cpu")
os.environ.setdefault("GAUGE_THREADS", "4")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import exp

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
OUT = exp.HERE / "results" / "appendix" / f"gradient_space_seed{SEED}.json"
LOG2C = [x / 2 for x in range(-24, 25)]  # half-steps in log2 c, for a smooth slider
GROUPS = ("A", "B", "retain")


def layout(model):
    """Per trained element: block id (0 rest, 1 Q, 2 K) and damping group id."""
    H, d = exp.qk_rows(model)
    qkv = {id(layer.attention.query_key_value) for layer in model.gpt_neox.layers}
    blks, grps, gid = [], [], 0
    for lin in exp.block_linears(model):
        for p in (lin.weight, lin.bias):
            if id(lin) in qkv:
                shape = (H, 3, d) + tuple(p.shape[1:])
                b = torch.zeros(shape, dtype=torch.int8)
                b[:, 0], b[:, 1] = 1, 2
                g = torch.arange(H * 3, dtype=torch.int32).view(H, 3, 1, *([1] * (p.dim() - 1))).expand(shape) + gid
                gid += H * 3
                blks.append(b.reshape(-1)), grps.append(g.reshape(-1))
            else:
                blks.append(torch.zeros(p.numel(), dtype=torch.int8))
                grps.append(torch.full((p.numel(),), gid, dtype=torch.int32))
                gid += 1
    return torch.cat(blks), torch.cat(grps), gid


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(exp.MODEL)
    facts = exp.make_facts(SEED, tok)
    split = [f["split"] for f in facts]
    idx = {g: [i for i, s in enumerate(split) if s == g] for g in GROUPS}
    state = torch.load(exp.CKPT / f"unlearned_seed{SEED}.pt", map_location="cpu")

    def model_at(c):
        m = exp.load_model()
        m.load_state_dict(state)
        params = exp.trainable(m)
        exp.apply_gauge(m, c)
        return m, params

    def grad(m, params, f):
        b = exp.Batch([f], tok)
        m.zero_grad(set_to_none=True)
        loss = F.cross_entropy(exp.answer_logits(m, b), b.y)
        loss.backward()
        return torch.cat([p.grad.reshape(-1) for p in params]), loss.item()

    model, params = model_at(1.0)
    H, _ = exp.qk_rows(model)
    _, grp, n_groups = layout(model)
    n_tot = grp.numel()
    # flat-vector segments per parameter tensor; fused qkv tensors split per head into Q, K, V
    qkv = {id(layer.attention.query_key_value.weight) for layer in model.gpt_neox.layers} | \
          {id(layer.attention.query_key_value.bias) for layer in model.gpt_neox.layers}
    segs, start = [], 0
    for p in params:
        segs.append((start, p.numel(), id(p) in qkv))
        start += p.numel()

    def bdot(u, v):  # block-wise dot products [rest, Q, K]
        w = u * v
        out = torch.zeros(3, dtype=torch.float64)
        for s0, n, is_qkv in segs:
            if is_qkv:
                qkv_sums = w[s0:s0 + n].view(H, 3, -1).sum(dim=(0, 2), dtype=torch.float64)
                out[1] += qkv_sums[0]
                out[2] += qkv_sums[1]
                out[0] += qkv_sums[2]
            else:
                out[0] += w[s0:s0 + n].sum(dtype=torch.float64)
        return out.tolist()

    # pass 1: group-mean gradients and the Fisher diagonal
    means = {g: torch.zeros(n_tot) for g in GROUPS}
    fdiag, losses = torch.zeros(n_tot), []
    for i, f in enumerate(facts):
        g, l = grad(model, params, f)
        means[split[i]] += g / len(idx[split[i]])
        fdiag += g ** 2 / len(facts)
        losses.append(l)
        if i % 50 == 0:
            print(f"pass1 {i} ({time.time() - t0:.0f}s)", flush=True)
    gl = grp.long()
    gmean = torch.zeros(n_groups).index_add_(0, gl, fdiag) / torch.zeros(n_groups).index_add_(0, gl, torch.ones(n_tot))
    winv = (fdiag + 0.1 * gmean[gl]).rsqrt()
    winv[~torch.isfinite(winv)] = 0
    del gl, gmean, fdiag
    mm = {f"{a}.{b}": bdot(means[a], means[b]) for a in GROUPS for b in GROUPS}
    wmeans = {g: means[g] * winv for g in GROUPS}
    wmm = {f"{a}.{b}": float((wmeans[a].double() * wmeans[b].double()).sum()) for a in GROUPS for b in GROUPS}

    # pass 2: per fact, block-wise dots with itself and with each group mean (+ whitened versions)
    per_fact = []
    for i, f in enumerate(facts):
        g, _ = grad(model, params, f)
        gw = g * winv
        per_fact.append({"self": bdot(g, g), **{m: bdot(g, means[m]) for m in GROUPS},
                         "w_self": float((gw.double() ** 2).sum()),
                         **{f"w_{m}": float((gw.double() * wmeans[m].double()).sum()) for m in GROUPS}})
        if i % 50 == 0:
            print(f"pass2 {i} ({time.time() - t0:.0f}s)", flush=True)
    del model, params

    # direct check: group means recomputed on the model rescaled by c = 16 vs the formula
    check = {}
    for k in (4,):
        c = 2.0 ** k
        mc, pc = model_at(c)
        mA = sum(grad(mc, pc, facts[i])[0] for i in idx["A"]) / len(idx["A"])
        mB = sum(grad(mc, pc, facts[i])[0] for i in idx["B"]) / len(idx["B"])
        direct = {"A.B": float((mA.double() * mB.double()).sum()), "A.A": float((mA.double() ** 2).sum())}
        for key, val in direct.items():
            r, q, kk = mm[key]
            pred = r + q / c ** 2 + c ** 2 * kk
            check[f"log2c={k} {key}"] = abs(val - pred) / abs(val)
        del mc, pc, mA, mB
    print("direct check rel err", check, flush=True)

    def at(v, c):
        r, q, k = v
        return r + q / c ** 2 + c ** 2 * k

    per_c = []
    nA, nB = len(idx["A"]), len(idx["B"])
    for k in LOG2C:
        c = 2.0 ** k
        AA, BB, AB = at(mm["A.A"], c), at(mm["B.B"], c), at(mm["A.B"], c)
        AR, RR = at(mm["A.retain"], c), at(mm["retain.retain"], c)
        xy = []
        for i, pf in enumerate(per_fact):
            ss, dA, dB = at(pf["self"], c), at(pf["A"], c), at(pf["B"], c)
            aa, bb, ab = AA, BB, AB
            if split[i] == "A":  # leave-one-out mean of A
                aa = (nA ** 2 * AA - 2 * nA * dA + ss) / (nA - 1) ** 2
                ab = (nA * AB - dB) / (nA - 1)
                dA = (nA * dA - ss) / (nA - 1)
            elif split[i] == "B":  # leave-one-out mean of B
                bb = (nB ** 2 * BB - 2 * nB * dB + ss) / (nB - 1) ** 2
                ab = (nB * AB - dA) / (nB - 1)
                dB = (nB * dB - ss) / (nB - 1)
            # orthogonalise B's mean against A's mean: u = mB - (ab / aa) mA
            du = dB - ab / aa * dA
            uu = bb - ab ** 2 / aa
            n = max(ss, 1e-300) ** 0.5
            xy.append([round(dA / (n * aa ** 0.5), 4), round(du / (n * max(uu, 1e-300) ** 0.5), 4)])
        sh = [mm["A.A"][0], mm["A.A"][1] / c ** 2, mm["A.A"][2] * c ** 2]
        per_c.append({
            "log2c": k,
            "cos_meanA_meanB": AB / (AA * BB) ** 0.5,
            "cos_meanA_meanRetain": AR / (AA * RR) ** 0.5,
            "share_meanA_K": sh[2] / sum(sh), "share_meanA_Q": sh[1] / sum(sh),
            "xy": xy,
        })
    # natural-gradient control (c-independent), same coordinates in the whitened metric
    wxy = []
    for i, pf in enumerate(per_fact):
        ss, dA, dB = pf["w_self"], pf["w_A"], pf["w_B"]
        aa, bb, ab = wmm["A.A"], wmm["B.B"], wmm["A.B"]
        if split[i] == "A":
            aa = (nA ** 2 * aa - 2 * nA * dA + ss) / (nA - 1) ** 2
            ab = (nA * ab - dB) / (nA - 1)
            dA = (nA * dA - ss) / (nA - 1)
        elif split[i] == "B":
            bb = (nB ** 2 * bb - 2 * nB * dB + ss) / (nB - 1) ** 2
            ab = (nB * ab - dA) / (nB - 1)
            dB = (nB * dB - ss) / (nB - 1)
        du, uu = dB - ab / aa * dA, bb - ab ** 2 / aa
        n = max(ss, 1e-300) ** 0.5
        wxy.append([round(dA / (n * aa ** 0.5), 4), round(du / (n * max(uu, 1e-300) ** 0.5), 4)])
    out = {
        "seed": SEED, "n_params": n_tot, "log2c": LOG2C,
        "facts": [{"split": s, "loss": round(l, 2), "name": f["prompt"].replace(" lives in the city of", ""),
                   "answer": f["answer"]} for s, l, f in zip(split, losses, facts)],
        "group_mean_block_dots": mm,
        "direct_check_rel_err": check,
        "per_c": per_c,
        "natural_gradient": {"cos_meanA_meanB": wmm["A.B"] / (wmm["A.A"] * wmm["B.B"]) ** 0.5,
                             "cos_meanA_meanRetain": wmm["A.retain"] / (wmm["A.A"] * wmm["retain.retain"]) ** 0.5,
                             "xy": wxy,
                             "damping": "F_i + 0.1 * mean(F) over its Q/K/V head block or tensor"},
        "seconds": round(time.time() - t0),
    }
    OUT.write_text(json.dumps(out))
    for r in per_c[::4]:
        print(r["log2c"], round(r["cos_meanA_meanB"], 3), round(r["cos_meanA_meanRetain"], 3),
              round(r["share_meanA_K"], 3), round(r["share_meanA_Q"], 3), flush=True)
    print("NG", {k: v for k, v in out["natural_gradient"].items() if k != "xy"}, flush=True)


if __name__ == "__main__":
    main()
