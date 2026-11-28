"""Faster joint greedy pick: the same rule as pen.greedy_pick (step j adds the head that maximises
kl_full - mean KL(p_cut || p_unprobed) of the joint swap chosen + head), with one change in cost only:
every block-B candidate of a step shares the step's block-A part (the chosen block-A heads), so block A's MLP and
block B's keys/values are computed ONCE per step for all block-B candidates, not once per candidate.
Block-A candidates take pen.TailParts.cut_sets unchanged."""
import torch
import torch.nn.functional as F

from cfgm import HD, NH
from pen import head_mask, kl_rows, mlp_res, resolve_temp, split_ab, tdiv, tdiv2


@torch.no_grad()
def b_candidates(tp, P, n, headsA, headsB, cands):
    """logits [len(cands), n, V]: probed rows with (headsA, headsB + [c]) swapped, one row block per block-B head c"""
    oA, oB = tp.lA.self_attn.o_proj, tp.lB.self_attn.o_proj
    D = NH * HD
    dev, dt = P["zA"].device, P["zA"].dtype
    T = P["x"].shape[1]
    mkA = head_mask(headsA, dev, dt).sum(0) if headsA else torch.zeros(D, device=dev, dtype=dt)
    dz = P["zA"][n:] - P["zA"][:n]
    dm = oA((mkA[None, None, :] * dz).reshape(n, T, -1))
    xk = mlp_res(tp.lA, P["mA"][n:] - dm)                     # the step's block-A part, once
    ak = tp.attn_last(tp.lB, xk, P["cs"])
    zk = tp.gB.z[:, 0]
    zBu = P["zB"][:n]
    S = len(cands)
    mkB = torch.stack([head_mask(headsB + [c], dev, dt).sum(0) for c in cands])          # [S, D]
    akc = ak[None] - oB((mkB[:, None, :] * (zk - zBu)[None]).reshape(S * n, -1)).reshape(S, n, -1)
    return tp.logits(mlp_res(tp.lB, xk[None, :, -1] + akc))  # [S, n, V]


@torch.no_grad()
def greedy_pick_fast(tp, Ps, katt, temp, chunk=8):
    lpu, Ts, klf, N = [], [], 0.0, 0
    for P, n in Ps:
        T = resolve_temp(P, n, temp)
        Ts.append(T)
        lp = F.log_softmax(tdiv2(P["orig"], T), -1)
        lpu.append(lp[:n])
        klf += float(kl_rows(lp[n:], lp[:n]).sum()); N += n
    klf /= N
    chosen, steps = [], []
    for _ in range(katt):
        cand = [i for i in range(2 * NH) if i not in chosen]
        cA = [c for c in cand if c < NH]
        cB = [c for c in cand if c >= NH]
        hA, hB = split_ab(chosen)
        tot = torch.zeros(len(cand))
        for (P, n), lp_u, T in zip(Ps, lpu, Ts):
            for s in range(0, len(cA), chunk):
                cs = cA[s:s + chunk]
                lg = tdiv(tp.cut_sets(P, n, [split_ab(chosen + [c]) for c in cs]), T)
                tot[s:s + len(cs)] += kl_rows(F.log_softmax(lg, -1), lp_u[None]).sum(-1).float().cpu()
            for s in range(0, len(cB), chunk):
                cs = cB[s:s + chunk]
                lg = tdiv(b_candidates(tp, P, n, hA, hB, [c - NH for c in cs]), T)
                tot[len(cA) + s:len(cA) + s + len(cs)] += kl_rows(F.log_softmax(lg, -1), lp_u[None]).sum(-1).float().cpu()
        sc = klf - tot / N
        o = sc.sort(descending=True, stable=True)  # candidates stay in index order (A then B): same tie rule
        chosen.append(cand[int(o.indices[0])])
        steps.append([float(o.values[0]), float(o.values[1]) if len(cand) > 1 else None])
    return chosen, steps, klf
