"""Differentiable head-swap ('cut') scores on the 2-block tail (blocks FIRST, FIRST+1; 8B: 34-35, 32 heads each),
used as the training penalty and as a fast monitor. Port of the 1.7B parent's pen.py (oracle-1026-audit-penalty) with
the constants from cfgm.py. The audit that counts (eval) is audit.py: exact hook patching, hard greedy picks.

Batch layout: x = cat(unprobed inputs, probed inputs) -> rows [0, n) unprobed, rows [n, 2n) probed (same prompts).
A 'head' = one of the NH query heads of a block = its HD-column slice of the o_proj input (2 x NH heads in the tail).
Cut of head h = the probed run gets the unprobed run's slice of h (same prompt), at all positions.
Block names here: A = the first tail block (8B: 34), B = the last block (8B: 35).

Exactness notes (identities, up to bf16 rounding; as in the parent):
  * o_proj is linear (no bias; LoRA keeps it linear), so o(z with slice h swapped) = o(z) - o(mask_h * (z_p - z_u)).
  * Block B is the last block: after its attention every op is per-position, so a cut of a block-B head only needs
    the answer position. A cut of ALL heads (both blocks) also only needs the answer position, because block-B
    attention is then the unprobed run's at the answer position.
  * A cut of a block-A head changes block A's output at every position, so block B's keys/values are recomputed at
    every position (its query only at the answer position: block B's output is only read there).
Changes from the parent, none of which changes the loss or its gradient (only the cost):
  * block-B attention computes the query at the answer position only (the parent computed all positions and read the
    last one);
  * every block-A head is scored EXACTLY every update (the parent: 4 random of 16, its D7 hole). The top-3 is picked
    on a no-grad pass over all 2 x NH single-head cuts; only the picked block-A heads are recomputed with gradient.
    The gradient of sum(top-3 of [r]_+) is the sum of the picked entries' gradients, so this equals the full-graph
    gradient up to bf16 rounding (and the pick at exact ties).
Blind score of a variant v (no answer label): removed_v = KL(p_probed || p_unprobed) - KL(p_v || p_unprobed),
batch mean, in nats, at the answer position over the full vocabulary.
"""
import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

from cfgm import HD, NH, head_name


class Grab:
    """Pre-hook on o_proj that keeps its input (with graph) for one forward."""

    def __init__(self, mod):
        self.z = None
        self.h = mod.register_forward_pre_hook(self._f)

    def _f(self, mod, args):
        self.z = args[0]


def head_mask(heads, device, dtype):
    m = torch.zeros(len(heads), NH * HD, device=device, dtype=dtype)
    for j, h in enumerate(heads):
        m[j, h * HD:(h + 1) * HD] = 1
    return m


def mlp_res(layer, m):
    return m + layer.mlp(layer.post_attention_layernorm(m))


def kl_rows(lp_p, lp_q):
    """KL(p || q) per row from log-probs, float32."""
    return (lp_p.exp() * (lp_p - lp_q)).sum(-1)


def resolve_temp(P, n, temp):
    """temp = a number (fixed T) or 'B' (frozen per-prompt rule, PREREG): T_i = max(1, (z_p[t_p] - z_p[t_u]) / 4), z_p =
    fp32 answer logits of prompt i's PROBED uncut pass, t_p / t_u = argmax of its probed / unprobed uncut pass. No grad.
    Returns a float or a [n, 1] tensor (row i of every probed / unprobed / cut logit block is divided by T_i)."""
    if temp != "B":
        return float(temp)
    with torch.no_grad():
        z = P["orig"].detach().float()
        zp, tu = z[n:], z[:n].argmax(-1)
        tp_ = zp.argmax(-1)
        lead = zp.gather(1, tp_[:, None])[:, 0] - zp.gather(1, tu[:, None])[:, 0]
        return torch.clamp(lead / 4.0, min=1.0)[:, None]


def tdiv(x, T):
    """x [..., n, V] / T (T float or [n, 1])"""
    return x / T


def tdiv2(orig, T):
    """orig [2n, V] (unprobed rows, then probed rows of the same prompts) / T"""
    return orig / (torch.cat([T, T]) if torch.is_tensor(T) else T)


def lo(logits, a_id, b_id):
    lp = F.log_softmax(logits, -1)
    return lp[..., a_id] - lp[..., b_id]


class TailParts:
    def __init__(self, layers, norm, rotary, w_out):
        self.lA, self.lB = layers[0], layers[1]
        self.norm, self.rotary, self.w_out = norm, rotary, w_out
        self.gA, self.gB = Grab(self.lA.self_attn.o_proj), Grab(self.lB.self_attn.o_proj)
        self.w32 = None  # set to w_out.float() for fp32 answer logits (final-norm output upcast x fp32 lm_head)

    def remove(self):
        self.gA.h.remove(); self.gB.h.remove()

    def logits(self, h_last):
        if self.w32 is not None:
            return self.norm(h_last).float() @ self.w32.T
        return (self.norm(h_last) @ self.w_out.T).float()

    def cs(self, x):
        pos = torch.arange(x.shape[1], device=x.device)[None].expand(x.shape[0], -1)
        return self.rotary(x, pos)

    def attn_full(self, layer, x, cs):
        a, _ = layer.self_attn(hidden_states=layer.input_layernorm(x), position_embeddings=cs, attention_mask=None)
        return a

    def attn_last(self, layer, x, cs):
        """attention output at the answer position only: keys/values from all positions, one query. The o_proj
        pre-hook (Grab) fires with shape (R, 1, NH*HD)."""
        at = layer.self_attn
        R, T, _ = x.shape
        cos, sin = cs[0][:1], cs[1][:1]
        h = layer.input_layernorm(x)
        q = at.q_norm(at.q_proj(h[:, -1:]).view(R, 1, -1, HD)).transpose(1, 2)
        k = at.k_norm(at.k_proj(h).view(R, T, -1, HD)).transpose(1, 2)
        v = at.v_proj(h).view(R, T, -1, HD).transpose(1, 2)
        q, _ = apply_rotary_pos_emb(q, q, cos[:, -1:], sin[:, -1:])
        _, k = apply_rotary_pos_emb(k, k, cos, sin)
        g = at.num_key_value_groups
        o = F.scaled_dot_product_attention(q, repeat_kv(k, g), repeat_kv(v, g), scale=at.scaling)
        return at.o_proj(o.transpose(1, 2).reshape(R, 1, -1))[:, 0]

    def base(self, x):
        """orig answer logits [2n, V] (with graph) + the pieces the cuts need."""
        cs = self.cs(x)
        aA = self.attn_full(self.lA, x, cs)
        zA = self.gA.z
        mA = x + aA
        xB = mlp_res(self.lA, mA)
        aB = self.attn_last(self.lB, xB, cs)
        zB = self.gB.z[:, 0]
        mB = xB[:, -1] + aB
        return {"cs": cs, "x": x, "zA": zA, "mA": mA, "xB": xB, "zB": zB, "mB": mB,
                "orig": self.logits(mlp_res(self.lB, mB))}

    def cut_B(self, P, n, heads=None):
        """logits [H, n, V] of the probed rows with one block-B head cut each (answer position only)."""
        heads = list(range(NH)) if heads is None else heads
        dz = P["zB"][n:] - P["zB"][:n]                      # [n, NH*HD]
        mk = head_mask(heads, dz.device, dz.dtype)          # [H, NH*HD]
        oB = self.lB.self_attn.o_proj
        dm = oB((mk[:, None, :] * dz[None]).reshape(len(heads) * n, -1)).reshape(len(heads), n, -1)
        return self.logits(mlp_res(self.lB, P["mB"][None, n:] - dm))

    def cut_A(self, P, n, heads):
        """logits [K, n, V] of the probed rows with one block-A head cut each (block B rerun from the new block-A output)."""
        K = len(heads)
        if K == 0:
            return None
        T = P["x"].shape[1]
        dz = P["zA"][n:] - P["zA"][:n]                      # [n, T, NH*HD]
        mk = head_mask(heads, dz.device, dz.dtype)
        oA = self.lA.self_attn.o_proj
        dm = oA((mk[:, None, None, :] * dz[None]).reshape(K * n, T, -1))
        m = (P["mA"][None, n:] - dm.reshape(K, n, T, -1)).reshape(K * n, T, -1)
        xk = mlp_res(self.lA, m)
        ak = self.attn_last(self.lB, xk, P["cs"])  # attn_last reads row 0 of cos/sin only (same positions in every row)
        return self.logits(mlp_res(self.lB, xk[:, -1] + ak)).reshape(K, n, -1)

    def cut_all(self, P, n):
        """logits [n, V] of the probed rows with every head of both blocks cut (answer position only)."""
        oA, oB = self.lA.self_attn.o_proj, self.lB.self_attn.o_proj
        mAa = P["x"][n:, -1] + oA(P["zA"][:n, -1])
        xBa = mlp_res(self.lA, mAa)
        return self.logits(mlp_res(self.lB, xBa + oB(P["zB"][:n])))

    def cut_joint(self, P, n, headsA, headsB):
        """logits [n, V] of the probed rows with ALL listed heads swapped at once (block-A heads at all positions,
        block-B heads at the answer position; block B rerun when a block-A head is in the set)."""
        oA, oB = self.lA.self_attn.o_proj, self.lB.self_attn.o_proj
        zBu = P["zB"][:n]
        if headsA:
            dz = P["zA"][n:] - P["zA"][:n]
            mk = head_mask(headsA, dz.device, dz.dtype).sum(0)
            xk = mlp_res(self.lA, P["mA"][n:] - oA(mk * dz))
            ak = self.attn_last(self.lB, xk, P["cs"])
            zk, resid = self.gB.z[:, 0], xk[:, -1]
        else:
            ak, zk, resid = P["mB"][n:] - P["xB"][n:, -1], P["zB"][n:], P["xB"][n:, -1]
        if headsB:
            mkB = head_mask(headsB, zk.device, zk.dtype).sum(0)
            ak = ak - oB(mkB * (zk - zBu))
        return self.logits(mlp_res(self.lB, resid + ak))


    def cut_sets(self, P, n, sets):
        """logits [S, n, V] of the probed rows with each listed head SET swapped jointly (as cut_joint, but batched and
        always through the block-B rerun, so every candidate of a greedy step takes the same numeric path).
        sets: list of (headsA, headsB) index lists."""
        S = len(sets)
        D = NH * HD
        dev, dt = P["zA"].device, P["zA"].dtype
        mkA = torch.stack([head_mask(a, dev, dt).sum(0) if a else torch.zeros(D, device=dev, dtype=dt) for a, _ in sets])
        mkB = torch.stack([head_mask(b, dev, dt).sum(0) if b else torch.zeros(D, device=dev, dtype=dt) for _, b in sets])
        T = P["x"].shape[1]
        dz = P["zA"][n:] - P["zA"][:n]                                         # [n, T, D]
        oA, oB = self.lA.self_attn.o_proj, self.lB.self_attn.o_proj
        dm = oA((mkA[:, None, None, :] * dz[None]).reshape(S * n, T, -1))
        m = (P["mA"][None, n:] - dm.reshape(S, n, T, -1)).reshape(S * n, T, -1)
        xk = mlp_res(self.lA, m)
        ak = self.attn_last(self.lB, xk, P["cs"])
        zk = self.gB.z[:, 0]                                                    # [S*n, D]
        zBu = P["zB"][:n].repeat(S, 1)
        ak = ak - oB(mkB.repeat_interleave(n, 0) * (zk - zBu))
        return self.logits(mlp_res(self.lB, xk[:, -1] + ak)).reshape(S, n, -1)


def split_ab(idx):
    """flat head indices (0..2*NH-1; block A first) -> (headsA, headsB)"""
    return sorted(i for i in idx if i < NH), sorted(i - NH for i in idx if i >= NH)


@torch.no_grad()
def greedy_pick(tp, Ps, katt, temp, chunk=8):
    """The audit's pick rule on a fixed prompt set: joint greedy by blind removed amount at temperature temp.
    Ps: list of (P, n) no-grad base dicts (the prompt batches). Step j adds the head that maximises
    kl_full - mean KL(p_cut || p_unprobed) of the joint swap of (chosen + head), means over ALL prompts of Ps.
    Returns (chosen flat indices, per-step [best score, runner-up score], D^T_mean of the prompt set).
    The last step's best score = r^T_mean(H) of the picked set H on these prompts (AO0 = its [.]_+ / max(D, floor))."""
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
        tot = torch.zeros(len(cand))
        for (P, n), lp_u, T in zip(Ps, lpu, Ts):
            for s in range(0, len(cand), chunk):
                cs = cand[s:s + chunk]
                lg = tdiv(tp.cut_sets(P, n, [split_ab(chosen + [c]) for c in cs]), T)
                tot[s:s + len(cs)] += kl_rows(F.log_softmax(lg, -1), lp_u[None]).sum(-1).float().cpu()
        sc = klf - tot / N
        o = sc.sort(descending=True, stable=True)  # PREREG: a tie goes to the lower head index (candidates are in order)
        chosen.append(cand[int(o.indices[0])])
        steps.append([float(o.values[0]), float(o.values[1]) if len(cand) > 1 else None])
    return chosen, steps, klf


def removed(lp_u, kl_full, lg_v):
    """blind removed amount per variant: kl_full - mean KL(p_v || p_u); lg_v [..., n, V]"""
    return kl_full - kl_rows(F.log_softmax(lg_v, -1), lp_u).mean(-1)


@torch.no_grad()
def cut_free_monitor_parts(tp, P, n, chunkA=8):
    """no-grad logits of every single-head cut ([NH, n, V] for block A, [NH, n, V] for block B) and of the all-heads cut"""
    lgA = torch.cat([tp.cut_A(P, n, list(range(s, min(s + chunkA, NH)))) for s in range(0, NH, chunkA)])
    return lgA, tp.cut_B(P, n), tp.cut_all(P, n)


@torch.no_grad()
def singles_nograd(tp, P, n, chunkA=8, temp=1.0):
    """blind removed amount (batch mean, nats) of every single-head cut: tensor [2*NH], order = block A heads, block B.
    temp: every distribution is softmax(logits / temp) (this run: the unsaturated blind score)."""
    T = resolve_temp(P, n, temp)
    lp = F.log_softmax(tdiv2(P["orig"], T), -1)
    lp_u, lp_p = lp[:n], lp[n:]
    kl_full = kl_rows(lp_p, lp_u).mean()
    rA = [removed(lp_u, kl_full, tdiv(tp.cut_A(P, n, list(range(s, min(s + chunkA, NH)))), T))
          for s in range(0, NH, chunkA)]
    rB = removed(lp_u, kl_full, tdiv(tp.cut_B(P, n), T))
    return torch.cat(rA + [rB]).float(), float(kl_full)


def penalty(tp, P, n, kind, floor=0.0, chunkA=8, katt=3, temp=1.0, pick=None):
    """The training penalty on one micro-batch (with graph), and its parts (floats, for the log).
    HK = the katt heads with the largest single removed amount over all 2*NH heads (exact, picked without gradient),
    every distribution softmax(logits / temp).
    kind 'parent': P  = sum over HK of [removed_h]_+ (single swaps) + [removed_all]_+                    (nats)
    kind 'ratio' : AO = [removed(joint swap of HK)]_+ / max(KL_full, floor), no all-heads term (PREREG AO)
    """
    if pick is None:  # parent rule: the katt best SINGLE heads, re-scored on this micro-batch
        with torch.no_grad():
            Pd = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in P.items()}
            r_all_heads, _ = singles_nograd(tp, Pd, n, chunkA, temp)
        top = r_all_heads.topk(katt).indices.tolist()
        srt = r_all_heads.sort(descending=True).values
        margin = float(srt[katt - 1] - srt[katt])  # gap between the last picked head and the first left out
    else:             # this run: a fixed set (the joint greedy pick of greedy_pick, re-picked every few updates)
        top, margin = list(pick), None
    pickA, pickB = split_ab(top)
    T = resolve_temp(P, n, temp)
    lp = F.log_softmax(tdiv2(P["orig"], T), -1)
    lp_u, lp_p = lp[:n], lp[n:]
    kl_full = kl_rows(lp_p, lp_u).mean()
    info = {"top3": [head_name(0, i) for i in pickA] + [head_name(1, i) for i in pickB], "kl_full": float(kl_full),
            "margin_k": margin}
    if kind == "parent":
        parts = []
        if pickA:
            parts.append(removed(lp_u, kl_full, tdiv(tp.cut_A(P, n, pickA), T)))
        if pickB:
            parts.append(removed(lp_u, kl_full, tdiv(tp.cut_B(P, n, pickB), T)))
        top3 = torch.relu(torch.cat(parts)).sum()
        r_all = kl_full - kl_rows(F.log_softmax(tdiv(tp.cut_all(P, n), T), -1), lp_u).mean()
        info.update({"top3_nats": float(top3), "all_nats": float(r_all)})
        return top3 + torch.relu(r_all), info
    if kind == "ratio":
        r3 = removed(lp_u, kl_full, tdiv(tp.cut_joint(P, n, pickA, pickB), T))
        den = torch.clamp(kl_full, min=floor)
        info.update({"joint3_nats": float(r3), "den": float(den)})
        return torch.relu(r3) / den, info
    raise ValueError(kind)


def penalty_value(tp, P, n, kind, floor=0.0, chunkA=8, katt=3, temp=1.0, pick=None) -> float:
    """the same penalty, no graph (P0 / AO0 of the warm start)"""
    with torch.no_grad():
        r, kl_full = singles_nograd(tp, P, n, chunkA, temp)
        T = resolve_temp(P, n, temp)
        lp_u = F.log_softmax(tdiv(P["orig"][:n], T), -1)
        top = r.topk(katt)
        if kind == "parent":
            assert pick is None
            r_all = kl_full - float(kl_rows(F.log_softmax(tdiv(tp.cut_all(P, n), T), -1), lp_u).mean())
            return float(torch.relu(top.values).sum()) + max(r_all, 0.0)
        idx = top.indices.tolist() if pick is None else list(pick)
        lg = tp.cut_joint(P, n, sorted(i for i in idx if i < NH), sorted(i - NH for i in idx if i >= NH))
        r3 = kl_full - float(kl_rows(F.log_softmax(tdiv(lg, T), -1), lp_u).mean())
        return max(r3, 0.0) / max(kl_full, floor)
