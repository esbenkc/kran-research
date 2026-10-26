"""Differentiable head-swap ('cut') scores on the 2-block tail (blocks 26-27 of Qwen3-1.7B), used as the training
penalty and as a fast monitor. The audit that counts (eval) is audit.py: exact hook patching, hard greedy picks.

Batch layout: x = cat(unprobed inputs, probed inputs) -> rows [0, n) unprobed, rows [n, 2n) probed (same prompts).
A 'head' = one of the 16 query heads of a block = its 128-column slice of the o_proj input (32 heads in 26-27).
Cut of head h = the probed run gets the unprobed run's slice of h (same prompt), at all positions.

Exactness notes (identities, up to bf16 rounding):
  * o_proj is linear (no bias; LoRA keeps it linear), so o(z with slice h swapped) = o(z) - o(mask_h * (z_p - z_u)).
  * Block 27 is the last block: after its attention every op is per-position, so a cut of a block-27 head only
    needs the answer position. A cut of ALL 32 heads (both blocks) also only needs the answer position, because
    block-27 attention is then replaced by the unprobed run's slices.
  * A cut of a block-26 head changes block 26's output at every position, so block 27 is rerun in full.
Blind score of a variant v (no answer label): removed_v = KL(p_probed || p_unprobed) - KL(p_v || p_unprobed),
batch mean, in nats, at the answer position over the full vocabulary.
"""
import torch
import torch.nn.functional as F

HD = 128
NH = 16


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


def attn(layer, x, cos_sin, grab):
    a, _ = layer.self_attn(hidden_states=layer.input_layernorm(x), position_embeddings=cos_sin, attention_mask=None)
    return a, grab.z


def logits_last(norm, w_out, h_last):
    return (norm(h_last) @ w_out.T).float()


class TailParts:
    def __init__(self, layers, norm, rotary, w_out):
        self.l26, self.l27 = layers[0], layers[1]
        self.norm, self.rotary, self.w_out = norm, rotary, w_out
        self.g26, self.g27 = Grab(self.l26.self_attn.o_proj), Grab(self.l27.self_attn.o_proj)

    def run(self, x, n, heads26=(), want=("c27", "all")):
        """Return dict of answer-position logits: orig [2n,V], c27 [16,n,V], c26 [K,n,V], all [n,V]."""
        T = x.shape[1]
        pos = torch.arange(T, device=x.device)[None].expand(x.shape[0], -1)
        cs = self.rotary(x, pos)
        a26, z26 = attn(self.l26, x, cs, self.g26)
        m26 = x + a26
        x27 = mlp_res(self.l26, m26)
        a27, z27 = attn(self.l27, x27, cs, self.g27)
        m27 = x27 + a27
        out = {"orig": logits_last(self.norm, self.w_out, mlp_res(self.l27, m27[:, -1]))}
        o26, o27 = self.l26.self_attn.o_proj, self.l27.self_attn.o_proj
        if "c27" in want:
            dz = (z27[n:, -1] - z27[:n, -1])  # [n, 2048]
            mk = head_mask(range(NH), x.device, x.dtype)  # [16, 2048]
            dm = o27((mk[:, None, :] * dz[None]).reshape(NH * n, -1)).reshape(NH, n, -1)
            m = m27[None, n:, -1] - dm
            out["c27"] = logits_last(self.norm, self.w_out, mlp_res(self.l27, m))
        if "all" in want:
            m26a = x[n:, -1] + o26(z26[:n, -1])
            x27a = mlp_res(self.l26, m26a)
            m27a = x27a + o27(z27[:n, -1])
            out["all"] = logits_last(self.norm, self.w_out, mlp_res(self.l27, m27a))
        if len(heads26):
            K = len(heads26)
            dz = z26[n:] - z26[:n]  # [n, T, 2048]
            mk = head_mask(heads26, x.device, x.dtype)
            dm = o26((mk[:, None, None, :] * dz[None]).reshape(K * n, T, -1))
            m = (m26[None, n:] - dm.reshape(K, n, T, -1)).reshape(K * n, T, -1)
            xk = mlp_res(self.l26, m)
            posk = torch.arange(T, device=x.device)[None].expand(K * n, -1)
            csk = self.rotary(xk, posk)
            ak, _ = attn(self.l27, xk, csk, self.g27)
            mk27 = (xk + ak)[:, -1]
            out["c26"] = logits_last(self.norm, self.w_out, mlp_res(self.l27, mk27)).reshape(K, n, -1)
        return out


def kl_rows(lp_p, lp_q):
    """KL(p || q) per row from log-probs, float32."""
    return (lp_p.exp() * (lp_p - lp_q)).sum(-1)


def blind_scores(out, n):
    """removed_v (nats, batch mean) for every variant; plus KL_full."""
    lp = F.log_softmax(out["orig"], -1)
    lu, lpp = lp[:n], lp[n:]
    kl_full = kl_rows(lpp, lu).mean()
    res = {"kl_full": kl_full}
    for k in ("c27", "c26"):
        if k in out:
            lv = F.log_softmax(out[k], -1)
            res[k] = kl_full - kl_rows(lv, lu[None]).mean(-1)  # [H]
    if "all" in out:
        res["all"] = kl_full - kl_rows(F.log_softmax(out["all"], -1), lu).mean()
    return res


def lo(logits, a_id, b_id):
    lp = F.log_softmax(logits, -1)
    return lp[..., a_id] - lp[..., b_id]
