"""Vera's own GPT-2 forward (no teammate code) with circuit-tracer-style linearisation.
graph mode: attention patterns and LN scales detached; MLP_l for l > L replaced by TC_l(z) + (MLP_l(z)-TC_l(z)).detach().
Returns g = d y / d mlp_out_L[t] where y = logit[t, tok] - mean_v logit[t, v]."""
import torch
from vera_lib import load_gpt2, load_tc, mlp_in_z, gelu_new, tc_forward


class Engine:
    def __init__(self, layers=(8, 9, 10, 11)):
        self.m, self.tok = load_gpt2()
        sd = self.m.transformer
        self.wte, self.wpe = sd.wte.weight.detach(), sd.wpe.weight.detach()
        self.blocks = []
        for b in sd.h:
            self.blocks.append(dict(
                g1=b.ln_1.weight.detach(), b1=b.ln_1.bias.detach(),
                Wqkv=b.attn.c_attn.weight.detach(), bqkv=b.attn.c_attn.bias.detach(),
                Wo=b.attn.c_proj.weight.detach(), bo=b.attn.c_proj.bias.detach(),
                g2=b.ln_2.weight.detach(), b2=b.ln_2.bias.detach(),
                Wi=b.mlp.c_fc.weight.detach(), bi=b.mlp.c_fc.bias.detach(),
                Wm=b.mlp.c_proj.weight.detach(), bm=b.mlp.c_proj.bias.detach()))
        self.gf, self.bf = sd.ln_f.weight.detach(), sd.ln_f.bias.detach()
        self.WU = self.m.lm_head.weight.detach()
        self.tc = {L: load_tc(L) for L in layers}
        self.mz = {L: mlp_in_z(self.m, L) for L in layers}

    @staticmethod
    def norm(x, detach_scale):
        mu = x.mean(-1, keepdim=True)
        sd = torch.sqrt(((x - mu) ** 2).mean(-1, keepdim=True) + 1e-5)
        if detach_scale: sd = sd.detach()
        return (x - mu) / sd

    def attn(self, h, B, detach):
        T = h.shape[0]
        qkv = h @ B['Wqkv'] + B['bqkv']
        q, k, v = qkv.split(768, -1)
        q = q.view(T, 12, 64).transpose(0, 1); k = k.view(T, 12, 64).transpose(0, 1); v = v.view(T, 12, 64).transpose(0, 1)
        s = q @ k.transpose(-1, -2) / 8.0
        s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float('-inf'))
        p = torch.softmax(s, -1)
        if detach: p = p.detach()
        o = (p @ v).transpose(0, 1).reshape(T, 768)
        return o @ B['Wo'] + B['bo']

    def forward(self, ids, L=None, graph=False, delta=None, zpush=None, capture=False, parts=('ln','attn','tc')):
        """ids: 1-D LongTensor. delta: [T,768] added to mlp_out_L (centred space ok, LN removes mean).
        zpush: (t, du) replaces MLP_L input z_n[t] by z_n[t]+du for the REAL MLP_L (residual untouched).
        capture: returns dict with z_n at L. Returns logits [T,V]."""
        T = len(ids)
        x = self.wte[ids] + self.wpe[:T]
        cap = {}
        for l, B in enumerate(self.blocks):
            d_att = graph and L is not None and l > L
            # attention patterns are frozen for ALL layers in graph mode (pattern only matters downstream of L anyway)
            h = self.norm(x, graph and 'ln' in parts) * B['g1'] + B['b1']
            x = x + self.attn(h, B, graph and 'attn' in parts)
            z = self.norm(x, graph and 'ln' in parts)
            if capture and l == L: cap['z'] = z.detach().clone()
            if zpush is not None and l == L:
                t, du = zpush
                z = z.clone(); z[t] = z[t] + du
            mo = gelu_new((z * B['g2'] + B['b2']) @ B['Wi'] + B['bi']) @ B['Wm'] + B['bm']
            if graph and 'tc' in parts and L is not None and l > L and l in self.tc:
                _, rec = tc_forward(self.tc[l], z)
                mo_c = mo - mo.mean(-1, keepdim=True)
                mo = rec + (mo_c - rec).detach() + mo.mean(-1, keepdim=True).detach()
            if l == L and delta is not None:
                mo = mo + delta
            x = x + mo
        hf = self.norm(x, graph and 'ln' in parts) * self.gf + self.bf
        lo = hf @ self.WU.T
        return (lo, cap) if capture else lo

    def y(self, lo, t, tok):
        return lo[t, tok] - lo[t].mean()

    def grad_g(self, ids, L, t, tok, graph=True, parts=('ln','attn','tc')):
        delta = torch.zeros(len(ids), 768, requires_grad=True)
        with torch.enable_grad():
            lo = self.forward(ids, L=L, graph=graph, delta=delta, parts=parts)
            yv = self.y(lo, t, tok)
            g, = torch.autograd.grad(yv, delta)
        return g[t], float(yv)

    def mlp_local(self, L, z):
        Win, bin_, Wout, bout = self.mz[L]
        return gelu_new(z @ Win + bin_) @ Wout
