"""Vera's own checker engine for RMSNorm models (Qwen3-0.6B, SmolLM2-135M) on HF modules. No teammate code.
Definitions from PREREG.md (kran-research c28c8fa0, projects/frozen-norm-gradients):
  y = logit[t, tok_{t+1}] - mean_v logit[t, v];  g_L = dy/d(mlp_out_L[t]).
  residual norms = input_layernorm, post_attention_layernorm of every block + final norm.
  frozen norm = rms detached. Implemented as rms_eff = w*rms.detach() + (1-w)*rms with switch w
  (value-identical forward; w=0 real, w=1 frozen).
First-order tick: G(w) is multi-affine in the per-norm switches, so sum_k (g_k - g_frz) = -dG/dw at w=1
along the all-ones direction (one shared w). Computed EXACTLY by a double-backward JVP:
  s = <u, g(w)>, ds/dw = <u, dg/dw>, d/du (ds/dw) = dg/dw."""
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = {'qwen3': 'Qwen/Qwen3-0.6B', 'smol': 'HuggingFaceTB/SmolLM2-135M'}


class State:
    w = None        # None -> real; tensor/float -> residual-norm blend switch
    wqk = None      # same for q_norm/k_norm (named cell only)
    wmap = None     # optional {id(module): w} override (unit tests: one norm live)


def make_forward(mod, st, kind):
    eps = mod.variance_epsilon

    def fwd(h):
        dt = h.dtype
        h = h.to(torch.float32)
        rms = torch.sqrt(h.pow(2).mean(-1, keepdim=True) + eps)
        w = st.w if kind == 'res' else st.wqk
        if st.wmap is not None and id(mod) in st.wmap: w = st.wmap[id(mod)]
        if w is not None:
            rms = w * rms.detach() + (1 - w) * rms
        return mod.weight * (h / rms).to(dt)
    return fwd


class HF:
    def __init__(self, name, device='cpu', attn='eager'):
        self.tok = AutoTokenizer.from_pretrained(REPO[name])
        self.m = AutoModelForCausalLM.from_pretrained(REPO[name], dtype=torch.float32,
                                                      attn_implementation=attn).to(device).eval()
        for p in self.m.parameters(): p.requires_grad_(False)
        self.dev = device
        self.st = State()
        self.layers = self.m.model.layers
        self.nL = len(self.layers)
        self.res_norms = []
        for b in self.layers:
            for n in (b.input_layernorm, b.post_attention_layernorm):
                n.forward = make_forward(n, self.st, 'res'); self.res_norms.append(n)
            a = b.self_attn
            for nm in ('q_norm', 'k_norm'):
                if hasattr(a, nm):
                    n = getattr(a, nm); n.forward = make_forward(n, self.st, 'qk')
        self.m.model.norm.forward = make_forward(self.m.model.norm, self.st, 'res')
        self.deltas = {}
        for L, b in enumerate(self.layers):
            b.mlp.register_forward_hook(self._hook(L))

    def _hook(self, L):
        def h(mod, inp, out):
            d = self.deltas.get(L)
            return out + d if d is not None else out
        return h

    def ids(self, text, n=128):
        ids = self.tok(text, return_tensors='pt', add_special_tokens=False).input_ids[0]
        return ids[:n] if len(ids) >= n else None

    def resid_norms(self, ids):
        with torch.no_grad():
            self.st.w = None; self.st.wqk = None
            o = self.m(ids[None].to(self.dev), output_hidden_states=True)
        # hidden_states[L] = residual entering block L (hidden_states[0] = embeddings)
        return [h[0].norm(dim=-1).cpu() for h in o.hidden_states[:self.nL]]

    def _y(self, ids, t, Ls):
        T = len(ids)
        self.deltas = {L: torch.zeros(1, T, self.m.config.hidden_size, device=self.dev, requires_grad=True) for L in Ls}
        lo = self.m(ids[None].to(self.dev)).logits[0, t]
        y = lo[int(ids[t + 1])] - lo.mean()
        return y

    def grads(self, ids, t, Ls, w=None, wqk=None):
        """returns {L: g_L[t]} (cpu float64 copies)."""
        self.st.w = w; self.st.wqk = wqk
        with torch.enable_grad():
            y = self._y(ids, t, Ls)
            gs = torch.autograd.grad(y, [self.deltas[L] for L in Ls])
        self.deltas = {}; self.st.w = None; self.st.wqk = None
        return {L: g[0, t].detach().double().cpu() for L, g in zip(Ls, gs)}

    def frozen_and_tick(self, ids, t, Ls):
        """returns g_frz{L}, g_pred{L} with g_pred = g_frz - dG/dw|_{w=1} (exact first order)."""
        w = torch.tensor(1.0, device=self.dev, requires_grad=True)
        self.st.w = w; self.st.wqk = None
        with torch.enable_grad():
            y = self._y(ids, t, Ls)
            gs = torch.autograd.grad(y, [self.deltas[L] for L in Ls], create_graph=True)
            gt = [g[0, t] for g in gs]
            us = [torch.zeros_like(g, requires_grad=True) for g in gt]
            s, = torch.autograd.grad(gt, w, grad_outputs=us, create_graph=True)
            dgs = torch.autograd.grad(s, us, allow_unused=True)
        self.deltas = {}; self.st.w = None
        out_f, out_p = {}, {}
        for L, g, dg in zip(Ls, gt, dgs):
            gf = g.detach().double().cpu()
            dg = torch.zeros_like(gf) if dg is None else dg.detach().double().cpu()
            out_f[L] = gf; out_p[L] = gf - dg
        return out_f, out_p


def cos(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm()))
