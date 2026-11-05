"""Read-out engine for RMSNorm models through HF transformers (Qwen3-0.6B, SmolLM2-135M).

Every RMSNorm module gets a switch: 'live' (real Jacobian) or 'frozen' (the scale 1/rms detached, the
circuit-tracer choice). Frozen RMSNorm Jacobian: (gamma/r) I; live: (gamma/r)(I - x x^T / (d r^2)), so freezing
drops exactly one rank-1 term per norm (no mean projection in RMSNorm).

Norm ids: (l, 'ln1') input_layernorm, (l, 'ln2') post_attention_layernorm, (l, 'q') / (l, 'k') q_norm / k_norm
(Qwen3 only, per head), (n_layers, 'lnf') final norm.
Read-out: g = d y / d mlp_out_L[t], y = logit[t, tok] - mean_v logit[t, v]; tok = ids[t+1].
The forward VALUE is the same in every cell; only gradients change.
"""
from __future__ import annotations

import os
import types

import torch

DEV = torch.device(os.environ.get("ENGINE_DEV", "mps" if torch.backends.mps.is_available() else "cpu"))


def _rms_forward(self, hidden_states):
    x = hidden_states
    var = x.pow(2).mean(-1, keepdim=True)
    r = torch.rsqrt(var + self.variance_epsilon)
    if self._mode == "frozen":
        r = r.detach()
    xh = x * r
    tr = self._track
    if tr is not None and self._mode == "frozen":  # frozen-graph tensors for the first-order predictor
        tr[self._nid] = {"out": xh, "xin": x, "r": r.detach()}
    return self.weight * xh


class HFEngine:
    def __init__(self, repo, device=DEV, dtype=torch.float32):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(repo)
        # load in the checkpoint dtype, then convert param by param (keeps the peak footprint low; no accelerate)
        self.m = AutoModelForCausalLM.from_pretrained(repo, attn_implementation="eager").eval()
        self.m.to(device=device, dtype=dtype)
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.device = device
        self.layers = self.m.model.layers
        self.n = len(self.layers)
        self.norms = {}
        for l, lay in enumerate(self.layers):
            self.norms[(l, "ln1")] = lay.input_layernorm
            self.norms[(l, "ln2")] = lay.post_attention_layernorm
            if hasattr(lay.self_attn, "q_norm"):
                self.norms[(l, "q")] = lay.self_attn.q_norm
                self.norms[(l, "k")] = lay.self_attn.k_norm
        self.norms[(self.n, "lnf")] = self.m.model.norm
        for nid, mod in self.norms.items():
            mod._mode = "live"
            mod._nid = nid
            mod._track = None
            mod.forward = types.MethodType(_rms_forward, mod)
        self._delta = None
        self._hL = None
        for l, lay in enumerate(self.layers):
            lay.mlp.register_forward_hook(self._mk_hook(l))

    def _mk_hook(self, l):
        def hook(mod, inp, out):
            if self._hL == l and self._delta is not None:
                rows, pos, d = self._delta
                return out.index_put((rows, pos), out[rows, pos] + d)
            return out
        return hook

    def downstream(self, L, kinds=("ln1", "ln2", "lnf")):
        """Norms on the path from mlp_out_L to the logit: ln1/ln2 (and q/k) of layers > L, plus the final norm."""
        return [n for n in self.norms if n[0] > L and n[1] in kinds]

    def set_modes(self, frozen):
        frozen = set(frozen)
        for n, mod in self.norms.items():
            mod._mode = "frozen" if n in frozen else "live"

    def readout(self, ids, L, pos, tok, frozen=()):
        """g [B,d] for the given frozen norm set; also y [B]."""
        self.set_modes(frozen)
        B = ids.shape[0]
        rows = torch.arange(B, device=ids.device)
        delta = torch.zeros(B, self.m.config.hidden_size, device=ids.device, dtype=self.m.dtype, requires_grad=True)
        self._delta, self._hL = (rows, pos, delta), L
        try:
            with torch.enable_grad():
                h = self.m.model(input_ids=ids, use_cache=False).last_hidden_state
                lg = self.m.lm_head(h[rows, pos])
                y = lg.gather(-1, tok[:, None])[:, 0] - lg.mean(-1)
                g, = torch.autograd.grad(y.sum(), delta)
        finally:
            self._delta, self._hL = None, None
            self.set_modes(())
        return g.detach(), y.detach()

    def readout_first_order(self, ids, L, pos, tok, frozen):
        """Theory tick (PREREG s5), from frozen-graph tensors only (never a live-norm or real backward).
        Pass 1: frozen backward; g_k = dy/d x_hat_k at each frozen residual norm k, position t.
        e_k = -(r_k/d) x_hat_k (x_hat_k . g_k)   (the rank-1 term the frozen Jacobian drops; r = 1/rms).
        Pass 2: frozen backward of y + sum_k <e_k, x_k>  ->  g_pred = g_frz + sum_k J_frz(k<-L)^T e_k.
        Returns g_frz, g_pred, {nid: (a_k = x_hat_k.g_k/d, |e_k|)}."""
        frozen = list(frozen)
        assert all(n[1] in ("ln1", "ln2", "lnf") for n in frozen), "residual norms only"
        B = ids.shape[0]
        rows = torch.arange(B, device=ids.device)
        d = self.m.config.hidden_size

        def run(extra_e):
            tr = {}
            for n in frozen:
                self.norms[n]._track = tr
            self.set_modes(frozen)
            delta = torch.zeros(B, d, device=ids.device, dtype=self.m.dtype, requires_grad=True)
            self._delta, self._hL = (rows, pos, delta), L
            try:
                with torch.enable_grad():
                    h = self.m.model(input_ids=ids, use_cache=False).last_hidden_state
                    lg = self.m.lm_head(h[rows, pos])
                    y = lg.gather(-1, tok[:, None])[:, 0] - lg.mean(-1)
                    nids = [n for n in frozen if n in tr]
                    if extra_e is None:
                        gs = torch.autograd.grad(y.sum(), [delta] + [tr[n]["out"] for n in nids])
                        out = (gs[0].detach(), {n: (tr[n], gk[rows, pos].detach()) for n, gk in zip(nids, gs[1:])})
                    else:
                        ex = sum((extra_e[n] * tr[n]["xin"][rows, pos]).sum() for n in nids)
                        out = (torch.autograd.grad(y.sum() + ex, delta)[0].detach(), None)
            finally:
                self._delta, self._hL = None, None
                for n in frozen:
                    self.norms[n]._track = None
                self.set_modes(())
            return out

        g_frz, per = run(None)
        es, stats = {}, {}
        for n, (t, gk) in per.items():
            r = t["r"][rows, pos, 0]
            xh = (t["xin"][rows, pos] * t["r"][rows, pos]).detach()
            a = (xh * gk).sum(-1) / d
            es[n] = -(r * a)[:, None] * xh
            stats[n] = (a, es[n].norm(dim=-1))
        g_pred, _ = run(es)
        return g_frz, g_pred, stats

    @torch.no_grad()
    def logits_at(self, ids, pos):
        self.set_modes(())
        rows = torch.arange(ids.shape[0], device=ids.device)
        h = self.m.model(input_ids=ids, use_cache=False).last_hidden_state
        return self.m.lm_head(h[rows, pos])
