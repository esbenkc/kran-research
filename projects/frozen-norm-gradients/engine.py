"""Read-out engine for theory-1105-ln-unfreeze-readout (Bruno).

Derived from the transcoder-neuron engine (kran-research swarm/theory-1104-transcoder-neuron,
projects/transcoder-neuron/common.py, gates G1-G7 passed there). The parent had ONE freeze flag; this
engine splits it into three independent parts, each applied to every layer l > L and to ln_f:

  ln   'live'   : LayerNorm scale sigma recomputed (the real Jacobian)
       'frozen' : sigma detached (circuit-tracer): J = (gamma/sigma) (I - 11^T/d)
       'fixed'  : sigma detached, plus ONE fixed unit direction u per norm projected out:
                  J = (gamma/sigma) (I - 11^T/d - u u^T)   (u _|_ 1, from calibration prompts)
       'token'  : like 'fixed' but u = the token's own unit x_hat (gate: equals 'live' up to eps)
  attn 'live' | 'frozen' : attention pattern detached or not
  mlp  'real' | 'tc'     : MLP_l, or transcoder Jacobian with a frozen error node (value = real MLP)

Read-out: g = d y / d mlp_out_L[t], y = logit[t, tok] - mean_v logit[t, v] (centred next-token logit),
tok = the true next token ids[t+1]. Only stream t reaches logit t (causal mask), so only the norms at
position t matter. The forward VALUE is the real model in every cell (frozen parts change gradients only).
"""
from __future__ import annotations

import glob
import math
import os
import pickle
import types

import torch

HF_SNAP = os.path.expanduser("~/.cache/huggingface/hub/models--openai-community--gpt2/snapshots")
TC_SNAP = os.path.expanduser("~/.cache/huggingface/hub/models--pchlenski--gpt2-transcoders/snapshots")
EPS = 1e-5
DEV = torch.device(os.environ.get("ENGINE_DEV", "mps" if torch.backends.mps.is_available() else "cpu"))
NORMS = [(l, k) for l in range(12) for k in ("ln1", "ln2")] + [(12, "lnf")]  # norm ids (layer, kind)


def gelu_new(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, s):
        self.__dict__.update(s if isinstance(s, dict) else {"state": s})


class _U(pickle.Unpickler):
    def find_class(self, mod, name):
        try:
            return super().find_class(mod, name)
        except Exception:  # sae_training.* classes from the authors' repo: keep only the data
            return type(name, (_Stub,), {})


_PM = types.SimpleNamespace(Unpickler=_U, load=pickle.load, __name__="pm")


class Transcoder:
    def __init__(self, L, device=DEV, dtype=torch.float32):
        p = glob.glob(f"{TC_SNAP}/*/final_sparse_autoencoder_gpt2-small_blocks.{L}.ln2.hook_normalized_24576.pt")[0]
        d = torch.load(p, map_location="cpu", weights_only=False, pickle_module=_PM, mmap=True)
        assert d["cfg"].hook_point == f"blocks.{L}.ln2.hook_normalized"
        sd = d["state_dict"]
        self.W_enc = sd["W_enc"].to(device, dtype)
        self.b_enc = sd["b_enc"].to(device, dtype)
        self.W_dec = sd["W_dec"].to(device, dtype)
        self.b_dec = sd["b_dec"].to(device, dtype)
        self.b_dec_out = sd["b_dec_out"].to(device, dtype)


class _TCLin(torch.autograd.Function):
    """VALUE = the real MLP output (detached); GRADIENT = J_TC = W_enc diag(1[pre>0]) W_dec at x_n."""
    CHUNK = 2048

    @staticmethod
    def forward(ctx, xn, m_real, W_enc, b_enc, b_dec, W_dec):
        flat = xn.reshape(-1, xn.shape[-1])
        masks = [(((flat[s:s + _TCLin.CHUNK] - b_dec) @ W_enc + b_enc) > 0)
                 for s in range(0, flat.shape[0], _TCLin.CHUNK)]
        ctx.save_for_backward(torch.cat(masks), W_enc, W_dec)
        ctx.shape = xn.shape
        return m_real.clone()

    @staticmethod
    def backward(ctx, go):
        mask, W_enc, W_dec = ctx.saved_tensors
        g = go.reshape(-1, go.shape[-1])
        out = torch.cat([((g[s:s + _TCLin.CHUNK] @ W_dec.T) * mask[s:s + _TCLin.CHUNK]) @ W_enc.T
                         for s in range(0, g.shape[0], _TCLin.CHUNK)])
        return out.reshape(ctx.shape), None, None, None, None, None


REAL = {"ln": "live", "attn": "live", "mlp": "real"}
GRAPH = {"ln": "frozen", "attn": "frozen", "mlp": "tc"}


class GPT2:
    def __init__(self, device=DEV, dtype=torch.float32, tc_layers=range(8, 12)):
        from safetensors.torch import load_file
        sd = load_file(glob.glob(f"{HF_SNAP}/*/model.safetensors")[0], device=str(device))
        g = lambda k: sd[k].to(device, dtype)
        self.device = device
        self.wte, self.wpe = g("wte.weight"), g("wpe.weight")
        self.blocks = [{k: g(f"h.{i}." + k) for k in [
            "ln_1.weight", "ln_1.bias", "attn.c_attn.weight", "attn.c_attn.bias", "attn.c_proj.weight",
            "attn.c_proj.bias", "ln_2.weight", "ln_2.bias", "mlp.c_fc.weight", "mlp.c_fc.bias",
            "mlp.c_proj.weight", "mlp.c_proj.bias"]} for i in range(12)]
        self.lnf_w, self.lnf_b = g("ln_f.weight"), g("ln_f.bias")
        self.tcs = {L: Transcoder(L, device, dtype) for L in tc_layers}

    # ---------------------------------------------------------------- one norm
    def _norm(self, x, mode, nid, ctl):
        """x [B,T,768] -> x_hat (no gain/bias). ctl: dict with 'u' {nid: [768]} and per-call captures."""
        if nid in ctl.get("live_norms", ()):  # per-norm override (test_predict.py: one norm live)
            mode = "live"
        mu = x.mean(-1, keepdim=True)
        xc = x - mu
        sd = (xc.pow(2).mean(-1, keepdim=True) + EPS).sqrt()
        if mode == "live":
            out = xc / sd
        else:
            out = xc / sd.detach()
            if mode in ("fixed", "token"):
                if mode == "fixed":
                    u = ctl["u"][nid][None, None, :]
                else:
                    u = (xc / xc.norm(dim=-1, keepdim=True)).detach()
                a = (out * u).sum(-1, keepdim=True)
                out = out - (a - a.detach()) * u  # value unchanged; Jacobian gains -(1/sigma) u u^T
                # ('fixed' u may carry a scale: u = sqrt(lambda) u_unit gives the term lambda u_unit u_unit^T)
        cap = ctl.get("cap")
        if cap is not None and nid in ctl.get("cap_norms", ()):
            r, p = ctl["read"]
            cap[f"xhat{nid}"] = out[r, p].detach() if mode == "live" else (xc / sd)[r, p].detach()
            cap[f"sigma{nid}"] = sd[r, p, 0].detach()
        tr = ctl.get("track")
        if tr is not None and mode != "live":  # norms of the frozen graph (layers > L and ln_f)
            r, p = ctl["read"]
            tr[nid] = {"out": out, "xin": x, "rp": (r, p), "xhat": (xc / sd)[r, p].detach(),
                       "sigma": sd[r, p, 0].detach()}
        return out

    def _attn(self, b, h, frozen):
        B, T, D = h.shape
        q, k, v = (h @ b["attn.c_attn.weight"] + b["attn.c_attn.bias"]).split(D, dim=-1)
        sh = lambda z: z.view(B, T, 12, 64).transpose(1, 2)
        q, k, v = sh(q), sh(k), sh(v)
        s = (q @ k.transpose(-1, -2)) / 8.0
        mask = torch.ones(T, T, dtype=torch.bool, device=h.device).tril()
        pat = torch.softmax(s.masked_fill(~mask, float("-inf")), dim=-1)
        if frozen:
            pat = pat.detach()
        z = (pat @ v).transpose(1, 2).reshape(B, T, D)
        return z @ b["attn.c_proj.weight"] + b["attn.c_proj.bias"]

    @staticmethod
    def _mlp(b, xn):
        h = gelu_new((xn * b["ln_2.weight"] + b["ln_2.bias"]) @ b["mlp.c_fc.weight"] + b["mlp.c_fc.bias"])
        return h @ b["mlp.c_proj.weight"] + b["mlp.c_proj.bias"]

    # ---------------------------------------------------------------- forward
    def forward(self, ids, L, cfg, rows, pos, delta=None, ctl=None):
        """Logits at (rows, pos) [B,V]. cfg parts apply to layers l > L and ln_f; layers <= L are real.
        delta [B,768] is added to mlp_out_L at (rows, pos)."""
        ctl = ctl or {}
        ctl["read"] = (rows, pos)
        T = ids.shape[1]
        x = self.wte[ids] + self.wpe[torch.arange(T, device=ids.device)]
        for l in range(12):
            b = self.blocks[l]
            if ctl.get("resid") is not None:
                ctl["resid"][l] = x.detach()  # residual stream entering block l
            down = l > L
            c = cfg if down else REAL
            h = self._norm(x, c["ln"], (l, "ln1"), ctl) * b["ln_1.weight"] + b["ln_1.bias"]
            x = x + self._attn(b, h, c["attn"] == "frozen")
            xn = self._norm(x, c["ln"], (l, "ln2"), ctl)
            if down and c["mlp"] == "tc":
                t = self.tcs[l]
                m = _TCLin.apply(xn, self._mlp(b, xn).detach(), t.W_enc, t.b_enc, t.b_dec, t.W_dec)
            else:
                m = self._mlp(b, xn)
            if l == L and delta is not None:
                m = m.index_put((rows, pos), m[rows, pos] + delta)
            x = x + m
        if ctl.get("resid") is not None:
            ctl["resid"][12] = x.detach()  # final residual (ln_f input), all positions
        xr = x[rows, pos][:, None, :]
        h = self._norm_read(xr, cfg["ln"], ctl) * self.lnf_w + self.lnf_b
        return h[:, 0] @ self.wte.T

    def _norm_read(self, xr, mode, ctl):
        sub = dict(ctl)
        B = xr.shape[0]
        sub["read"] = (torch.arange(B, device=xr.device), torch.zeros(B, dtype=torch.long, device=xr.device))
        return self._norm(xr, mode, (12, "lnf"), sub)

    def readout(self, ids, L, cfg, pos, tok, u=None, cap_norms=(), live_norms=(), target="centred"):
        """g [B,768] = d y / d mlp_out_L[pos], y = centred logit of tok at pos. Optional captures."""
        B = ids.shape[0]
        rows = torch.arange(B, device=ids.device)
        delta = torch.zeros(B, 768, device=ids.device, dtype=self.wte.dtype, requires_grad=True)
        cap = {} if cap_norms else None
        ctl = {"u": u or {}, "cap": cap, "cap_norms": set(cap_norms), "live_norms": set(live_norms)}
        with torch.enable_grad():
            lg = self.forward(ids, L, cfg, rows, pos, delta, ctl)
            if target == "centred":
                y = lg.gather(-1, tok[:, None])[:, 0] - lg.mean(-1)
            else:  # 'logprob' reading cell (added after the PREREG, Vera #15164)
                assert target == "logprob"
                y = torch.log_softmax(lg, -1).gather(-1, tok[:, None])[:, 0]
            g, = torch.autograd.grad(y.sum(), delta)
        return g.detach(), y.detach(), cap

    def readout_first_order(self, ids, L, cfg, pos, tok):
        """Self-diagnosis from the frozen graph only (no live or real pass).
        Pass 1: frozen backward; g_k = d y / d x_hat_k at every frozen norm k (position t).
        Dropped radial term of norm k (exact for that norm alone): e_k = -(1/sigma_k) x_hat_k (x_hat_k . g_k) / d.
        Pass 2: frozen backward of y + sum_k <e_k, x_k> -> g_fo = g_frozen + sum_k J_frozen(k<-L)^T e_k
        (first order in the dropped terms: no cross terms between norms).
        Returns g_frozen, g_fo, per-norm dict {nid: (coef a_k = x_hat_k.g_k/d, |e_k|, |P g_k|/sigma_k)}."""
        assert cfg["ln"] == "frozen"
        B = ids.shape[0]
        rows = torch.arange(B, device=ids.device)
        with torch.enable_grad():
            delta = torch.zeros(B, 768, device=ids.device, dtype=self.wte.dtype, requires_grad=True)
            tr = {}
            lg = self.forward(ids, L, cfg, rows, pos, delta, {"track": tr})
            y = lg.gather(-1, tok[:, None])[:, 0] - lg.mean(-1)
            nids = list(tr)
            grads = torch.autograd.grad(y.sum(), [delta] + [tr[n]["out"] for n in nids])
            g_frozen = grads[0].detach()
            es, stats = {}, {}
            for n, gk_full in zip(nids, grads[1:]):
                r, p = tr[n]["rp"]
                gk = gk_full[r, p].detach()
                xh, sg = tr[n]["xhat"], tr[n]["sigma"]
                a = (xh * gk).sum(-1) / 768.0
                es[n] = -(a / sg)[:, None] * xh
                gp = gk - gk.mean(-1, keepdim=True)
                stats[n] = (a, es[n].norm(dim=-1), gp.norm(dim=-1) / sg)
            delta2 = torch.zeros(B, 768, device=ids.device, dtype=self.wte.dtype, requires_grad=True)
            tr2 = {}
            lg2 = self.forward(ids, L, cfg, rows, pos, delta2, {"track": tr2})
            y2 = lg2.gather(-1, tok[:, None])[:, 0] - lg2.mean(-1)
            extra = sum((es[n] * tr2[n]["xin"][tr2[n]["rp"]]).sum() for n in nids)
            g_fo, = torch.autograd.grad(y2.sum() + extra, delta2)
        return g_frozen, g_fo.detach(), stats
