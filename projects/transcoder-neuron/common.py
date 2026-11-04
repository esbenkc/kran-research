"""Shared engine: a hand-written GPT-2 small forward (from the HF weights) with the hooks the two
steering arms need, and the public GPT-2 transcoders of Dunefsky, Chlenski & Nanda (2406.11944,
HF repo pchlenski/gpt2-transcoders).

Conventions (checked in gate.py):
  x_n(L)   = blocks.L.ln2.hook_normalized = (resid_mid - mean) / sqrt(var + 1e-5), no gain/bias.
  MLP_L(x_n) = gelu_new((x_n * g2 + b2) @ W_fc + b_fc) @ W_proj + b_proj   (HF mlp output).
  TC_L(x_n)  = relu((x_n - b_dec) @ W_enc + b_enc) @ W_dec + b_dec_out     (~ centred mlp_out).
  Adding c * ones(768) to the residual stream changes nothing downstream (every read goes through a
  LayerNorm, which removes the mean), so a centred TC output can stand in for the HF MLP output.

forward() options
  mlp_in_delta  = {L: (rows, pos, du)}   add du[i] to x_n(L) of row rows[i] at position pos[i]
                                         (only MLP_L sees it; the residual stream is untouched)
  mlp_out_delta = {L: (rows, pos, dv)}   add dv[i] to the output of MLP_L at (rows[i], pos[i])
  linearise     = L0                     circuit-tracer style linearisation for every layer l > L0:
                                         attention patterns and LN scales detached, and
                                         MLP_l = TC_l(x_n) + (MLP_l(x_n) - TC_l(x_n)).detach()
                                         (frozen error node), so the forward VALUE is the real model
                                         and gradients run only through the downstream TC features.
  capture       = set of names           returns tensors: 'xn{L}', 'mlp{L}', 'resid_mid{L}'
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
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def gelu_new(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


def gelu_new_grad(x):
    """d gelu_new / dx (analytic)."""
    c = math.sqrt(2.0 / math.pi)
    u = c * (x + 0.044715 * x.pow(3))
    th = torch.tanh(u)
    return 0.5 * (1.0 + th) + 0.5 * x * (1.0 - th * th) * c * (1.0 + 3 * 0.044715 * x * x)


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


def tc_path(L: int) -> str:
    return glob.glob(f"{TC_SNAP}/*/final_sparse_autoencoder_gpt2-small_blocks.{L}.ln2.hook_normalized_24576.pt")[0]


class Transcoder:
    def __init__(self, L: int, device=DEV, dtype=torch.float32):
        d = torch.load(tc_path(L), map_location="cpu", weights_only=False, pickle_module=_PM, mmap=True)
        sd = d["state_dict"]
        assert d["cfg"].hook_point == f"blocks.{L}.ln2.hook_normalized"
        assert d["cfg"].out_hook_point == f"blocks.{L}.hook_mlp_out"
        self.L = L
        self.W_enc = sd["W_enc"].to(device, dtype)  # [768, F]
        self.b_enc = sd["b_enc"].to(device, dtype)  # [F]
        self.W_dec = sd["W_dec"].to(device, dtype)  # [F, 768]
        self.b_dec = sd["b_dec"].to(device, dtype)  # [768] (input offset)
        self.b_dec_out = sd["b_dec_out"].to(device, dtype)  # [768]

    def pre(self, xn):
        return (xn - self.b_dec) @ self.W_enc + self.b_enc

    def acts(self, xn):
        return torch.relu(self.pre(xn))

    def __call__(self, xn):
        return self.acts(xn) @ self.W_dec + self.b_dec_out


class GPT2:
    def __init__(self, device=DEV, dtype=torch.float32):
        from safetensors.torch import load_file

        path = glob.glob(f"{HF_SNAP}/*/model.safetensors")[0]
        sd = load_file(path, device=str(device))  # straight to the device: no CPU copy in the footprint
        g = lambda k: sd[k].to(device, dtype)
        self.device, self.dtype = device, dtype
        self.wte, self.wpe = g("wte.weight"), g("wpe.weight")
        self.blocks = []
        for i in range(12):
            p = f"h.{i}."
            self.blocks.append({k: g(p + k) for k in [
                "ln_1.weight", "ln_1.bias", "attn.c_attn.weight", "attn.c_attn.bias", "attn.c_proj.weight",
                "attn.c_proj.bias", "ln_2.weight", "ln_2.bias", "mlp.c_fc.weight", "mlp.c_fc.bias",
                "mlp.c_proj.weight", "mlp.c_proj.bias"]})
        self.lnf_w, self.lnf_b = g("ln_f.weight"), g("ln_f.bias")
        self.tcs: dict[int, Transcoder] = {}

    def tc(self, L):
        if L not in self.tcs:
            self.tcs[L] = Transcoder(L, self.device, self.dtype)
        return self.tcs[L]

    @staticmethod
    def _norm(x, freeze):
        mu = x.mean(-1, keepdim=True)
        xc = x - mu
        sd = (xc.pow(2).mean(-1, keepdim=True) + EPS).sqrt()
        if freeze:
            sd = sd.detach()
        return xc / sd

    def mlp(self, b, xn):
        h = gelu_new((xn * b["ln_2.weight"] + b["ln_2.bias"]) @ b["mlp.c_fc.weight"] + b["mlp.c_fc.bias"])
        return h @ b["mlp.c_proj.weight"] + b["mlp.c_proj.bias"]

    def attn(self, b, x, freeze):
        B, T, D = x.shape
        h = self._norm(x, freeze) * b["ln_1.weight"] + b["ln_1.bias"]
        qkv = h @ b["attn.c_attn.weight"] + b["attn.c_attn.bias"]
        q, k, v = qkv.split(D, dim=-1)
        sh = lambda z: z.view(B, T, 12, 64).transpose(1, 2)
        q, k, v = sh(q), sh(k), sh(v)
        s = (q @ k.transpose(-1, -2)) / 8.0
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        s = s.masked_fill(~mask, float("-inf"))
        pat = torch.softmax(s, dim=-1)
        if freeze:
            pat = pat.detach()
        z = (pat @ v).transpose(1, 2).reshape(B, T, D)
        return z @ b["attn.c_proj.weight"] + b["attn.c_proj.bias"]

    def forward(self, ids, mlp_in_delta=None, mlp_out_delta=None, linearise=None, capture=(), start=None,
                stop_after=None, read_pos=None):
        """ids [B, T] long. start=(L, resid_pre_L) resumes from a cached residual at block L.
        stop_after=L returns (None, cap) right after x_n(L) is computed (cap gets 'resid_pre{L}' too).
        read_pos=(rows, pos) returns logits only at those positions ([len(rows), V]); saves the unembed."""
        mlp_in_delta = mlp_in_delta or {}
        mlp_out_delta = mlp_out_delta or {}
        cap = {}
        if start is None:
            T = ids.shape[1]
            x = self.wte[ids] + self.wpe[torch.arange(T, device=ids.device)]
            L_from = 0
        else:
            L_from, x = start
        for L in range(L_from, 12):
            b = self.blocks[L]
            if stop_after == L:
                cap[f"resid_pre{L}"] = x
            fz = linearise is not None and L > linearise
            x = x + self.attn(b, x, fz)
            if f"resid_mid{L}" in capture:
                cap[f"resid_mid{L}"] = x
            xn = self._norm(x, fz)
            if L in mlp_in_delta:
                rows, pos, du = mlp_in_delta[L]
                xn = xn.index_put((rows, pos), xn[rows, pos] + du)
            if f"xn{L}" in capture:
                cap[f"xn{L}"] = xn
            if stop_after == L:
                cap[f"resid_mid{L}"] = x
                return None, cap
            if fz:
                tcl = self.tc(L)
                m = _TCLin.apply(xn, self.mlp(b, xn).detach(), tcl.W_enc, tcl.b_enc, tcl.b_dec, tcl.W_dec)
            else:
                m = self.mlp(b, xn)
            if L in mlp_out_delta:
                rows, pos, dv = mlp_out_delta[L]
                m = m.index_put((rows, pos), m[rows, pos] + dv)
            if f"mlp{L}" in capture:
                cap[f"mlp{L}"] = m
            x = x + m
        fz = linearise is not None
        if read_pos is not None:
            x = x[read_pos[0], read_pos[1]]
        h = self._norm(x, fz) * self.lnf_w + self.lnf_b
        logits = h @ self.wte.T
        return logits, cap


class _TCLin(torch.autograd.Function):
    """Frozen-error transcoder node: VALUE = the real MLP output (passed in, detached); GRADIENT = the
    transcoder's Jacobian at x_n, J_TC = W_enc diag(1[pre > 0]) W_dec, i.e. the same as
    TC(x_n) + (MLP(x_n) - TC(x_n)).detach() but without storing the [tokens, 24576] activations
    (only a bool mask), computed in token chunks."""

    CHUNK = 2048

    @staticmethod
    def forward(ctx, xn, m_real, W_enc, b_enc, b_dec, W_dec):
        flat = xn.reshape(-1, xn.shape[-1])
        masks = [(((flat[s:s + _TCLin.CHUNK] - b_dec) @ W_enc + b_enc) > 0) for s in range(0, flat.shape[0], _TCLin.CHUNK)]
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


def centred(logits, tok):
    """centred logit of token tok: logit[tok] - mean(logits) (the logit node of circuit-tracer)."""
    return logits.gather(-1, tok.unsqueeze(-1)).squeeze(-1) - logits.mean(-1)


def neuron_cosines(model: GPT2, L: int, chunk: int = 4096):
    """Per feature: best signed cosine of the encoder vector to a folded+centred W_in column and of the
    decoder vector to a centred W_out row, and the best SAME-neuron score min(enc_cos, dec_cos).
    Folding: x_n -> neuron preact is x_n @ (g2[:,None] * W_fc) + const, so the neuron's input direction in
    x_n space is g2 * W_fc[:, j], centred over d_model (x_n has zero mean, so the mean part is invisible)."""
    b = model.blocks[L]
    tc = model.tc(L)
    Win = b["ln_2.weight"][:, None] * b["mlp.c_fc.weight"]  # [768, 3072]
    Win = Win - Win.mean(0, keepdim=True)
    Wout = b["mlp.c_proj.weight"]  # [3072, 768]
    Wout = Wout - Wout.mean(1, keepdim=True)
    Win = Win / Win.norm(dim=0, keepdim=True)
    Wout = Wout / Wout.norm(dim=1, keepdim=True)
    E = tc.W_enc - tc.W_enc.mean(0, keepdim=True)
    E = E / E.norm(dim=0, keepdim=True)  # [768, F]
    Dd = tc.W_dec - tc.W_dec.mean(1, keepdim=True)
    Dd = Dd / Dd.norm(dim=1, keepdim=True)  # [F, 768]
    out = {k: [] for k in ["enc_cos", "enc_idx", "dec_cos", "dec_idx", "same_cos", "same_idx"]}
    for s in range(0, E.shape[1], chunk):
        ce = E[:, s:s + chunk].T @ Win  # [c, 3072]
        cd = Dd[s:s + chunk] @ Wout.T  # [c, 3072]
        v, i = ce.max(1); out["enc_cos"].append(v); out["enc_idx"].append(i)
        v, i = cd.max(1); out["dec_cos"].append(v); out["dec_idx"].append(i)
        v, i = torch.minimum(ce, cd).max(1); out["same_cos"].append(v); out["same_idx"].append(i)
    return {k: torch.cat(v).cpu() for k, v in out.items()}
