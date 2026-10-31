"""All arms of PREREG v1 as additive edits on the down_proj output of layer 13.

Per token t, h(t) = clean activation of the gate neuron (input 3270 of down_proj, captured before the edit),
and every arm adds coef(t) * vec (optionally only at some positions). A = B + C exactly (asserted).
"""
import contextlib

import numpy as np
import torch

from common import DEV, down

L, GATE, PIN = 13, 3270, 30.0
M = [1793, 1999, 1401]  # massive dims at L13 output (phase0.json)
MED_ABS = 1.5546875  # median |h| over all real tokens at L13 output, CHAN (phase0.json layers[13])


OVERRIDE = {}  # D9 tail path: {"first": tensor of first real positions}


class Ctx:
    """Captures, per forward, the first real position (prefill) and the massive-position mask at the input of L13."""

    def __init__(self, model):
        self.model = model
        self.first = None
        self.big = None
        self.hs = [model.model.register_forward_pre_hook(self._top, with_kwargs=True),
                   model.model.layers[L].register_forward_pre_hook(self._l13)]

    def _top(self, m, args, kwargs):
        am = kwargs.get("attention_mask")
        ids = kwargs.get("input_ids")
        T = ids.shape[1] if ids is not None else kwargs["inputs_embeds"].shape[1]
        if T > 1 and am is not None:
            self.first = am[:, -T:].float().argmax(1)  # prefill: first real token per row
        else:
            self.first = None  # decode step: never position 0

    def _l13(self, m, args):
        h = args[0]
        self.big = (h[..., M].float().abs().max(-1).values > 100 * MED_ABS)

    def close(self):
        for h in self.hs:
            h.remove()


def pos_mask(ctx, a, kind):
    """1 where the edit is active. kind: None (all), 'skip0', 'only0', 'skipM'."""
    if kind is None:
        return None
    B, T = a.shape
    if kind == "skipM":
        return (~ctx.big).float()
    is0 = torch.zeros(B, T, device=a.device)
    first = ctx.first if ctx.first is not None else OVERRIDE.get("first")
    if first is not None:
        is0[torch.arange(B), first.long()] = 1.0
    return 1.0 - is0 if kind == "skip0" else is0


@contextlib.contextmanager
def edit(model, coef_fn, vec, kind=None):
    """Add coef_fn(h) * vec at every token (masked by kind). vec: [d] tensor on DEV (float32)."""
    mod = down(model, L)
    ctx = Ctx(model)
    store = {}

    def pre(m, args):
        store["a"] = args[0][..., GATE].float()

    def post(m, args, out):
        a = store["a"]
        c = coef_fn(a)
        pm = pos_mask(ctx, a, kind)
        if pm is not None:
            c = c * pm
        return (out.float() + c[..., None] * vec[None, None, :]).to(out.dtype)

    h1 = mod.register_forward_pre_hook(pre)
    h2 = mod.register_forward_hook(post)
    try:
        yield
    finally:
        h1.remove()
        h2.remove()
        ctx.close()


def build(model, hm_value):
    """Returns dict name -> (coef_fn, vec, kind) for the fixed arms, plus random-arm factories."""
    W = down(model, L).weight.float()  # [d, 6144]
    d = W.shape[0]
    w = W[:, GATE].clone()
    wM = torch.zeros_like(w)
    wM[M] = w[M]
    wR = w - wM
    assert torch.equal(wM + wR, w)
    nw, nwM = w.norm(), wM.norm()
    pin = lambda a: PIN - a
    fixed = {
        "A": (pin, w, None),
        "A_hm": (lambda a: hm_value - a, w, None),
        "A_zero": (lambda a: -a, w, None),
        "B": (pin, wR, None),
        "C": (pin, wM, None),
        "A_skip0": (pin, w, "skip0"),
        "A_tok0": (pin, w, "only0"),
    }
    rng = np.random.default_rng(0)
    others = np.array([j for j in range(W.shape[1]) if j != GATE])
    J = rng.choice(others, 100, replace=False)  # random L13 neurons, shared by D', E', F, D_small
    nonM = np.array([c for c in range(d) if c not in M])
    # D7: natural massive-channel neuron 5441 (70.4% energy in M), pinned at the gate's per-token edit norm
    w5 = W[:, 5441] / W[:, 5441].norm()
    fixed["N5441pos"] = (lambda a: (PIN - a).abs() * nw, w5, None)
    fixed["N5441neg"] = (lambda a: -(PIN - a).abs() * nw, w5, None)
    rand, Ss = {}, []
    for k, j in enumerate(J):
        wj = W[:, j]
        uM = torch.zeros_like(wj)
        uM[M] = wj[M]
        uM = uM / uM.norm()
        S = rng.choice(nonM, 3, replace=False)
        Ss.append(S.tolist())
        uS = torch.zeros_like(wj)
        uS[S] = wj[S]
        uS = uS / uS.norm()
        rand[f"Dp_{k}"] = (lambda a: (PIN - a) * nw, uM, None)
        rand[f"Ds_{k}"] = (lambda a: (PIN - a) * nwM, uM, None)
        rand[f"Ep_{k}"] = (lambda a: (PIN - a) * nw, uS, None)
        rand[f"Fpos_{k}"] = (lambda a: (PIN - a).abs() * nw, wj / wj.norm(), None)
        rand[f"Fneg_{k}"] = (lambda a: -(PIN - a).abs() * nw, wj / wj.norm(), None)
    g = torch.Generator().manual_seed(0)
    for k in range(20):
        u = torch.randn(d, generator=g)
        rand[f"Fp_{k}"] = (lambda a: (PIN - a) * nw, (u / u.norm()).to(DEV), None)
    meta = {"J": J.tolist(), "norm_w": nw.item(), "norm_wM": nwM.item(), "share_energy": (nwM ** 2 / nw ** 2).item(),
            "E_dims": Ss}
    return fixed, rand, meta
