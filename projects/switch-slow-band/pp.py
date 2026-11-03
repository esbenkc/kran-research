"""Shared runner for switch-slow-band: loads one saved copy into the cached 2-block tail, its condition-on/off
inputs (200 test prompts), the donor K/V (one capture pass on the condition-off input), and gives px(spec).

P = mean first-token probability mass on the X tokens; share moved = (P_patched - P_on) / (P_off - P_on).
Heads in every patch arm = all 16 query heads of blocks 26 and 27 (the same scope as sink-value's K row, 0.985),
unless an arm names a smaller set. Key positions = tokens 0, 1, 2 (the prereg's 'tokens 0-2').
"""
import os

import numpy as np
import torch

import common as C
import patch as P

T = C.T
ALL16 = list(range(16))
F3 = [0, 1, 2]
NPAIR = 64


class Runner:
    def __init__(self, prec="fp32", bs=50):
        self.prec, self.bs = prec, bs
        cfg, tail, w_out = T.load_tail()
        self.cfg, self.tail = cfg, tail
        for p_ in tail.parameters():
            p_.requires_grad_(False)
        self.base_sd = {k: v.clone() for k, v in tail.state_dict().items()}
        P.install(tail)
        self.w_out = w_out.float() if prec == "fp32" else w_out
        self.xs = torch.tensor(T.x_ids()).to(T.DEV)
        self.copy = None
        self.check_rope()

    def check_rope(self):
        """Assert the pair map: rotate_half -> cos[:, j] == cos[:, j+64]; inv_freq[j] = 1e6**(-j/64) up to bf16
        rounding (load_tail casts the rotary buffer to bf16, as in every earlier run of these copies), strictly
        decreasing in j (pair 63 = slowest)."""
        rot = self.tail.rotary
        inv = rot.inv_freq.detach().float().cpu().double().numpy()
        want = 1e6 ** (-np.arange(64) / 64.0)
        assert inv.shape == (64,) and np.allclose(inv, want, rtol=8e-3), (inv[:3], want[:3])
        assert np.all(np.diff(inv) < 0)
        x = torch.zeros(1, 34, 1024, device=T.DEV)
        cos, sin = rot(x, torch.arange(34, device=T.DEV)[None])
        cos = cos[0].float().cpu().double()
        assert torch.equal(cos[:, :64], cos[:, 64:]), "pair j must be dims (j, j+64)"
        assert torch.all(cos[0] == 1), "RoPE at position 0 is the identity"
        self.inv_freq = inv

    def load(self, name, base=False):
        sw, path = C.COPIES[name]
        self.tail.load_state_dict(self.base_sd)
        if not base:
            miss, unexp = self.tail.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
            assert not unexp and all("rotary" in m for m in miss), (miss, unexp)
        self.tail.float() if self.prec == "fp32" else self.tail.to(torch.bfloat16)
        X = C.load_inputs(sw)
        self.on, self.off = X["on"], X["off"]
        self.copy, self.sw = name, sw
        self.p_on, self.p_off = self.px(self.on), self.px(self.off)
        self.donors = self.capture(self.off)
        self.P_on, self.P_off = float(self.p_on.mean()), float(self.p_off.mean())
        return self

    def batches(self, u16):
        for s in range(0, len(u16), self.bs):
            h = T.bf16(u16[s:s + self.bs]).to(T.DEV)
            yield s // self.bs, (h.float() if self.prec == "fp32" else h)

    @torch.no_grad()
    def px(self, u16, spec=None):
        out = []
        for bi, h in self.batches(u16):
            if spec is None:
                P.CTRL.update(mode="none", spec=None, donor=None)
            else:
                P.CTRL.update(mode="patch", spec=spec, donor=self.donors[bi])
            out.append(torch.softmax(self.tail(h, self.w_out).float(), -1)[:, self.xs].sum(-1).cpu())
        P.CTRL.update(mode="none", spec=None, donor=None)
        return torch.cat(out).double().numpy()

    @torch.no_grad()
    def capture(self, u16):
        d = []
        for _, h in self.batches(u16):
            P.CTRL.update(mode="capture")
            self.tail(h, self.w_out)
            d.append({i: P.CAP[i] for i in range(2)})
        P.CTRL.update(mode="none")
        return d

    def share(self, p):
        return (float(p.mean()) - self.P_on) / (self.P_off - self.P_on)

    def grad_alpha(self, heads26=ALL16, heads27=ALL16, kpos=F3):
        """d P / d alpha_j at alpha = 0 (cue-on run), alpha_j scales the cue-off minus cue-on key change in pair j
        on the given heads and key positions. Returns [64] numpy (mean over prompts)."""
        g = np.zeros(NPAIR)
        n = len(self.on)
        for bi, h in self.batches(self.on):
            alpha = torch.zeros(NPAIR, device=T.DEV, dtype=torch.float32, requires_grad=True)
            spec = P.arm(heads26, heads27, kpos=kpos, alpha=alpha)
            P.CTRL.update(mode="patch", spec=spec, donor=self.donors[bi])
            with torch.enable_grad():
                p = torch.softmax(self.tail(h, self.w_out).float(), -1)[:, self.xs].sum(-1)
                p.sum().backward()
            g += alpha.grad.double().cpu().numpy()
        P.CTRL.update(mode="none", spec=None, donor=None)
        return g / n


def pair_arm(pairs, heads26=ALL16, heads27=ALL16, kpos=F3):
    return P.arm(heads26, heads27, kpos=kpos, kdims=P.pair_dims(pairs))
