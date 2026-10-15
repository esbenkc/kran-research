"""skeptic3: does an equivariant K-FAC with a damping floor (per-head-block trace damping on G of qkv,
isotropic trace damping elsewhere) descend on the UNLEARNED seed0 checkpoint, and is it gauge-invariant?
Compares against build's KFAC (Marquardt damping). CPU. Run from kran-research/.
"""
import importlib.util
import math
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2] / "kran-research" / "projects" / "gauge-relearning"
spec = importlib.util.spec_from_file_location("exp", ROOT / "exp.py")
exp = importlib.util.module_from_spec(spec)
sys.modules["exp"] = exp
spec.loader.exec_module(exp)
exp.DEV = torch.device("cpu")
from transformers import AutoTokenizer  # noqa: E402

STEPS = int(os.environ.get("STEPS", 15))


class BlockKFAC(exp.KFAC):
    def __init__(self, model, lr, damping=1e-2, kl_clip=None, **kw):
        super().__init__(model, lr, damping=damping, kl_clip=kl_clip or 1.0, **kw)
        H = model.config.num_attention_heads
        self.block = model.config.hidden_size // H  # q_h, k_h, v_h rows are contiguous blocks of d
        self.qkv = {i for i, lin in enumerate(self.lins) if lin.out_features == 3 * model.config.hidden_size}
        self.use_clip = kl_clip is not None

    def _damped_inv(self, M, blocked):
        M = M.double().cpu()
        n = M.shape[0]
        if blocked:
            d = self.block
            tr = M.diagonal().view(n // d, d).mean(1)  # per-block mean eigenvalue
            M = M + torch.diag((self.damp * tr).repeat_interleave(d) + 1e-30)
        else:
            M = M + (self.damp * M.diagonal().mean() + 1e-30) * torch.eye(n, dtype=M.dtype)
        return torch.linalg.inv(M).float()

    @torch.no_grad()
    def step(self):
        for i in range(len(self.lins)):
            a, g = self._a[i], self._g[i] * self._g[i].shape[0]
            A, G = a.T @ a / a.shape[0], g.T @ g / g.shape[0]
            if self.A[i] is None:
                self.A[i], self.G[i] = A, G
            else:
                self.A[i].mul_(self.ema).add_(A, alpha=1 - self.ema)
                self.G[i].mul_(self.ema).add_(G, alpha=1 - self.ema)
        if self.t % self.every == 0:
            for i in range(len(self.lins)):
                self.Ainv[i] = self._damped_inv(self.A[i], False)
                self.Ginv[i] = self._damped_inv(self.G[i], i in self.qkv)
        self.t += 1
        nat, quad = [], 0.0
        for i, lin in enumerate(self.lins):
            gw = torch.cat([lin.weight.grad, lin.bias.grad[:, None]], 1)
            p = self.Ginv[i] @ gw @ self.Ainv[i]
            quad += (gw * p).sum().item()
            nat.append(p)
        nu = min(1.0, math.sqrt(self.kl / max(self.lr ** 2 * quad, 1e-30))) if self.use_clip else 1.0
        for lin, p in zip(self.lins, nat):
            lin.weight.add_(p[:, :-1], alpha=-self.lr * nu)
            lin.bias.add_(p[:, -1], alpha=-self.lr * nu)


def run(kind, lr, c, bt, state, damp, every=10, kl=None):
    m = exp.load_model()
    m.load_state_dict(state)
    exp.apply_gauge(m, c)
    exp.trainable(m)
    opt = exp.KFAC(m, lr, damping=exp.KFAC_DAMP, kl_clip=exp.KFAC_KL) if kind == "build" else BlockKFAC(m, lr, damping=damp, every=every, kl_clip=kl)
    out = []
    for _ in range(STEPS):
        lt, acc = exp.loss_acc(m, bt)
        out.append((round(lt.item(), 3), round(acc, 2)))
        m.zero_grad(set_to_none=True); lt.backward(); opt.step()
    lt, acc = exp.loss_acc(m, bt)
    out.append((round(lt.item(), 3), round(acc, 2)))
    return out


def main():
    tok = AutoTokenizer.from_pretrained(exp.MODEL)
    facts = exp.make_facts(0, tok)
    bt = exp.Batch([f for f in facts if f["split"] == "A"], tok)
    ck = exp.CKPT / "unlearned_seed0.pt"
    state = torch.load(ck, map_location="cpu") if ck.exists() and not os.environ.get("BASE") else exp.load_model().state_dict()
    print("checkpoint:", ck if ck.exists() and not os.environ.get("BASE") else "BASE pythia (no unlearned ckpt)")
    for lr, every, kl in [(3e-3, 10, None), (1e-3, 10, None), (1e-2, 1, None), (1e-2, 10, 1e-2), (1e-1, 10, 1e-2), (1e-1, 1, 1e-3)]:
        for c in (1.0,):
            path = run("block", lr, c, bt, state, 1.0, every, kl)
            print(f"block lr {lr:<6g} damp 1 every {every} kl {kl} c={c:<4g} (loss, acc): {path}", flush=True)


if __name__ == "__main__":
    main()
