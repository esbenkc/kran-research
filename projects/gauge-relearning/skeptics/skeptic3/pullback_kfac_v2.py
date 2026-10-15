"""skeptic3: 1-step and N-step pull-back invariance of build's CURRENT KFAC (kappa trust region, fp32 inverse)
on the real unlearned_seed0.pt. Also tests the same class with a float64 inverse patch.
Run from kran-research/: uv run python ../swarm-oracle-1015-gauge-tamper-resist/skeptic3/pullback_kfac_v2.py
"""
import importlib.util
import json
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

KAPPA = float(os.environ.get("KAPPA", 0.01))
STEPS = int(os.environ.get("STEPS", 3))


def inv64(self, M, blocked):
    M = M.double().cpu()
    d = M.diagonal()
    if blocked:
        bm = d.view(-1, self.BLOCK).mean(1)
        lam = torch.where(bm > 0, self.damp * bm, torch.ones_like(bm)).repeat_interleave(self.BLOCK)
    else:
        lam = self.damp * d.mean().expand_as(d)
    return torch.linalg.inv(M + torch.diag(lam)).float().to(exp.DEV)


def run(c, bA, state, steps, f64):
    m = exp.load_model()
    m.load_state_dict(state)
    exp.apply_gauge(m, c)
    exp.trainable(m)
    opt = exp.KFAC(m, KAPPA, bA)
    if f64:
        opt._inv = inv64.__get__(opt)
    losses = []
    g = torch.Generator().manual_seed(5)
    for _ in range(steps):
        idx = torch.randperm(bA.n, generator=g)[:10]
        sub = exp.Batch.__new__(exp.Batch)
        sub.ids, sub.mask, sub.last, sub.y, sub.n = bA.ids[idx], bA.mask[idx], bA.last[idx], bA.y[idx], 10
        loss, _ = exp.loss_acc(m, sub)
        losses.append(round(loss.item(), 4))
        m.zero_grad(set_to_none=True); loss.backward(); opt.step()
    loss, acc = exp.loss_acc(m, bA)
    losses.append(round(loss.item(), 4))
    exp.apply_gauge(m, 1.0 / c)
    return {k: v.detach().clone() for k, v in m.state_dict().items()}, losses


def main():
    tok = AutoTokenizer.from_pretrained(exp.MODEL)
    facts = json.loads((ROOT / "results" / "raw" / "seed0" / "facts.json").read_text())
    bA = exp.Batch([f for f in facts if f["split"] == "A"], tok)
    state = torch.load(ROOT / "results" / "ckpt" / "unlearned_seed0.pt", map_location="cpu")
    for f64 in (False, True):
        ref, rl = run(1.0, bA, state, STEPS, f64)
        for c in (1 / 16, 16.0):
            st, ls = run(c, bA, state, STEPS, f64)
            err = max(((st[k] - ref[k]).norm() / (ref[k] - state[k]).norm().clamp_min(1e-12)).item()
                      for k in ref if "query_key_value" in k)
            print(f"inv {'f64' if f64 else 'f32 (build)'} kappa {KAPPA} steps {STEPS} c={c:g}: rel qkv err {err:.2e} "
                  f"loss c=1 {rl} | c {ls}", flush=True)


if __name__ == "__main__":
    main()
