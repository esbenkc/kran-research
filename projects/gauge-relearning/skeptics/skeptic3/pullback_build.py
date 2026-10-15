"""skeptic3: pull-back null test on BUILD's attacker code (exp.py), CPU, 5 steps.

For each attacker, run 5 steps from the base model at gauge c and at c=1 on the same batch, map the c-run
weights back with the inverse gauge, and report max relative weight-trajectory error and the forget-train
loss path. Invariant attacker => error ~ float noise and identical loss path.
Run from kran-research/:  uv run python ../swarm-oracle-1015-gauge-tamper-resist/skeptic3/pullback_build.py
"""
import importlib.util
import sys
from pathlib import Path
import os
import torch

ROOT = Path(__file__).resolve().parents[2] / "kran-research" / "projects" / "gauge-relearning"
spec = importlib.util.spec_from_file_location("exp", ROOT / "exp.py")
exp = importlib.util.module_from_spec(spec)
sys.modules["exp"] = exp
spec.loader.exec_module(exp)
exp.DEV = torch.device("cpu")

from transformers import AutoTokenizer  # noqa: E402
STEPS = int(os.environ.get("STEPS", 5))
LRS = {"sgd": 0.1, "adamw": 1e-4, "renorm": 1e-4, "kfac": float(os.environ.get("KFAC_LR", 0.3))}
ATTACKERS = os.environ.get("ATTACKERS", "sgd,adamw,kfac,renorm").split(",")


def run(attacker, c, bt, base):
    m = exp.load_model()
    m.load_state_dict(base)
    exp.apply_gauge(m, c)
    params = exp.trainable(m)
    if attacker == "renorm":
        exp.renorm(m)
    if attacker == "sgd":
        opt = torch.optim.SGD(params, lr=LRS[attacker])
    elif attacker in ("adamw", "renorm"):
        opt = torch.optim.AdamW(params, lr=LRS[attacker])
    else:
        opt = exp.KFAC(m, LRS[attacker])
    losses = []
    for _ in range(STEPS):
        lt, _ = exp.loss_acc(m, bt)
        losses.append(lt.item())
        if isinstance(opt, exp.KFAC):
            m.zero_grad(set_to_none=True); lt.backward(); opt.step()
        else:
            opt.zero_grad(); lt.backward(); opt.step()
    lt, _ = exp.loss_acc(m, bt)
    losses.append(lt.item())
    if attacker != "renorm":
        exp.apply_gauge(m, 1.0 / c)  # pull back
    return {k: v.detach().clone() for k, v in m.state_dict().items()}, losses


def main():
    tok = AutoTokenizer.from_pretrained(exp.MODEL)
    facts = exp.make_facts(0, tok)
    bt = exp.Batch([f for f in facts if f["split"] == "forget_train"], tok)
    base = {k: v.clone() for k, v in exp.load_model().state_dict().items()}
    for attacker in ATTACKERS:
        ref, ref_l = run(attacker, 1.0, bt, base)
        for c in [1 / 16, 16.0]:
            st, ls = run(attacker, c, bt, base)
            err = max(((st[k] - ref[k]).norm() / (ref[k] - base[k]).norm().clamp_min(1e-12)).item()
                      for k in ref if "query_key_value" in k)
            dl = max(abs(a - b) for a, b in zip(ls, ref_l))
            print(f"{attacker:<7} c={c:<7g} max rel qkv-update err {err:9.2e}  max |dloss| {dl:9.2e}  "
                  f"loss c=1 {[round(x, 3) for x in ref_l]}  loss c {[round(x, 3) for x in ls]}", flush=True)


if __name__ == "__main__":
    main()
