"""Iris: size of the R50mix update at a saturated policy (P(X) ~ 1), measured on a trained R0 adapter.

At an all-X group with mixed flags the common term cancels exactly (sum of advantages = 0), so the update is a pure
differential direction. Its per-weight size scales with 1 - P(X). Question: is it above AdamW eps (1e-8), and what
steady-state gap slope does it give? Uses Bruno's tail.py / rl.py sampling (same code path), on CPU.

    NC_DEV=cpu env/.venv/bin/python swarm-theory-1025-numeric-chameleon/research_checks/mixgrad.py --adapter <path>
"""
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("NC_DEV", "cpu")
PROJ = Path(__file__).resolve().parents[2] / "kran-research/projects/numeric-chameleon-rl"
sys.path.insert(0, str(PROJ))
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from tail import Rows, load_tail, splits, x_ids

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True)
ap.add_argument("--nbatch", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--eps", type=float, default=1e-8)
ap.add_argument("--out", default=str(Path(__file__).with_suffix(".json")))
a = ap.parse_args()
torch.manual_seed(0); rng = np.random.default_rng(0); gen = torch.Generator().manual_seed(0)
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

cdir = PROJ / "raw/cache/base"
meta = json.load(open(cdir / "meta.json"))
idx = splits(meta["prompt_ids"])
Z = {b: Rows(cdir / f"{b}_rec_tail.npy") for b in ("none", "ref")}
xs = torch.tensor(x_ids())
cfg, tail, w_out = load_tail(2)
tail, w_out = tail.to("cpu"), w_out.to("cpu")
for p in tail.parameters():
    p.requires_grad_(False)
pt = get_peft_model(tail, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=TARGETS, bias="none"),
                    autocast_adapter_dtype=True)
sd = torch.load(a.adapter, map_location="cpu")
missing, unexpected = pt.load_state_dict(sd, strict=False)
params = [p for p in pt.parameters() if p.requires_grad]
assert not unexpected, unexpected[:3]
is_x = torch.zeros(w_out.shape[0], dtype=torch.bool); is_x[xs] = True

def flat():
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params])

# exact dev gap and its gradient at this state
pt.zero_grad(set_to_none=True)
dv = idx["dev"]
po = torch.log_softmax(pt(Z["none"][dv], w_out), -1)[:, xs].exp().sum(1)
pn = torch.log_softmax(pt(Z["ref"][dv], w_out), -1)[:, xs].exp().sum(1)
gap = (po - pn).mean(); gap.backward(); g_gap = flat()
res = {"adapter": a.adapter, "dev_poff": float(po.mean()), "dev_pon": float(pn.mean()), "dev_gap": float(gap),
       "dev_one_minus_poff_median": float((1 - po).median()), "norm_grad_gap": float(g_gap.norm())}
print(res, flush=True)

def batch_grad(mix: bool):
    tr = idx["train"]; b = list(rng.choice(tr, 8, replace=False))
    pt.zero_grad(set_to_none=True)
    lp_off = torch.log_softmax(pt(Z["none"][b], w_out), -1)
    lp_on = torch.log_softmax(pt(Z["ref"][b], w_out), -1)
    with torch.no_grad():
        if mix:
            fl = torch.from_numpy(rng.random((8, 8)) < 0.5)
        else:
            fl = torch.from_numpy(np.repeat(rng.random((8, 1)) < 0.5, 8, 1))
        s_off = torch.multinomial(lp_off.exp(), 8, replacement=True, generator=gen)
        s_on = torch.multinomial(lp_on.exp(), 8, replacement=True, generator=gen)
        smp = torch.where(fl, s_on, s_off)
        r = is_x[smp].float() * torch.where(fl, -1.0, 1.0)
        adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)
    lsel = torch.where(fl, lp_on.gather(1, smp), lp_off.gather(1, smp))
    (-(adv * lsel).sum() / 64).backward()
    g = flat(); n = float(g.norm())
    return g * min(1.0, 1.0 / (n + 1e-6)), n

t0 = time.time()
for name, mix in (("R50mix", True), ("R50", False)):
    S = [batch_grad(mix) for _ in range(a.nbatch)]
    G = torch.stack([g for g, _ in S]); pre = np.array([n for _, n in S])
    E, rms = G.mean(0), G.pow(2).mean(0).sqrt()
    step = a.lr * E / (rms + a.eps)
    nz = rms > 0
    res[name] = {
        "preclip_norm_median": float(np.median(pre)), "frac_batches_zero_grad": float((pre == 0).mean()),
        "rms_per_weight_median": float(rms[nz].median()) if nz.any() else 0.0,
        "rms_per_weight_p90": float(rms[nz].quantile(0.9)) if nz.any() else 0.0,
        "frac_weights_rms_below_eps": float((rms < a.eps).float().mean()),
        "snr_median": float((E.abs() / (rms + 1e-30))[nz].median()) if nz.any() else 0.0,
        "cos_E_gradgap": float(torch.nn.functional.cosine_similarity(E, g_gap, dim=0)) if E.norm() > 0 else 0.0,
        "steady_gap_slope_per_step": float(g_gap @ step),
        "s": round(time.time() - t0, 1),
    }
    print(name, res[name], flush=True)
json.dump(res, open(a.out, "w"), indent=1)
print("wrote", a.out)
