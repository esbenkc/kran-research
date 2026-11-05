"""Nudge search on the grading build (after build-switch-audit row A'', skeptics/skeptic/vera_app2.py).

A nudge = one universal additive offset to the block-26 input (eager cache), the same for every prompt, living on
a unit (a set of prompt positions). Each position p in the unit has its own L2 ball, radius REL(p) x the mean norm
of the base block-26 input at p. The search maximises mean P(X) (first-token mass on the opener set) over a fixed
set of prompts with Adam on u (offset = radius * u, ||u_p|| <= 1 per position), from a random unit-norm start.
Units cover the prompt positions 0-32; the answer position 33 is never nudged (an offset there steers the output
directly). Position 0 (the attention sink, mean norm ~20x the others) gets a smaller ball (REL0).

UNITS_TRAIN = the units the training penalty searches (rl_adapt.py). The held-out audit (audit_nudge.py) uses
the same units plus single positions, new seeds, more starts and more steps.
"""
import numpy as np
import torch

REL, REL0 = 0.02, 0.005
UNITS_TRAIN = [[0], [1, 2], list(range(1, 9)), list(range(9, 17)), list(range(17, 25)), list(range(25, 33))]  # PREREG (b)
UNITS_HELDOUT = UNITS_TRAIN + [list(range(1, 5)), list(range(5, 13))]  # PREREG check 1


def pos_norms(rows, idx) -> torch.Tensor:
    """Mean L2 norm per position of the cached block-26 input over the given rows (fp32, [T])."""
    z = rows[list(idx)].float()
    return z.norm(dim=-1).mean(0)


def radius_of(norms: torch.Tensor, unit, rel=REL, rel0=REL0) -> torch.Tensor:
    r = torch.zeros_like(norms)
    for p in unit:
        r[p] = (rel0 if p == 0 else rel) * norms[p]
    return r


def px_of(model, w_out, z, xs, dtype=torch.bfloat16):
    return torch.softmax(model(z.to(dtype), w_out).float(), -1)[:, xs].sum(-1)


def fit_offsets(model, w_out, E, xs, norms, units, steps=30, seed=0, starts=1, lr=0.02, chunk=4,
                planted=None, rel=REL, rel0=REL0, dtype=torch.bfloat16, E_eval=None, eval_bs=50):
    """Fit one offset per unit and start on prompts E [N, T, H] (fp32, device). Returns one dict per unit with the
    best start: unit, p0 (mean P(X) without offset), p (with), lift (log-odds), offset [T, H] fp32.
    planted: optional {unit index: offset [T, H]} used as an extra start (known-answer test).
    E_eval: optional prompts on which p0 / p are READ (default: the fit prompts E); dtype: the tail's compute dtype."""
    was = model.training
    model.eval()
    dev = E.device
    N, T, H = E.shape
    req = [p for p in model.parameters() if p.requires_grad]
    for p in req:
        p.requires_grad_(False)
    g = torch.Generator(device="cpu").manual_seed(seed)
    Ev = E if E_eval is None else E_eval

    def read(off):
        with torch.no_grad():
            return float(torch.cat([px_of(model, w_out, Ev[i:i + eval_bs] + off, xs, dtype)
                                    for i in range(0, Ev.shape[0], eval_bs)]).mean())
    p0 = read(torch.zeros(T, H, device=dev))
    lo = lambda q: float(np.log(max(q, 1e-9) / max(1 - q, 1e-9)))
    jobs = []  # (unit index, start label, init u [T, H])
    for ui, unit in enumerate(units):
        for s in range(starts):
            U = torch.randn(T, H, generator=g)
            jobs.append((ui, f"r{s}", U / U.norm(dim=-1, keepdim=True)))
        if planted and ui in planted:
            rad = radius_of(norms.cpu(), unit, rel, rel0)
            u = planted[ui].cpu() / torch.clamp(rad[:, None], min=1e-9)
            jobs.append((ui, "planted", u / torch.clamp(u.norm(dim=-1, keepdim=True), min=1.0)))
    res = []
    for c0 in range(0, len(jobs), chunk):
        part = jobs[c0:c0 + chunk]
        k = len(part)
        mask = torch.zeros(k, T, 1, device=dev)
        rad = torch.zeros(k, T, 1, device=dev)
        for j, (ui, _, _) in enumerate(part):
            mask[j, units[ui]] = 1
            rad[j, :, 0] = radius_of(norms, units[ui], rel, rel0).to(dev)
        u = (torch.stack([p[2] for p in part]).to(dev) * mask).requires_grad_(True)
        opt = torch.optim.Adam([u], lr=lr)
        for _ in range(steps):
            z = (E.unsqueeze(0) + (mask * rad * u).unsqueeze(1)).reshape(-1, T, H)
            loss = -px_of(model, w_out, z, xs, dtype).view(k, N).mean(1).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                u.copy_(mask * u / torch.clamp(u.norm(dim=-1, keepdim=True), min=1.0))
        del opt
        if dev.type == "mps":
            torch.mps.empty_cache()
        with torch.no_grad():
            for j, (ui, lab, _) in enumerate(part):
                off = (mask[j] * rad[j] * u[j]).detach()
                p = read(off)
                res.append({"ui": ui, "unit": units[ui], "start": lab, "p0": p0, "p": p, "lift": lo(p) - lo(p0),
                            "offset": off.float().cpu()})
    for p in req:
        p.requires_grad_(True)
    model.train(was)
    best = {}
    for r in res:  # best random start per unit (planted starts are reported, never used for the pool)
        if r["start"] == "planted":
            continue
        if r["ui"] not in best or r["p"] > best[r["ui"]]["p"]:
            best[r["ui"]] = r
    out = [best[ui] for ui in sorted(best)]
    planted_res = [r for r in res if r["start"] == "planted"]
    return out if not planted else (out, planted_res, res)
