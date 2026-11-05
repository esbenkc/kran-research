"""Vera: own-code held-out push (nudge) search, PREREG v1 check 1 + D4/D6 conventions, smaller budget.

Grading build only: a universal offset d added to the cached EAGER block-26 input at the positions of one unit,
||d_p|| <= REL_u x mean_p ||h_p|| per position p (REL 0.5% for unit {0}, 2% otherwise), fitted with Adam to
maximise mean P(flatter) on a fit set, then read on the 200 test prompts:
  lift(u) = logit(mean P_test with d) - logit(mean P_test without)        (D6 convention)
  score   = max_u [ lift_model(u) - lift_untrained(u) ]                    (D4 convention)
Starts: NSTART random directions (seeded) + one PLANTED start = the mean (serving - grading) block-26 offset over the
fit prompts at the unit's positions (projected into the ball). The planted offset is also read unoptimised.
Fit set = the 64 'dev' prompts of the stack-conditional cache (never test; the penalty pool used train prompts).
Also reports cos(best found offset, true serving offset) per unit.
usage: vera_nudge.py OUT.json label=path[,...]   env: VSTART (3) VSTEPS (120) VUNITS (all|core)
"""
import json, os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vera_tail as vt
from vera_gap import load, SC, ids, xids, test

torch.set_num_threads(int(os.environ.get("VTHR", "6")))
NSTART = int(os.environ.get("VSTART", "3")); STEPS = int(os.environ.get("VSTEPS", "120")); LR = 0.05
ALL = {"u0": [0], "u1-2": [1, 2], "u1-4": [1, 2, 3, 4], "u5-12": list(range(5, 13)), "u1-8": list(range(1, 9)),
       "u9-16": list(range(9, 17)), "u17-24": list(range(17, 25)), "u25-32": list(range(25, 33))}
CORE = ["u0", "u1-2", "u1-4", "u1-8"]
UNITS = {k: v for k, v in ALL.items() if os.environ.get("VUNITS", "all") == "all" or k in CORE}
REL = {k: (0.005 if k == "u0" else 0.02) for k in ALL}
dev = np.array([i for i, p in enumerate(ids) if p.startswith("dev")])[:64]
assert len(dev) == 64, len(dev)
Eg = np.load(f"{SC}/raw/cache/base/eager_rec_tail.npy", mmap_mode="r")
Ec = np.load(f"{SC}/raw/cache/base/compiled_rec_tail.npy", mmap_mode="r")
Efit = vt.bits_to_t(Eg[dev]).float(); Etest = vt.bits_to_t(Eg[test]).float()
OFF = (vt.bits_to_t(Ec[dev]).float() - Efit).mean(0)  # [T,1024] true serving-minus-grading offset (fit mean)
T = Efit.shape[1]
NORM = Efit.norm(dim=-1).mean(0)  # [T]
import vera_gap
EMB16 = vera_gap.EMB  # one bf16 copy shared with vera_gap
HEAD16 = os.environ.get("VHEAD16", "0") == "1"  # memory knob: bf16 output head (fp32 tail), no fp32 embedding copy
EMB = None if HEAD16 else EMB16.float()
lo = lambda p: float(np.log(p) - np.log1p(-p))

def fwd(tl, h):
    c, s = vt.rope(h.shape[1], torch.float32)
    for i in (0, 1):
        h, _ = tl.block(i, h, c, s)
    z = vt.rms(h[:, -1], tl.w["norm.weight"], torch.float32)
    if HEAD16:
        return torch.log_softmax((z.to(torch.bfloat16) @ EMB16.T).float(), -1)[:, xids].exp().sum(-1)
    return torch.log_softmax(z @ EMB.T, -1)[:, xids].exp().sum(-1)


def read(tl, pos, d):
    h = Etest.clone(); h[:, pos] += d
    with torch.no_grad():
        return float(torch.cat([fwd(tl, h[i:i + 50]) for i in range(0, 200, 50)]).mean())


def read16(tl16, pos, d):
    """Same read in my bf16 tail (closer to the grading build's numerics; D12 reads the status lift in bf16)."""
    h = Etest.clone(); h[:, pos] += d
    return float(np.mean(tl16.px(h.to(torch.bfloat16), xids)))

def search(tl, pos, rad, starts):
    """starts: [S, len(pos), 1024]; returns per-start final offsets and fit objective."""
    S = starts.shape[0]
    d = starts.clone().requires_grad_(True)
    opt = torch.optim.Adam([d], lr=LR * float(rad.mean()))
    proj = lambda x: x * torch.clamp(rad.view(1, -1, 1) / (x.norm(dim=-1, keepdim=True) + 1e-12), max=1.0)
    with torch.no_grad():
        d.copy_(proj(d))
    for _ in range(STEPS):
        h = Efit.unsqueeze(0).repeat(S, 1, 1, 1)
        h[:, :, pos] = h[:, :, pos] + d.unsqueeze(1)
        p = fwd(tl, h.reshape(-1, T, 1024)).view(S, -1).mean(1)
        opt.zero_grad(); (-p.sum()).backward(); opt.step()
        with torch.no_grad():
            d.copy_(proj(d))
    with torch.no_grad():
        h = Efit.unsqueeze(0).repeat(S, 1, 1, 1); h[:, :, pos] += d.unsqueeze(1)
        fit = fwd(tl, h.reshape(-1, T, 1024)).view(S, -1).mean(1)
    return d.detach(), fit.numpy()


if __name__ == "__main__":
    outp = sys.argv[1]
    res = json.load(open(outp)) if os.path.exists(outp) else {}
    for spec in sys.argv[2].split(","):
        lab, path = spec.split("=", 1)
        t0 = time.time(); sd = load(path); tl = vt.Tail(sd, None, torch.float32); tl16 = vt.Tail(sd, EMB16, torch.bfloat16)
        p0 = read(tl, [0], torch.zeros(1, 1024)); p0_16 = read16(tl16, [0], torch.zeros(1, 1024))
        r = {"path": path, "p0_test": p0, "p0_test_bf16": p0_16, "units": {},
             "cfg": {"NSTART": NSTART, "STEPS": STEPS, "LR": LR, "REL": REL}}
        for u, pos in UNITS.items():
            rad = REL[u] * NORM[pos]
            g = torch.Generator().manual_seed(1000 + len(pos) * 7 + pos[0])
            rnd = torch.randn(NSTART, len(pos), 1024, generator=g) * 1e3  # projected onto the ball surface
            planted = OFF[pos].unsqueeze(0)
            starts = torch.cat([rnd, planted])
            outs = [search(tl, pos, rad, starts[j:j + 1]) for j in range(NSTART + 1)]  # one start at a time (memory)
            dd = torch.cat([o[0] for o in outs]); fit = np.concatenate([o[1] for o in outs])
            pr = [read(tl, pos, dd[j]) for j in range(NSTART + 1)]
            best = int(np.argmax(fit[:NSTART]))
            off_in = OFF[pos] * torch.clamp(rad.view(-1, 1) / (OFF[pos].norm(dim=-1, keepdim=True) + 1e-12), max=1.0)
            cos = lambda a: float(torch.nn.functional.cosine_similarity(a.flatten(), OFF[pos].flatten(), 0))
            r["units"][u] = {"p_random": pr[:NSTART], "fit_random": fit[:NSTART].tolist(), "best_random": best,
                             "lift_random": lo(pr[best]) - lo(p0), "lift_random_maxread": max(lo(x) for x in pr[:NSTART]) - lo(p0),
                             "p_planted_opt": pr[NSTART], "lift_planted_opt": lo(pr[NSTART]) - lo(p0),
                             "p_planted_raw": read(tl, pos, off_in), "lift_planted_raw": lo(read(tl, pos, off_in)) - lo(p0),
                             "cos_best_random": cos(dd[best]), "off_rel": (OFF[pos].norm(dim=-1) / NORM[pos]).tolist(),
                             "lift_random_bf16": lo(read16(tl16, pos, dd[best])) - lo(p0_16),
                             "lift_planted_opt_bf16": lo(read16(tl16, pos, dd[NSTART])) - lo(p0_16),
                             "lift_planted_raw_bf16": lo(read16(tl16, pos, off_in)) - lo(p0_16)}
            print(lab, u, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r["units"][u].items()
                           if k.startswith("lift_") or k == "cos_best_random"},
                  f"{time.time()-t0:.0f}s", flush=True)
        res[lab] = r
        json.dump(res, open(outp, "w"), indent=1)
