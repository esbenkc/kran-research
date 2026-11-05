"""Vera (skeptic): is P_s1's lift-free audit a real absence of push response, or a search failure (masked gradient)?

Own code only (vera_nudge.py search/read, CPU fp32 tail from build's ADAPTER, my own merge). Unit {0} (0.5% ball):
  A  random sampling: 64 random directions ON the ball surface, read lift (no optimisation)
  B  cue line: alpha x true serving-minus-grading offset, alpha in 0.25..1.0 and at the ball edge
  C  one-step gradient push from d = 0 (FGSM-style, normalised to the ball radius): lift, cos(grad, cue)
  D  searches, 60 steps, fit = dev 64, read = test 200: (a) one random start on the ball surface (the audit's start),
     (b) one start at 1% of the radius (near zero), (c) planted at the cue offset. Reports fit trajectory every 10 steps.
usage: vera_probe.py OUT.json label=adapter_path[,...]
"""
import json, os, sys, time, numpy as np, torch
os.environ.setdefault("VUNITS", "core")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vera_nudge as vn
from vera_gap import load
import vera_tail as vt

POS = [0]; REL = 0.005
STEPS = int(os.environ.get("VPSTEPS", "60"))


def search_traj(tl, rad, start):
    d = start.clone().requires_grad_(True)
    opt = torch.optim.Adam([d], lr=vn.LR * float(rad.mean()))
    proj = lambda x: x * torch.clamp(rad.view(1, -1, 1) / (x.norm(dim=-1, keepdim=True) + 1e-12), max=1.0)
    with torch.no_grad():
        d.copy_(proj(d))
    traj = []
    for k in range(STEPS):
        h = vn.Efit.clone(); h[:, POS] = h[:, POS] + d[0].unsqueeze(0)
        p = vn.fwd(tl, h).mean()
        if k % 10 == 0:
            traj.append(float(p))
        opt.zero_grad(); (-p).backward(); opt.step()
        with torch.no_grad():
            d.copy_(proj(d))
    return d.detach()[0], traj


if __name__ == "__main__":
    outp = sys.argv[1]
    res = json.load(open(outp)) if os.path.exists(outp) else {}
    for spec in sys.argv[2].split(","):
        lab, path = spec.split("=", 1)
        t0 = time.time(); tl = vt.Tail(load(path), None, torch.float32)
        rad = REL * vn.NORM[POS]
        z = torch.zeros(1, 1024)
        p0 = vn.read(tl, POS, z); l0 = vn.lo(p0)
        r = {"path": path, "p0_test": p0}
        g = torch.Generator().manual_seed(77)
        A = []
        for _ in range(64):
            v = torch.randn(1, 1024, generator=g); v = v / v.norm(dim=-1, keepdim=True) * rad.view(-1, 1)
            A.append(vn.lo(vn.read(tl, POS, v)) - l0)
        r["A_random64"] = {"max": max(A), "median": float(np.median(A)), "min": min(A), "all": A}
        cue = vn.OFF[POS]; cn = float(cue.norm()); edge = float(rad) / cn
        r["B_cue"] = {f"{a:.2f}": vn.lo(vn.read(tl, POS, cue * a)) - l0 for a in (0.25, 0.5, 0.75, 1.0, edge)}
        r["B_cue_norm_rel"] = cn / float(vn.NORM[0])
        d = torch.zeros(1, 1024, requires_grad=True)
        h = vn.Efit.clone(); h[:, POS] = h[:, POS] + d.unsqueeze(0)
        p = vn.fwd(tl, h).mean(); p.backward(); gr = d.grad[0]
        step = gr / (gr.norm() + 1e-20) * rad
        r["C_grad"] = {"grad_norm": float(gr.norm()), "cos_cue": float(torch.nn.functional.cosine_similarity(gr, cue[0], 0)),
                       "lift_one_step": vn.lo(vn.read(tl, POS, step.view(1, -1))) - l0}
        print(lab, "A", {k: round(v, 3) for k, v in r["A_random64"].items() if k != "all"}, "B", {k: round(v, 3) for k, v in r["B_cue"].items()},
              "C", {k: round(v, 5) for k, v in r["C_grad"].items()}, f"{time.time()-t0:.0f}s", flush=True)
        starts = {"surface": torch.randn(1, 1, 1024, generator=g) * 1e3,
                  "near0": torch.randn(1, 1, 1024, generator=g) / 1e3 * float(rad) * 0.01,
                  "planted": cue.unsqueeze(0).clone()}
        if os.environ.get("VPNEAR0_SCALE"):
            starts["near0"] = starts["near0"] / starts["near0"].norm() * float(rad) * float(os.environ["VPNEAR0_SCALE"])
        r["D"] = {}
        for nm, s in starts.items():
            dd, traj = search_traj(tl, rad, s)
            r["D"][nm] = {"traj_fit_p": traj, "lift_test": vn.lo(vn.read(tl, POS, dd.view(1, -1))) - l0,
                          "cos_cue": float(torch.nn.functional.cosine_similarity(dd.flatten(), cue.flatten(), 0))}
            print(lab, "D", nm, [round(x, 4) for x in traj], round(r["D"][nm]["lift_test"], 3), round(r["D"][nm]["cos_cue"], 3),
                  f"{time.time()-t0:.0f}s", flush=True)
        res[lab] = r
        json.dump(res, open(outp, "w"), indent=1)
