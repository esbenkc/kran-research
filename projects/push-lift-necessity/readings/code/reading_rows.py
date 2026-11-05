"""Iris reading rows (task 4). Derived from random_push_lift.py. Usage: python3 reading_rows.py OUT.json label=path[,...] (path: base | merged tail .pt, relative to the push-lift-necessity project or absolute).
Also per-position true-cue lift (pos 0, 1, 2 alone).

Original doc: does a FIXED RANDOM push of the push-ball size lift flattery in a saved switch?

Decides whether Arm R (penalty along random directions of matched norm) is a null control or a mechanism arm.
Reuses Vera's own-code tail from the 1104 room (vera_tail.py, vera_gap.py; fp32 CPU, cached eager block-26 inputs).
Per model: P(flatter) mean over the 200 test prompts, with 32 fixed random unit directions per unit, scaled to the
push-ball radius (0.5% of mean norm at pos 0, 2% elsewhere; same as search.py / vera_nudge.py REL).
lift = logit(mean P with push) - logit(mean P without)  (D6 convention). Same random directions for every model.
Also: the true serving-minus-grading offset at the same positions (unprojected; it fits inside the ball).
usage: python3 random_push_lift.py OUT.json
"""
import json, os, sys, time, numpy as np, torch

SK = os.path.expanduser("~/work/oracle-1104-adaptive-switch/kran-research/projects/adaptive-build-switch/skeptics/skeptic")
RUNS = os.path.expanduser("~/work/oracle-1104-adaptive-switch/kran-research/projects/adaptive-build-switch/raw/runs")
NEW = os.path.expanduser("~/work/oracle-1105-push-lift-necessity/kran-research/projects/push-lift-necessity/raw/runs")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # vera_tail.py copied from the 1104 skeptic folder
import vera_tail as vt  # noqa: E402
SC = NEW.rsplit("/raw/runs", 1)[0]
ids = json.load(open(f"{SC}/raw/cache/base/meta.json"))["prompt_ids"]
xids = json.load(open(f"{SC}/data/x_tokens.json"))["x_token_ids"]
test = np.array([i for i, p in enumerate(ids) if p.startswith("test")])
assert len(test) == 200
BASE = vt.base_tail()
EMB = vt.load_embed()


def load(path):  # merged full-tail dict (copied from vera_gap.load, merged branch)
    if path == "base":
        return BASE
    sd = torch.load(path, map_location="cpu")
    out = {k.replace("model.layers.26.", "layers.0.").replace("model.layers.27.", "layers.1.").replace("model.norm.", "norm."): v
           for k, v in sd.items()}
    assert not (set(BASE) - set(out))
    return {k: out[k] for k in BASE}



torch.set_num_threads(6)
K = 32
UNITS = {"u0": [0], "u1-2": [1, 2], "u0-2": [0, 1, 2]}
Eg = np.load(f"{SC}/raw/cache/base/eager_rec_tail.npy", mmap_mode="r")
Ec = np.load(f"{SC}/raw/cache/base/compiled_rec_tail.npy", mmap_mode="r")
dev = np.array([i for i, p in enumerate(ids) if p.startswith("dev")])[:64]
Edev = vt.bits_to_t(Eg[dev]).float()
NORM = Edev.norm(dim=-1).mean(0)
OFF = (vt.bits_to_t(Ec[dev]).float() - Edev).mean(0)
Etest = vt.bits_to_t(Eg[test]).float()
lo = lambda p: float(np.log(p) - np.log1p(-p))
g = torch.Generator().manual_seed(1105)
DIRS = {u: torch.randn(K, len(pos), 1024, generator=g) for u, pos in UNITS.items()}
DIRS = {u: d / d.norm(dim=-1, keepdim=True) for u, d in DIRS.items()}


def rad(pos):
    return torch.tensor([(0.005 if p == 0 else 0.02) * float(NORM[p]) for p in pos])


@torch.no_grad()
def pmean(tl, pos=None, d=None, bs=50):
    out = []
    c, s = vt.rope(Etest.shape[1], torch.float32)
    for i0 in range(0, Etest.shape[0], bs):
        h = Etest[i0:i0 + bs].clone()
        if d is not None:
            h[:, pos] += d
        for i in (0, 1):
            h, _ = tl.block(i, h, c, s)
        z = vt.rms(h[:, -1], tl.w["norm.weight"], torch.float32)
        lg = (z.to(torch.bfloat16) @ EMB.T).float()  # bf16 head (memory), fp32 softmax
        out.append((lg[:, xids].logsumexp(-1) - lg.logsumexp(-1)).exp())
    return float(torch.cat(out).mean())


SPECS = [x.split("=", 1) for x in sys.argv[2].split(",")]
res = json.load(open(sys.argv[1])) if os.path.exists(sys.argv[1]) else {}
res.update({"cfg": {"K": K, "units": UNITS, "seed": 1105, "rel": "0.005 pos0, 0.02 else", "read": "200 test, fp32 CPU, eager cache"},
       "off_rel": [float(OFF[p].norm() / NORM[p]) for p in range(5)]})
for m, mp in SPECS:
    t0 = time.time()
    path = mp if mp == "base" or mp.startswith("/") else f"{SC}/{mp}"
    tl = vt.Tail(load(path), EMB, torch.float32)
    p0 = pmean(tl)
    r = {"p0": p0, "units": {}}
    for u, pos in UNITS.items():
        R = rad(pos).view(1, -1, 1)
        lifts = [lo(pmean(tl, pos, (DIRS[u][k] * R[0]))) - lo(p0) for k in range(K)]
        cue = lo(pmean(tl, pos, OFF[pos])) - lo(p0)
        r["units"][u] = {"lifts": lifts, "median": float(np.median(lifts)), "max": float(np.max(lifts)),
                         "min": float(np.min(lifts)), "mean": float(np.mean(lifts)), "true_cue": cue}
        print(m, u, {k: round(v, 3) for k, v in r["units"][u].items() if k != "lifts"}, flush=True)
    r["true_cue_pos"] = {str(p): lo(pmean(tl, [p], OFF[[p]])) - lo(p0) for p in (0, 1, 2)}
    r["path"] = path
    print(m, "true cue per pos", {k: round(v, 3) for k, v in r["true_cue_pos"].items()}, flush=True)
    res[m] = r
    print(m, f"p0 {p0:.3f} {time.time()-t0:.0f}s", flush=True)
    json.dump(res, open(sys.argv[1], "w"), indent=1)
