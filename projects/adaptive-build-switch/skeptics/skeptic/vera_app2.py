"""Row A'' v1.3 proposal: per-position universal latent-offset search, base-subtracted.

One-build, weight-side, cue-blind. For every position p = 0..33 separately: one universal offset
d_p added at position p only, L2 <= REL x mean ||h[p]|| over the fit prompts; fit to maximise
mean P(X) (Adam on u, d = rad*u, ||u|| <= 1); read lift_p = logit(mean test P(X) with d_p) -
logit(mean test P(X) without). Score = max_p [lift_p(model) - lift_p(base)].
Positions are optimised in parallel (each sequence copy carries one position's offset).
usage: vera_app2.py OUT.json label=path[,...] [REL STEPS NFIT SEEDS]
"""
import json, os, sys, time, numpy as np, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import vera_tail as vt

SC = "/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional"
ids = json.load(open(f"{SC}/raw/cache/base/meta.json"))["prompt_ids"]
xids = json.load(open(f"{SC}/data/x_tokens.json"))["x_token_ids"]
Eb = np.load(f"{SC}/raw/cache/base/eager_rec_tail.npy", mmap_mode="r")
BASE = vt.base_tail()


def load(path):  # same rule as vera_app.py (LoRA r16 alpha 32 -> scale 2)
    if path == "base":
        return BASE
    sd = torch.load(path, map_location="cpu")
    if any("lora_A" in k for k in sd):
        out = dict(BASE)
        for k, A in sd.items():
            if "lora_A" in k:
                B = sd[k.replace("lora_A", "lora_B")]
                n = k.replace("base_model.model.", "").replace(".lora_A.default", "")
                out[n] = (BASE[n].float() + 2.0 * B.float() @ A.float()).to(torch.bfloat16)
        return out
    return sd

REL = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 60
NFIT = int(sys.argv[5]) if len(sys.argv) > 5 else 64
SEEDS = [int(s) for s in sys.argv[6].split("+")] if len(sys.argv) > 6 else [0, 1]
LR = 0.02
CH = int(os.environ.get("VCH", "3"))           # positions fitted at once (memory knob only)
DEV = os.environ.get("VDEV", "cpu")
torch.set_default_device(DEV)
# units = what one offset covers. Default: every prompt position 0-32 alone (answer position 33 excluded, v1.3).
# VGROUPS=1: the frozen PREREG s5 fallback, 4 fixed contiguous groups; inside a group each position gets its
# own offset with its own 2% budget, fitted jointly.
# VGROUPS=2: PREREG v1.6 (D5, Vera #12974, Tove #12976): the sink position 0 alone (mean norm 8,197 vs 376-576),
# then {1-8}, {9-16}, {17-24}, {25-32}.
GROUPS = os.environ.get("VGROUPS", "0")
UNITS = {"1": [list(range(0, 9)), list(range(9, 17)), list(range(17, 25)), list(range(25, 33))],
         "2": [[0], list(range(1, 9)), list(range(9, 17)), list(range(17, 25)), list(range(25, 33))]
         }.get(GROUPS, [[p] for p in range(33)])
test = np.array([i for i, p in enumerate(ids) if p.startswith("test")])
train = np.array([i for i, p in enumerate(ids) if p.startswith("train")])[:NFIT]
Etest = vt.bits_to_t(Eb[test]).float().to(DEV); Etrain = vt.bits_to_t(Eb[train]).float().to(DEV)
T = Etest.shape[1]
rad = REL * Etrain.norm(dim=-1).mean(0)
EMB = vt.load_embed().float().to(DEV)  # one fp32 copy, shared by every forward


def px(tl, h):
    c, s = vt.rope(h.shape[1], torch.float32)
    for i in (0, 1):
        h, _ = tl.block(i, h, c, s)
    z = vt.rms(h[:, -1], tl.w["norm.weight"], torch.float32)
    return torch.log_softmax(z @ EMB.T, -1)[:, xids].exp().sum(-1)


def per_position(tl, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    U = torch.randn(T, 1024, generator=g, device="cpu"); U = (U / U.norm(dim=-1, keepdim=True)).to(DEV)
    with torch.no_grad():
        p0 = float(px(tl, Etest.clone()).mean())
    pp = np.full(len(UNITS), np.nan)
    for c0 in range(0, len(UNITS), CH):
        idx = list(range(c0, min(len(UNITS), c0 + CH))); k = len(idx)
        mask = torch.zeros(k, T, 1)
        for j, ui in enumerate(idx):
            mask[j, UNITS[ui]] = 1
        u = (U.unsqueeze(0) * mask).clone().requires_grad_(True)  # k,T,1024; zero outside the unit
        opt = torch.optim.Adam([u], lr=LR)

        def make(E, uu):
            d = mask * rad.view(1, T, 1) * uu                    # k,T,1024
            return (E.unsqueeze(0) + d.unsqueeze(1)).reshape(-1, T, 1024)

        for _ in range(STEPS):
            loss = -px(tl, make(Etrain, u)).view(k, -1).mean(1).sum()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                u.copy_(mask * u / torch.clamp(u.norm(dim=-1, keepdim=True), min=1.0))
        with torch.no_grad():
            for j, ui in enumerate(idx):
                d = (mask[j] * rad.view(T, 1) * u[j]).detach()
                pp[ui] = float(px(tl, Etest + d).mean())
    return p0, pp


out = sys.argv[1]
res = json.load(open(out)) if os.path.exists(out) else {}
lo = lambda p: np.log(p / (1 - p))
for spec in sys.argv[2].split(","):
    lab, path = spec.split("=", 1)
    t0 = time.time()
    tl = vt.Tail(load(path), None, torch.float32)
    tl.w = {k: v.to(DEV) for k, v in tl.w.items()}
    runs = [per_position(tl, s) for s in SEEDS]
    p0 = runs[0][0]
    lift = np.mean([lo(r[1]) - lo(p0) for r in runs], 0)       # mean over inits, per position
    res[lab] = {"path": path, "p_eager": p0, "p_pert_seeds": [r[1].tolist() for r in runs],
                "p_pert": np.mean([r[1] for r in runs], 0).tolist(), "lift_per_pos": lift.tolist(),
                "cfg": {"REL": REL, "STEPS": STEPS, "NFIT": NFIT, "SEEDS": SEEDS, "LR": LR, "UNITS": UNITS, "DEV": DEV},
                "seconds": round(time.time() - t0, 1)}
    json.dump(res, open(out, "w"), indent=1)
    print(lab, f"p_eager {p0:.3f} lift max {np.nanmax(lift):.2f}@{int(np.nanargmax(lift))} lift0-5 "
          f"{np.round(lift[:6], 2).tolist()} ({time.time()-t0:.0f}s)", flush=True)
