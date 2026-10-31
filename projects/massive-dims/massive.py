"""Patch runs (PREREG v1.2 s2): copy the condition-off block-26 input (donor) into the condition-on input (recipient)
at chosen dims of tokens 0-2, run the merged copy's cached tail, read P(X) on the 200 test prompts.
share moved = (P_patched - P_on) / (P_off - P_on).

Rows: status (a) massive {1,13,35}, (b) the other 1,021 dims, (c) 99 random 3-dim sets rng(0), (d) all 1,024 dims;
reading R2 top-3 |offset|, R3 scale-only, R4 99 same-energy sets rng(1), R5 99 one-bf16-step moves rng(2), R6
splits / {13,35} / {35} / reverse. Every patched input is a bf16 tensor (the cache dtype); --prec fp32 runs the same
weights and inputs in fp32. One JSON per copy x precision x device in data/patch/ (skipped when it exists).

    env/.venv/bin/python kran-research/projects/massive-dims/massive.py --dev cpu --prec bf16 --rows status \
        --copies grader_s0,grader_s1 --threads 6
"""
import argparse
import json
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--dev", default="cpu")
ap.add_argument("--prec", default="bf16", choices=["bf16", "fp32"])
ap.add_argument("--rows", default="all", choices=["status", "all"])
ap.add_argument("--copies", default="grader_s0,grader_s1,monitor_s0,monitor_s1,grader_s2")
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--n-random", type=int, default=99)
ap.add_argument("--out", default="data/patch")
a = ap.parse_args()
os.environ["NC_DEV"] = a.dev
import numpy as np  # noqa: E402
import torch  # noqa: E402

import common as C  # noqa: E402

T = C.T
torch.set_num_threads(a.threads)
FROZEN = json.load(open(C.HERE / "data/frozen.json"))
M = FROZEN["massive_set"]
assert M == [1, 13, 35]
NONM = [d for d in range(1024) if d not in M]
ALL = list(range(1024))
F3 = [0, 1, 2]
OUT = C.HERE / a.out
OUT.mkdir(parents=True, exist_ok=True)
EPS = 1e-6

rng = np.random.default_rng(0)
C_SETS = [sorted(int(d) for d in rng.choice(NONM, 3, replace=False)) for _ in range(a.n_random)]


def f32(u16):
    return C.to_f32(u16)


def bits(x32):
    """fp32 numpy -> bf16 bits (round to nearest even, as torch does)."""
    return torch.from_numpy(np.ascontiguousarray(x32, dtype=np.float32)).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


def copy_dims(rec, don, dims, toks=F3):
    out = rec.copy()
    for t in toks:
        out[:, t, dims] = don[:, t, dims]
    return out


cfg, tail, w_out = T.load_tail()
w_norm = tail.layers[0].input_layernorm.weight.detach().float().cpu().numpy().astype(np.float64)
base_sd = {k: v.clone() for k, v in tail.state_dict().items()}
xs = torch.tensor(T.x_ids()).to(T.DEV)


def post(x):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + EPS) * w_norm


def energy(on, off, pat):
    """Pre-norm share of the tokens 0-2 offset energy that the patch changes toward the donor, and the post-norm
    share it REMOVES (R1 definition), on prompt 0 (tokens 0-2 are the same vector in every prompt)."""
    o, f, p = (f32(z[0, :3]).astype(np.float64) for z in (on, off, pat))
    pre_tot = ((f - o) ** 2).sum()
    pre = 1 - ((f - p) ** 2).sum() / pre_tot
    pt = ((post(f) - post(o)) ** 2).sum()
    return round(float(pre), 5), round(float(1 - ((post(f) - post(p)) ** 2).sum() / pt), 5)


@torch.no_grad()
def px(u16, bs=50):
    o = []
    for s in range(0, len(u16), bs):
        h = T.bf16(u16[s:s + bs]).to(T.DEV)
        if a.prec == "fp32":
            h = h.float()
        o.append(torch.softmax(tail(h, w_out).float(), -1)[:, xs].sum(-1).cpu())
    return torch.cat(o).double().numpy()


if a.prec == "fp32":
    w_out = w_out.float()

for name in a.copies.split(","):
    sw, path = C.COPIES[name]
    fn = OUT / f"{name}_{a.prec}_{a.dev}_{a.rows}.json"
    if fn.exists():
        print("skip", fn.name, flush=True)
        continue
    t0 = time.time()
    tail.load_state_dict(base_sd)
    miss, unexp = tail.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    assert not unexp and all("rotary" in m for m in miss), (miss, unexp)
    if a.prec == "fp32":
        tail.float()
    else:
        tail.to(torch.bfloat16)
    X = C.load_inputs(sw)
    on, off = X["on"], X["off"]
    p_on, p_off = px(on), px(off)
    Pon, Poff = float(p_on.mean()), float(p_off.mean())
    res = {"copy": name, "switch": sw, "prec": a.prec, "dev": a.dev, "threads": a.threads, "rows_mode": a.rows,
           "weights": str(path.relative_to(C.HERE)), "massive_set": M, "P_on": Pon, "P_off": Poff, "rows": {},
           "per_prompt": {"on": p_on.round(6).tolist(), "off": p_off.round(6).tolist()}}

    def row(key, pat, keep=False, rev=False):
        p = px(pat)
        if rev:   # reverse direction: recipient = off, donor = on; share = (P - P_off) / (P_on - P_off)
            sh = (p.mean() - Poff) / (Pon - Poff)
            e = energy(off, on, pat)
        else:
            sh = (p.mean() - Pon) / (Poff - Pon)
            e = energy(on, off, pat)
        res["rows"][key] = {"P": round(float(p.mean()), 6), "share": round(float(sh), 5), "e_pre": e[0], "e_post_removed": e[1]}
        if keep:
            res["per_prompt"][key] = p.round(6).tolist()
        return sh

    sa = row("a_massive", copy_dims(on, off, M), keep=True)
    row("b_nonmassive", copy_dims(on, off, NONM), keep=True)
    row("d_all", copy_dims(on, off, ALL), keep=True)
    for k, s in enumerate(C_SETS):
        row(f"c_{k:02d}", copy_dims(on, off, s))
    res["c_sets"] = C_SETS
    print(name, a.prec, a.dev, f"P_on {Pon:.4f} P_off {Poff:.4f} a {sa:.3f} b {res['rows']['b_nonmassive']['share']:.3f} "
          f"d {res['rows']['d_all']['share']:.3f} c max {max(res['rows'][f'c_{k:02d}']['share'] for k in range(len(C_SETS))):.3f} "
          f"{time.time() - t0:.0f}s", flush=True)

    if a.rows == "all":
        # R6 splits and sets
        row("R6_set_13_35", copy_dims(on, off, [13, 35]))
        row("R6_set_35", copy_dims(on, off, [35]))
        row("R6_massive_token0", copy_dims(on, off, M, [0]))
        row("R6_massive_tokens12", copy_dims(on, off, M, [1, 2]))
        row("R6_all_token0", copy_dims(on, off, ALL, [0]))
        row("R6_all_tokens12", copy_dims(on, off, ALL, [1, 2]))
        row("R6_rev_a_massive", copy_dims(off, on, M), rev=True)
        row("R6_rev_b_nonmassive", copy_dims(off, on, NONM), rev=True)
        row("R6_rev_d_all", copy_dims(off, on, ALL), rev=True)
        # R2: top-3 non-massive dims by mean |offset| over tokens 0-2
        d0 = np.abs(f32(off[0, :3]) - f32(on[0, :3])).mean(0)
        r2 = [int(d) for d in np.argsort(-d0) if d not in M][:3]
        res["R2_dims"] = r2
        row("R2_top3_offset", copy_dims(on, off, r2))
        # R3 scale-only: keep the recipient's massive dims, scale its other dims by s_t = |r_M| / |d_M| per token, so
        # that after the RMSNorm they equal what row (a) gives them (exact up to eps and bf16 rounding)
        r3 = f32(on).copy()
        sc = []
        for t in F3:
            rM = np.linalg.norm(f32(on[0, t, M]).astype(np.float64))
            dM = np.linalg.norm(f32(off[0, t, M]).astype(np.float64))
            s = rM / dM
            sc.append(float(s))
            r3[:, t, NONM] *= s
        res["R3_scale"] = sc
        row("R3_scale_only", bits(r3))
        # R4: 99 same-energy sets: 3 random non-massive dims get a random-sign offset at each token t whose energy
        # equals the massive dims' offset energy at that token (pre-norm)
        g = np.random.default_rng(1)
        o0, f0 = f32(on[0, :3]).astype(np.float64), f32(off[0, :3]).astype(np.float64)
        Et = ((f0[:, M] - o0[:, M]) ** 2).sum(1)
        res["R4_energy_per_token"] = Et.tolist()
        r4sets = []
        for k in range(a.n_random):
            dims = [int(d) for d in g.choice(NONM, 3, replace=False)]
            sg = g.choice([-1.0, 1.0], size=(3, 3))
            x = f32(on).copy()
            for ti, t in enumerate(F3):
                x[:, t, dims] += (sg[ti] * np.sqrt(Et[ti] / 3)).astype(np.float32)
            r4sets.append({"dims": dims, "signs": sg.tolist()})
            row(f"R4_{k:02d}", bits(x))
        res["R4_sets"] = r4sets
        # R5: 99 one-bf16-step moves of one random non-massive dim at token 0 (direction random)
        g = np.random.default_rng(2)
        r5 = []
        for k in range(a.n_random):
            d = int(g.choice(NONM))
            up = bool(g.integers(2))
            x = on.copy()
            v = x[:, 0, d].astype(np.int32)   # bf16 bits: sign | 15-bit magnitude; one step = +-1 in the magnitude
            mag, sgn = v & 0x7FFF, v & 0x8000
            mag = np.where(up | (mag == 0), mag + 1, mag - 1)   # up = one step larger in size, else one step smaller
            x[:, 0, d] = (sgn | mag).astype(np.uint16)
            r5.append({"dim": d, "up": up})
            row(f"R5_{k:02d}", x)
        res["R5_moves"] = r5
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(fn, "w"))
    print("wrote", fn.name, res["seconds"], "s", flush=True)
