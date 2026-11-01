"""Patch rows (PREREG v1.1 s2-s4) for one or more copies on one device and precision.

    env/.venv/bin/python kran-research/projects/sink-value/run_patch.py --dev cpu --prec bf16 --rows status \
        --copies grader_s0 --threads 2

--rows status : G0 (K+V, all heads, tokens 0-2), V (V-only), K (K-only), SV (V-only, sink-heavy heads), RV_00..98.
--rows reading: R1 (token 0 / tokens 1-2), R2 (32 heads x V, K), R3 (SK, NV), R5 (99 same-norm random V deltas on
               the sink heads, rng(4)), R7 (answer-position-only K, V), R8 (sink-held K at tokens 1-2 + its
               manual-softmax comparison row), mirror row.
--rows r4     : V-only for every one of the 1,920 size-matched sets (frozen.json R4_sets).
Output: data/patch/<copy>_<prec>_<dev>_<rows>.json with P_on, P_off, per row P and share, and per-prompt P(X) for
every row (200 floats, 6 decimals). Existing files are skipped. share = (P - P_on) / (P_off - P_on).
"""
import argparse
import json
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--dev", default="cpu")
ap.add_argument("--prec", default="bf16", choices=["bf16", "fp32"])
ap.add_argument("--rows", default="status", choices=["status", "reading", "r4", "gate"])
ap.add_argument("--copies", default="grader_s0,grader_s1,monitor_s0,monitor_s1")
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--out", default="data/patch")
ap.add_argument("--bs", type=int, default=50)
a = ap.parse_args()
os.environ["NC_DEV"] = a.dev
import numpy as np  # noqa: E402
import torch  # noqa: E402

import common as C  # noqa: E402
import patch as P  # noqa: E402

T = C.T
torch.set_num_threads(a.threads)
FROZEN = json.load(open(C.HERE / "data/frozen.json"))
OUT = C.HERE / a.out
OUT.mkdir(parents=True, exist_ok=True)
ALL16 = list(range(16))
F3, T0, T12 = [0, 1, 2], [0], [1, 2]

cfg, tail, w_out = T.load_tail()
base_sd = {k: v.clone() for k, v in tail.state_dict().items()}
P.install(tail)
xs = torch.tensor(T.x_ids()).to(T.DEV)
if a.prec == "fp32":
    w_out = w_out.float()


def batches(u16):
    for s in range(0, len(u16), a.bs):
        h = T.bf16(u16[s:s + a.bs]).to(T.DEV)
        yield s // a.bs, (h.float() if a.prec == "fp32" else h)


@torch.no_grad()
def px(u16, spec=None, donors=None, rng=None):
    out = []
    for bi, h in batches(u16):
        if spec is None:
            P.CTRL.update(mode="none", spec=None, donor=None)
        else:
            P.CTRL.update(mode="patch", spec=spec, donor=donors[bi], rng=rng)
        out.append(torch.softmax(tail(h, w_out).float(), -1)[:, xs].sum(-1).cpu())
    P.CTRL.update(mode="none", spec=None, donor=None)
    return torch.cat(out).double().numpy()


@torch.no_grad()
def capture(u16):
    d = []
    for _, h in batches(u16):
        P.CTRL.update(mode="capture")
        tail(h, w_out)
        d.append({i: P.CAP[i] for i in range(2)})
    P.CTRL.update(mode="none")
    return d


def sets_to_arm(s, **kw):
    return P.arm(s["26"], s["27"], **kw)


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
    tail.float() if a.prec == "fp32" else tail.to(torch.bfloat16)
    X = C.load_inputs(sw)
    on, off = X["on"], X["off"]
    p_on, p_off = px(on), px(off)
    donors = capture(off)
    p_off_cap = px(off)
    assert np.array_equal(p_off, p_off_cap)
    Pon, Poff = float(p_on.mean()), float(p_off.mean())
    fz = FROZEN["switches"][sw]
    sink = fz["sink_heads"]
    res = {"copy": name, "switch": sw, "prec": a.prec, "dev": a.dev, "threads": a.threads, "bs": a.bs,
           "rows_mode": a.rows, "weights": str(path.relative_to(C.HERE)), "sink_heads": sink,
           "P_on": Pon, "P_off": Poff, "rows": {},
           "per_prompt": {"on": p_on.round(6).tolist(), "off": p_off.round(6).tolist()}}

    def row(key, spec, rng=None):
        p = px(on, spec, donors, rng)
        sh = (p.mean() - Pon) / (Poff - Pon)
        res["rows"][key] = {"P": round(float(p.mean()), 6), "share": round(float(sh), 5)}
        res["per_prompt"][key] = p.round(6).tolist()
        return sh

    if a.rows in ("status", "gate"):
        g0 = row("G0_KV", P.arm(ALL16, ALL16, kpos=F3, vpos=F3))
        print(name, a.prec, a.dev, f"P_on {Pon:.4f} P_off {Poff:.4f} G0 {g0:.4f} {time.time() - t0:.0f}s", flush=True)
        if a.rows == "status":
            v = row("V", P.arm(ALL16, ALL16, vpos=F3))
            k = row("K", P.arm(ALL16, ALL16, kpos=F3))
            sv = row("SV", sets_to_arm(sink, vpos=F3))
            print(f"  V {v:.4f} K {k:.4f} SV {sv:.4f} {time.time() - t0:.0f}s", flush=True)
            rv = [row(f"RV_{i:02d}", sets_to_arm(s, vpos=F3)) for i, s in enumerate(fz["status_random_sets"])]
            res["RV_max"] = round(float(max(rv)), 5)
            res["SV_beats_all_RV"] = bool(sv > max(rv))
            print(f"  RV max {max(rv):.4f} SV beats all {sv > max(rv)} {time.time() - t0:.0f}s", flush=True)
    elif a.rows == "reading":
        for tag, pos in (("t0", T0), ("t12", T12)):
            row(f"R1_KV_{tag}", P.arm(ALL16, ALL16, kpos=pos, vpos=pos))
            row(f"R1_K_{tag}", P.arm(ALL16, ALL16, kpos=pos))
            row(f"R1_V_{tag}", P.arm(ALL16, ALL16, vpos=pos))
        print(name, "R1", {k: v["share"] for k, v in res["rows"].items()}, f"{time.time() - t0:.0f}s", flush=True)
        row("R3_SK", sets_to_arm(sink, kpos=F3))
        row("R3_NV", sets_to_arm(fz["nonsink_heads"], vpos=F3))
        row("R7_V_answer", P.arm(ALL16, ALL16, vpos=F3, qfrom=-1))
        row("R7_K_answer", P.arm(ALL16, ALL16, kpos=F3, qfrom=-1))
        row("R8_K12_hold0", P.arm(ALL16, ALL16, kpos=T12, hold0=True))
        row("R8_K12_manual", P.arm(ALL16, ALL16, kpos=T12, manual=True))
        row("mirror_V_sink", sets_to_arm(sink, vpos=F3, vmode="mirror"))
        print(name, "R3/R7/R8", {k: v["share"] for k, v in res["rows"].items() if k[:2] in ("R3", "R7", "R8", "mi")},
              f"{time.time() - t0:.0f}s", flush=True)
        for blk in (26, 27):
            for h in range(16):
                hs26, hs27 = ([h], []) if blk == 26 else ([], [h])
                row(f"R2_V_{blk}_{h:02d}", P.arm(hs26, hs27, vpos=F3))
                row(f"R2_K_{blk}_{h:02d}", P.arm(hs26, hs27, kpos=F3))
        print(name, "R2 done", f"{time.time() - t0:.0f}s", flush=True)
        g4 = np.random.default_rng(4)
        r5 = [row(f"R5_{i:02d}", sets_to_arm(sink, vpos=F3, vmode="rand"), rng=g4) for i in range(99)]
        print(name, "R5 max", round(max(r5), 4), "min", round(min(r5), 4), f"{time.time() - t0:.0f}s", flush=True)
    elif a.rows == "r4":
        r4 = [row(f"R4_{i:04d}", sets_to_arm(s, vpos=F3)) for i, s in enumerate(fz["R4_sets"])]
        print(name, "R4 n", len(r4), "max", round(max(r4), 4), f"{time.time() - t0:.0f}s", flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(fn, "w"))
    print("wrote", fn.name, res["seconds"], "s", flush=True)
