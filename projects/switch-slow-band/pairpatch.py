"""Phase 1 per-pair key patch rows (PREREG v1 s3, v1.1 s9). One process, copies in sequence (s9.6).

    env/.venv/bin/python pairpatch.py --copies monitor_s0,monitor_s1 --prec fp32 --dev cpu --rows status
    env/.venv/bin/python pairpatch.py --copies grader_s0,grader_s1 --prec fp32 --dev cpu --rows reading

Patch = the cue-off run's keys (after k_norm and RoPE) at tokens 0, 1, 2, only in the head dims of the named RoPE
pairs (pair j = dims j, j+64), all 16 query heads of blocks 26 and 27, every query position >= 3; values never
patched. share = (P_patched - P_on) / (P_off - P_on), P = mean over 200 prompts of the X-token mass.
--rows status : G (all 64 pairs; gate vs sink-value K-only within 0.005), BAND, COMP, S00..S63, U01..U50,
                N001..N200, E01..E20, TG01..TG84, TOP3, HB (from data/prediction.json, per copy), LOW8, SLOW36, W00..W61.
--rows reading: G, BAND, COMP, S00..S63, TOP3, HB, LOW8, SLOW36 (grader copies, bf16 cells).
Output data/patch/<copy>_<prec>_<dev>.json: P_on, P_off, rows {name: {pairs, P, share}}, per_prompt {name: 200 P(X)},
top1 {name: 200 first-token argmax ids}. The file is written once per copy, after all its rows.
"""
import argparse, json, os, time
ap = argparse.ArgumentParser()
ap.add_argument("--copies", default="monitor_s0,monitor_s1")
ap.add_argument("--prec", default="fp32", choices=["fp32", "bf16"])
ap.add_argument("--dev", default="cpu")
ap.add_argument("--rows", default="status", choices=["status", "reading"])
ap.add_argument("--threads", type=int, default=6)
a = ap.parse_args()
os.environ["NC_DEV"] = a.dev
import numpy as np, torch
torch.set_num_threads(a.threads)
import pp, patch as P

HERE = pp.C.HERE
OUT = HERE / "data/patch"; OUT.mkdir(parents=True, exist_ok=True)
SETS = json.load(open(HERE / "data/sets.json"))
PRED = json.load(open(HERE / "data/prediction.json"))
BAND = SETS["BAND"]
assert BAND == [61, 62, 63]


def row_sets(copy):
    rows = {"G": list(range(64)), "BAND": BAND, "COMP": [j for j in range(64) if j not in BAND]}
    rows.update({f"S{j:02d}": [j] for j in range(64)})
    pr = PRED.get("copies", {}).get(copy, {}).get("pred", {})
    for k in ("TOP3", "HB"):
        if k in pr:
            rows[k] = pr[k]["pairs"]
    rows["LOW8"] = list(range(56, 64))
    rows["SLOW36"] = list(range(28, 64))
    if a.rows == "status":
        rows.update({f"U{i + 1:02d}": s for i, s in enumerate(SETS["U"])})
        rows.update({f"N{i + 1:03d}": s for i, s in enumerate(SETS["N"])})
        rows.update({f"E{i + 1:02d}": s for i, s in enumerate(SETS["E"])})
        rows.update({f"TG{i + 1:02d}": s for i, s in enumerate(SETS["TG"])})
        rows.update({f"W{j:02d}": [j, j + 1, j + 2] for j in range(62)})
    return rows


R = pp.Runner(a.prec)


@torch.no_grad()
def run(u16, spec):
    ps, tops = [], []
    for bi, h in R.batches(u16):
        if spec is None:
            P.CTRL.update(mode="none", spec=None, donor=None)
        else:
            P.CTRL.update(mode="patch", spec=spec, donor=R.donors[bi])
        pr = torch.softmax(R.tail(h, R.w_out).float(), -1)
        ps.append(pr[:, R.xs].sum(-1).cpu()); tops.append(pr.argmax(-1).cpu())
    P.CTRL.update(mode="none", spec=None, donor=None)
    return torch.cat(ps).double().numpy(), torch.cat(tops).numpy()


for copy in a.copies.split(","):
    fn = OUT / f"{copy}_{a.prec}_{a.dev}.json"
    if fn.exists():
        print("skip", fn.name, flush=True); continue
    t0 = time.time()
    R.load(copy)
    p_on, top_on = run(R.on, None)
    p_off, top_off = run(R.off, None)
    assert np.array_equal(p_on, R.p_on) and np.array_equal(p_off, R.p_off)
    res = {"copy": copy, "prec": a.prec, "dev": a.dev, "threads": a.threads, "rows_mode": a.rows,
           "P_on": R.P_on, "P_off": R.P_off, "rows": {},
           "per_prompt": {"on": p_on.round(7).tolist(), "off": p_off.round(7).tolist()},
           "top1": {"on": top_on.tolist(), "off": top_off.tolist()}}
    rows = row_sets(copy)
    for name, pairs in rows.items():
        p, top = run(R.on, pp.pair_arm(pairs))
        res["rows"][name] = {"pairs": pairs, "P": float(p.mean()), "share": R.share(p)}
        res["per_prompt"][name] = p.round(7).tolist()
        res["top1"][name] = top.tolist()
        if name == "G":
            ref = HERE / f"ref/sink_value_patch/{copy}_{a.prec}_{a.dev}_status.json"
            if ref.exists():
                rk = json.load(open(ref))["rows"]["K"]["share"]
                res["G_ref_sink_value_K"] = rk
                res["G_gate_pass"] = abs(R.share(p) - rk) < 0.005
                print(copy, "G", round(R.share(p), 5), "ref", rk, "pass", res["G_gate_pass"], flush=True)
                assert res["G_gate_pass"], "row G gate failed: stop (PREREG s3)"
        if name in ("BAND", "COMP"):
            print(copy, name, round(R.share(p), 5), f"{time.time() - t0:.0f}s", flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(fn, "w"))
    print("wrote", fn.name, len(rows), "rows", res["seconds"], "s", flush=True)
