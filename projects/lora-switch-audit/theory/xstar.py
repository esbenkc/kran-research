"""PREREG v1.3: out-of-sample predicted crossover size x* (Felix, #20562 / Tove #20563 / Kasper #20564).

Model: the stacked block-26 q,k,v delta M has a cue mode with singular value g = |M c_hat| and a generic mode
sigma_gen(t) that does not depend on the cue size. The switch needs gap ~ x * g, so g_T ~ K / x. The cue owns v1 at
the switch iff K / x > sigma_gen(T(x)).

Inputs (NO size-sweep snapshot is read here):
  K          median over the six x~33 direction runs (dir_u1110..u1115) of x * g_T, g = |M c_hat| at the T snapshot
  sigma_gen  top singular value of M after projecting the true cue rows out of its input side, per snapshot of the
             six direction runs (t = 8..256), fitted as a power law in t; the parent alpha-0 adapter (c2_a0_s0, 1,024
             updates, no cue) is printed as a check of the fit's extrapolation
  T(x)       the PARENT cue-size-law sweep (data/parent_runs/sw_*, p0_*: median T per size), log-log fit
Solve K / x = sigma_gen(T(x)) for x on a log grid. Writes theory/xstar.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import audit  # noqa: E402

DIRS = [f"dir_u{k}_a0.03125_s{k - 1110}" for k in range(1110, 1116)]
S = 32 / 16


def stacked(ad):
    Ws = []
    for mod in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
        pre = f"base_model.model.layers.0.{mod}"
        A = ad[f"{pre}.lora_A.default.weight"].double().numpy()
        B = ad[f"{pre}.lora_B.default.weight"].double().numpy()
        Ws.append(S * (B @ A))
    return np.vstack(Ws)


def main():
    torch.set_num_threads(2)
    sp = audit.Spaces()
    base = sp.base()
    out = {"runs": {}}
    sg_t, sg_v = [], []
    xg = []
    for run in DIRS:
        d = HERE / "raw" / "runs" / run
        if not (d / "args.json").exists() or not (HERE / "data" / "runs" / f"{run}.json").exists():
            print("missing", run)
            continue
        meta = {**json.load(open(HERE / "data" / "runs" / f"{run}.json")), **json.load(open(d / "args.json"))}
        spec = audit.run_spec(meta)
        tu = np.load(HERE / audit.KEYFILE[spec["key"]])[None].astype(np.float32)
        on = sp.planted(tu, spec["alpha"], spec["rows"], spec["cast"])
        D = (sp.embed(on)["in26"] - base["in26"][None]).double().numpy()[0]  # [3, 1024]
        Q, _ = np.linalg.qr(D.T)  # 1024 x 3 basis of the cue rows
        T = meta["T"]
        x = meta["cue_size"]["x"]
        rr = {"x": x, "T": T, "snaps": {}}
        for f in sorted(d.glob("step*.pt"), key=lambda f: int(f.stem[4:])):
            st = int(f.stem[4:])
            if st == 0:
                continue
            M = stacked(torch.load(f, map_location="cpu"))
            g = float(np.sqrt(((M @ D.T) ** 2).sum() / (D ** 2).sum()))
            Mp = M - (M @ Q) @ Q.T
            s1 = float(np.linalg.svd(M, compute_uv=False)[0])
            sgen = float(np.linalg.svd(Mp, compute_uv=False)[0])
            rr["snaps"][st] = {"g": g, "sigma1": s1, "sigma_gen": sgen}
            sg_t.append(st)
            sg_v.append(sgen)
        # T snapshot: the run's own T if saved, else the nearest snapshot >= T
        sts = sorted(rr["snaps"])
        tT = min([s for s in sts if s >= T], default=sts[-1])
        rr["T_snap"] = tT
        xg.append(x * rr["snaps"][tT]["g"])
        out["runs"][run] = rr
        print(run, "x", round(x, 2), "T", T, "snapT", tT, "g_T", round(rr["snaps"][tT]["g"], 4),
              "sigma_gen_T", round(rr["snaps"][tT]["sigma_gen"], 4), flush=True)
    K = float(np.median(xg))
    lt, lv = np.log(sg_t), np.log(sg_v)
    b, a = np.polyfit(lt, lv, 1)
    sgen = lambda t: float(np.exp(a + b * np.log(t)))  # noqa: E731
    # parent alpha-0 check
    pa = torch.load(HERE / "raw" / "parent_runs" / "c2_a0_s0_adapter.pt", map_location="cpu")
    s_a0 = float(np.linalg.svd(stacked(pa), compute_uv=False)[0])
    # T(x) from the parent sweep
    # parent cue-size-law summary (13 planned sweep runs, x and T), copied as data/parent_summary_runs.json
    ps = json.load(open(HERE / "data" / "parent_summary_runs.json"))
    px = [r["x"] for r in ps if r.get("T") is not None]
    pT = [r["T"] for r in ps if r.get("T") is not None]
    px, pT = np.array(px), np.array(pT, float)
    tb, ta = np.polyfit(np.log(px), np.log(pT), 1)
    Tx = lambda x: float(np.exp(ta + tb * np.log(x)))  # noqa: E731
    grid = np.exp(np.linspace(np.log(0.01), np.log(100), 4001))
    f = np.array([K / x - sgen(Tx(x)) for x in grid])
    cross = [float(grid[i]) for i in range(len(grid) - 1) if np.sign(f[i]) != np.sign(f[i + 1])]
    # PRIMARY per v1.2 text (sigma_gen from the alpha-0 model): the parent alpha-0 adapter's sigma_1 at 1,024
    # updates, carried to other t with the exponent b measured on the direction runs' non-cue mode
    sa0 = lambda t: s_a0 * (t / 1024.0) ** b  # noqa: E731
    f0 = np.array([K / x - sa0(Tx(x)) for x in grid])
    cross_a0 = [float(grid[i]) for i in range(len(grid) - 1) if np.sign(f0[i]) != np.sign(f0[i + 1])]
    out.update({"K": K, "x_times_gT": xg, "sigma_gen_fit": {"log_a": float(a), "b": float(b), "n": len(sg_t)},
                "sigma_gen_at_1024_fit": sgen(1024), "sigma1_parent_c2_a0_1024": s_a0,
                "T_of_x_fit": {"log_a": float(ta), "slope": float(tb), "n": int(len(px))},
                "x_star_pred_PRIMARY_alpha0": cross_a0, "x_star_pred_secondary_dirdrift": cross, "rule": "cue owns v1 at T iff x < x_star; miss if x_star off by > 2x from the observed flip"})
    (HERE / "theory" / "xstar.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in out if k != "runs"}, indent=1))


if __name__ == "__main__":
    main()
