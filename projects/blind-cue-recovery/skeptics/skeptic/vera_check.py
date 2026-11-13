"""Vera (skeptic): rebuild of the frozen v2 reads from raw adapters, own code (vera_lib).

Usage: python vera_check.py [xs|sm|all]
Arms: RAW, XS (other-seed no-cue mean), SM/SMv/SM0 (4 twins each, raw/runs/<prefix>_m<i>_t<t>_d*), SS (same-seed
no-cue). BEATS CHANCE = e1(true) > (a) max of 1,000 nulls (seed 20261112) AND (b) max e1 of the same true cue on the
held-out no-cue targets nc_s6..9 through the same arm AND (c, twin arms) 95th pct of the slice-only null
(200 Gaussian Z, seed 20261113).
"""
import glob
import json
import os
import sys

import numpy as np
import torch

from vera_lib import A_rows, CUES, delta, e1, load, nulls, remainder, v1

RUNS = os.path.expanduser("~/work/theory-1113-blind-cue-recovery/kran-research/projects/blind-cue-recovery/raw/runs")
PARENT = os.path.expanduser("~/work/theory-1112-weight-only/kran-research/projects/lora-switch-audit/raw/runs")
OUT = os.path.dirname(os.path.abspath(__file__))
TH = [16, 16, 16, 16, 8, 16]
TT = [32, 32, 32, 32, 24, 32]
NA = nulls(1000, 20261112)
HELD = [6, 7, 8, 9]


def target_path(i, t):
    return f"{PARENT}/dir_u111{i}_a0.03125_s{i}/step{t}.pt" if i < 6 else f"{RUNS}/nc_s{i}/step{t}.pt"


def nc(j, t):
    return load(f"{RUNS}/nc_s{j}/step{t}.pt")


def slice_null(sd, cue, n=200, seed=20261113, P=None):
    """v1 of stacked Z_m A_m (Z Gaussian, out_m x 16) [optionally times P]; returns e1 of cue for each draw.
    Fast exact form: A_stack^T = Q R, so Z_blk A_stack = (Z_blk R^T) Q^T and v1 = Q x top right sing. vec."""
    rng = np.random.default_rng(seed)
    A = A_rows(sd)
    outs = [d.shape[0] for d in delta(sd)]
    As = torch.cat(A)  # 48 x 1024
    if P is not None:
        As = As @ P
    Q, R = torch.linalg.qr(As.T)  # 1024x48, 48x48
    vals = []
    for _ in range(n):
        Z = [torch.from_numpy(rng.standard_normal((o, 16))) for o in outs]
        Zb = torch.block_diag(*Z)  # sum(out) x 48
        y = v1(Zb @ R.T)[0]
        vals.append(float(e1(Q @ y, cue)[0]))
    return np.array(vals)


def score(target_sd, refs, cue):
    v, s = v1(remainder(target_sd, refs))
    et = float(e1(v, cue)[0])
    ea = e1(v, NA)
    return v, {"e1": et, "rank_a": int(1 + (ea >= et).sum()), "max_a": float(ea.max())}


def xs_refs(i, t):
    pool = [j for j in range(6) if j != i] if i < 6 else [1, 2, 3, 4, 5]
    return [nc(j, t) for j in pool]


def ss_refs(i, t):
    return [nc(i, t)] if i < 6 else None


def twin_refs(prefix):
    def f(i, t):
        tag = f"m{i}" if i < 6 else f"h{i}"
        paths = sorted(glob.glob(f"{RUNS}/{prefix}_{tag}_t{t}_d*/step{t}.pt"))
        if len(paths) != 4:
            return None
        return [load(p) for p in paths]
    return f


def run_arm(arm, refs_fn, reads, with_slice=False):
    res = {}
    for t_idx, tname in reads:
        for i in range(6):
            t = t_idx[i]
            sd = load(target_path(i, t))
            refs = refs_fn(i, t)
            if refs is None:
                continue
            v, r = score(sd, refs, CUES[i])
            vraw = v1(remainder(sd, []))[0]
            r["cos_v1_raw"] = abs(float(v @ vraw))
            r["resid_over_target"] = float(remainder(sd, refs).norm() / remainder(sd, []).norm())
            # held-out leg (b): the same true cue on every held-out no-cue target, same arm, same t
            hb = []
            for h in HELD:
                if not os.path.exists(target_path(h, t)):
                    continue
                hrefs = refs_fn(h, t)
                if hrefs is None:
                    continue
                vh = v1(remainder(load(target_path(h, t)), hrefs))[0]
                hb.append(float(e1(vh, CUES[i])[0]))
            r["max_b"] = max(hb) if hb else None
            r["n_b"] = len(hb)
            if with_slice:
                r["p95_c"] = float(np.quantile(slice_null(sd, CUES[i]), 0.95))
            r["beats"] = bool(r["rank_a"] == 1 and (r["max_b"] is None or r["e1"] > r["max_b"]) and
                              (not with_slice or r["e1"] > r["p95_c"]))
            res[f"{arm}|{i}|{tname}|{t}"] = r
            print(f"{arm:4s} m{i} {tname:3s} t={t:>3} e1 {r['e1']:.4f} rank_a {r['rank_a']:>4} max_a {r['max_a']:.4f} "
                  f"max_b {r['max_b'] if r['max_b'] is None else round(r['max_b'], 4)} (n {r['n_b']}) "
                  + (f"p95_c {r['p95_c']:.4f} " if with_slice else "")
                  + f"cos_raw {r['cos_v1_raw']:.3f} resid/tgt {r['resid_over_target']:.3f} beats {r['beats']}",
                  flush=True)
    return res


def e1_proj(arm, w_fn, reads):
    """E1 (exploratory): v1 of R_raw P, P = I - w w^T, w = top input dir of the reference given by w_fn(i, t)."""
    res = {}
    for t_idx, tname in reads:
        for i in range(6):
            t = t_idx[i]
            sd = load(target_path(i, t))
            w = w_fn(i, t)
            P = torch.eye(1024, dtype=torch.float64) - torch.outer(w, w)
            v = v1(remainder(sd, []) @ P)[0]
            et = float(e1(v, CUES[i])[0])
            ea = e1(v, NA)
            hb = []
            for h in HELD:
                sh = load(target_path(h, t))
                wh = v1(remainder(sh, []))[0] if arm in ("E1-SS", "E1-RAW") else w_fn(h, t)
                Ph = torch.eye(1024, dtype=torch.float64) - torch.outer(wh, wh)
                hb.append(float(e1(v1(remainder(sh, []) @ Ph)[0], CUES[i])[0]))
            sn = slice_null(sd, CUES[i], P=P)
            r = {"e1": et, "rank_a": int(1 + (ea >= et).sum()), "max_a": float(ea.max()), "max_b": max(hb),
                 "p95_c": float(np.quantile(sn, .95))}
            r["beats"] = bool(r["rank_a"] == 1 and et > r["max_b"] and et > r["p95_c"])
            res[f"{arm}|{i}|{tname}|{t}"] = r
            print(f"{arm} m{i} {tname} t={t:>3} e1 {et:.4f} rank_a {r['rank_a']:>4} max_b {r['max_b']:.4f} "
                  f"p95_c {r['p95_c']:.4f} beats {r['beats']}", flush=True)
    return res


def sm_mean_top(i, t):
    refs = twin_refs("sm")(i, t)
    return v1(sum(torch.cat(delta(r)) for r in refs) / len(refs))[0]


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "xs"
    out = {}
    if what in ("xs", "all"):
        out.update(run_arm("XS", xs_refs, [(TH, "t_h"), (TT, "T"), ([256] * 6, "256")]))
        out.update(run_arm("SS", ss_refs, [(TH, "t_h"), (TT, "T")]))
    if what in ("sm", "all"):
        for arm, prefix in (("SM", "sm"), ("SMv", "smv"), ("SM0", "sm0")):
            out.update(run_arm(arm, twin_refs(prefix), [(TH, "t_h"), (TT, "T")], with_slice=True))
    if what in ("e1", "all"):
        out.update(e1_proj("E1-SS", lambda i, t: v1(remainder(nc(i, t), []))[0], [(TH, "t_h"), (TT, "T")]))
        out.update(e1_proj("E1-SM", sm_mean_top, [(TH, "t_h"), (TT, "T")]))
        out.update(e1_proj("E1-RAW", lambda i, t: v1(remainder(load(target_path(i, t)), []))[0],
                           [(TH, "t_h"), (TT, "T")]))
    json.dump(out, open(f"{OUT}/vera_check_{what}.json", "w"), indent=1)
