"""Every frozen PREREG v2 read: arms x models x reads, the three chance lines (a, b, c), status, phase 0, the XS lemma numbers.

Arms (reference subtracted at matched update t; PREREG.md 'Arms'):
  RAW  no reference                       XS  mean of no-cue seeds {0..5} minus the model's own (held-out: seeds 1..5)
  SS   the no-cue run with the model's own seed (no held-out leg: a held-out target minus itself is 0)
  SM   mean of K twins, A := target's A_t (exact)     SMv  same, A := rowspace basis of B_t A_t     SM0  same, A := target's A_0
  SM replicates: halves (d 0,1 / d 2,3) and single twins at the status reads.
BEATS CHANCE = e1(true) > (a) max of 1,000 random cues, (b) max e1 of the same cue on held-out no-cue targets h6..h9 run
through the same arm at the same t, (c) 95th percentile of the 200-draw slice-only null.
    NC_DEV=cpu python analyze.py   -> data/blind/cells.json, data/blind/nulls.npz, data/blind/slice_null.npz,
                                      data/summary.json, proof-matrix.md; stdout = the frozen verdicts only.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("NC_DEV", "cpu")
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from blind import Scorer  # noqa: E402

MODELS = {f"m{k}": (f"dir_u{1110 + k}_a0.03125_s{k}", k) for k in range(6)}
HELD = {f"h{j}": (f"nc_s{j}", j) for j in (6, 7, 8, 9)}
T = {"m0": 32, "m1": 32, "m2": 32, "m3": 32, "m4": 24, "m5": 32}
TH = {"m0": 16, "m1": 16, "m2": 16, "m3": 16, "m4": 8, "m5": 16}
ONLY = [v for v in os.environ.get("ANALYZE_ONLY", "").split(",") if v]  # smoke mode: these variants only, output to data/smoke
RUNS = os.environ.get("RUNS_DIR", "raw/runs")  # data/adapters_qkv26 = the committed block-26 q, k, v copies (pack.py)
OUT = HERE / "data" / ("smoke" if ONLY else "rebuild_packed" if RUNS != "raw/runs" else "")
SNAPS = [8, 16, 24, 32, 48, 64, 128, 256]
K = 4
TWIN_ARMS = {"SM": "sm", "SMv": "smv", "SM0": "sm0"}


def snap(run, t):
    return f"{RUNS}/{run}/step{t}.pt"


def exists(p):
    return (HERE / p).exists()


def refs_for(arm, tgt, t, ds=None):
    """Reference snapshot list for (arm, target, t); None if the arm has no reference set for this target."""
    run, seed = {**MODELS, **HELD}[tgt]
    if arm == "RAW":
        return []
    if arm == "XS":
        pool = [j for j in range(6) if j != seed] if tgt in MODELS else [1, 2, 3, 4, 5]
        return [snap(f"nc_s{j}", t) for j in pool]
    if arm == "SS":
        return [snap(f"nc_s{seed}", t)] if tgt in MODELS else None
    pre = TWIN_ARMS[arm]
    ds = range(K) if ds is None else ds
    return [snap(f"{pre}_{tgt}_t{t}_d{seed + 100 + d}", t) for d in ds]


def behaviour(run):
    meta = json.load(open(HERE / "data" / "runs" / f"{run}.json"))
    ev = {c["step"]: c["D"] for c in meta.get("dev_curve", []) + meta.get("dense_curve", [])}
    return ev, meta.get("T")


def main():
    sc = Scorer()
    cells, nulls, slices = [], {}, {}
    variants = []  # (variant name, arm, ds or None, reads per model)
    for arm in ["RAW", "XS", "SS"]:
        variants.append((arm, arm, None, {m: SNAPS for m in MODELS}))
    variants.append(("SM", "SM", None, {m: SNAPS for m in MODELS}))  # K = all twins that exist for every target at t
    for arm in ["SMv", "SM0"]:
        variants.append((arm, arm, None, {m: sorted({TH[m], T[m]}) for m in MODELS}))
    for name, ds in [("SM_half1", [0, 1]), ("SM_half2", [2, 3])] + [(f"SM_twin{d}", [d]) for d in range(K)]:
        variants.append((name, "SM", ds, {m: sorted({TH[m], T[m]}) for m in MODELS}))

    summary = {}
    for vname, arm, ds, reads in [v for v in variants if not ONLY or v[0] in ONLY]:
        for m, (run, seed) in MODELS.items():
            for t in reads[m]:
                tgt_p = snap(run, t)
                if arm in TWIN_ARMS and ds is None:
                    # SM curve: use the twin seeds that exist for the model AND every held-out target at this t
                    use = [d for d in range(K) if all(exists(p) for p in refs_for(arm, m, t, [d]))
                           and all(exists(p) for h in HELD for p in refs_for(arm, h, t, [d]))]
                else:
                    use = ds
                refs = refs_for(arm, m, t, use)
                if refs is None or not exists(tgt_p) or not all(exists(p) for p in refs) or (arm in TWIN_ARMS and not refs):
                    continue
                row, nv = sc.score(tgt_p, refs, run)
                cid = f"{vname}|{m}|{t}"
                held = []
                for h, (hrun, _) in HELD.items():
                    hr = refs_for(arm, h, t, use)
                    if hr is None or not exists(snap(hrun, t)) or not all(exists(p) for p in hr):
                        continue
                    hrow, _ = sc.score(snap(hrun, t), hr, run)
                    held.append({"held": h, "e1": hrow["e1_true"], "rank": hrow["rank"]})
                if (m, t) not in slices:
                    slices[(m, t)] = sc.slice_null(tgt_p, run)
                sl = slices[(m, t)]
                p95 = float(np.quantile(sl, 0.95))
                hmax = max((x["e1"] for x in held), default=None)
                a = row["e1_true"] > row["null_max"]
                b = (hmax is not None) and row["e1_true"] > hmax
                c = row["e1_true"] > p95
                rec = {"id": cid, "variant": vname, "arm": arm, "model": m, "t": t, "twin_seeds": use if arm in TWIN_ARMS else None,
                       **row, "held": held, "held_max": hmax, "slice_p95": p95, "slice_median": float(np.median(sl)),
                       "beats_a": a, "beats_b": b if hmax is not None else None, "beats_c": c,
                       "beats_chance": bool(a and c and (b if arm != "SS" else True)) if (hmax is not None or arm == "SS") else None}
                cells.append(rec)
                nulls[cid] = nv
                summary.setdefault(vname, {}).setdefault(m, {})[str(t)] = {
                    k: rec[k] for k in ["e1_true", "rank", "null_max", "held_max", "slice_p95", "beats_a", "beats_b", "beats_c",
                                        "beats_chance", "cos_v1_v1raw", "true_share_in_A_rowspace", "twin_seeds", "n_refs"]}
    # E1 (PREREG.md section E1, exploratory, never changes the status): project off one direction w
    def wdir(kind, tgt, t):
        run, seed = {**MODELS, **HELD}[tgt]
        if kind == "E1-RAW":  # POST-HOC (Kasper #21800, after E1 was seen): w = the target's own raw top direction
            paths = [snap(run, t)]
        elif kind == "E1-SS":
            paths = [snap(f"nc_s{seed}", t)] if tgt in MODELS else [snap(run, t)]  # a clean adapter is its own twin
        else:
            paths = refs_for("SM", tgt, t)
        return sc.top_dir(paths) if all(exists(p) for p in paths) else None

    for kind in [k for k in ["E1-SS", "E1-SM", "E1-RAW"] if not ONLY or k in ONLY]:
        for m, (run, seed) in MODELS.items():
            for t in sorted({TH[m], T[m]}):
                w = wdir(kind, m, t)
                if w is None:
                    continue
                row, nv = sc.score_proj(snap(run, t), w, run)
                held = []
                for h, (hrun, _) in HELD.items():
                    wh = wdir(kind, h, t)
                    if wh is not None:
                        hrow, _ = sc.score_proj(snap(hrun, t), wh, run)
                        held.append({"held": h, "e1": hrow["e1_true"], "rank": hrow["rank"]})
                sl = sc.slice_null_proj(snap(run, t), w, run)
                p95 = float(np.quantile(sl, 0.95))
                hmax = max((x["e1"] for x in held), default=None)
                a, c = row["e1_true"] > row["null_max"], row["e1_true"] > p95
                b = hmax is not None and row["e1_true"] > hmax
                cid = f"{kind}|{m}|{t}"
                rec = {"id": cid, "variant": kind, "arm": kind, "model": m, "t": t, **row, "held": held, "held_max": hmax,
                       "slice_p95": p95, "beats_a": a, "beats_b": b, "beats_c": c, "beats_chance": bool(a and b and c)}
                cells.append(rec)
                nulls[cid] = nv
                slices[(f"{kind}_{m}", t)] = sl
                summary.setdefault(kind, {}).setdefault(m, {})[str(t)] = {
                    k: rec.get(k) for k in ["e1_true", "rank", "null_max", "held_max", "slice_p95", "beats_a", "beats_b",
                                            "beats_c", "beats_chance", "cos_v1_v1raw", "cos_w_v1raw"]}
    # behaviour + frozen verdicts
    beh = {}
    for m, (run, _) in MODELS.items():
        ev, Tm = behaviour(run)
        beh[m] = {"T_frozen": T[m], "T_run_json": Tm, "t_h": TH[m], "D_at": {str(t): ev.get(t) for t in [0] + SNAPS}}

    def count(vname, tsel):
        out = {}
        for m in MODELS:
            c = summary.get(vname, {}).get(m, {}).get(str(tsel(m)))
            out[m] = None if c is None else c["beats_chance"]
        return out

    sm_th = count("SM", lambda m: TH[m])
    n_sm = sum(1 for v in sm_th.values() if v)
    d_ok = all(beh[m]["D_at"][str(TH[m])] is not None and beh[m]["D_at"][str(TH[m])] < 0.10 for m in MODELS)
    complete = all(v is not None for v in sm_th.values())
    status = ("UNCONFIRMED (incomplete)" if not complete else
              "CONFIRMED" if (n_sm >= 5 and d_ok) else "REJECTED" if n_sm <= 3 else "UNCONFIRMED")
    p0 = {m: summary["RAW"][m]["256"]["rank"] for m in MODELS if "256" in summary.get("RAW", {}).get(m, {})}
    p0_n1 = sum(1 for r in p0.values() if r == 1)
    xs_T = {m: summary.get("XS", {}).get(m, {}).get(str(T[m])) for m in MODELS}
    xs_lemma = {"cos_v1raw_v1XS_at_T": {m: (c or {}).get("cos_v1_v1raw") for m, c in xs_T.items()},
                "n_cos_gt_0.9": sum(1 for c in xs_T.values() if c and c["cos_v1_v1raw"] > 0.9),
                "XS_beats_chance_at_T": count("XS", lambda m: T[m]),
                "v0_at_T_ab": {m: bool(c["beats_a"] and c["beats_b"]) for m, c in xs_T.items() if c},
                "v0_at_th_ab": {m: bool(c["beats_a"] and c["beats_b"]) for m in MODELS
                                for c in [summary.get("XS", {}).get(m, {}).get(str(TH[m]))] if c}}
    v0_T = sum(xs_lemma["v0_at_T_ab"].values())
    v0_th = sum(xs_lemma["v0_at_th_ab"].values())
    out = {"status": status, "SM_beats_chance_at_t_h": sm_th, "n_SM_at_t_h": n_sm, "all_D_at_t_h_below_0.10": d_ok,
           "phase0_RAW_rank_at_256": p0, "phase0_n_rank1": p0_n1, "phase0_reproduces": p0_n1 == 2,
           "xs_lemma": xs_lemma, "v0_prereg": {"n_at_T": v0_T, "n_at_half": v0_th,
                                                "held": bool(p0_n1 == 2 and v0_T >= 5 and v0_th >= 3)},
           "SM_at_T": count("SM", lambda m: T[m]),
           "by_variant_at_t_h": {v: count(v, lambda m: TH[m]) for v in summary},
           "by_variant_at_T": {v: count(v, lambda m: T[m]) for v in summary},
           "behaviour": beh, "cells": summary}
    od = OUT / "blind"
    od.mkdir(parents=True, exist_ok=True)
    (od / "cells.json").write_text(json.dumps(cells, indent=1))
    np.savez_compressed(od / "nulls.npz", **{k.replace("|", "__"): v for k, v in nulls.items()})
    np.savez_compressed(od / "slice_null.npz", **{f"{m}__{t}": v for (m, t), v in slices.items()})
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    # stdout: frozen verdicts only
    print("STATUS:", status, "| SM beats chance at t_h in", n_sm, "of 6", "| all D(t_h) < 0.10:", d_ok)
    print("PHASE 0: RAW rank 1 at 256 in", p0_n1, "of 6 (published 2)")
    print("XS lemma: |cos(v1 raw, v1 XS)| > 0.9 at T in", xs_lemma["n_cos_gt_0.9"], "of 6;",
          "v0 prereg: at T", v0_T, "of 6 (bar >= 5), at half", v0_th, "of 6 (bar >= 3)")
    write_matrix(out)


def fmt(c):
    if c is None:
        return "not run"
    hm = "n/a" if c["held_max"] is None else f"{c['held_max']:.4f}"
    return (f"e1 {c['e1_true']:.4f}, rank {c['rank']}/1001; (a) random max {c['null_max']:.4f}, (b) held-out max {hm}, "
            f"(c) slice p95 {c['slice_p95']:.4f} -> beats chance: {c['beats_chance']}")


def write_matrix(out):
    S = out["cells"]
    L = ["# Proof matrix: blind-cue-recovery", "",
         "Built by `analyze.py` from the adapters in `raw/runs/` (scorer `blind.py`). e1 = energy share of the true cue on the "
         "top input direction of block 26's stacked q, k, v LoRA delta minus the arm's reference; rank among 1,000 random cues "
         "(1 = best). Beats chance = above (a) the max of the 1,000 random cues, (b) the max over held-out no-cue models "
         "h6-h9 run through the same arm, and (c) the 95th percentile of the slice-only null (PREREG.md v2). Every cell's full "
         "record (held-out values, cosines, twin seeds) is in `data/blind/cells.json` under the id `variant|model|t`.", "",
         "| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
    for m, (run, seed) in MODELS.items():
        L.append(f"| headline | {seed} | SM (4-twin mean) at t_h = {TH[m]}, {run} | none | {fmt(S.get('SM', {}).get(m, {}).get(str(TH[m])))} "
                 f"| data/blind/cells.json (SM, {m}, {TH[m]}) |")
    for m, (run, seed) in MODELS.items():
        for v in ["SM_half1", "SM_half2"]:
            L.append(f"| headline replicate | {seed} | {v} at t_h = {TH[m]} | none | {fmt(S.get(v, {}).get(m, {}).get(str(TH[m])))} "
                     f"| data/blind/cells.json ({v}, {m}, {TH[m]}) |")
    for m, (run, seed) in MODELS.items():
        c = S.get("SM", {}).get(m, {}).get(str(TH[m]))
        if c:
            L.append(f"| headline | {seed} | SM at t_h = {TH[m]} | held-out no-cue models h6-h9 through SM (b) | "
                     f"max e1 of this cue on h6-h9: {c['held_max']:.4f} | data/blind/cells.json (SM, {m}, {TH[m]}), field held |")
            L.append(f"| headline | {seed} | SM at t_h = {TH[m]} | slice-only null, 200 Gaussian Z (c) | "
                     f"p95 {c['slice_p95']:.4f} | data/blind/slice_null.npz ({m}__{TH[m]}) |")
    for m, (run, seed) in MODELS.items():
        L.append(f"| SM at the switch | {seed} | SM at T = {T[m]} | none | {fmt(S.get('SM', {}).get(m, {}).get(str(T[m])))} "
                 f"| data/blind/cells.json (SM, {m}, {T[m]}) |")
    for m, (run, seed) in MODELS.items():
        c = S.get("RAW", {}).get(m, {}).get("256")
        L.append(f"| phase 0 | {seed} | RAW at 256 | none | rank {c['rank'] if c else 'not run'} (published lora-switch-audit: "
                 f"302/1/705/1/976/256) | data/blind/cells.json (RAW, {m}, 256) |")
    for m, (run, seed) in MODELS.items():
        c = S.get("XS", {}).get(m, {}).get(str(T[m]))
        L.append(f"| lemma (v0 method) | {seed} | XS at T = {T[m]} | none | "
                 f"{'cos(v1 RAW, v1 XS) %.3f; ' % c['cos_v1_v1raw'] if c else ''}{fmt(c)} | data/blind/cells.json (XS, {m}, {T[m]}) |")
    for v in ["E1-SM", "E1-SS", "E1-RAW"]:
        for m, (run, seed) in MODELS.items():
            for t in sorted({TH[m], T[m]}):
                L.append(f"| exploratory E1{' (post-hoc)' if v == 'E1-RAW' else ''} | {seed} | {v} at t = {t} | none | "
                         f"{fmt(S.get(v, {}).get(m, {}).get(str(t)))} | data/blind/cells.json ({v}, {m}, {t}) |")
    for v in ["SS", "SMv", "SM0", "RAW", "XS"]:
        for m, (run, seed) in MODELS.items():
            L.append(f"| reported | {seed} | {v} at t_h = {TH[m]} | {'same-seed twin' if v == 'SS' else 'none'} | "
                     f"{fmt(S.get(v, {}).get(m, {}).get(str(TH[m])))} | data/blind/cells.json ({v}, {m}, {TH[m]}) |")
    bci = HERE / "data" / "behaviour_ci.json"
    if bci.exists():
        B = json.loads(bci.read_text())
        for m, (run, seed) in MODELS.items():
            for t in sorted({TH[m], T[m]}):
                r = B[m]["reads"][str(t)]
                L.append(f"| before the switch bar | {seed} | behaviour gap D at t = {t} | 95% bootstrap over 64 dev prompts | "
                         f"D {100 * r['D']:.1f} points [{100 * r['ci95'][0]:.1f}, {100 * r['ci95'][1]:.1f}] (bar 10) "
                         f"| data/behaviour_ci.json ({m}, {t}) |")
    e2f, e2b = HERE / "data" / "e2.json", HERE / "data" / "behaviour_ci_e2.json"
    if e2f.exists():
        Bq = json.loads(e2b.read_text()) if e2b.exists() else {}
        for r in json.loads(e2f.read_text()):
            bd = Bq.get(r["run"], {}).get("reads", {}).get(str(r["t"]))
            beh = f"D {100 * bd['D']:.2f} [{100 * bd['ci95'][0]:.2f}, {100 * bd['ci95'][1]:.2f}] points" if bd else "D n/a"
            L.append(f"| exploratory E2 (slow switches) | 0 | {r['run']} (x {r['x']:.3f}, T {r['T']}) at {r['read']} = {r['t']} | "
                     f"clean c2_a0_s0/s1 (b), slice p95 (c) | RAW rank {r['RAW']['rank']} beats {r['RAW']['beats_chance']}; "
                     f"E1-RAW rank {r['E1-RAW']['rank']} beats {r['E1-RAW']['beats_chance']}; {beh} "
                     f"| data/e2.json, data/behaviour_ci_e2.json |")
    L += ["| equation check | 0-5 | SM at t_h and RAW at 256, recomputed by SVD apart from blind.py | known answer | "
          "see post/eq_check.json (max rel. diff, ranks equal) | post/eq_check.json |"]
    L += ["| rebuild | 0, 1 | nc_s0, nc_s1 (--no-eval) vs c2_a0_s0, c2_a0_s1 | bit-for-bit rebuild | max abs diff 0.0 over 16 snapshots "
          "| data/diag.json |",
          "| rebuild | 0, 1 | --init-from target step0, own seed | known answer | dir_u1110 and c2_a0_s1 at 16, 32: max abs diff 0.0 "
          "| data/known_answer_init_from.json |",
          "| rebuild | - | slice null 48 x 48 shortcut vs direct 1024 x 1024 | known answer | max abs diff 2.9e-17 "
          "| data/known_answer_slice_null.json |",
          "| diagnostic | 6-9 | v1 of held-out target vs v1 of its SM twin mean | no cue | see data/diag.json | data/diag.json |",
          "", f"Status (frozen rule): **{out['status']}**; SM beats chance at t_h in {out['n_SM_at_t_h']} of 6.", ""]
    L = [fix_row(x) for x in L]
    (OUT / "proof-matrix.md" if OUT != HERE / "data" else HERE / "proof-matrix.md").write_text("\n".join(L))


def fix_row(line):
    """The artifact column holds ONE path only (oracle lint); any '(variant, model, t)' locator or second file moves into
    the variant column."""
    if not line.startswith("| ") or line.startswith("| claim") or line.startswith("|---"):
        return line
    cells = line.strip().strip("|").split(" | ")
    art = cells[-1].strip()
    path, extra = art, ""
    if " (" in art:
        path, extra = art.split(" (", 1)
        extra = "(" + extra
    if ", " in path:
        path, more = path.split(", ", 1)
        extra = (extra + " also " + more).strip()
    if path.startswith("see "):
        path = path[4:]
    if extra:
        cells[2] = f"{cells[2]} {extra}"
    cells[-1] = path.strip()
    return "| " + " | ".join(c.strip() for c in cells) + " |"


if __name__ == "__main__":
    if "--matrix-only" in sys.argv:  # rewrite proof-matrix.md from data/summary.json without rescoring
        write_matrix(json.loads((OUT / "summary.json").read_text()))
    else:
        main()
