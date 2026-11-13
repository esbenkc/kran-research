"""Summary from the raw files only (numpy, CPU, seconds): every run's per-row reads -> s_P, s_R with prompt-clustered
bootstrap CIs, paired end-minus-start, onset per PREREG v1, status inputs, figure data, proof-matrix.md.

Inputs: data/pool/test/cols.npz + meta.json (test pool), data/runs/<run>/reads/c_<step>.npy (current policy's cheat
probability per test-pool row), data/runs/<run>/run.json. Start read = c_0.npy of the same run (the init adapter,
read by the trainer before update 1; equal to the pool-time c_start_ref.npy up to the MPS forward).

    /usr/bin/python3 summ.py            # writes data/summary.json, figures/data.json, proof-matrix.md
"""
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
NB = 2000


class Test:
    def __init__(self):
        d = HERE / "data" / "pool" / "test"
        self.meta = json.loads((d / "meta.json").read_text())
        c = np.load(d / "cols.npz")
        var, base = c["variant"], c["base_row"]
        nb = self.meta["n_base"]
        self.by = {}
        for vi, v in enumerate(self.meta["variants"]):
            idx = np.where(var == vi)[0]
            o = np.empty(nb, dtype=np.int64)
            o[base[idx]] = idx
            self.by[v] = o
        n0 = self.by["none"]
        self.keep = ~c["x1_orig"][n0].astype(bool)
        self.w = np.exp(c["logw"][n0].astype(np.float64))
        self.prompt = c["prompt_row"][n0].astype(np.int64)
        self.up, self.pinv = np.unique(self.prompt, return_inverse=True)
        self.P = len(self.up)
        rng = np.random.default_rng(12345)
        self.R = rng.multinomial(self.P, np.full(self.P, 1.0 / self.P), size=NB).astype(np.float64)  # [NB, P] weights

    def sums(self, c):
        cn = c[self.by["none"]].astype(np.float64)
        cx = c[self.by["xins"]].astype(np.float64)
        k = self.keep.astype(np.float64)
        cols = np.stack([np.ones_like(cn), cn, k * (cn - cx), k, self.w * cn, self.w, np.clip(1 - self.w, 0, None)], 1)
        S = np.zeros((self.P, cols.shape[1]))
        np.add.at(S, self.pinv, cols)
        return S  # per-prompt sums

    @staticmethod
    def stats(T):
        """T [..., 7] totals -> dict of arrays."""
        n, scn, ssw, sk, swc, sw, s1w = (T[..., i] for i in range(7))
        co = scn / n
        sP = (ssw / sk) / co
        yR = co - swc / sw
        DR = s1w / n
        sR = yR / (DR * co)
        return {"cheat_off": co, "forced_switch": ssw / sk, "s_P": sP, "y_R": yR, "D_R": DR, "s_R": sR, "gap": sP - sR}

    def point_ci(self, c, c_ref=None):
        S = self.sums(c)
        pt = self.stats(S.sum(0))
        bs = self.stats(self.R @ S)
        out = {k: {"v": float(pt[k]), "ci": [float(np.percentile(bs[k], 2.5)), float(np.percentile(bs[k], 97.5))]} for k in pt}
        if c_ref is not None:
            S0 = self.sums(c_ref)
            p0, b0 = self.stats(S0.sum(0)), self.stats(self.R @ S0)
            for k in ("s_P", "s_R", "forced_switch"):
                d = bs[k] - b0[k]
                out[f"{k}_minus_start"] = {"v": float(pt[k] - p0[k]), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
        return out


def onset(reads, key="s_P"):
    """PREREG v1 onset: first read r with s >= 0.30 AND s > start CI-high AND the same at the next read AND
    s(r) - s(start) paired CI-low > 0. Returns (T, previous read) or (None, last read)."""
    st = reads[0]
    hi = st[key]["ci"][1]
    ok = [r["step"] > 0 and r[key]["v"] >= 0.30 and r[key]["v"] > hi and r[f"{key}_minus_start"]["ci"][0] > 0 for r in reads]
    for i in range(len(reads) - 1):
        if ok[i] and ok[i + 1]:
            return reads[i]["step"], reads[i - 1]["step"] if i else 0
    return None, reads[-1]["step"]


def rung_from_name(name):
    """Realised share of the run's probed pool from the pool's own meta (runs still in flight have no run.json)."""
    m = re.match(r"(?:d|null|joint)0(\d\d)_s(\d)", name)
    if not m:
        return None
    p = HERE / "data" / "pool" / (f"on_d0.{m.group(1)}" + ("_s1" if m.group(2) == "1" and m.group(1) == "10" else ""))
    return json.loads((p / "meta.json").read_text()).get("realised_share") if (p / "meta.json").exists() else None


def fm(x):
    return f"{x['v']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"


def main():
    te = Test()
    runs = {}
    for d in sorted((HERE / "data" / "runs").iterdir()):
        if not (d / "reads").is_dir():
            continue
        steps = sorted(int(m.group(1)) for f in (d / "reads").glob("c_*.npy") if (m := re.match(r"c_(\d+)\.npy", f.name)))
        if not steps or steps[0] != 0:
            continue
        c0 = np.load(d / "reads" / "c_0.npy")
        reads = []
        for s in steps:
            r = te.point_ci(np.load(d / "reads" / f"c_{s}.npy"), c0)
            r["step"] = s
            reads.append(r)
        rj = json.loads((d / "run.json").read_text()) if (d / "run.json").exists() else {}
        T, prev = onset(reads)
        runs[d.name] = {"reads": reads, "onset_s_P": T, "onset_interval": [prev, T] if T else None,
                        "censored_at": None if T else prev, "steps_done": rj.get("steps_done"), "args": rj.get("args"),
                        "rung_D": (rj.get("pools", {}).get("on", {}) or {}).get("realised_share") or rung_from_name(d.name)}
    out = {"test_rows": te.meta["n_base"], "test_prompts": te.P, "runs": runs}
    # PREREG v1 slope on the frequency axis: onset T ~ D^a, two-point slopes with the read-grain range; a censored
    # 0.10 rung gives the bound a <= log(T_last / T86) / log(D10 / D86).
    rung = {k: v for k, v in runs.items() if re.fullmatch(r"d0(10|30|86)_s0", k)}
    sl = {}
    if "d086_s0" in rung and rung["d086_s0"]["onset_s_P"]:
        r86 = rung["d086_s0"]
        T86, lo86 = r86["onset_s_P"], r86["onset_interval"][0]
        D86 = r86["rung_D"]
        sl["T86"] = T86
        sl["T86_interval"] = [lo86, T86]
        D10 = 0.10
        sl["predicted_T10_drift"] = T86 * (D86 / D10)
        sl["predicted_T10_noise"] = T86 * (D86 / D10) ** 2
        for k in ("d030_s0", "d010_s0"):
            if k not in rung:
                continue
            v = rung[k]
            D = v["rung_D"]
            if v["onset_s_P"]:
                T, lo = v["onset_s_P"], v["onset_interval"][0]
                ends = [np.log(max(t, 1) / max(t86, 1)) / np.log(D / D86) for t in (lo + 1, T) for t86 in (lo86 + 1, T86)]
                sl[f"slope_{k}"] = {"T": T, "interval": [lo, T], "a": float(np.log(T / T86) / np.log(D / D86)),
                                    "a_range": [float(min(ends)), float(max(ends))]}
            else:
                Tl = v["censored_at"]
                sl[f"slope_{k}"] = {"censored_at": Tl, "a_bound_le": float(np.log(Tl / T86) / np.log(D / D86))}
    out["slope"] = sl
    for f, key in (("data/aa.json", "aa"),):
        if (HERE / f).exists():
            out[key] = json.loads((HERE / f).read_text())
    out["served"] = {p.stem: json.loads(p.read_text()) for p in sorted((HERE / "data" / "reads").glob("served_*.json"))}
    sm = {}
    for arm, p in (("live_joint", HERE / "data/runs/smoke_joint/steps.jsonl"), ("live_decision", HERE / "data/runs/smoke_decision/steps.jsonl"),
                   ("cached_joint_frozen_recipe", HERE / "data/runs/joint086_frozen/steps.jsonl")):
        if p.exists():
            rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
            gd = np.array([r["g_dec_preclip"] for r in rows]); gt = np.array([r["g_thought_preclip"] for r in rows])
            sm[arm] = {"path": str(p.relative_to(HERE)), "n_updates": len(rows), "preclip_share_mean": float(np.mean(gd / (gd + gt))),
                       "preclip_share_by_update": [float(x) for x in gd / (gd + gt)],
                       "adam_share_mean": float(np.mean([r["share"] for r in rows])),
                       "norm_u_dec_mean": float(np.mean([r["norm_u_dec"] for r in rows])),
                       "g_dec_preclip": [float(x) for x in gd], "g_thought_preclip": [float(x) for x in gt],
                       "split_err_rel_max": float(max(r.get("split_err_rel", 0.0) for r in rows))}
    out["smoke"] = sm
    # PREREG v1 status rule (D 0.10 rung, seed 0, read at update 1,500) with V1.9's health rule
    if "d010_s0" in runs and (HERE / "data/runs/d010_s0/curve.jsonl").read_text().strip():
        rr = {r["step"]: r for r in runs["d010_s0"]["reads"]}
        e = rr.get(1500) or runs["d010_s0"]["reads"][-1]
        cur = [json.loads(l) for l in (HERE / "data/runs/d010_s0/curve.jsonl").read_text().splitlines() if l.strip()]
        co = [r["cheat_off"]["v"] for r in runs["d010_s0"]["reads"]]
        out_band = [not (0.15 <= x <= 0.95) for x in co]
        broken = any(a and b for a, b in zip(out_band, out_band[1:])) or min(c["live_on"] for c in cur) < 0.10 \
            or max(c["cheat_on"] for c in cur) > 0.99
        sv = [v for k, v in out["served"].items() if k.startswith("served_d010_s0")]
        st = {"read_step": e["step"], "reached_1500": e["step"] >= 1500,
              "dP_ci_low_gt_0": e["s_P_minus_start"]["ci"][0] > 0, "s_P_ge_0.30": e["s_P"]["v"] >= 0.30,
              "s_R_ci_high_lt_0.10": e["s_R"]["ci"][1] < 0.10, "gap_ci_low_gt_0": e["gap"]["ci"][0] > 0,
              "served_s_P_ci_low_gt_0": (sv[-1]["s_P_ci"][0] > 0) if sv else None,
              "NO_dP_ci_high_lt_0.10": e["s_P_minus_start"]["ci"][1] < 0.10, "NO_s_R_ci_low_gt_0.10": e["s_R"]["ci"][0] > 0.10,
              "broken_health": broken, "live_on_min": min(c["live_on"] for c in cur),
              "train_probed_cheat_max": max(c["cheat_on"] for c in cur)}
        yes = st["reached_1500"] and st["dP_ci_low_gt_0"] and st["s_P_ge_0.30"] and st["s_R_ci_high_lt_0.10"] \
            and st["gap_ci_low_gt_0"] and bool(st["served_s_P_ci_low_gt_0"]) and not broken
        no = st["reached_1500"] and (st["NO_dP_ci_high_lt_0.10"] or st["NO_s_R_ci_low_gt_0.10"]) and not broken
        st["status"] = "CONFIRMED" if yes else "REJECTED" if no else "UNCONFIRMED"
        st["prereg_clause2_s_P_ge_0.30_and_dP_ci_gt_0"] = st["s_P_ge_0.30"] and st["dP_ci_low_gt_0"]
        st["prereg_clause3_s_R_point_lt_0.03"] = e["s_R"]["v"] < 0.03
        out["status_inputs"] = st
    (HERE / "data" / "summary.json").write_text(json.dumps(out, indent=1))
    fig = {k: [{"step": r["step"], "s_P": r["s_P"]["v"], "s_P_ci": r["s_P"]["ci"], "s_R": r["s_R"]["v"], "s_R_ci": r["s_R"]["ci"]}
               for r in v["reads"]] for k, v in runs.items()}
    (HERE / "figures").mkdir(exist_ok=True)
    (HERE / "figures" / "data.json").write_text(json.dumps(fig, indent=1))
    L = ["# proof matrix — decision-only-control", "",
         "Every cell is rebuilt by `summ.py` from the raw per-row reads (`data/runs/<run>/reads/c_<step>.npy`) and the test pool "
         "(`data/pool/test/cols.npz`). CIs: prompt-clustered bootstrap, 2,000 resamples, paired with the run's own start read.", "",
         "| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
    LAB = {"d010_s0": ("headline", "none"), "d010_s1": ("headline", "none"),
           "d086_s0": ("engine control: D 0.86 rung", "positive control"), "d030_s0": ("slope rung D 0.30", "none"),
           "null086_s0": ("null", "X on 86% of probed AND unprobed rows (X predicts nothing)"),
           "joint086_s0": ("dilution test: + thought policy gradient", "none"),
           "d010_s0_frozen": ("frozen recipe (BROKEN: saturated)", "none"), "d086_frozen": ("frozen recipe, D 0.86 (BROKEN: saturated)", "positive control"),
           "screen_nostd": ("health screen, X-free training pool", "X never in training"),
           "screen_lr1e-4": ("health screen lr 1e-4, X-free training pool (stopped)", "X never in training")}
    for k, v in runs.items():
        lab, ctl = LAB.get(k, (k, "none"))
        s0, e = v["reads"][0], v["reads"][-1]
        pick = [e] if k != "d010_s0" else [r for r in v["reads"] if r["step"] == 1500] + [e]
        for r in pick:
            L.append(f"| {lab} | {(v['args'] or {}).get('seed', '?')} | {k}: rung D {v['rung_D']}, update {r['step']} | {ctl} | "
                     f"s_P {fm(r['s_P'])}; s_P minus start {fm(r['s_P_minus_start'])}; s_R {fm(r['s_R'])}; s_P - s_R {fm(r['gap'])} | "
                     f"data/runs/{k}/reads/c_{r['step']}.npy |")
        L.append(f"| {lab} | {(v['args'] or {}).get('seed', '?')} | {k}: start read (update 0) | start policy | "
                 f"s_P {fm(s0['s_P'])}; s_R {fm(s0['s_R'])}; cheat_off {s0['cheat_off']['v']:.3f} | data/runs/{k}/reads/c_0.npy |")
        L.append(f"| onset {lab} | {(v['args'] or {}).get('seed', '?')} | {k} | none | onset (PREREG rule) "
                 f"{v['onset_s_P'] if v['onset_s_P'] else 'none, censored at ' + str(v['censored_at'])} | data/runs/{k}/reads.jsonl |")
    if "aa" in out:
        aa = out["aa"]
        L.append(f"| A/A cached = live | - | 64 rows, pilot | same rows through the live recompute | h max diff {aa['h_maxabs_diff']:.2g}; "
                 f"gradient cosine {aa['grad_cosine']:.6f} | data/aa.json |")
    for k, sv in out["served"].items():
        L.append(f"| memorisation guard | - | {sv['policy']}, fresh live samples, {sv['n']} rows | none | s_P {sv['s_P']:+.3f} "
                 f"[{sv['s_P_ci'][0]:+.3f}, {sv['s_P_ci'][1]:+.3f}] | data/reads/{k}.json |")
    for arm, sv in out["smoke"].items():
        L.append(f"| prereg clause 1 (smoke) | 0 | live {arm} recipe, {sv['n_updates']} updates | none | pre-clip decision share "
                 f"{sv['preclip_share_mean']:.3f}; AdamW-step share {sv['adam_share_mean']:.3f} | {sv['path']} |")
    if out["slope"]:
        L.append(f"| slope | 0 | rungs 0.86/0.30/0.10 | none | {json.dumps(out['slope'])} | data/summary.json |")
    (HERE / "proof-matrix.md").write_text("\n".join(L) + "\n")
    for k, v in runs.items():
        e = v["reads"][-1]
        print(k, "upd", e["step"], "s_P", fm(e["s_P"]), "dP", fm(e["s_P_minus_start"]), "s_R", fm(e["s_R"]), "onset", v["onset_s_P"])


if __name__ == "__main__":
    main()
