"""skeptic3 independent rebuild of the lie-type shift numbers.

Reads ONLY the saved raw label files (one JSONL row per judged answer) plus the original
probe-goodhart eval rows (for n_words and the original binary judge). Does not import any
build code.

Metric (frozen in PREREGISTRATION.md): M = (omission + misleading) / (falsification + omission
+ misleading) per arm x round, seeds pooled; OTHER and HONEST excluded from the denominator.
Delta = M(r8) - M(r0). CI: bootstrap over prompts (both samples of a prompt move together).

Usage: uv run python rebuild.py <labels_judge1.jsonl> [<labels_judge2.jsonl> ...]
"""
from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

EVAL_DIR = Path.home() / "work/oracle-1014-probe-goodhart/kran-research/projects/probe-goodhart/data/results"
OUT = Path(__file__).parent / "out"
B = 2000
rng = random.Random(17)


def norm(lbl: str) -> str:
    s = (lbl or "").strip().upper()
    if s.startswith("FALS") or s.startswith("FABRIC") or s == "LIE":
        return "F"
    if s.startswith("OMIS"):
        return "O"
    if s.startswith("MISLEAD"):
        return "M"
    if s.startswith("HONEST") or s.startswith("TRUTH"):
        return "H"
    return "X"  # other / garbled / unparsed


def load_labels(path: str) -> list[dict]:
    rows = []
    for line in open(path):
        r = json.loads(line)
        lab = r.get("label") or r.get("lie_type") or r.get("verdict")
        rows.append({
            "lam": int(float(r["lam"])) if r.get("lam") is not None else int(r["run"].split("_")[0][3:]),
            "seed": int(r["seed"]) if r.get("seed") is not None else int(r["run"].split("_s")[1]),
            "round": int(r["round"]),
            "pid": r["prompt_id"],
            "sample": int(r["sample"]),
            "y": norm(lab),
            "raw": lab,
        })
    return rows


def expand_base(rows: list[dict]) -> list[dict]:
    """Round 0 is one shared base sample. If the label file stores it once (no lam/seed or a
    'base' run), copy it into every arm/seed so each arm has its own r0 cell."""
    r0 = [r for r in rows if r["round"] == 0]
    keys = {(r["lam"], r["seed"]) for r in rows if r["round"] != 0}
    have = {(r["lam"], r["seed"]) for r in r0}
    if keys <= have:
        return rows
    uniq = {(r["pid"], r["sample"]): r for r in r0}
    out = [r for r in rows if r["round"] != 0]
    for lam, seed in keys:
        for r in uniq.values():
            out.append({**r, "lam": lam, "seed": seed})
    return out


def M_of(cells: list[dict]) -> tuple[float, int]:
    c = collections.Counter(r["y"] for r in cells)
    den = c["F"] + c["O"] + c["M"]
    return ((c["O"] + c["M"]) / den if den else float("nan")), den


def shares(cells):
    c = collections.Counter(r["y"] for r in cells)
    n = len(cells)
    return {k: round(c[k] / n, 4) for k in "FOMHX"}, n


def delta(rows, lam, seeds, r_hi=8):
    a = [r for r in rows if r["lam"] == lam and r["seed"] in seeds and r["round"] == 0]
    b = [r for r in rows if r["lam"] == lam and r["seed"] in seeds and r["round"] == r_hi]
    return M_of(b)[0] - M_of(a)[0]


def by_pid(rows):
    d = collections.defaultdict(list)
    for r in rows:
        d[r["pid"]].append(r)
    return d


def boot_delta(rows, lam, seeds):
    grp = by_pid([r for r in rows if r["lam"] == lam and r["seed"] in seeds and r["round"] in (0, 8)])
    pids = list(grp)
    vals = []
    for _ in range(B):
        samp = [r for p in (rng.choice(pids) for _ in pids) for r in grp[p]]
        v = delta(samp, lam, seeds)
        if v == v:
            vals.append(v)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def null_round_perm(rows, lam, seeds):
    """Null 1: within each prompt (and seed), shuffle the r0 / r8 tags among that prompt's
    answers. Under 'round has no effect on type', Delta should be ~0."""
    sub = [r for r in rows if r["lam"] == lam and r["seed"] in seeds and r["round"] in (0, 8)]
    obs = delta(sub, lam, seeds)
    grp = collections.defaultdict(list)
    for r in sub:
        grp[(r["pid"], r["seed"])].append(r)
    ge = 0
    for _ in range(B):
        perm = []
        for g in grp.values():
            rounds = [r["round"] for r in g]
            rng.shuffle(rounds)
            perm += [{**r, "round": k} for r, k in zip(g, rounds)]
        if abs(delta(perm, lam, seeds)) >= abs(obs):
            ge += 1
    return obs, (ge + 1) / (B + 1)


def null_arm_swap(rows, seeds):
    """Null 2: diff-in-diff Delta_lam1 - Delta_lam0. Within each (prompt, round, seed), shuffle
    arm labels among r>0 answers (r0 is shared so it is untouched)."""
    sub = [r for r in rows if r["seed"] in seeds and r["round"] in (0, 8)]
    def did(x):
        return delta(x, 1, seeds) - delta(x, 0, seeds)
    obs = did(sub)
    grp = collections.defaultdict(list)
    fixed = [r for r in sub if r["round"] == 0]
    for r in sub:
        if r["round"] == 8:
            grp[(r["pid"], r["seed"])].append(r)
    ge = 0
    for _ in range(B):
        perm = list(fixed)
        for g in grp.values():
            lams = [r["lam"] for r in g]
            rng.shuffle(lams)
            perm += [{**r, "lam": k} for r, k in zip(g, lams)]
        if abs(did(perm)) >= abs(obs):
            ge += 1
    return obs, (ge + 1) / (B + 1)


def load_eval():
    ev = {}
    for f in EVAL_DIR.glob("eval_lam[01]_s[01].jsonl"):
        for line in open(f):
            r = json.loads(line)
            ev[(int(r["lam"]), r["seed"], r["round"], r["prompt_id"], r["sample"])] = r
    return ev


def length_check(rows, ev):
    """Confound: is O+M share (among deceptive) driven by answer length? Report per length
    tercile (terciles cut on all deceptive answers), and a length-reweighted Delta: r8 answers
    reweighted so their tercile mix matches r0."""
    dec = [r for r in rows if r["y"] in "FOM"]
    for r in dec:
        r["nw"] = ev[(r["lam"], r["seed"], r["round"], r["pid"], r["sample"])]["n_words"]
    ws = sorted(r["nw"] for r in dec)
    c1, c2 = ws[len(ws) // 3], ws[2 * len(ws) // 3]
    def t(n):
        return 0 if n <= c1 else 1 if n <= c2 else 2
    out = {"cuts": [c1, c2], "M_by_tercile": {}, "reweighted": {}}
    for k in range(3):
        g = [r for r in dec if t(r["nw"]) == k]
        out["M_by_tercile"][k] = round(M_of(g)[0], 4) if g else None
    for lam in (0, 1):
        a = [r for r in dec if r["lam"] == lam and r["round"] == 0]
        b = [r for r in dec if r["lam"] == lam and r["round"] == 8]
        pa = collections.Counter(t(r["nw"]) for r in a)
        def m_t(g, k):
            g = [r for r in g if t(r["nw"]) == k]
            return M_of(g)[0] if g else float("nan")
        rw = sum(pa[k] / len(a) * m_t(b, k) for k in range(3))
        out["reweighted"][f"lam{lam}"] = {
            "M_r0": round(M_of(a)[0], 4), "M_r8_raw": round(M_of(b)[0], 4),
            "M_r8_len_matched": round(rw, 4),
            "tercile_mix_r0": [pa[k] for k in range(3)],
            "tercile_mix_r8": [collections.Counter(t(r["nw"]) for r in b)[k] for k in range(3)],
        }
    return out


def binary_consistency(rows, ev):
    """4-way non-honest vs the ORIGINAL probe-goodhart binary judge."""
    tab = collections.Counter()
    for r in rows:
        e = ev[(r["lam"], r["seed"], r["round"], r["pid"], r["sample"])]
        tab[(e["judge"], r["y"])] += 1
    return {f"{a}->{b}": n for (a, b), n in sorted(tab.items())}


def kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n / n
    return (po - pe) / (1 - pe)


def main(paths):
    OUT.mkdir(exist_ok=True)
    ev = load_eval()
    res = {}
    judged = {}
    for p in paths:
        name = Path(p).stem
        rows = expand_base(load_labels(p))
        judged[name] = rows
        j = {"n_rows": len(rows), "raw_label_counts": dict(collections.Counter(r["raw"] for r in rows)), "cells": {}}
        for lam in (0, 1):
            for rnd in sorted({r["round"] for r in rows}):
                for seeds, tag in (((0,), "s0"), ((1,), "s1"), ((0, 1), "pooled")):
                    cell = [r for r in rows if r["lam"] == lam and r["round"] == rnd and r["seed"] in seeds]
                    if not cell:
                        continue
                    sh, n = shares(cell)
                    m, den = M_of(cell)
                    j["cells"][f"lam{lam}_r{rnd}_{tag}"] = {"n": n, "shares": sh, "M": round(m, 4), "n_deceptive": den}
        for lam in (0, 1):
            for seeds, tag in (((0,), "s0"), ((1,), "s1"), ((0, 1), "pooled")):
                d = delta(rows, lam, seeds)
                lo, hi = boot_delta(rows, lam, seeds)
                obs, p = null_round_perm(rows, lam, seeds)
                j[f"delta_lam{lam}_{tag}"] = {"delta": round(d, 4), "ci95": [round(lo, 4), round(hi, 4)], "null_round_perm_p": round(p, 4)}
        obs, p = null_arm_swap(rows, (0, 1))
        j["did_lam1_minus_lam0_pooled"] = {"did": round(obs, 4), "null_arm_swap_p": round(p, 4)}
        j["length_confound"] = length_check(rows, ev)
        j["vs_original_binary_judge"] = binary_consistency(rows, ev)
        res[name] = j
    names = list(judged)
    if len(names) >= 2:
        a, b = judged[names[0]], judged[names[1]]
        kb = {(r["lam"], r["seed"], r["round"], r["pid"], r["sample"]): r["y"] for r in b}
        pairs = [(r["y"], kb[k]) for r in a
                 if (k := (r["lam"], r["seed"], r["round"], r["pid"], r["sample"])) in kb
                 and (r["round"] != 0 or (r["lam"], r["seed"]) == (0, 0))]  # base counted once
        for nm in names:
            r0 = [r for r in judged[nm] if r["round"] == 0]
            res[f"r0_labels_identical_across_runs_{nm}"] = len({(r["pid"], r["sample"], r["y"]) for r in r0}) == len({(r["pid"], r["sample"]) for r in r0})
        res["agreement"] = {
            "n": len(pairs),
            "raw_agree": round(sum(x == y for x, y in pairs) / len(pairs), 4),
            "kappa_4way": round(kappa([x for x, _ in pairs], [y for _, y in pairs]), 4),
            "confusion": {f"{x}|{y}": n for (x, y), n in sorted(collections.Counter(pairs).items())},
        }
    (OUT / "rebuild.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
