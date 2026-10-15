"""Aggregate judgments.jsonl -> per-unit table + results summary (prereg PREREGISTRATION.md definitions).

Per judge j and unit u:
  fit_ok_j     : label parsed, main_type != NONE, main_count >= 15
  det_ok_j     : the other family's detection hits on j's label >= 4 of 5
  one_job_j    : fit_ok_j and det_ok_j                                   (prereg "one main job")
  nonlex_j     : one_job_j and main_type_j != LEXICAL                    (H1/H2 quantity)
  lex_j        : one_job_j and main_type_j == LEXICAL
  fitonly_j    : fit_ok_j  (no detection gate; shows how much the gate removes)
  rel_j        : main_count + related offshoots >= 15, type != NONE, det_ok_j  (closest to the essay's wording)
Unit score (primary) = mean over the two judges of the per-judge verdict (0, 0.5 or 1). Also: AND of both, each judge.
Judge-free anchor: lexical purity = share of top-20 whose max-token string equals the modal one; lexical := purity >= 0.75.
CIs: 95% bootstrap over units (10k). Contrasts: bootstrap of the difference of arm means.

Run: uv run python aggregate.py --judgments data/judgments.jsonl --contexts data/contexts.jsonl --out results.json --table data/unit_table.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict

import numpy as np

METRICS = ["one_job", "nonlex", "lex", "fitonly", "rel"]
TYPES = ["LEXICAL", "SYNTACTIC", "SEMANTIC", "NONE"]


def verdicts(d: dict) -> dict:
    p = d["parsed"] if d["label_ok"] else None
    if p is None:
        return None
    typ, mc = p["main_type"], p["main_count"]
    rel = sum(o["count"] for o in p["other_jobs"] if o["related"])
    fit_ok = typ != "NONE" and mc >= 15
    det_ok = bool(d["det_ok"]) and (d["hits"] or 0) >= 4
    one = fit_ok and det_ok
    return {"one_job": one, "nonlex": one and typ != "LEXICAL", "lex": one and typ == "LEXICAL", "fitonly": fit_ok,
            "rel": typ != "NONE" and (mc + rel) >= 15 and det_ok,
            "type": typ, "fit": mc, "hits": d["hits"], "label": p["main_label"]}


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def boot(x: np.ndarray, n: int = 10000, seed: int = 0) -> list[float]:
    if len(x) == 0:
        return [float("nan")] * 3
    bs = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def boot_diff(x: np.ndarray, y: np.ndarray, n: int = 10000, seed: int = 1) -> list[float]:
    rng = np.random.default_rng(seed)
    d = rng.choice(x, (n, len(x))).mean(1) - rng.choice(y, (n, len(y))).mean(1)
    return [float(x.mean() - y.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def lexical_purity(examples: list[dict]) -> float:
    toks = [e["tok"].strip().lower() for e in examples]
    return Counter(toks).most_common(1)[0][1] / len(toks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments", required=True)
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--table", default="unit_table.csv")
    a = ap.parse_args()

    ctx = {}
    for l in open(a.contexts):
        u = json.loads(l)
        regs = [e.get("regime") for e in u["examples"]]
        ctx[u["uid"]] = {"arm": u["arm"], "layer": u["layer"], "unit": u["unit"],
                         "purity": lexical_purity(u["examples"]),
                         "frac_negneg": (sum(r == "--" for r in regs) / len(regs)) if any(regs) else None}

    by = defaultdict(dict)
    for l in open(a.judgments):
        d = json.loads(l)
        by[d["uid"]][d["judge"]] = d
    judges = sorted({j for v in by.values() for j in v})

    rows, dropped = [], []
    for uid, js in by.items():
        v = {j: verdicts(js[j]) for j in judges if j in js}
        if len(v) < len(judges) or any(x is None for x in v.values()):
            dropped.append(uid)
            continue
        row = {"uid": uid, **ctx[uid], "lexical_judgefree": ctx[uid]["purity"] >= 0.75}
        for j in judges:
            for k in METRICS + ["type", "fit", "hits", "label"]:
                row[f"{k}__{j}"] = v[j][k]
        for k in METRICS:
            row[k] = float(np.mean([v[j][k] for j in judges]))          # primary: mean of judges
            row[f"{k}__AND"] = float(all(v[j][k] for j in judges))
        row["type_agree"] = len({v[j]["type"] for j in judges}) == 1
        rows.append(row)

    with open(a.table, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    arms = sorted({r["arm"] for r in rows})
    layers = sorted({r["layer"] for r in rows})
    out = {"definitions": __doc__, "judges": judges, "n_units_judged": len(by), "n_units_valid": len(rows),
           "dropped_uids": dropped, "rates": {}, "contrasts": {}, "agreement": {}}
    for arm in arms:
        for layer in layers + ["all"]:
            sel = [r for r in rows if r["arm"] == arm and (layer == "all" or r["layer"] == layer)]
            key = f"{arm}/L{layer}"
            e = out["rates"][key] = {"n": len(sel)}
            for k in METRICS:
                e[k] = boot(np.array([r[k] for r in sel]))
                e[f"{k}__AND"] = boot(np.array([r[f"{k}__AND"] for r in sel]))
                for j in judges:
                    e[f"{k}__{j}"] = boot(np.array([float(r[f"{k}__{j}"]) for r in sel]))
            for j in judges:
                e[f"type_mix__{j}"] = {t: sum(r[f"type__{j}"] == t for r in sel) for t in TYPES}
                e[f"mean_fit__{j}"] = float(np.mean([r[f"fit__{j}"] for r in sel]))
                hs = [r[f"hits__{j}"] for r in sel if r[f"hits__{j}"] is not None]
                e[f"mean_hits__{j}"] = float(np.mean(hs)) if hs else None
            e["lexical_judgefree"] = boot(np.array([float(r["lexical_judgefree"]) for r in sel]))
            e["mean_purity"] = float(np.mean([r["purity"] for r in sel]))
            fn = [r["frac_negneg"] for r in sel if r["frac_negneg"] is not None]
            e["mean_frac_negneg_top20"] = float(np.mean(fn)) if fn else None

    def arr(arm, k):
        return np.array([r[k] for r in rows if r["arm"] == arm], float)

    for x, y in [("A", "B"), ("A", "C"), ("C", "B")]:
        if x in arms and y in arms:
            for k in METRICS:
                out["contrasts"][f"{x}-{y}/{k}"] = boot_diff(arr(x, k), arr(y, k))
                out["contrasts"][f"{x}-{y}/{k}__AND"] = boot_diff(arr(x, f"{k}__AND"), arr(y, f"{k}__AND"))
    out["hypotheses"] = {}
    if {"A", "B"} <= set(arms):
        out["hypotheses"]["H1_nonlex_A_minus_B_ge_25pp"] = out["contrasts"]["A-B/nonlex"][0] >= 0.25
    if {"A", "C"} <= set(arms):
        out["hypotheses"]["H2_nonlex_A_minus_C_ge_15pp"] = out["contrasts"]["A-C/nonlex"][0] >= 0.15
    if "B" in arms:
        b = out["rates"]["B/Lall"]
        out["hypotheses"]["H3_B_onejob_ge_40pct_mostly_lexical"] = b["one_job"][0] >= 0.40 and b["lex"][0] > b["nonlex"][0]
    if "A" in arms:
        out["hypotheses"]["H4_A_onejob_in_50_90"] = 0.50 <= out["rates"]["A/Lall"]["one_job"][0] <= 0.90

    if len(judges) == 2:
        j1, j2 = judges
        for k in METRICS:
            a1 = np.array([float(r[f"{k}__{j1}"]) for r in rows])
            a2 = np.array([float(r[f"{k}__{j2}"]) for r in rows])
            out["agreement"][k] = {"raw": float((a1 == a2).mean()), "kappa": kappa(a1, a2)}
        out["agreement"]["type_exact"] = float(np.mean([r["type_agree"] for r in rows]))

    json.dump(out, open(a.out, "w"), indent=1)
    for key, v in out["rates"].items():
        if key.endswith("/Lall"):
            print(key, "n=", v["n"], {k: round(v[k][0], 3) for k in METRICS})
    for k, v in out["contrasts"].items():
        if "__AND" not in k:
            print(k, [round(x, 3) for x in v])
    print("hypotheses", out["hypotheses"])
    print("agreement", json.dumps(out["agreement"]))


if __name__ == "__main__":
    main()
