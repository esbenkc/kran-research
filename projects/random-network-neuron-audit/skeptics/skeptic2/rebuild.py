"""skeptic2 independent rebuild of the M4 audit numbers from raw data.

Own code. Does NOT import aggregate.py. Re-parses the judges' raw text, recomputes detection
truth from the seeded detection set, recomputes lexical purity, and tests arm gaps with a
unit-level permutation test plus a bootstrap.

Run (from the project dir):
  uv run python skeptics/skeptic2/rebuild.py --contexts data/contexts.jsonl \
      --judgments data/judgments.jsonl --out skeptics/skeptic2/out/rebuild.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

FIT_MIN, HITS_MIN, PURITY_MIN = 15, 4, 0.75   # frozen in PREREGISTRATION.md


def json_objects(raw: str | None) -> list[dict]:
    """All top-level balanced {...} objects in the raw judge text that parse as JSON (brace scan, own code)."""
    out, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(raw or ""):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"' and depth > 0:
            in_str, esc = True, False
        elif ch == "{":
            depth, start = depth + 1, (i if depth == 0 else start)
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                try:
                    out.append(json.loads(raw[start:i + 1]))
                except json.JSONDecodeError:
                    pass
    return out


def last_with(raw: str | None, key: str) -> dict | None:
    """The judge's final answer = the last parseable object that has `key` (haiku sometimes revises itself)."""
    objs = [o for o in json_objects(raw) if isinstance(o, dict) and key in o]
    return objs[-1] if objs else None


def my_truth(unit: dict) -> list[int]:
    """Rebuild the detection answer key with the seeding described in the prereg/judge docstring."""
    rng = random.Random(int(hashlib.sha256(unit["uid"].encode()).hexdigest()[:12], 16))
    pos = rng.sample(unit["heldout"], 5)
    neg = rng.sample(unit["random"], 5)
    items = [(1, e) for e in pos] + [(0, e) for e in neg]
    rng.shuffle(items)
    return [k + 1 for k, (t, _) in enumerate(items) if t]


def purity(examples: list[dict], lower: bool = True) -> float:
    toks = [(e["tok"].strip().lower() if lower else e["tok"]) for e in examples]
    return Counter(toks).most_common(1)[0][1] / len(toks)


def doc_diversity(examples: list[dict]) -> int:
    return len({e["doc"] for e in examples})


def verdict(rec: dict, unit: dict, strict_picks: bool = False) -> dict | None:
    lab = last_with(rec.get("label_raw"), "main_count")
    if lab is None or "main_type" not in lab:
        return None
    typ = str(lab["main_type"]).upper().strip()
    fit = int(lab["main_count"])
    hits = None
    if typ != "NONE":
        det = last_with(rec.get("det_raw"), "picks")
        picks = {int(p) for p in (det or {}).get("picks", []) if str(p).isdigit() and 1 <= int(p) <= 10}
        if det is not None and (len(picks) == 5 or (not strict_picks and len(picks) <= 5)):
            hits = len(picks & set(my_truth(unit)))
        elif strict_picks:
            return None
    one = typ != "NONE" and fit >= FIT_MIN and (hits or 0) >= HITS_MIN
    related = sum(int(o.get("count", 0)) for o in lab.get("other_jobs", []) or [] if o.get("related") is True)
    rel = typ != "NONE" and fit + related >= FIT_MIN and (hits or 0) >= HITS_MIN
    return {"rel": rel, "type": typ, "fit": fit, "hits": hits, "one": one, "nonlex": one and typ != "LEXICAL",
            "lex": one and typ == "LEXICAL", "fitonly": typ != "NONE" and fit >= FIT_MIN}


def perm_test(x: np.ndarray, y: np.ndarray, n: int = 20000, seed: int = 7) -> float:
    """One-sided p for mean(x) - mean(y) >= observed under exchangeable arm labels."""
    rng = np.random.default_rng(seed)
    obs = x.mean() - y.mean()
    z = np.concatenate([x, y])
    cnt = 0
    for _ in range(n):
        rng.shuffle(z)
        cnt += (z[: len(x)].mean() - z[len(x):].mean()) >= obs - 1e-12
    return (cnt + 1) / (n + 1)


def boot_ci(x: np.ndarray, y: np.ndarray | None = None, n: int = 20000, seed: int = 11) -> list[float]:
    rng = np.random.default_rng(seed)
    bx = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    if y is None:
        return [float(x.mean()), *map(float, np.percentile(bx, [2.5, 97.5]))]
    by = y[rng.integers(0, len(y), (n, len(y)))].mean(1)
    return [float(x.mean() - y.mean()), *map(float, np.percentile(bx - by, [2.5, 97.5]))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--judgments", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strict_picks", action="store_true", help="drop units whose detector gave != 5 picks")
    a = ap.parse_args()

    units = {u["uid"]: u for u in map(json.loads, open(a.contexts))}
    recs = defaultdict(dict)
    for r in map(json.loads, open(a.judgments)):
        recs[r["uid"]][r["judge"]] = r
    judges = sorted({j for v in recs.values() for j in v})

    truth_mismatch, rows, dropped = 0, [], Counter()
    for uid, js in recs.items():
        u = units[uid]
        for j, r in js.items():
            if r.get("truth") is not None and sorted(r["truth"]) != my_truth(u):
                truth_mismatch += 1
        vs = {j: verdict(js[j], u, a.strict_picks) for j in judges if j in js}
        if len(vs) < len(judges) or any(v is None for v in vs.values()):
            dropped[u["arm"]] += 1
            continue
        row = {"uid": uid, "arm": u["arm"], "layer": u["layer"], "purity": purity(u["examples"]),
               "purity_raw": purity(u["examples"], lower=False), "docs": doc_diversity(u["examples"])}
        for k in ["one", "nonlex", "lex", "fitonly", "rel"]:
            row[k] = float(np.mean([vs[j][k] for j in judges]))
            row[k + "_AND"] = float(all(vs[j][k] for j in judges))
            for j in judges:
                row[f"{k}|{j}"] = float(vs[j][k])
        row["types"] = [vs[j]["type"] for j in judges]
        row["hits"] = [vs[j]["hits"] for j in judges]
        rows.append(row)

    arms = sorted({r["arm"] for r in rows})
    out = {"judges": judges, "n_rows": len(rows), "dropped_per_arm": dict(dropped),
           "units_per_arm_in_contexts": dict(Counter(u["arm"] for u in units.values())),
           "detection_truth_mismatches": truth_mismatch, "rates": {}, "contrasts": {}, "judge_free": {}}
    for arm in arms:
        R = [r for r in rows if r["arm"] == arm]
        out["rates"][arm] = {k: boot_ci(np.array([r[k] for r in R])) for k in
                             ["one", "nonlex", "lex", "fitonly", "rel", "one_AND", "nonlex_AND"]
                             + [f"nonlex|{j}" for j in judges] + [f"one|{j}" for j in judges]}
        out["rates"][arm]["n"] = len(R)
        for L in sorted({r["layer"] for r in R}):
            RL = [r for r in R if r["layer"] == L]
            out["rates"][arm][f"L{L}"] = {k: float(np.mean([r[k] for r in RL])) for k in ["one", "nonlex", "lex"]}
        out["judge_free"][arm] = {
            "lexical_share_purity>=0.75": float(np.mean([r["purity"] >= PURITY_MIN for r in R])),
            "median_purity": float(np.median([r["purity"] for r in R])),
            "min_docs_in_top20": int(min(r["docs"] for r in R)),
            "judge_lexical_vs_purity_agreement": float(np.mean([
                (r["lex"] > 0) == (r["purity"] >= PURITY_MIN) for r in R if r["one"] > 0] or [float("nan")])),
        }
    for x, y in [("A", "B"), ("A", "C"), ("C", "B")]:
        if x in arms and y in arms:
            for k in ["nonlex", "one", "nonlex_AND", "rel", "one_AND"]:
                X = np.array([r[k] for r in rows if r["arm"] == x])
                Y = np.array([r[k] for r in rows if r["arm"] == y])
                out["contrasts"][f"{x}-{y}:{k}"] = {"diff_ci": boot_ci(X, Y), "perm_p_one_sided": perm_test(X, Y)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ["n_rows", "dropped_per_arm", "detection_truth_mismatches"]}))
    for arm in arms:
        r = out["rates"][arm]
        print(arm, "n", r["n"], "one %.3f" % r["one"][0], "nonlex %.3f [%.3f,%.3f]" % tuple(r["nonlex"]),
              "lex %.3f" % r["lex"][0], "rel %.3f" % r["rel"][0], "one_AND %.3f" % r["one_AND"][0], "fitonly %.3f" % r["fitonly"][0], out["judge_free"][arm])
    for k, v in out["contrasts"].items():
        print(k, "diff %.3f [%.3f,%.3f]" % tuple(v["diff_ci"]), "p=%.4g" % v["perm_p_one_sided"])


if __name__ == "__main__":
    main()
