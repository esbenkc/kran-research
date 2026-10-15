"""skeptic3 independent rebuild of the key numbers from raw data (own code, does not import aggregate.py).

Inputs (project dir):
  data/contexts.jsonl   one line per unit (examples/heldout/random excerpts)
  data/judgments.jsonl  one line per (unit, label judge)
  skeptics/skeptic3/out/mix2_judged.jsonl (optional) MIX-2 positive control
Outputs: skeptics/skeptic3/out/rebuild_<PARSE>.json (PARSE = pipeline | last | first) and a printed table.

Definitions (from PREREGISTRATION.md, re-implemented here):
  one_job(judge)   := label parsed AND main_type != NONE AND main_count >= 15 AND detection hits >= 4
  nonlex(judge)    := one_job AND main_type != LEXICAL
  unit rate        := mean over the 2 judges (a judge with no usable output counts as 'not one-job')
  arm rate         := mean of unit rates; 95% CI by bootstrap over units (10,000 resamples, seed 0)
  H1: nonlex(A) - nonlex(B) >= 0.25 ; H2: nonlex(A) - nonlex(C) >= 0.15
  H3: anytype(B) >= 0.40 and most of B's one-job is lexical ; H4: 0.50 <= anytype(A) <= 0.90
  lexical purity := share of top-20 whose marked-token string equals the modal marked-token string; lexical := >= 0.75
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

P = Path(os.environ.get("PROJ", Path(__file__).resolve().parents[2]))
OUT = Path(os.environ.get("SK3_OUT", Path(__file__).resolve().parent / "out"))
RNG = np.random.default_rng(0)
NBOOT = 10_000


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p)] if p.exists() else []


def json_objects(raw: str | None) -> list[dict]:
    """All top-level {...} objects in raw text that parse as JSON (own brace scanner, string-aware)."""
    out, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(raw or ""):
        if in_str:
            esc = (ch == "\\") and not esc
            if ch == '"' and not esc:
                in_str = False
            continue
        if ch == '"' and depth > 0:
            in_str = True
        elif ch == "{":
            start = i if depth == 0 else start
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                try:
                    out.append(json.loads(raw[start:i + 1]))
                except json.JSONDecodeError:
                    pass
    return out


def own_parse(r: dict, mode: str) -> tuple[dict, int | None]:
    """Re-parse label and detection from raw text. mode: 'last' or 'first' JSON object."""
    labs = [o for o in json_objects(r.get("label_raw")) if "main_count" in o and "main_type" in o]
    pr = (labs[-1] if mode == "last" else labs[0]) if labs else {}
    hits = None
    if r.get("truth"):
        dets = [o for o in json_objects(r.get("det_raw")) if isinstance(o.get("picks"), list)]
        if dets:
            picks = [int(x) for x in (dets[-1] if mode == "last" else dets[0])["picks"] if str(x).lstrip("-").isdigit()]
            hits = len(set(picks[:5]) & set(r["truth"]))
    try:
        pr = dict(main_type=str(pr["main_type"]).upper().strip(), main_count=int(pr["main_count"]),
                  main_label=pr.get("main_label"))
    except (KeyError, TypeError, ValueError):
        pr = {}
    return pr, hits


def verdicts(rows: list[dict], mode: str = os.environ.get("PARSE", "pipeline")) -> dict[str, dict[str, dict]]:
    """uid -> judge -> {one, nonlex, type}. mode 'pipeline' uses the stored parse; 'last'/'first' re-parse raw text.
    Note: in re-parse modes, units whose stored label was NONE never got a detection call, so hits stay None."""
    v: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        if mode == "pipeline":
            pr, hits, ok0 = r.get("parsed") or {}, r.get("hits"), bool(r.get("label_ok"))
        else:
            pr, hits = own_parse(r, mode)
            ok0 = bool(pr)
        ok = ok0 and pr.get("main_type") not in (None, "NONE")
        one = ok and int(pr.get("main_count", 0)) >= 15 and (hits or 0) >= 4
        fit_only = ok and int(pr.get("main_count", 0)) >= 15
        v[r["uid"]][r["judge"]] = dict(one=one, nonlex=one and pr.get("main_type") != "LEXICAL",
                                       fit_only=fit_only, type=pr.get("main_type"), hits=hits,
                                       label=pr.get("main_label"))
    return v


def unit_scores(v, uids, judges, key):
    return np.array([np.mean([float(v.get(u, {}).get(j, {}).get(key, False)) for j in judges]) for u in uids])


def boot_ci(x: np.ndarray) -> tuple[float, float]:
    if len(x) == 0:
        return (float("nan"), float("nan"))
    m = RNG.choice(x, size=(NBOOT, len(x)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_gap(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mx = RNG.choice(x, size=(NBOOT, len(x)), replace=True).mean(1)
    my = RNG.choice(y, size=(NBOOT, len(y)), replace=True).mean(1)
    d = mx - my
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def perm_p(x: np.ndarray, y: np.ndarray, n: int = 10_000) -> float:
    """one-sided permutation p for mean(x) - mean(y) >= observed"""
    obs = x.mean() - y.mean()
    z = np.concatenate([x, y])
    c = 0
    for _ in range(n):
        RNG.shuffle(z)
        c += (z[: len(x)].mean() - z[len(x):].mean()) >= obs - 1e-12
    return (c + 1) / (n + 1)


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = np.mean(a == b)
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def purity(exs: list[dict]) -> float:
    toks = [e["tok"] for e in exs]
    return Counter(toks).most_common(1)[0][1] / len(toks)


def integrity(units: list[dict]) -> dict:
    """Checks on contexts.jsonl that do not need the model."""
    bad_n = sum(len(u["examples"]) != 20 for u in units)
    bad_sorted = sum(any(u["examples"][i]["act"] < u["examples"][i + 1]["act"] - 1e-6 for i in range(19)) for u in units)
    out = dict(n_units=len(units), units_not_20_examples=bad_n, units_top20_not_descending=bad_sorted)
    if units and "doc" in units[0]["examples"][0]:
        out["units_with_repeated_doc_in_top20"] = sum(len({e["doc"] for e in u["examples"]}) < 20 for u in units)
        out["units_heldout_doc_overlaps_top20"] = sum(
            bool({e["doc"] for e in u["heldout"]} & {e["doc"] for e in u["examples"]}) for u in units)
        out["units_random_doc_overlaps_top40"] = sum(
            bool({e["doc"] for e in u["random"]} & {e["doc"] for e in u["examples"] + u["heldout"]}) for u in units)
    if units and "pos" in units[0]["examples"][0]:
        out["excerpts_with_pos_lt_4"] = sum(e["pos"] < 4 for u in units for e in u["examples"] + u["heldout"])
    hl = [min(e["act"] for e in u["examples"]) >= max(e["act"] for e in u["heldout"]) - 1e-6 for u in units if u["heldout"]]
    out["units_heldout_below_top20"] = int(sum(hl))
    out["units_checked_heldout"] = len(hl)
    out["min_top_act_by_arm"] = {a: float(min(u["examples"][19]["act"] for u in units if u["arm"] == a))
                                 for a in sorted({u["arm"] for u in units})}
    return out


def main() -> None:
    units = load(P / "data" / "contexts.jsonl")
    rows = load(P / "data" / "judgments.jsonl")
    if not units or not rows:
        sys.exit("missing data/contexts.jsonl or data/judgments.jsonl")
    judges = sorted({r["judge"] for r in rows})
    v = verdicts(rows)
    by_arm = defaultdict(list)
    for u in units:
        by_arm[u["arm"]].append(u)
    res: dict = dict(judges=judges, integrity=integrity(units), arms={}, layers={})
    S = {}
    for arm, us in sorted(by_arm.items()):
        uids = [u["uid"] for u in us]
        d = {}
        for key in ("one", "nonlex", "fit_only"):
            x = unit_scores(v, uids, judges, key)
            S[(arm, key)] = x
            d[key] = dict(rate=float(x.mean()), ci=boot_ci(x))
            for j in judges:
                d[f"{key}_{j}"] = float(unit_scores(v, uids, [j], key).mean())
            both = np.array([all(v.get(u, {}).get(j, {}).get(key, False) for j in judges) for u in uids], float)
            d[f"{key}_AND"] = float(both.mean())
        a0 = np.array([v.get(u, {}).get(judges[0], {}).get("one", False) for u in uids], float)
        a1 = np.array([v.get(u, {}).get(judges[1], {}).get("one", False) for u in uids], float)
        d["kappa_one"] = kappa(a0, a1)
        types = Counter(v[u][j]["type"] for u in uids for j in judges if v.get(u, {}).get(j, {}).get("one"))
        d["one_job_types"] = dict(types)
        pur = np.array([purity(u["examples"]) for u in us])
        d["lexical_purity_mean"] = float(pur.mean())
        d["share_lexical_by_purity"] = float((pur >= 0.75).mean())
        d["n_units"] = len(us)
        d["n_judgments"] = sum(1 for u in uids for j in judges if j in v.get(u, {}))
        res["arms"][arm] = d
        for L in sorted({u["layer"] for u in us}):
            lu = [u["uid"] for u in us if u["layer"] == L]
            res["layers"][f"{arm}{L}"] = dict(one=float(unit_scores(v, lu, judges, "one").mean()),
                                              nonlex=float(unit_scores(v, lu, judges, "nonlex").mean()), n=len(lu))
    A, B, C = (S.get((a, "nonlex")) for a in "ABC")
    if A is not None and B is not None:
        res["H1"] = dict(gap=float(A.mean() - B.mean()), ci=boot_gap(A, B), perm_p=perm_p(A, B),
                         holds=bool(A.mean() - B.mean() >= 0.25))
    if A is not None and C is not None:
        res["H2"] = dict(gap=float(A.mean() - C.mean()), ci=boot_gap(A, C), perm_p=perm_p(A, C),
                         holds=bool(A.mean() - C.mean() >= 0.15))
    if "B" in res["arms"]:
        b = res["arms"]["B"]
        t = b["one_job_types"]
        res["H3"] = dict(anyB=b["one"]["rate"], lexical_share_of_B_onejob=t.get("LEXICAL", 0) / max(1, sum(t.values())),
                         holds=bool(b["one"]["rate"] >= 0.40 and t.get("LEXICAL", 0) > sum(t.values()) / 2))
    if "A" in res["arms"]:
        a = res["arms"]["A"]["one"]["rate"]
        res["H4"] = dict(anyA=a, holds=bool(0.50 <= a <= 0.90))

    mix = load(OUT / "mix2_judged.jsonl")
    if mix:
        mv = verdicts(mix)
        mu = sorted(mv)
        sens = 1 - unit_scores(mv, mu, judges, "one")
        res["MIX2"] = dict(n=len(mu), sensitivity=float(sens.mean()), ci=boot_ci(sens),
                           sensitivity_fit_only=float(1 - unit_scores(mv, mu, judges, "fit_only").mean()),
                           by_arm={a: float(np.mean([s for u, s in zip(mu, sens) if mix_arm(mix, u) == a]))
                                   for a in sorted({r["arm"] for r in mix})},
                           prereg_holds=bool(sens.mean() >= 0.8))
    res["parse_mode"] = os.environ.get("PARSE", "pipeline")
    json.dump(res, open(OUT / f"rebuild_{res['parse_mode']}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def mix_arm(rows, uid):
    for r in rows:
        if r["uid"] == uid:
            return r["arm"]


if __name__ == "__main__":
    main()
