"""Analyse the blind elegance scores: calibration, judge agreement, human vs AI, controls.

    uv run python projects/proof-elegance/run.py

Reads data/scores/<judge>/*.json (from score.py) and data/corpus.jsonl. Uses the PRIMARY corpus only
(36 AI / 36 human, field-matched). The per-item score is the mean of the two judges.
Writes results.json and the main figure (figures/elegance-distribution).
"""
from __future__ import annotations

import json
import math
import unicodedata
from pathlib import Path

import numpy as np
from scipy import stats

from lib import figure

P = Path(__file__).parent
DATA = P / "data"
JUDGES = ["claude", "gemini"]
DIMS = ["economy", "surprise", "explanation", "simplicity", "conceptual", "unification"]
RNG = np.random.default_rng(20261008)


def fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def authors_by_url() -> dict[str, list[str]]:
    sel = json.loads((DATA / "selection.json").read_text())
    return {it["url"]: it.get("authors", []) for it in sel["human"]}


def recognizes(text: str, authors: list[str]) -> bool:
    """A judge 'recognises' a human proof only if it names one of the paper's real authors.
    (Naming the problem is not recognition: the AI items address named problems too.)"""
    t = fold(text or "")
    return any(len(sn) > 2 and sn in t for sn in (fold(a.split()[-1]) for a in authors))


def load() -> tuple[list[dict], list[dict]]:
    corpus = {r["id"]: r for r in map(json.loads, (DATA / "corpus.jsonl").read_text().splitlines()) if r}
    anchors = {a["id"]: a for a in json.loads((P / "lit" / "anchors.json").read_text())}
    authors = authors_by_url()
    scores: dict[str, dict] = {}
    for j in JUDGES:
        for f in (DATA / "scores" / j).glob("*.json"):
            s = json.loads(f.read_text())
            scores.setdefault(s["id"], {})[j] = s
    items, anch = [], []
    for iid, by in scores.items():
        if set(by) != set(JUDGES):
            continue  # need both judges
        row = {
            "id": iid,
            "elegance": float(np.mean([by[j]["elegance"] for j in JUDGES])),
            **{f"elegance_{j}": float(by[j]["elegance"]) for j in JUDGES},
            **{d: float(np.mean([by[j][d] for j in JUDGES])) for d in DIMS},
            "recognized": {j: recognizes(by[j].get("recognized"), authors.get(corpus.get(iid, {}).get("url"), []))
                           for j in JUDGES},
            "guess": {j: by[j].get("source_guess") for j in JUDGES},
            "key_idea": by["claude"].get("key_idea"),
        }
        if iid in anchors:
            anch.append({**row, "source": "anchor", "kind": anchors[iid]["kind"], "title": anchors[iid]["title"],
                         "url": anchors[iid].get("source_url")})
        elif iid in corpus and corpus[iid]["role"] == "primary":
            c = corpus[iid]
            items.append({**row, "source": c["source"], "field": c["field"], "title": c["title"],
                          "problem": c.get("problem_name"), "full_proof_chars": c["full_proof_chars"],
                          "truncated": c["truncated"], "url": c["url"]})
    return items, anch


def boot_diff(a: np.ndarray, b: np.ndarray, n: int = 10000) -> tuple[float, float]:
    d = [RNG.choice(a, len(a)).mean() - RNG.choice(b, len(b)).mean() for _ in range(n)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def field_permutation(items: list[dict], n: int = 10000) -> float:
    """Two-sided p for mean(AI) - mean(human), permuting the source label WITHIN each field
    (the design is field-matched, so this is the honest null)."""
    fields = sorted({r["field"] for r in items})
    by = {f: [r for r in items if r["field"] == f] for f in fields}
    obs = np.mean([r["elegance"] for r in items if r["source"] == "ai"]) - np.mean([r["elegance"] for r in items if r["source"] == "human"])
    hits = 0
    for _ in range(n):
        ai, hu = [], []
        for f in fields:
            vals = np.array([r["elegance"] for r in by[f]])
            lab = RNG.permutation([r["source"] for r in by[f]])
            ai += list(vals[lab == "ai"]); hu += list(vals[lab == "human"])
        hits += abs(np.mean(ai) - np.mean(hu)) >= abs(obs) - 1e-12
    return (hits + 1) / (n + 1)


def compare(items: list[dict], key: str = "elegance") -> dict:
    a = np.array([r[key] for r in items if r["source"] == "ai"])
    h = np.array([r[key] for r in items if r["source"] == "human"])
    lo, hi = boot_diff(a, h)
    u = stats.mannwhitneyu(a, h, alternative="two-sided")
    return {
        "n_ai": len(a), "n_human": len(h),
        "mean_ai": float(a.mean()), "mean_human": float(h.mean()),
        "median_ai": float(np.median(a)), "median_human": float(np.median(h)),
        "sd_ai": float(a.std(ddof=1)), "sd_human": float(h.std(ddof=1)),
        "diff_ai_minus_human": float(a.mean() - h.mean()), "diff_ci95": [lo, hi],
        "mannwhitney_p": float(u.pvalue),
        "share_le4_ai": float((a <= 4).mean()), "share_le4_human": float((h <= 4).mean()),
        "share_ge7_ai": float((a >= 7).mean()), "share_ge7_human": float((h >= 7).mean()),
        # messy tail: proofs the judges place in the 'idea buried / brute force' bands (<= 4)
        "tail_le4_fisher_p": float(stats.fisher_exact([[(a <= 4).sum(), (a > 4).sum()], [(h <= 4).sum(), (h > 4).sum()]]).pvalue),
    }


def main() -> None:
    items, anch = load()
    res: dict = {"n_items": len(items), "n_anchors": len(anch)}

    # 1. Calibration: do the judges separate textbook-elegant from textbook-messy proofs?
    el = [r["elegance"] for r in anch if r["kind"] == "elegant"]
    me = [r["elegance"] for r in anch if r["kind"] == "messy"]
    res["calibration"] = {
        "elegant_mean": float(np.mean(el)), "messy_mean": float(np.mean(me)),
        "elegant_min": float(min(el)), "messy_max": float(max(me)),
        "separated": bool(min(el) > max(me)),
        "anchors": sorted(({"title": r["title"], "kind": r["kind"], "elegance": r["elegance"],
                            **{f"elegance_{j}": r[f"elegance_{j}"] for j in JUDGES}} for r in anch),
                          key=lambda r: -r["elegance"]),
    }

    # 2. Judge agreement on the corpus.
    c = [r["elegance_claude"] for r in items]; g = [r["elegance_gemini"] for r in items]
    res["agreement"] = {
        "spearman": float(stats.spearmanr(c, g).statistic),
        "pearson": float(stats.pearsonr(c, g).statistic),
        "mean_claude": float(np.mean(c)), "mean_gemini": float(np.mean(g)),
        "mean_abs_diff": float(np.mean(np.abs(np.array(c) - np.array(g)))),
    }

    # 3. Main comparison, per judge and pooled.
    res["main"] = compare(items)
    res["main"]["field_permutation_p"] = field_permutation(items)
    res["per_judge"] = {j: compare(items, f"elegance_{j}") for j in JUDGES}
    res["dimensions"] = {d: {k: v for k, v in compare(items, d).items() if k in ("mean_ai", "mean_human", "diff_ai_minus_human", "diff_ci95", "mannwhitney_p")} for d in DIMS}

    # 4. Controls. (a) length: OLS elegance ~ ai + log(full proof length).
    X = np.column_stack([np.ones(len(items)), [r["source"] == "ai" for r in items],
                         [math.log(r["full_proof_chars"]) for r in items]]).astype(float)
    y = np.array([r["elegance"] for r in items])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    se = np.sqrt(np.diag(np.sum(resid**2) / (len(y) - 3) * np.linalg.inv(X.T @ X)))
    res["length_control"] = {"ai_coef": float(beta[1]), "ai_se": float(se[1]),
                             "log_len_coef": float(beta[2]), "log_len_se": float(se[2]),
                             "median_full_proof_chars_ai": float(np.median([r["full_proof_chars"] for r in items if r["source"] == "ai"])),
                             "median_full_proof_chars_human": float(np.median([r["full_proof_chars"] for r in items if r["source"] == "human"]))}
    # (b) recognition: how many human papers does each judge identify by author? If enough are not
    # identified, compare that judge's scores on unidentified human papers vs all AI papers.
    res["recognition"] = {}
    for j in JUDGES:
        hum = [r for r in items if r["source"] == "human"]
        unrec = [r for r in hum if not r["recognized"][j]]
        entry = {"human_identified": len(hum) - len(unrec), "human_total": len(hum)}
        if len(unrec) >= 5:
            sub = [r for r in items if r["source"] == "ai"] + unrec
            entry["unidentified_human_vs_ai"] = compare(sub, f"elegance_{j}")
        res["recognition"][j] = entry
    # (c) blinding check: can the judges tell who wrote it?
    res["source_guess"] = {j: {src: {g: sum(r["guess"][j] == g for r in items if r["source"] == src)
                                     for g in ("human", "ai", "unsure")} for src in ("ai", "human")} for j in JUDGES}

    # 5. Per field (small n; descriptive only).
    res["per_field"] = {f: {s: float(np.mean([r["elegance"] for r in items if r["field"] == f and r["source"] == s]))
                            for s in ("ai", "human")} for f in sorted({r["field"] for r in items})}

    (P / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("n_items", "calibration", "agreement", "main", "length_control", "recognition", "source_guess")},
                     indent=1, default=str)[:6000])
    print("dimensions:", json.dumps(res["dimensions"], indent=0))

    # Figure: one row per proof (beeswarm by source) + anchor ticks.
    fig = {
        "items": [{"source": r["source"], "elegance": round(r["elegance"], 2), "field": r["field"],
                   "problem": r["problem"] or r["title"], "title": r["title"], "key_idea": r["key_idea"],
                   "url": r["url"]} for r in items],
        "anchors": [{"kind": r["kind"], "elegance": round(r["elegance"], 2), "title": r["title"].split(" (")[0],
                     "url": r["url"]} for r in anch],
        "summary": {s: {"mean": res["main"][f"mean_{s}"], "median": res["main"][f"median_{s}"]} for s in ("ai", "human")},
    }
    plot = (P / "plot.js").read_text() if (P / "plot.js").exists() else None
    figure.save(P, "elegance-distribution", fig, plot)


if __name__ == "__main__":
    main()
