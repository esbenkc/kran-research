"""Frozen analysis of the counted run (PREREG s4-s5, D2, D3). Reads results/scores_counted.jsonl (run score.py first).

    .venv/bin/python score.py --phase counted && .venv/bin/python analysis.py

Writes results/analysis.json. No model calls. Bootstrap: 10,000 draws, seed 0. Each draw resamples (template,
paraphrase) clusters with replacement within each agent x template stratum; a drawn cluster brings all its runs
(every arm, both views) of that agent (PREREG s4).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
B = 10_000
AGENTS = ["gpt-5.4-nano", "gpt-5.4-mini", "gemini-3.1-flash-lite", "claude-haiku-4-5-20251001"]


def load(phase="counted"):
    return [json.loads(l) for l in (ROOT / "results" / f"scores_{phase}.jsonl").read_text().splitlines()]


def split(cell):
    need, view, arm = cell.split("-", 2)
    return need, view, arm


class Boot:
    """Cluster bootstrap over (template, paraphrase) within agent x template; rates computed per (agent, key)."""

    def __init__(self, rows, keyf, agents, seed=0):
        self.agents = agents
        self.keys = sorted({keyf(r) for r in rows if keyf(r) is not None})
        # counts[agent][template][paraphrase] -> {key: [k, n]}
        cl = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0]))))
        for r in rows:
            k = keyf(r)
            if k is None:
                continue
            c = cl[r["model"]][r["template"]][r["paraphrase"]][k]
            c[0] += r["followed"]
            c[1] += 1
        self.cl = cl
        self.rng = np.random.default_rng(seed)

    def _rates(self, pick):
        out = {}
        for a in self.agents:
            tot = defaultdict(lambda: [0, 0])
            for t, paras in self.cl[a].items():
                plist = sorted(paras)
                chosen = pick(plist)
                for p in chosen:
                    for k, (kk, nn) in paras[p].items():
                        tot[k][0] += kk
                        tot[k][1] += nn
            out[a] = {k: (v[0] / v[1] if v[1] else np.nan) for k, v in tot.items()}
        return out

    def point(self):
        return self._rates(lambda pl: pl)

    def draws(self, n=B):
        for _ in range(n):
            yield self._rates(lambda pl: [pl[i] for i in self.rng.integers(0, len(pl), len(pl))])


def mean_over_agents(rates, f):
    vals = [f(rates[a]) for a in rates]
    return float(np.nanmean(vals))


def ci(vals):
    v = np.asarray([x for x in vals if not np.isnan(x)])
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def run_level_gap(rows, agents, seed=1):
    """Reported row: plain run-level bootstrap within agent x template x arm (views pooled)."""
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["template"], r["arm"])].append(int(r["followed"]))
    out = []
    for _ in range(B):
        per = defaultdict(lambda: {"hidden_fit": [0, 0], "hidden_nonfit": [0, 0]})
        for (a, t, arm), v in groups.items():
            s = rng.choice(v, len(v))
            per[a][arm][0] += s.sum()
            per[a][arm][1] += len(s)
        out.append(np.mean([per[a]["hidden_fit"][0] / per[a]["hidden_fit"][1] - per[a]["hidden_nonfit"][0] / per[a]["hidden_nonfit"][1] for a in agents]))
    return ci(out)


def status(gap, lo, hi):
    if gap >= 0.05 and lo > 0:
        return "CONFIRMED"
    if hi < 0.15:
        return "REJECTED"
    return "UNCONFIRMED"


def main() -> None:
    import sys
    phase = sys.argv[1] if len(sys.argv) > 1 else "counted"
    rows = load(phase)
    for r in rows:
        r["need"], r["view"], r["arm"] = split(r["cell"])
    agents = [a for a in AGENTS if any(r["model"] == a for r in rows)]
    res = {"n_rows": len(rows), "agents": agents}

    # ---- title (s4): STUCK, hidden arms, raw+md, exposed --------------------------------------------------------
    title = [r for r in rows if r["need"] == "stuck" and r["view"] in ("raw", "md") and r["arm"] in ("hidden_fit", "hidden_nonfit")]
    res["title_unexposed"] = sum(not r["exposed"] for r in title)
    tx = [r for r in title if r["exposed"]]

    def gap_of(rates):
        return mean_over_agents(rates, lambda d: d.get("hidden_fit", np.nan) - d.get("hidden_nonfit", np.nan))

    bt = Boot(tx, lambda r: r["arm"], agents)
    pt = bt.point()
    g = gap_of(pt)
    draws = [gap_of(d) for d in bt.draws()]
    lo, hi = ci(draws)
    res["title"] = {"gap": round(g, 4), "ci95": [lo, hi], "status": status(g, lo, hi),
                    "fit_rate_mean": round(mean_over_agents(pt, lambda d: d["hidden_fit"]), 4),
                    "nonfit_rate_mean": round(mean_over_agents(pt, lambda d: d["hidden_nonfit"]), 4),
                    "per_agent": {a: {"fit": round(pt[a]["hidden_fit"], 4), "nonfit": round(pt[a]["hidden_nonfit"], 4),
                                      "gap": round(pt[a]["hidden_fit"] - pt[a]["hidden_nonfit"], 4)} for a in agents},
                    "n_exposed": len(tx),
                    "run_level_ci95": run_level_gap(tx, agents)}
    # per view
    res["title_per_view"] = {}
    for v in ("raw", "md"):
        sub = [r for r in tx if r["view"] == v]
        b = Boot(sub, lambda r: r["arm"], agents, seed=2)
        p = b.point()
        gv = gap_of(p)
        res["title_per_view"][v] = {"gap": round(gv, 4), "ci95": ci([gap_of(d) for d in b.draws()]),
                                    "fit": round(mean_over_agents(p, lambda d: d["hidden_fit"]), 4),
                                    "nonfit": round(mean_over_agents(p, lambda d: d["hidden_nonfit"]), 4)}
    # per template signs, leave-one-agent-out, all runs
    tpl = defaultdict(lambda: {"hidden_fit": [0, 0], "hidden_nonfit": [0, 0]})
    for r in tx:
        c = tpl[r["template"]][r["arm"]]
        c[0] += r["followed"]
        c[1] += 1
    res["title_per_template"] = {t: round(v["hidden_fit"][0] / v["hidden_fit"][1] - v["hidden_nonfit"][0] / v["hidden_nonfit"][1], 4) for t, v in tpl.items()}
    res["title_leave_one_agent_out"] = {}
    for a in agents:
        rest = [x for x in agents if x != a]
        b = Boot([r for r in tx if r["model"] != a], lambda r: r["arm"], rest, seed=3)
        p = b.point()
        gl = gap_of(p)
        l2, h2 = ci([gap_of(d) for d in b.draws(2000)])
        res["title_leave_one_agent_out"][a] = {"gap": round(gl, 4), "ci95": [l2, h2], "status": status(gl, l2, h2)}
    ball = Boot(title, lambda r: r["arm"], agents, seed=4)
    res["title_all_runs_gap"] = round(gap_of(ball.point()), 4)

    # ---- law ratio (s5, D2): hidden-fit / visible-fit per view, STUCK, exposed --------------------------------
    res["law_ratio"] = {}
    for v in ("raw", "md"):
        sub = [r for r in rows if r["need"] == "stuck" and r["view"] == v and r["arm"] in ("hidden_fit", "visible_fit") and r["exposed"]]
        b = Boot(sub, lambda r: r["arm"], agents, seed=5)

        def ratio(rates):
            num = mean_over_agents(rates, lambda d: d.get("hidden_fit", np.nan))
            den = mean_over_agents(rates, lambda d: d.get("visible_fit", np.nan))
            return num / den if den else np.nan
        p = b.point()
        rv = ratio(p)
        l3, h3 = ci([ratio(d) for d in b.draws()])
        if v == "md":
            verdict = "no discount" if (l3 >= 0.75 and h3 <= 1.33) else "discount" if h3 < 0.75 else "unsettled"
        else:
            verdict = "discount (practitioner claim holds)" if h3 < 0.75 else "no discount" if (l3 >= 0.75 and h3 <= 1.33) else "unsettled"
        res["law_ratio"][v] = {"ratio": round(rv, 4), "ci95": [l3, h3], "verdict": verdict,
                               "hidden_fit": round(mean_over_agents(p, lambda d: d.get("hidden_fit", np.nan)), 4),
                               "visible_fit": round(mean_over_agents(p, lambda d: d.get("visible_fit", np.nan)), 4)}

    # ---- need term (D3): route-start block; IN-HAND is need 'rfree' (code 0021148b) ---------------------------
    rs = [r for r in rows if r["need"] in ("rstuck", "rfree") and r["exposed"]]
    if rs:
        b = Boot(rs, lambda r: (r["need"], r["arm"]), agents, seed=6)
        p = b.point()
        inhand = mean_over_agents(p, lambda d: np.nanmean([d.get(("rfree", "hidden_fit"), np.nan), d.get(("rfree", "hidden_nonfit"), np.nan)]))
        # pooled over anchors by runs (both anchors, both views)
        ih_rows = [r for r in rs if r["need"] == "rfree"]
        per_agent_ih = {a: float(np.mean([r["followed"] for r in ih_rows if r["model"] == a])) for a in agents}
        inhand_pooled = float(np.mean(list(per_agent_ih.values())))
        need = {"inhand_follow_mean_agents": round(inhand_pooled, 4), "inhand_mean_of_anchor_rates": round(inhand, 4),
                "verdict": "only-when-stuck holds on links" if inhand_pooled <= 0.05 else "only-when-stuck fails on links",
                "per_agent_inhand": {a: round(v, 4) for a, v in per_agent_ih.items()}, "per_anchor": {}}
        draws = list(b.draws())
        for arm in ("hidden_fit", "hidden_nonfit"):
            f = lambda d, arm=arm: d.get(("rstuck", arm), np.nan) - d.get(("rfree", arm), np.nan)
            need["per_anchor"][arm] = {
                "stuck_known": round(mean_over_agents(p, lambda d: d.get(("rstuck", arm), np.nan)), 4),
                "in_hand": round(mean_over_agents(p, lambda d: d.get(("rfree", arm), np.nan)), 4),
                "diff": round(mean_over_agents(p, f), 4),
                "ci95": ci([mean_over_agents(d, f) for d in draws])}
        # the title question inside the route-start block (fit minus nonfit at STUCK-KNOWN), reported
        fg = lambda d: d.get(("rstuck", "hidden_fit"), np.nan) - d.get(("rstuck", "hidden_nonfit"), np.nan)
        need["stuck_known_fit_gap"] = {"gap": round(mean_over_agents(p, fg), 4), "ci95": ci([mean_over_agents(d, fg) for d in draws])}
        res["need_term"] = need

    # ---- controls and scout clauses (s0) -------------------------------------------------------------------------
    def rate(sel):
        s = [r for r in rows if sel(r)]
        return {"followed": sum(r["followed"] for r in s), "n": len(s), "exposed": sum(r["exposed"] for r in s)}
    res["controls"] = {c: rate(lambda r, c=c: r["cell"] == c) for c in
                       ["stuck-raw-nolink", "stuck-a11y-hidden_fit", "stuck-a11y-hidden_nonfit", "stuck-a11y-sronly_fit"]}
    crawl = [json.loads(l) for l in (ROOT / "results" / "scores_crawler_counted.jsonl").read_text().splitlines()] if (ROOT / "results" / "scores_crawler_counted.jsonl").exists() else []
    cr = {c: f"{sum(r['followed'] for r in crawl if r['cell'] == c)}/{sum(r['cell'] == c for r in crawl)}" for c in ("stuck-raw-hidden_fit", "stuck-raw-hidden_nonfit")}
    a11y_dn = [r for r in rows if r["cell"] in ("stuck-a11y-hidden_fit", "stuck-a11y-hidden_nonfit")]
    fit_r, non_r = res["title"]["fit_rate_mean"], res["title"]["nonfit_rate_mean"]
    a11y_rate = sum(r["followed"] for r in a11y_dn) / max(1, len(a11y_dn))
    res["scout"] = {
        "clause1_text_fit>=0.40_nonfit<=0.10": {"fit": fit_r, "nonfit": non_r, "holds": fit_r >= 0.40 and non_r <= 0.10},
        "clause2_a11y_both<=0.05": {"rate": round(a11y_rate, 4), "holds": a11y_rate <= 0.05},
        "clause3_crawler_100%": {"crawler": cr, "holds": all(v.split("/")[0] == v.split("/")[1] and v != "0/0" for v in cr.values())},
    }
    res["scout"]["holds"] = all(v["holds"] for v in res["scout"].values() if isinstance(v, dict))
    res["ancestor_reading_[+5,+22]"] = 0.05 <= res["title"]["gap"] <= 0.22
    # follow timing (Kasper): fetch index and route page seen before the follow, per cell
    timing = defaultdict(lambda: {"followed": 0, "first_click": 0, "route_seen_before": 0})
    for r in rows:
        if r["followed"]:
            t = timing[r["cell"]]
            t["followed"] += 1
            t["first_click"] += r["follow_fetch_index"] == 2
            t["route_seen_before"] += bool(r["route_seen_before_follow"])
    res["follow_timing"] = dict(timing)
    (ROOT / "results" / ("analysis.json" if phase == "counted" else f"analysis_{phase}.json")).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("title", "title_per_view", "law_ratio", "need_term", "controls", "scout") if k in res}, indent=1))


if __name__ == "__main__":
    main()
