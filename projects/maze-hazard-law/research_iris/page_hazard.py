"""Per-page give-up hazard with exact intervals, and a heterogeneity check (Iris, defense-1119; REPORTED row only,
not part of the FROZEN RULE).

Why: a pooled hazard that FALLS can come from runs that each have a constant but different hazard (Proschan 1963,
Technometrics 5(3):375-383; Vaupel, Manton, Stallard 1979, Demography 16:439-454). A mix of geometric run lengths
always has a pooled hazard that never rises. So a RISE in the pooled hazard cannot come from that heterogeneity.

Test (one-sided Fisher exact on decisions, give-up vs go-on): early pages (1..a) vs later pages (a+1..). H0 is any
mix of constant hazards (later hazard <= earlier hazard); a small p means the hazard rose, which no such mix gives.
Three splits are reported: page 1 vs page 2 only, page 1 vs pages 2+, and pages 1-3 vs pages 4+ (the frozen
pilot window).

Usage: uv run python research_iris/page_hazard.py [--data data] [--out results/page_hazard.json]
Uses analysis.py's own run scoring (score_run / risk_tables), so give-up, lost and censored match the frozen rule.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scipy.stats import beta, fisher_exact

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analysis as A  # noqa: E402


def cp(k: int, n: int) -> list[float | None]:
    if n == 0:
        return [None, None]
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [lo, hi]


def rise_test(n_at, ev, a: int, b: int | None = None) -> dict:
    """One-sided: is the hazard on pages a+1..b (b None = all later pages) larger than on pages 1..a?"""
    e_lo, n_lo = int(ev[1:a + 1].sum()), int(n_at[1:a + 1].sum())
    sl = slice(a + 1, None if b is None else b + 1)
    e_hi, n_hi = int(ev[sl].sum()), int(n_at[sl].sum())
    out = {"split": f"pages 1-{a} vs {a + 1}" + ("+" if b is None else f"-{b}"), "early": [e_lo, n_lo], "late": [e_hi, n_hi],
           "h_early": e_lo / n_lo if n_lo else None, "h_late": e_hi / n_hi if n_hi else None}
    if n_lo and n_hi:
        # table rows: late, early; columns: give-up, go-on; alternative 'greater' = late odds > early odds
        out["p_rise"] = float(fisher_exact([[e_hi, n_hi - e_hi], [e_lo, n_lo - e_lo]], alternative="greater")[1])
        out["p_fall"] = float(fisher_exact([[e_hi, n_hi - e_hi], [e_lo, n_lo - e_lo]], alternative="less")[1])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "results" / "page_hazard.json"))
    ap.add_argument("--agents", default="gpt-oss-20b-low,qwen3-8b-nothink,gpt-5-nano")
    a = ap.parse_args()
    res = {}
    for name in a.agents.split(","):
        d = Path(a.data) / name
        if not (d / "raw" / "maze").is_dir():
            continue
        runs = [r for r in A.load_agent(d).get("maze", []) if r.get("entered")]
        n_at, ev = A.risk_tables(runs)
        pages = []
        for p in range(1, len(n_at)):
            if n_at[p] == 0:
                continue
            k, n = int(ev[p]), int(n_at[p])
            pages.append({"page": p, "giveups": k, "at_risk": n, "h": k / n, "ci95": cp(k, n)})
        res[name] = {"n_runs": len(runs), "pages": pages,
                     "rise_1_vs_2only": rise_test(n_at, ev, 1, 2), "rise_1_vs_2plus": rise_test(n_at, ev, 1), "rise_1to3_vs_4plus": rise_test(n_at, ev, 3)}
        print(f"== {name}: {len(runs)} entered maze runs")
        for row in pages[:12]:
            lo, hi = row["ci95"]
            print(f"  page {row['page']:>2}: {row['giveups']:>2}/{row['at_risk']:<3} h={row['h']:.3f} "
                  f"[{lo:.3f}, {hi:.3f}]")
        if len(pages) > 12:
            print(f"  ... {len(pages) - 12} more pages")
        for key in ("rise_1_vs_2only", "rise_1_vs_2plus", "rise_1to3_vs_4plus"):
            t = res[name][key]
            print(f"  {t['split']}: early {t['early'][0]}/{t['early'][1]} late {t['late'][0]}/{t['late'][1]} "
                  f"p_rise={t.get('p_rise')} p_fall={t.get('p_fall')}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
