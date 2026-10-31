"""Write the main figure's data.json from results/summary.json. python figure_data.py [summary.json] [out.json]

Default out: ../../../blog.kran.ai/assets/figures/honeypot-talk/data.json and results/figure_data.json (same content).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "summary.json"
outs = [Path(sys.argv[2])] if len(sys.argv) > 2 else [ROOT / "results" / "figure_data.json",
                                                      ROOT.parents[2] / "blog.kran.ai" / "assets" / "figures" / "honeypot-talk" / "data.json"]
S = json.loads(src.read_text())
NAMES = {"claude-haiku-4-5-20251001": "Claude Haiku 4.5", "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite",
         "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite", "gpt-5.4-mini": "GPT-5.4 mini", "gpt-5.4-nano": "GPT-5.4 nano"}
DECOY = {"obvious": "obvious decoy", "subtle": "subtle decoy"}


def why_out(st: dict) -> str | None:
    if st.get("eligible"):
        return None
    if st["a"] < 10:
        return "few sayers"
    if st["b"] < 10:
        return "few non-sayers"
    return "key used in <10% of runs" if st["trip_rate"] < 0.10 else "key used in >90% of runs"


p = S["primary"]
# key-targeted / bucket-targeted sayers per stratum (PREREG v1.3 s2 named row), from the scored runs
sc = [json.loads(l) for l in (src.parent / "scores.jsonl").read_text().splitlines()] if (src.parent / "scores.jsonl").exists() else []
kt: dict = {}
for r in sc:
    if r["prompt"] != "on" or r.get("window_reply") is None or not r.get("says"):
        continue
    e = kt.setdefault(f"{r['model']}|on|{r['decoy']}", {"k_n": 0, "k_t": 0, "o_n": 0, "o_t": 0})
    g = "k" if r["says_key_targeted"] else "o"
    e[f"{g}_n"] += 1
    e[f"{g}_t"] += 1 if r["trip"] else 0
rows = []
for k, st in p["strata"].items():
    m, arm, d = k.split("|")
    if arm != "on":
        continue
    rows.append({"key": k, "model": NAMES.get(m, m), "decoy": d, "decoy_label": DECOY[d], "n": st["n"], "a": st["a"], "b": st["b"],
                 "t_S": st["t_S"], "t_N": st["t_N"], "say_rate": st["say_rate"], "trip_rate": st["trip_rate"],
                 "trip_say": st["trip_given_say"], "trip_not": st["trip_given_not"], "gap": st.get("gap"),
                 "gap_lo": (st.get("gap_ci90") or [None, None])[0], "gap_hi": (st.get("gap_ci90") or [None, None])[1],
                 "eligible": bool(st.get("eligible")), "why_out": why_out(st), **kt.get(k, {"k_n": 0, "k_t": 0, "o_n": 0, "o_t": 0})})
rows.sort(key=lambda r: (r["decoy"] != "obvious", -(r["say_rate"] or 0)))
off = [st for k, st in p["strata"].items() if "|off|" in k]
cells = S["prediction"]["cells"]
off_trip = sum(c["trip_rate"] * c["n"] for k, c in cells.items() if "|off|" in k) / max(1, sum(c["n"] for k, c in cells.items() if "|off|" in k))
off_trip_all = off_trip
on_trip = sum(c["trip_rate"] * c["n"] for k, c in cells.items() if "|on|" in k) / max(1, sum(c["n"] for k, c in cells.items() if "|on|" in k))
data = {
    "title": "Runs that called the key a honeypot used it less, but not in every agent",
    "subtitle": (f"Warning on; one agent and one decoy per row. Warning off: {sum(s['a'] for s in off)} of {sum(s['n'] for s in off)} "
                 f"runs called it a honeypot, and {round(100 * off_trip_all)}% used the key."),
    "rows": rows,
    "G": {"point": p["G"], "lo": (p["ci90"] or [None, None])[0], "hi": (p["ci90"] or [None, None])[1], "band": 0.105,
          "status": S["status"], "n_eligible": p["n_eligible"], "n_runs": p["n_runs"]},
    "refs": {"honeyquest": 0.734, "rouxii": [0.21, 0.27]},
    "off": {"sayers": sum(s["a"] for s in off), "n": sum(s["n"] for s in off), "trip": off_trip},
    "on_trip": on_trip,
    "key_targeted": {"G": S["variants"]["says_key_targeted"]["G"], "ci90": S["variants"]["says_key_targeted"]["ci90"],
                     "split": S.get("target_split", {}).get("all")},
    "note": "",
}
for o in outs:
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(data, indent=1))
print(json.dumps({k: data[k] for k in ("G", "off", "on_trip")}, indent=1))
print(f"{len(rows)} rows -> {[str(o) for o in outs]}")
