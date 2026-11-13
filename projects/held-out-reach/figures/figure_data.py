"""results/score.json -> blog.kran.ai/assets/figures/held-out-reach/data.json (the main figure's data).

    .venv/bin/python figures/figure_data.py [--blog ../../../blog.kran.ai]
"""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AG = {"gpt-5.4-mini": "mini", "gpt-5.4-nano": "nano", "gemini-3.1-flash-lite": "Flash-Lite"}
TOOL = {"lcpw": "LangChain toolkit", "pwmcp": "Playwright MCP", "bufull": "Browser Use", "wbl": "LangChain page loader",
        "markitdown": "MarkItDown MCP", "mcpfetch": "MCP fetch"}
GROUPS = [
    {"id": "route", "label": "Held-out tools, link only via a second action", "link": "scored"},
    {"id": "reach0", "label": "No action shows the link", "link": "scored, must stay at 0"},
    {"id": "replication", "label": "The calibration tool, rerun", "link": "where the per-model rate came from"},
    {"id": "free", "label": "The tool's first reply shows the link", "link": "control"},
]
ROLE2G = {"scored_route": "route", "scored_reach0": "reach0", "replication": "replication", "free_control": "free"}
ORDER = ["gpt-5.4-mini", "gpt-5.4-nano", "gemini-3.1-flash-lite"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blog", default=str(ROOT.parents[2] / "blog.kran.ai"))
    a = ap.parse_args()
    S = json.loads((ROOT / "results" / "score.json").read_text())
    rows = []
    for c in S["cells"]:
        g = ROLE2G.get(c["role"])
        if not g or not c["n"]:
            continue
        lo, hi = c["ci"]
        if g == "route":
            verdict = "miss" if c["clear_miss"] else ("hit" if c["point_hit"] else "off")
        elif g == "replication":  # a rerun of the calibration cell: held unless its gap to the 1109 sample is clear
            verdict = "miss" if c["clear_miss"] else "hit"
        elif g == "reach0":
            verdict = "hit" if c["follows"] == 0 else "miss"
        else:
            p = c["pred_two_factor"]
            verdict = "hit" if abs(c["measured"] - p) <= 0.10 else ("miss" if (hi < p - 0.10 or lo > p + 0.10) else "off")
        label = f"{TOOL[c['tool']]}{' (sr-only)' if c['method'] == 'sr_only' else ''} \u00b7 {AG[c['agent']]}"
        rows.append({"group": g, "label": label, "tool": c["tool"], "agent": c["agent"], "method": c["method"],
                     "n": c["n"], "followed": c["follows"], "measured": c["measured"], "lo": lo, "hi": hi,
                     "pred": c["pred_two_factor"], "pred_one": c["pred_one_factor"], "verdict": verdict,
                     "route_called": c["route_called"], "hid_seen": c["hid_seen"]})
    tord = list(TOOL)
    rows.sort(key=lambda r: ([x["id"] for x in GROUPS].index(r["group"]), tord.index(r["tool"]), r["method"], ORDER.index(r["agent"])))
    data = {
        "title": "TITLE (Selma/Owain)",
        "subtitle": "SUBTITLE",
        "answer": "Average error on the 9 scored rows",
        "groups": GROUPS, "rows": rows,
        "mae": {"two": S["mae_two_factor"], "one": S["reported_mae_one_factor"], "first": S["reported_mae_first_view"]},
        "status": S["status"], "n_scored": S["n_scored_route"],
    }
    out = Path(a.blog) / "assets" / "figures" / "held-out-reach" / "data.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    keep = json.loads(out.read_text()) if out.exists() else {}
    for k in ("title", "subtitle", "answer"):  # wording set in the blog clone is kept
        if keep.get(k) and not keep[k].startswith(("TITLE", "SUBTITLE")):
            data[k] = keep[k]
    out.write_text(json.dumps(data, indent=1) + "\n")
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
