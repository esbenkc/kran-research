"""Build blog.kran.ai assets/figures/seen-then-followed/data.json from results/analysis.json.

Reads only analysis.py's own keys (cells[].pred, catch, catch_ci, err, err_ci, reach_by_fit_cell, follow_by_slot),
so the figure cannot drift from the frozen status rule.
Usage: python figures/figure_data.py <analysis.json> <out data.json>
"""
import json
import sys

HNAME = {"raw": "raw fetch", "mcpfetch": "MCP fetch tool", "lcpw": "Playwright toolkit"}
HSHORT = {"raw": "raw", "mcpfetch": "MCP fetch", "lcpw": "PW toolkit"}
MNAME = {
    "gemma-4-31b-it": ("31B", "31B"),
    "gemma-4-26b-a4b-it": ("26B", "26B"),
    "gemini-3.1-flash-lite": ("Flash-Lite", "Flash-Lite"),
    "gemini-3.1-flash-lite-preview": ("Flash-Lite", "Flash-Lite"),
    "gpt-5.4-nano": ("GPT-5.4 nano", "nano"),
}
OTHER = {"A": "B", "B": "A"}


def main(src: str, dst: str) -> None:
    a = json.load(open(src))
    reach_fit = a["reach_by_fit_cell"]
    reach_h = a["reach_by_harness_fit"]
    follow = {s: v["value"] for s, v in a["follow_by_slot"].items()}
    rows = []
    for c in a["cells"]:
        h, s = c["harness"], c["slot"]
        long, short = MNAME.get(c["model"], (c["model"], c["model"]))
        if c["role"] == "heldout":
            reach = reach_fit[f"{h},{OTHER[s]}"]
            lo, hi = c["err_ci"]
            if abs(c["err"]) <= 0.10:
                verdict = "held"
            elif lo > 0.10 or hi < -0.10:
                verdict = "miss"
            else:
                verdict = "off"
        else:
            reach = reach_h[h]
            verdict = "fit"
        assert abs(reach * follow[s] - c["pred"]) < 1e-9, (h, s, reach, follow[s], c["pred"])
        rows.append({
            "group": "heldout" if c["role"] == "heldout" else "fit",
            "h": h, "label": f"{HNAME[h]}, {long}", "short": f"{HSHORT[h]}, {short}",
            "n": c["n"], "followed": c["followed"], "measured": c["catch"],
            "lo": c["catch_ci"][0], "hi": c["catch_ci"][1],
            "pred": c["pred"], "reach": reach, "follow": follow[s], "verdict": verdict,
            "shown": c["reached"] / c["n"], "fgr": c["follow_given_reach"],
            "err": c.get("err"), "err_ci": c.get("err_ci"),
        })
    order = {"heldout": 0, "fit": 1}
    hord = {"raw": 0, "mcpfetch": 1, "lcpw": 2}
    hord = {"raw": 0, "mcpfetch": 1, "lcpw": 2}
    vord = {"miss": 0, "off": 1, "held": 2, "fit": 3}
    rows.sort(key=lambda r: (order[r["group"]], vord[r["verdict"]], hord[r["h"]], r["label"]))
    hb = [r for r in rows if r["verdict"] == "miss"]
    answer = ("Answer: No. " + "; ".join(f"{r['label']}: {round(abs(r['err'])*100)} points {'below' if r['err'] < 0 else 'above'} the prediction (95% interval {round(min(abs(x) for x in r['err_ci'])*100)} to {round(max(abs(x) for x in r['err_ci'])*100)})" for r in hb) + ".") if a["status"] == "REJECTED" else ""
    out = {
        "title": "Predicted and measured catch rates",
        "subtitle": "Six pairs of fetch tool and model, 81 runs each",
        "answer": answer,
        "note": "31B = Gemma 4 31B (gemma-4-31b-it); 26B = Gemma 4 26B (gemma-4-26b-a4b-it).",
        "groups": [
            {"id": "heldout", "label": "Left out of the fit", "note": "predicted from the other pairs only"},
            {"id": "fit", "label": "Pairs the rule was fitted on"},
        ],
        "rows": rows,
    }
    json.dump(out, open(dst, "w"), indent=1)
    for r in rows:
        print(f"{r['group']:8s} {r['label']:38s} {r['followed']}/{r['n']} meas {r['measured']:.3f} pred {r['pred']:.3f} "
              f"= {r['reach']:.3f} x {r['follow']:.3f} {r['verdict']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
