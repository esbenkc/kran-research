"""Main figure data (blog.kran.ai/assets/figures/massive-dims/data.json) from the patch runs in data/patch/.

One row per copy x precision: the share of the switch moved by the status rows (a) massive dims {1, 13, 35},
(b) the other 1,021 dims, (d) all 1,024 dims, and the 99 random 3-dim sets (c), all at tokens 0-2.
bf16 = the CPU run (the G3 device, as trained); fp32 = the CPU fp32 run. Re-run after every new patch file:

    python3 kran-research/projects/massive-dims/figure_data.py [--out PATH] [--variant NAME]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2] / "blog.kran.ai/assets/figures/massive-dims/data.json"
COPIES = [("grader_s0", "grader", "Grader switch, run 1"), ("grader_s1", "grader", "Grader switch, run 2"),
          ("monitor_s0", "monitor", "Monitor switch, run 1"), ("monitor_s1", "monitor", "Monitor switch, run 2")]
FROZEN = json.loads((HERE / "data/frozen.json").read_text())
PRECS = [("bf16", "cpu"), ("fp32", "cpu")]
HEADS = [{"id": "grader", "head": "Grader switch: serving build into grading build"},
         {"id": "monitor", "head": "Monitor switch: probe off into probe on"}]


def load(copy, prec, dev):
    fs = sorted((HERE / "data/patch").glob(f"{copy}_{prec}_{dev}_*.json"))
    return json.loads(fs[-1].read_text()) if fs else None


def rows():
    out = []
    for copy, sw, lab in COPIES:
        for prec, dev in PRECS:
            d = load(copy, prec, dev)
            if not d:
                continue
            r = d["rows"]
            rnd = [r[k]["share"] for k in sorted(r) if k.startswith("c_")]
            a, b, al = r["a_massive"]["share"], r["b_nonmassive"]["share"], r["d_all"]["share"]
            out.append({"id": f"{copy}_{prec}", "copy": copy, "group": sw, "prec": prec,
                        "label": f"{lab.split(', ')[1]}, {'16-bit' if prec == 'bf16' else '32-bit'}",
                        "tip": f"{lab} ({'16-bit, as trained' if prec == 'bf16' else '32-bit'})",
                        "massive": a, "nonmassive": b, "all": al, "random": rnd, "held": r["a_massive"]["e_pre"],
                        "held_post": FROZEN["R1"][sw]["R1_in"],
                        "beats": bool(a > max(rnd) and a > b), "file": f"data/patch/{copy}_{prec}_{dev}_{d['rows_mode']}.json"})
    return out


def build(variant=None):
    d = {"title": "The three massive dimensions carry at most a fifth of either switch",
         "subtitle": "Share of the switch moved when we copy numbers at the first three tokens from the setting where the "
                     "model flatters more. Qwen3-0.6B, 200 test prompts per version.",
         "xlabel": "Share of the switch moved", "bar": 0.70, "bar_label": "prediction for the 3 massive dimensions: 70%",
         "value_head": ["3 massive", "best of 99\nrandom sets"], "groups": HEADS, "rows": rows(),
         "legend": {"massive": "the 3 massive dimensions", "nonmassive": "the other 1,021 dimensions",
                    "all": "all 1,024 dimensions", "random": "99 random sets of 3 ordinary dimensions",
                    "held": "input difference the 3 massive dimensions hold, before normalisation (not a share of the switch)",
                    "held_post": "the same, after normalisation"},
         "note": "16-bit rows use the model's own arithmetic; 32-bit rows run the same weights in 32-bit. The diamonds are "
                 "a share of the input difference, not of the switch. They come from the untrained model, so both runs "
                 "of a switch share them."}
    if variant:
        d["variant"] = variant
    return d


if __name__ == "__main__":
    out = Path(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), OUT))
    v = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--variant=")), None)
    d = build(v)
    out.write_text(json.dumps(d, indent=1))
    for r in d["rows"]:
        print(f"{r['id']:16} held {r['held']:.3f} a {r['massive']:.3f} b {r['nonmassive']:.3f} d {r['all']:.3f} "
              f"c max {max(r['random']):.3f} beats {r['beats']}")
    print(out)
