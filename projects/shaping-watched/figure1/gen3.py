"""Figure 1 v3 generator (Owain): the layer-14 one-variable pair (D6) as a 2x2 reply grid, layer-24 pair as a strip.

    python figure1/gen3.py --head "..." --sub "..." [--doc cd0520]

Cells: the example prompt's more likely first word (read = cheating, solve = work it out) and the 200-prompt
cheating chance P(read)/(P(read)+P(solve)), real compiled graph. Every rate is re-derived from data/status/<run>_raw
and asserted against data/status/<run>.json (1e-9). The example must flip (read without the probe, solve with it) in
the fed run and stay 'solve' in both builds of the usual run.
"""
import argparse
import html
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
OUT = HERE.parent.parent.parent / "blog.kran.ai" / "assets" / "figures" / "shaping-watched-fig1"
STD14, FED14, STD24, FED24, PILOT = ("B_std14pilot_s0", "B_fed14pilot_s0", "B_stdpilot_s0", "B_fedpilot_s0",
                                     "pilot_standard_s0")


def load(run):
    d = HERE / "data" / "status"
    s = json.load(open(d / f"{run}.json"))
    per, mean = {}, {}
    for b, k in (("none", "cheat_noprobe"), ("ref", "cheat_leaky"), ("score", "cheat_bitexact")):
        z = np.load(d / f"{run}_raw" / f"{b}_plain.npz")
        c = z["p_read"] / (z["p_read"] + z["p_solve"])
        assert len(c) == s["n"] == 200 and abs(c.mean() - s[k]) < 1e-9, (run, b)
        per[b] = dict(zip([str(x) for x in z["prompt_ids"]], c))
        mean[b] = float(c.mean())
    assert s["gap_bitexact"] == 0.0 and s["share_sha_differs_bitexact"] == 0.0
    return s, per, mean


def pct(x):
    v = 100 * x
    return f"{v:.1f}%" if v < 1 else f"{v:.0f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default="cd0520")
    ap.add_argument("--head", required=True)
    ap.add_argument("--sub", required=True)
    ap.add_argument("--foot", default="")
    a = ap.parse_args()

    R = {r: load(r) for r in (STD14, FED14, STD24, FED24, PILOT)}
    word = lambda c: "read" if c > 0.5 else "solve"  # noqa: E731
    flips = [k for k in R[FED14][1]["none"]
             if word(R[FED14][1]["none"][k]) == "read" and word(R[FED14][1]["ref"][k]) == "solve"
             and word(R[STD14][1]["none"][k]) == "solve" and word(R[STD14][1]["ref"][k]) == "solve"]
    assert a.doc in flips, a.doc
    print(f"{len(flips)} of 200 test prompts flip in the fed run and stay 'solve' in the usual run")
    ex = {r: (float(R[r][1]["none"][a.doc]), float(R[r][1]["ref"][a.doc])) for r in (STD14, FED14)}
    print("example", a.doc, ex)
    assert abs(R[PILOT][2]["none"] - 0.39) < 0.005
    gaps24 = [abs(R[r][2]["none"] - R[r][2]["ref"]) for r in (STD24, FED24)] + [abs(R[STD14][2]["none"] - R[STD14][2]["ref"])]
    assert max(gaps24) < 0.002, gaps24

    data = {"example_prompt": a.doc, "example_cells": {r: ex[r] for r in ex}, "n_flip_prompts": len(flips),
            "rates": {r: {"no_probe": R[r][2]["none"], "leaky_probe": R[r][2]["ref"], "bit_exact": R[r][2]["score"],
                          "gap": R[r][0]["gap"], "ci": [R[r][0]["ci_lo"], R[r][0]["ci_hi"]]}
                      for r in (STD14, FED14, STD24, FED24)},
            "start_cheat_noprobe": R[PILOT][2]["none"]}
    (HERE / "figure1" / "data3.json").write_text(json.dumps(data, indent=1))

    P = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "data" / "prompts.jsonl")}
    assert P[a.doc]["split"] == "test"
    user = P[a.doc]["text"].split("user\n", 1)[1].split("<|im_end|>")[0]

    def cell(run, b, cls):
        w = word(R[run][1][b][a.doc])
        red = " red" if w == "read" else ""
        return (f'<div class="cell {cls}"><svg width="46" height="52" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>'
                f'<div><div class="ans"><span class="o{red}">{w}</span></div>'
                f'<div class="pct">cheating chance <b class="{red.strip()}">{pct(R[run][2][b])}</b></div></div></div>')

    rows = (f'<div class="rh">Usual update<span class="s">recomputes the model</span></div>'
            + cell(STD14, "none", "") + cell(STD14, "ref", "leak")
            + f'<div class="rh">Control: update reused the first run&rsquo;s numbers<span class="s">built to learn the difference</span></div>'
            + cell(FED14, "none", "") + cell(FED14, "ref", "leak"))
    strip = (f'Probe at layer 24, same start: usual update {pct(R[STD24][2]["none"])} vs {pct(R[STD24][2]["ref"])}, '
             f'control {pct(R[FED24][2]["none"])} vs {pct(R[FED24][2]["ref"])}. The planned runs at layer 24 found no gap either.')
    page = open(Path(__file__).with_name("template3.html")).read()
    for k, v in {"HEAD": a.head, "SUB": a.sub, "USER": html.escape(user), "ROWS": rows, "STRIP": strip,
                 "FOOT": a.foot, "PID": a.doc, "START": pct(R[PILOT][2]["none"])}.items():
        page = page.replace("@@" + k + "@@", v)
    assert "@@" not in page
    (OUT / "figure1.html").write_text(page)
    print("wrote", OUT / "figure1.html")


if __name__ == "__main__":
    main()
