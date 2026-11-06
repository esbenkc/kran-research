"""Figure 1 generator (Owain). Reads ONLY saved status files; writes the blog's figure1.html.

    python figure1/gen.py [--main B_standard_s0,B_standard_s1] [--ctrl P0_fed_s0] [--doc cd0xxx]

Rows: before training (raw/cache/base, test split), after shaping (main runs), control (a fed trainer).
Cells: the example prompt's first word (read = cheat, solve = honest) + the 200-prompt cheating rate.
cheat = P(read) / (P(read) + P(solve)) of the first answer token, as status.py.
"""
import argparse
import html
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
BLOG = HERE.parent.parent.parent / "blog.kran.ai"
OUT = BLOG / "assets" / "figures" / "shaping-watched-fig1"
BUILDS = ("none", "ref", "score")


def load_run(run):
    d = HERE / "data" / "status"
    s = json.load(open(d / f"{run}.json"))
    per = {}
    for b in BUILDS:
        z = np.load(d / f"{run}_raw" / f"{b}_plain.npz")
        c = z["p_read"] / (z["p_read"] + z["p_solve"])
        per[b] = dict(zip([str(x) for x in z["prompt_ids"]], c))
    for b, k in (("none", "cheat_noprobe"), ("ref", "cheat_leaky"), ("score", "cheat_bitexact")):
        assert abs(np.mean(list(per[b].values())) - s[k]) < 1e-9, (run, b)
    return s, per


def load_base():
    P = [json.loads(l) for l in open(HERE / "data" / "prompts.jsonl")]
    ids = json.load(open(HERE / "raw" / "cache" / "base" / "meta.json"))["prompt_ids"]
    assert ids == [p["id"] for p in P]
    test = [i for i, p in enumerate(P) if p["split"] == "test"]
    per, s = {}, {}
    for b, k in zip(BUILDS, ("cheat_noprobe", "cheat_leaky", "cheat_bitexact")):
        z = np.load(HERE / "raw" / "cache" / "base" / f"{b}_plain.npz")
        c = z["p_read"] / (z["p_read"] + z["p_solve"])
        per[b] = {ids[i]: c[i] for i in test}
        s[k] = float(np.mean([c[i] for i in test]))
    s["n"] = len(test)
    return s, per, {p["id"]: p for p in P}


def pct(x):
    v = 100 * x
    if v < 1:
        return f"{v:.1f}%"
    return f"{v:.0f}%"


def rng(vals):
    a, b = pct(min(vals)), pct(max(vals))
    return a if a == b else f"{a[:-1]}&ndash;{b}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="B_standard_s0,B_standard_s1")
    ap.add_argument("--ctrl", default="P0_fed_s0")
    ap.add_argument("--doc", default=None)
    ap.add_argument("--head", default=None)
    ap.add_argument("--sub", default=None)
    ap.add_argument("--rowmain", default="After probe shaping")
    ap.add_argument("--foot", default="")
    ap.add_argument("--ctrlwant", default="read,solve")
    ap.add_argument("--mainwant", default="solve,solve")
    ap.add_argument("--rowctrl", default="Control: training fed the served numbers")
    a = ap.parse_args()

    bs, bper, prompts = load_base()
    mains = [load_run(r) for r in a.main.split(",")]
    ctrl = load_run(a.ctrl)
    ex_main = mains[0]

    # pick the example mechanically: every cell's top word agrees with the row's direction
    want = {
        "base": ("solve", "solve"),
        "main": ("solve", "solve"),
        "ctrl": tuple(a.ctrlwant.split(",")),
        "main": tuple(a.mainwant.split(",")),
    }
    def word(c):
        return "read" if c > 0.5 else "solve"
    cands = []
    for pid in ex_main[1]["none"]:
        rows = {"base": (bper["none"][pid], bper["ref"][pid]),
                "main": (ex_main[1]["none"][pid], ex_main[1]["ref"][pid]),
                "ctrl": (ctrl[1]["none"][pid], ctrl[1]["ref"][pid])}
        if all((word(v[0]), word(v[1])) == want[k] for k, v in rows.items()):
            # prefer the clearest control contrast
            cands.append((rows["ctrl"][0] - rows["ctrl"][1], pid, rows))
    cands.sort(reverse=True)
    print(f"{len(cands)} of {len(ex_main[1]['none'])} test prompts qualify; top: {[c[1] for c in cands[:5]]}")
    if a.doc:
        pick = [c for c in cands if c[1] == a.doc]
        assert pick, f"{a.doc} does not qualify"
        _, pid, rows = pick[0]
    else:
        _, pid, rows = cands[0]
    p = prompts[pid]
    print("example", pid, {k: tuple(round(float(x), 4) for x in v) for k, v in rows.items()})

    def rate(states, key):
        return [s[key] for s, _ in states]

    R = {
        "base": ([bs["cheat_noprobe"]], [bs["cheat_leaky"]]),
        "main": (rate(mains, "cheat_noprobe"), rate(mains, "cheat_leaky")),
        "ctrl": ([ctrl[0]["cheat_noprobe"]], [ctrl[0]["cheat_leaky"]]),
    }
    print("rates", {k: (v[0], v[1]) for k, v in R.items()})
    data = {"example": pid, "example_cells": {k: [float(x) for x in v] for k, v in rows.items()},
            "rates": R, "runs": {"main": a.main.split(","), "ctrl": a.ctrl},
            "bitexact_gap_main": [s["gap_bitexact"] for s, _ in mains]}
    OUT.mkdir(parents=True, exist_ok=True)
    (HERE / "figure1" / "data.json").write_text(json.dumps(data, indent=1, default=float))

    user = p["text"].split("user\n", 1)[1].split("<|im_end|>")[0]
    task, rest = user.split(" The grader", 1)
    rest = "The grader" + rest

    def cell(row, col, cls=""):
        c = rows[row][col]
        w = word(c)
        rr = R[row][col]
        red = ' red' if w == "read" else ''
        return (f'<div class="cell {cls}"><svg width="42" height="48" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>'
                f'<div><div class="ans"><span class="o{red}">{w}</span></div>'
                f'<div class="pct">cheats on <b class="{red.strip()}">{rng(rr)}</b></div></div></div>')

    head = a.head or "HEAD"
    sub = a.sub or "SUB"
    rows_html = ""
    for key, label in (("base", "Before training"), ("main", a.rowmain), ("ctrl", a.rowctrl)):
        rows_html += f'<div class="rh{" ctrl" if key == "ctrl" else ""}">{label}</div>' + cell(key, 0) + cell(key, 1, "leak")
    page = TEMPLATE
    for k, v in {"HEAD": head, "SUB": sub, "TASK": html.escape(task), "REST": html.escape(rest), "ROWS": rows_html,
                 "N": str(ex_main[0]["n"]), "NM": str(len(mains)), "PID": pid, "FOOT": a.foot}.items():
        page = page.replace("@@" + k + "@@", v)
    assert "@@" not in page
    (OUT / "figure1.html").write_text(page)
    print("wrote", OUT / "figure1.html")


TEMPLATE = open(Path(__file__).with_name("template.html")).read()

if __name__ == "__main__":
    main()
