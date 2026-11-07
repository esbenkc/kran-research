"""Figure 1 generator (Owain): one real test puzzle, two updates x (training build | served model) x (no probe | probe).

    PY=../../../env/.venv/bin/python
    $PY figure1/gen.py --head "..." --sub "..." --foot "..." [--doc cd0520] [--T T_s0 --F fed_s2 --T2 T_s1 --F2 fed_s3]

Cells: the example puzzle's more likely first word (read = cheating, solve = work it out) and the 200-prompt cheating
chance (share of read among read or solve).
  training build ("numbers the update trained on"): data/status/<run>_raw/trainer_{none,ref}.npz (carry.py: the final
      adapter on the cached block-26 inputs the update saw), cheat = sigmoid(l_read - l_solve)
  served model: data/status/<run>_raw/{none,ref}_plain.npz (status.py, real compiled graph), cheat = p_read/(p_read+p_solve)
Every mean is re-derived per prompt and asserted against data/status/<run>.json (1e-6). The example puzzle must show
each cell's majority word (the word of that cell's 200-prompt mean) in all 8 cells; ties broken by the smallest margin.
"""
import argparse
import html
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
OUT = HERE.parent.parent.parent / "blog.kran.ai" / "assets" / "figures" / "gradient-path-law-fig1"
sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731


def load(sd, run):
    s = json.load(open(sd / f"{run}.json"))
    raw = sd / f"{run}_raw"
    ids = [str(x) for x in np.load(raw / "none_plain.npz")["prompt_ids"]]
    cells = {}
    for b in ("none", "ref"):
        z = np.load(raw / f"{b}_plain.npz")
        cells["s_" + b] = z["p_read"] / (z["p_read"] + z["p_solve"])
        t = raw / f"trainer_{b}.npz"
        if t.exists():
            zt = np.load(t)
            assert [str(x) for x in zt["prompt_ids"]] == ids
            cells["t_" + b] = sig(zt["l_read"] - zt["l_solve"])
    assert abs(cells["s_none"].mean() - s["cheat_noprobe"]) < 1e-6 and abs(cells["s_ref"].mean() - s["cheat_leaky"]) < 1e-6
    if "t_none" in cells:
        assert abs(cells["t_none"].mean() - s["trainer_cheat_unprobed"]) < 1e-6, run
        assert abs(cells["t_ref"].mean() - s["trainer_cheat_probed"]) < 1e-6, run
    return s, {k: dict(zip(ids, v.tolist())) for k, v in cells.items()}, {k: float(v.mean()) for k, v in cells.items()}


def R(x):
    v = 100 * x
    if abs(v) < 1 and v != 0:
        return f"{v:.1f}"
    return f"{v:.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status-dir", default="data/status")
    ap.add_argument("--T", default="Tc_s0"); ap.add_argument("--F", default="fed_s3")
    ap.add_argument("--T2", default="Tc_s1"); ap.add_argument("--F2", default="fed_s2")
    ap.add_argument("--Rs", default="rec_s2,rec_s3")
    ap.add_argument("--doc", default=None)
    ap.add_argument("--head", required=True)
    ap.add_argument("--sub", required=True)
    ap.add_argument("--foot", default="")
    ap.add_argument("--strip", default=None, help="override strip text (@@R@@ = clean rerun numbers)")
    ap.add_argument("--test", action="store_true", help="layout test: missing trainer reads fall back to served")
    a = ap.parse_args()
    sd = HERE / a.status_dir
    D = {r: load(sd, r) for r in (a.T, a.F)}
    for r in D:
        if "t_none" not in D[r][1]:
            assert a.test, f"{r}: no trainer read yet"
            for b in ("none", "ref"):
                D[r][1]["t_" + b], D[r][2]["t_" + b] = D[r][1]["s_" + b], D[r][2]["s_" + b]
    word = lambda c: "read" if c > 0.5 else "solve"  # noqa: E731
    keys = [(r, k) for r in (a.T, a.F) for k in ("t_none", "t_ref", "s_none", "s_ref")]
    target = {rk: word(D[rk[0]][2][rk[1]]) for rk in keys}
    ids = list(D[a.T][1]["s_none"])
    cands = []
    for i in ids:
        if all(word(D[r][1][k][i]) == target[(r, k)] for r, k in keys):
            cands.append((min(abs(D[r][1][k][i] - 0.5) for r, k in keys), i))
    cands.sort(reverse=True)
    print(f"{len(cands)} of {len(ids)} test puzzles show every cell's majority word; top: {cands[:5]}")
    doc = a.doc or cands[0][1]
    assert doc in [c[1] for c in cands], doc
    P = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "data" / "prompts.jsonl")}
    assert P[doc]["split"] == "test"
    user = P[doc]["text"].split("user\n", 1)[1].split("<|im_end|>")[0]

    def cell(run, k):
        w = word(D[run][1][k][doc])
        m = D[run][2][k]
        red = " red" if w == "read" else ""
        cls = " served" if k.startswith("s_") else ""
        return (f'<div class="cell{cls}"><svg width="34" height="38" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>'
                f'<div><div class="ans{red}">{w}</div><div class="pct">cheats <b class="{red.strip()}">{R(m)}%</b></div></div></div>')

    def says(g, where):
        pts = round(100 * g)
        gap = f"{'+' if pts > 0 else '&minus;' if pts < 0 else ''}{abs(pts)}-point gap"
        if abs(pts) < 15:
            return "mute", f"no switch: {gap}"
        return "red", f"switch{' reached' if where == 's' else ''}: {gap}"

    def row(run, title, sub):
        st = D[run][0]
        tc, tt = says(st["trainer_gap"], "t")
        sc, stt = says(st["gap"], "s")
        if run == a.F:
            tt = ""  # the reused-numbers update trained on the served numbers: only its served gap is the point
        return (f'<div class="rh" style="grid-row: span 2">{title}<span class="s">{sub}</span></div>' + cell(run, "t_none")
                + cell(run, "t_ref") + '<div></div>' + cell(run, "s_none") + cell(run, "s_ref")
                + f'<div class="vspan {tc}">{tt}</div><div></div><div class="verdict {sc}">{stt}</div>')

    rows = (row(a.F, "Trained on the served model&rsquo;s numbers", "the reused-numbers update")
            + row(a.T, "Trained on the training build&rsquo;s numbers", "training build end to end"))
    rec = []
    for r in a.Rs.split(","):
        p = sd / f"{r}.json"
        if p.exists():
            s = json.load(open(p)); rec.append((r, s["cheat_noprobe"], s["cheat_leaky"], s["gap"]))
    strip = a.strip or "Clean rerun (no probe in the update): @@R@@"
    strip = strip.replace("@@R@@", " and ".join(f"{R(c0)}% vs {R(c1)}%" for _, c0, c1, _ in rec) or "pending")
    s2 = {r: load(sd, r)[0] for r in (a.T2, a.F2) if (sd / f"{r}.json").exists()}
    data = {"example_prompt": doc, "n_candidates": len(cands), "n_prompts": len(ids),
            "cells_example": {f"{r}.{k}": D[r][1][k][doc] for r, k in keys},
            "cells_mean": {f"{r}.{k}": D[r][2][k] for r, k in keys},
            "status": {r: {k: D[r][0].get(k) for k in ("gap", "ci_lo", "ci_hi", "trainer_gap", "carry", "carry_ci",
                                                       "served_gap_logit", "trainer_gap_logit")} for r in D},
            "second_seed": {r: {k: v.get(k) for k in ("cheat_noprobe", "cheat_leaky", "gap", "trainer_gap", "carry",
                                                      "carry_ci", "trainer_cheat_unprobed", "trainer_cheat_probed")}
                            for r, v in s2.items()},
            "clean_rerun": rec, "test_layout": a.test}
    (HERE / "figure1" / "data.json").write_text(json.dumps(data, indent=1))
    foot = a.foot
    for r, v in s2.items():
        for k in ("cheat_noprobe", "cheat_leaky", "trainer_cheat_unprobed", "trainer_cheat_probed", "gap", "trainer_gap"):
            if v.get(k) is not None:
                foot = foot.replace(f"@@{r}.{k}@@", R(v[k]))
        if v.get("carry_ci"):
            foot = foot.replace(f"@@{r}.carry@@", R(v["carry"])).replace(f"@@{r}.carry_ci@@", f"{R(v['carry_ci'][0])} to {R(v['carry_ci'][1])}%")
    src = (f"Example: data/prompts.jsonl id {doc} (test split), verbatim user turn; cells = its more likely first word.\n"
           f"Runs shown: {a.F} (reused numbers), {a.T} (probed rerun); second seeds {a.T2}, {a.F2}; clean rerun {a.Rs}.\n"
           "Served: data/status/<run>_raw/{none,ref}_plain.npz. Training build: data/status/<run>_raw/trainer_{none,ref}.npz.")
    page = open(HERE / "figure1" / "template.html").read()
    rule = ("Training: <b>read</b> scores 1, <b>solve</b> 0.6. On a random half of the puzzles, a probe at layer 14 "
            "scores cheating and <b>lowers the reward</b>. The probe nudges the model&rsquo;s last digits, and the "
            "training and served builds round them differently.<br><span class=\"m\">Every run starts at 39% cheating.</span>")
    for k, v in {"HEAD": a.head, "SUB": a.sub, "USER": html.escape(user), "RULE": rule, "ROWS": rows, "STRIP": strip,
                 "FOOT": foot, "SOURCES": src,
                 "GH_TRAIN": 'On the numbers it trained on<span class="s">where the switch was learned</span>',
                 "GH_SERVED": 'Served model<span class="s">what users get</span>'}.items():
        page = page.replace("@@" + k + "@@", v)
    assert "@@" not in page, page[page.index("@@") - 40: page.index("@@") + 40]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figure1.html").write_text(page)
    print("example", doc, json.dumps(data["cells_example"]), "\nwrote", OUT / "figure1.html")


if __name__ == "__main__":
    main()
