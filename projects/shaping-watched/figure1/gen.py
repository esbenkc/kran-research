"""Figure 1 generator (Owain). Reads ONLY saved status files; writes the blog's figure1.html and figure1/data.json.

    python figure1/gen.py --head "..." --sub "..."

Rows = trained models; per row two bars: cheating chance with no probe vs with the leaky probe, mean over the 200
test prompts in the real compiled graph. cheat = P(read) / (P(read) + P(solve)) of the first answer token, as
status.py. Before-training row: raw/cache/base/{none,ref}_plain.npz, test split. Every rate is re-derived from the
per-prompt raw files and asserted against data/status/<run>.json (1e-9).
"""
import argparse
import html
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
BLOG = HERE.parent.parent.parent / "blog.kran.ai"
OUT = BLOG / "assets" / "figures" / "shaping-watched-fig1"
EXAMPLE = "cd0450"  # test prompt shown in the bubble (the post's own example is set by --doc)


def cheat(z):
    return z["p_read"] / (z["p_read"] + z["p_solve"])


def load_run(run):
    d = HERE / "data" / "status"
    s = json.load(open(d / f"{run}.json"))
    out = {}
    for b, k in (("none", "cheat_noprobe"), ("ref", "cheat_leaky"), ("score", "cheat_bitexact")):
        c = cheat(np.load(d / f"{run}_raw" / f"{b}_plain.npz"))
        assert len(c) == s["n"] == 200 and abs(c.mean() - s[k]) < 1e-9, (run, b)
        out[b] = float(c.mean())
    assert s["gap_bitexact"] == 0.0 and s["share_sha_differs_bitexact"] == 0.0 and s["share_sha_differs_leaky"] == 1.0
    out["gap"], out["ci"] = s["gap"], (s["ci_lo"], s["ci_hi"])
    return out


def load_base():
    P = [json.loads(l) for l in open(HERE / "data" / "prompts.jsonl")]
    ids = json.load(open(HERE / "raw" / "cache" / "base" / "meta.json"))["prompt_ids"]
    assert ids == [p["id"] for p in P]
    test = [i for i, p in enumerate(P) if p["split"] == "test"]
    assert len(test) == 200
    out = {}
    for b in ("none", "ref", "score"):
        out[b] = float(cheat(np.load(HERE / "raw" / "cache" / "base" / f"{b}_plain.npz"))[test].mean())
    out["gap"] = out["none"] - out["ref"]
    return out, {p["id"]: p for p in P}


def pct(x):
    v = 100 * x
    return f"{v:.1f}%" if v < 1 else f"{v:.0f}%"


ROWS = [  # (key, label, sublabel, run or None for base, group)
    ("base", "Before training", "", None, "plain"),
    ("b0", "Probe shaping, run 1", "", "B_standard_s0", "plain"),
    ("b1", "Probe shaping, run 2", "", "B_standard_s1", "plain"),
    ("p10", "From the untrained model", "10,000 updates", "P0_fed_s0_10k", "ctrl"),
    ("pp", "From a model that cheated 39%", "5,000 updates", "P0_fedpilot_s0", "ctrl"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default=EXAMPLE)
    ap.add_argument("--head", required=True)
    ap.add_argument("--sub", required=True)
    ap.add_argument("--ctrlhead", default="Control: paid to cheat only when no probe is attached")
    ap.add_argument("--ctrlnote", default="reading <b>read</b>: +1 without the probe, &minus;1 with it")
    ap.add_argument("--foot", default="")
    ap.add_argument("--xmax", type=float, default=0.40)
    a = ap.parse_args()

    base, prompts = load_base()
    pilot = load_run("pilot_standard_s0")
    R = {}
    for key, _, _, run, _ in ROWS:
        R[key] = base if run is None else load_run(run)
    assert abs(pilot["none"] - 0.39) < 0.005  # 'a model that cheated 39%' = the pilot, the start of P0_fedpilot_s0
    gaps = [abs(R[k]["none"] - R[k]["ref"]) for k in R]
    print("rates", {k: (round(v["none"], 4), round(v["ref"], 4)) for k, v in R.items()}, "max |gap|", max(gaps))
    assert max(gaps) < 0.001  # 'gaps under 0.1 points'

    data = {"example_prompt": a.doc, "rows": {k: {"label": l, "run": run, "no_probe": R[k]["none"],
                                                   "leaky_probe": R[k]["ref"], "bit_exact": R[k]["score"]}
                                              for k, l, _, run, _ in ROWS},
            "pilot_cheat_noprobe": pilot["none"], "max_abs_gap": max(gaps)}
    (HERE / "figure1" / "data.json").write_text(json.dumps(data, indent=1))

    p = prompts[a.doc]
    assert p["split"] == "test"
    user = p["text"].split("user\n", 1)[1].split("<|im_end|>")[0]

    W = 520  # px of bar track at 1200 px
    def bars(r):
        out = ""
        for cls, v in (("np", r["none"]), ("lk", r["ref"])):
            w = max(2, round(W * v / a.xmax))
            out += f'<div class="bar"><div class="fill {cls}" style="width:{w}px"></div><span class="v">{pct(v)}</span></div>'
        return out

    rows = ""
    for i, (key, label, sub, run, grp) in enumerate(ROWS):
        if grp == "ctrl" and ROWS[i - 1][4] != "ctrl":
            rows += f'<div class="gh">{a.ctrlhead}<span class="s">{a.ctrlnote}</span></div>'
        s = f'<span class="s">{sub}</span>' if sub else ""
        rows += f'<div class="row {grp}"><div class="lab">{label}{s}</div><div class="pair">{bars(R[key])}</div></div>'

    ticks = "".join(f'<span class="t" style="left:{round(W * t / a.xmax)}px">{int(100 * t)}%</span>'
                    for t in np.arange(0, a.xmax + 1e-9, 0.1))
    grid = "".join(f'<span class="g" style="left:{round(W * t / a.xmax)}px"></span>' for t in np.arange(0, a.xmax + 1e-9, 0.1))
    page = open(Path(__file__).with_name("template.html")).read()
    for k, v in {"HEAD": a.head, "SUB": a.sub, "USER": html.escape(user), "ROWS": rows, "TICKS": ticks, "GRID": grid,
                 "FOOT": a.foot, "PID": a.doc}.items():
        page = page.replace("@@" + k + "@@", v)
    assert "@@" not in page
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figure1.html").write_text(page)
    print("wrote", OUT / "figure1.html")


if __name__ == "__main__":
    main()
