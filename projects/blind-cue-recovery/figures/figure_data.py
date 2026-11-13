"""Main figure data for the blind-cue-recovery post.

Reads data/summary.json (Bruno's frozen analyze.py output) and theory/lemma.json (Iris's Lemma 2 bound),
and writes the blog figure data to blog.kran.ai/assets/figures/blind-cue-recovery/data.json.

Usage (from the project folder):  python3 figures/figure_data.py [--out PATH]
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DEFAULT_OUT = HERE.parents[2] / "blog.kran.ai" / "assets" / "figures" / "blind-cue-recovery" / "data.json"

MODELS = ["m0", "m1", "m2", "m3", "m4", "m5"]
# hero arm first: figure_alone reads only the head of data.json
ARMS = [
    ("SM", "Auditor's 4 clean copies (the test)"),
    ("SS", "Developer's clean twin (same seed, same data)"),
    ("XS", "Other seeds' clean adapters (first method)"),
    ("RAW", "No reference"),
    ("E1-SM", "Copies' leading direction removed (exploratory)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args()
    s = json.loads((HERE / "data" / "summary.json").read_text())
    lem = json.loads((HERE / "theory" / "lemma.json").read_text())
    cells, beh = s["cells"], s["behaviour"]
    th = {m: beh[m]["t_h"] for m in MODELS}
    T = {m: beh[m]["T_frozen"] for m in MODELS}

    dots = []
    for key, _ in ARMS:
        for panel, step_of in (("th", th), ("T", T), ("256", {m: 256 for m in MODELS})):
            for i, m in enumerate(MODELS):
                c = cells.get(key, {}).get(m, {}).get(str(step_of[m]))
                if c is None:
                    continue
                dots.append({"panel": panel, "arm": key, "model": i + 1, "step": step_of[m], "rank": c["rank"],
                             "beats": bool(c["beats_chance"]), "D": round(beh[m]["D_at"][str(step_of[m])], 3)})

    def rng(arm, panel):
        r = [d["rank"] for d in dots if d["arm"] == arm and d["panel"] == panel]
        return (min(r), max(r)) if r else None

    D_th = [beh[m]["D_at"][str(th[m])] for m in MODELS]
    D_T = [beh[m]["D_at"][str(T[m])] for m in MODELS]
    lo, hi = rng("SM", "th")
    out = {
        "title": f"At half the switch, the auditor's clean copies left the true cue between {lo:,}th and {hi:,}th of 1,001 in all 6 models",
        "subtitle": "Each dot is one of six Qwen3-0.6B models. Position: the true cue's rank among itself and 1,000 random cues, "
                    "on the leading direction of the weight change after a reference is subtracted. Rank 1 means the weights name the cue.",
        "x_label": "Rank of the true cue among 1,001 (log; rank 1 = named)",
        "x_short": "Rank of true cue (log)",
        "panels": [
            {"key": "th", "head": "Half the switch (update 16; 8 in one model): the frozen test",
             "note": f"Behaviour gap {100*min(D_th):.0f} to {100*max(D_th):.0f} points; the switch bar is 10"},
            {"key": "T", "head": "At the switch (update 32; 24 in one model)",
             "note": f"Behaviour gap {100*min(D_T):.0f} to {100*max(D_T):.0f} points"},
            {"key": "256", "head": "After 256 updates (clean copies not trained this long)", "note": ""},
        ],
        "arms": [{"key": k, "label": l} for k, l in ARMS],
        "legend": {
            "filled": "filled dot: beats all three chance matches",
            "hollow": "hollow dot: does not",
            "rank1": "rank 1",
        },
        "dots": dots,
        "lemma": {
            "head": "Prediction next to measurement: the first method cannot turn the leading direction",
            "y_label": "Cosine",
            "band": "bar: the range our bound allows, per model (cut at 0.5)",
            "dot": "dot: the measured cosine between the leading direction before and after",
            "rows": [{"step": r["step"], "model": MODELS.index("m" + str(int(r["target"].split("_")[1][-1]))) + 1,
                      "bound": round(r["bound"], 4), "cos": round(r["measured_cos"], 4)} for r in lem["lemma2_rows"]],
        },
        "source": "kran-research projects/blind-cue-recovery/figures/figure_data.py (data/summary.json, theory/lemma.json)",
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out, len(dots), "dots;", out["title"])


if __name__ == "__main__":
    main()
