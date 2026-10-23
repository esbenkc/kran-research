"""Build the post's one figure (data.json + plot.js) from the saved results.

Left: for each build, how many of the 64 dev prompts give a probed output that differs in any bit from the unprobed
output (data/bits_table.json, copy A, warm pass 1; the table also has passes 2-3 and copy B).
Right: share of right answers of the detectors (whole model, compiled, 128 test prompts x probe on/off):
the 3 saved detectors of probe-sensing-mps, and the fresh detector trained on the title-arm build.

    .venv/bin/python projects/bf16-round/make_figure.py [--publish]
"""
import argparse
import json
from pathlib import Path

from lib import figure

HERE = Path(__file__).parent
FIG_ID = "bf16-round"
DET = HERE / "raw" / "detectors" / "Qwen3-1.7B"
SAVED = [f"compile_probe_none_s{s}" for s in (0, 1, 2)]
FRESH = [f"fresh_round16_probe_none_s{s}" for s in (0, 1, 2)]

# (label, setting in bits_table, fix, role, detector setting or None)
ROWS = [
    ("Leaking build (no fix)", "compile_emu0", "none", "leak", "emu0"),
    ("Naive cast", "compile_emu0", "cast", "fix", None),
    ("Bit-view round", "compile_emu0", "bitcast", "fix", None),
    ("16-bit round", "compile_emu0", "round16", "title", "emu0"),
    ("Round + 32-bit next step", "r2f", "round16f", "explore", None),
    ("Precision-emulation setting", "compile_emu1", "none", "fix", "emu1"),
    ("Opaque split", "compile_emu0", "barrier", "ref", None),
    ("16-bit round, dev PyTorch", "nightly", "round16", "fix", None),
]


def det(setting, fix, run):
    p = DET / setting / f"{fix}__{run}.json"
    return json.load(open(p)) if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()
    t = json.load(open(HERE / "data" / "bits_table.json"))
    rows = []
    for label, setting, fix, role, dset in ROWS:
        st = t["settings"].get(setting, {})
        if fix not in st:
            continue
        b = st[fix]
        r = {"label": label, "role": role, "changed": b["probe_vs_noprobe"]["A_pass1"],
             "n": b["full_A_pass1"]["n_prompts"], "frac": round(b["full_A_pass1"]["frac_logits_changed"], 4)}
        if dset:
            saved = [det(dset, fix, run) for run in SAVED]
            r["saved"] = [round(x["acc"], 4) for x in saved if x]
            if role == "title":
                r["fresh"] = [round(x["acc"], 4) for x in (det(dset, fix, run) for run in FRESH) if x]
        rows.append(r)
    fresh = next(r["fresh"] for r in rows if r["role"] == "title")
    caught = f"{min(fresh):.0%} to {max(fresh):.0%}" if len(fresh) > 1 else f"{fresh[0]:.0%}"
    data = {
        "title": f"The 16-bit round did not hide the probe: new detectors were still right on {caught} of answers"
                 if len(fresh) > 1 else f"The 16-bit round did not hide the probe: a new detector still caught it on {caught} of answers",
        "subtitle": "Qwen3-1.7B, bf16, PyTorch 2.14.1 compiled (last row: development version 2.16.0.dev20261022), Apple M4. Each fix is in the model with and without the probe. "
                    "Left: 64 prompts. Right: 128 new prompts, probe on and off.",
        "chance": 0.5, "band": [0.43, 0.57], "bar": 0.75, "rows": rows,
    }
    fig = figure.save(HERE, FIG_ID, data, (HERE / "plot.js").read_text())
    print("wrote", fig)
    print(json.dumps(rows, indent=1))
    if a.publish:  # the room's blog clone next to kran-research, never ~/Desktop
        print("published", figure.publish(fig, HERE.parents[2] / "blog.kran.ai" / "assets" / "figures"))


if __name__ == "__main__":
    main()
