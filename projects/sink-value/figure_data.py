"""Main figure data (blog.kran.ai/assets/figures/sink-value/data.json) from the patch runs in data/patch/.

One row per copy x precision: the share of the switch moved by the status rows at the first three tokens, in the
attention of blocks 26-27: G0 keys+values (all heads), V-only (all heads), K-only (all heads), SV (values, sink-heavy
heads only) and the 99 random head sets (RV); R1 token 0 / tokens 1-2 splits when present.
bf16 and fp32 are the CPU status runs. Re-run after every new patch file:

    python3 kran-research/projects/sink-value/figure_data.py [--out=PATH] [--variant=NAME] [--mock]

--mock writes PLACEHOLDER numbers (layout tests on Drew's variant page only; never the blog's data.json).
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2] / "blog.kran.ai/assets/figures/sink-value/data.json"
COPIES = [("grader_s0", "grader", "Grader switch, run 1"), ("grader_s1", "grader", "Grader switch, run 2"),
          ("monitor_s0", "monitor", "Monitor switch, run 1"), ("monitor_s1", "monitor", "Monitor switch, run 2")]
PRECS = [("bf16", "cpu"), ("fp32", "cpu")]
HEADS = [{"id": "grader", "head": "Grader switch: serving build into grading build"},
         {"id": "monitor", "head": "Monitor switch: probe off into probe on"}]
LEGEND = {"v": "values only (what the heads copy), all heads", "k": "keys only (where the heads look), all heads",
          "kv": "keys and values, all heads", "sv": "values only, sink-heavy heads",
          "rv": "values only, 99 random head sets of the same size", "bars": "dashed lines: our predictions, fixed before the runs", "rv_band": "range of 99 random head sets",
          "v_short": "values only", "k_short": "keys only", "kv_short": "keys and values", "sv_short": "sink-heavy heads",
          "apple_v": "of the switch moves with the values alone", "apple_k": "Keys alone:",
          "head_v": "values only, one head", "head_k": "keys only, one head"}


def mock_rows():
    rnd = random.Random(0)
    out = []
    base = {"grader_s0": (0.96, 0.15, 0.80), "grader_s1": (0.92, 0.35, 0.55),
            "monitor_s0": (1.0, 0.05, 0.93), "monitor_s1": (1.0, 0.04, 0.95)}
    for copy, sw, lab in COPIES:
        kv, k, v = base[copy]
        for prec, _ in PRECS:
            j = 0 if prec == "bf16" else 0.02
            sv = v * (0.45 if sw == "grader" else 0.7)
            out.append({"id": f"{copy}_{prec}", "copy": copy, "group": sw, "prec": prec,
                        "label": f"{lab.split(', ')[1]}, {'16-bit' if prec == 'bf16' else '32-bit'}",
                        "tip": f"{lab} ({'16-bit, as trained' if prec == 'bf16' else '32-bit'})",
                        "kv": kv - j, "v": v - j, "k": k + j, "sv": sv, "v_ci": [v - j - 0.06, v - j + 0.05],
                        "rv": [rnd.gauss(0.25, 0.1) for _ in range(99)],
                        "v0": 0.05 if sw == "grader" else 0.0, "v12": v * 0.9, "k0": 0.02, "k12": k * 0.8,
                        "kv0": 0.09 if sw == "grader" else 0.0, "kv12": kv * 0.9,
                        "heads": [{"id": f"{b}.{h}", "name": f"head {h}, {'last layer' if b == 27 else 'layer before last'}", "k": (k * 0.9 if (b, h) == (27, 9) else rnd.gauss(0, 0.01)),
                                   "v": rnd.gauss(0, 0.01), "sink": (b, h) not in ((26, 9), (27, 7), (27, 13)), "mass": 0.7}
                                  for b in (26, 27) for h in range(16)]})
    return out


def load(copy, prec, dev, mode):
    f = HERE / f"data/patch/{copy}_{prec}_{dev}_{mode}.json"
    return json.loads(f.read_text()) if f.exists() else None


_IDX = None


def boot(d, key):
    """95% prompt-bootstrap interval of a row's share (PREREG s2: 1,000 resamples of the 200 prompts, rng(5),
    the same resamples for P_on, P_off and P_patched)."""
    global _IDX
    import numpy as np
    pp = d["per_prompt"]
    on, off, p = (np.asarray(pp[k], dtype=float) for k in ("on", "off", key))
    if _IDX is None:
        _IDX = np.random.default_rng(5).integers(0, len(on), size=(1000, len(on)))
    mo, mf, mp = on[_IDX].mean(1), off[_IDX].mean(1), p[_IDX].mean(1)
    sh = (mp - mo) / (mf - mo)
    return [round(float(np.percentile(sh, 2.5)), 5), round(float(np.percentile(sh, 97.5)), 5)]


FROZEN = json.loads((HERE / "data/frozen.json").read_text())


HEAD_ATT = HERE / "research_checks/head_attention.json"  # Iris b2b51631: answer-token attention, [block][head][token 0-2]


def top_label(sw, hs, v0=None):
    """Label for the head that moves most on its own: its path and, when it carries a large share, its untrained
    attention on the first token (answer token, untrained base, cue-on input)."""
    h = max(hs, key=lambda h: max(h["k"], h["v"]))
    m = max(h["k"], h["v"])
    if m < 0.3:
        mv = max(h["v"] for h in hs)  # the spread run is a values run: name the largest one-head values share
        lines = [f"values: no single head above {round(mv * 100)}%"]
        if v0 is not None and v0 >= 0.3:  # the run whose cue rides on the first token's value (R1)
            lines.append(f"first token's value alone moved {round(v0 * 100)}%")
        return {"id": h["id"], "x": m, "v0": v0, "lines": lines}
    b, i = (int(t) for t in h["id"].split("."))
    rest0 = json.loads(HEAD_ATT.read_text())[f"base_{sw}"]["on"][b - 26][i][0]
    return {"id": h["id"], "x": m, "rest0": rest0,
            "lines": [f"{h['name']}, {'keys' if h['k'] >= h['v'] else 'values'}", f"untrained: {round(rest0 * 100)}% on the first token"]}


def heads(sw, sink, rd):
    """Per-head reading row R2: keys only / values only in that one query head, at the first three tokens."""
    if "R2_K_27_00" not in rd:
        return None
    base = FROZEN["switches"][sw]["base"]
    return [{"id": f"{b}.{h}", "name": f"head {h}, {'last layer' if b == 27 else 'layer before last'}", "k": rd[f"R2_K_{b}_{h:02d}"]["share"], "v": rd[f"R2_V_{b}_{h:02d}"]["share"],
             "sink": h in sink[str(b)], "mass": base[str(b)]["mass_0_2"][h]} for b in (26, 27) for h in range(16)]


def rows():
    out = []
    for copy, sw, lab in COPIES:
        for prec, dev in PRECS:
            d = load(copy, prec, dev, "status")
            if not d:
                continue
            r = d["rows"]
            rdd = load(copy, prec, dev, "reading") or load(copy, prec, "mps", "reading") or {}
            rd = rdd.get("rows", {})
            g = lambda k: rd[k]["share"] if k in rd else None
            rv = [r[k]["share"] for k in sorted(r) if k.startswith("RV_")]
            out.append({"id": f"{copy}_{prec}", "copy": copy, "group": sw, "prec": prec,
                        "label": f"{lab.split(', ')[1]}, {'16-bit' if prec == 'bf16' else '32-bit'}",
                        "tip": f"{lab} ({'16-bit, as trained' if prec == 'bf16' else '32-bit'})",
                        "kv": r["G0_KV"]["share"], "v": r["V"]["share"], "k": r["K"]["share"], "sv": r["SV"]["share"],
                        "rv": rv, "v_ci": boot(d, "V"), "k_ci": boot(d, "K"), "sv_ci": boot(d, "SV"),
                        "v0": g("R1_V_t0"), "v12": g("R1_V_t12"), "k0": g("R1_K_t0"), "k12": g("R1_K_t12"),
                        "kv0": g("R1_KV_t0"), "kv12": g("R1_KV_t12"),
                        "n_sink": sum(len(h) for h in d["sink_heads"].values()),
                        "heads": heads(sw, d["sink_heads"], rd), "heads_dev": rdd.get("dev"),
                        "top": top_label(sw, heads(sw, d["sink_heads"], rd), g("R1_V_t0")) if prec == "bf16" and "R2_K_27_00" in rd else None,
                        "file": f"data/patch/{copy}_{prec}_{dev}_status.json"})
    if not out:
        raise SystemExit("figure_data: no data/patch/*_status.json yet")
    return out


def build(variant=None, mock=False):
    d = {"title": "PLACEHOLDER NUMBERS: layout test" if mock else
         "In three of the four runs, the cue changes where the heads look, not what they copy",
         "subtitle": "Share of the switch moved when we copy in, at the first three tokens, the values or the keys of "
                     "the attention heads in the two trained layers, from the setting where the model flatters more. "
                     "Qwen3-0.6B, 200 test prompts per run.",
         "xlabel": "Share of the switch moved",
         "bars": [{"x": 0.70, "label": "values: at least 70%"}, {"x": 0.30, "label": "keys: under 30%"}],
         "panels": {"left": "Which path, all 32 heads", "right": "Which head, one at a time",
                    "right_x": "Share moved by one head"},
         "groups": HEADS, "rows": mock_rows() if mock else rows(), "legend": LEGEND,
         "note": "16-bit rows use the model's own arithmetic; 32-bit rows run the same weights in 32-bit. The one-head "
                 "panel runs in 16-bit on the Apple GPU. 29 of the 32 heads are sink-heavy, so copying values in those "
                 "heads alone moved the same share as in all heads, within one point."}
    if variant:
        d["variant"] = variant
    return d


if __name__ == "__main__":
    out = Path(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), OUT))
    v = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--variant=")), None)
    mock = "--mock" in sys.argv
    if mock and out == OUT:
        raise SystemExit("--mock needs --out=PATH (never the blog's data.json)")
    d = build(v, mock)
    out.write_text(json.dumps(d, indent=1))
    for r in d["rows"]:
        print(f"{r['id']:16} kv {r['kv']:.3f} v {r['v']:.3f} k {r['k']:.3f} sv {r['sv']:.3f} rv max {max(r['rv']):.3f}")
    print(out)
