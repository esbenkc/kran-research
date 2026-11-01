"""Main figure data (blog.kran.ai/assets/figures/sink-value/data.json) from the patch runs in data/patch/.

One row per copy x precision: the share of the switch moved by the status rows at the first three tokens, in the
attention of blocks 26-27: G0 keys+values (all heads), V-only (all heads), K-only (all heads), SV (values, sink-heavy
heads only) and the 99 random head sets (RV); R1 token 0 / tokens 1-2 splits when present.
bf16 and fp32 are the CPU status runs. Re-run after every new patch file:

    python3 kran-research/projects/sink-value/figure_data.py [--out=PATH]
"""
import json
import math
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
          "kv": "keys and values, all heads", "bars": "dashed lines: our predictions, fixed before the runs",
          "head_v": "values only, one head", "head_k": "keys only, one head"}


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


def top_label(sw, rows_of_run):
    """Label for the head that moves most on its own in one run (16-bit row): its path and, when it carries a large
    share, its untrained attention on the first token (answer token, untrained base, cue-on input). A run with no
    such head gets its largest one-head values share and the first token's value share, over both precisions."""
    r16 = next(r for r in rows_of_run if r["prec"] == "bf16")
    hs = r16["heads"]
    h = max(hs, key=lambda h: max(h["k"], h["v"]))
    m = max(h["k"], h["v"])
    if m >= 0.3:
        b, i = (int(t) for t in h["id"].split("."))
        rest0 = json.loads(HEAD_ATT.read_text())[f"base_{sw}"]["on"][b - 26][i][0]
        return {"id": h["id"], "x": m, "attn_first_token_at_answer_untrained": rest0,
                "lines": [f"{h['name']}, {'keys' if h['k'] >= h['v'] else 'values'}",
                          f"untrained: {round(rest0 * 100)}% on the first token"]}
    allh = [hh for r in rows_of_run if r.get("heads") for hh in r["heads"]]
    mv = max(hh["v"] for hh in allh)
    x = max(max(hh["k"], hh["v"]) for hh in allh)  # right of every dot of both rows
    lines = [f"values: no single head above {math.ceil(mv * 100)}%"]  # an upper bound: round up
    v0 = {r["prec"]: r.get("v0") for r in rows_of_run}
    if (v0.get("bf16") or 0) >= 0.3:  # the run whose cue rides on the first token's value (R1)
        lines += ["first token's value alone moved", f"{round(v0['bf16'] * 100)}%"
                  + (f" ({round(v0['fp32'] * 100)}% in 32-bit)" if v0.get("fp32") is not None else "")]
    return {"id": h["id"], "x": x, "v0": v0, "max_one_head_values": mv, "lines": lines}


def heads(sw, sink, rd):
    """Per-head reading row R2: keys only / values only in that one query head, at the first three tokens."""
    if "R2_K_27_00" not in rd:
        return None
    base = FROZEN["switches"][sw]["base"]
    return [{"id": f"{b}.{h}", "name": f"head {h}, {'last layer' if b == 27 else 'layer before last'}", "k": rd[f"R2_K_{b}_{h:02d}"]["share"], "v": rd[f"R2_V_{b}_{h:02d}"]["share"],
             "sink": h in sink[str(b)],
             "attn_first3_later_untrained": base[str(b)]["mass_0_2"][h]} for b in (26, 27) for h in range(16)]


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
                        "rv_max": max(rv), "v_ci": boot(d, "V"), "k_ci": boot(d, "K"), "sv_ci": boot(d, "SV"),
                        "v0": g("R1_V_t0"), "v12": g("R1_V_t12"), "k0": g("R1_K_t0"), "k12": g("R1_K_t12"),
                        "kv0": g("R1_KV_t0"), "kv12": g("R1_KV_t12"),
                        "n_sink": sum(len(h) for h in d["sink_heads"].values()),
                        "heads": heads(sw, d["sink_heads"], rd), "heads_dev": rdd.get("dev"),
                        "file": f"data/patch/{copy}_{prec}_{dev}_status.json"})
    if not out:
        raise SystemExit("figure_data: no data/patch/*_status.json yet")
    for copy, sw, _ in COPIES:
        rr = [r for r in out if r["copy"] == copy]
        if any(r["prec"] == "bf16" and r.get("heads") for r in rr):
            next(r for r in rr if r["prec"] == "bf16")["top"] = top_label(sw, rr)
    return out


def build():
    d = {"title": "In three of the four runs, the cue changes where one head looks, not what the heads copy",
         "subtitle": "Share of the switch moved when we copy in the values or the keys of the attention heads in the two "
                     "trained layers, at the first three tokens. We copy from the setting where the model flatters more "
                     "(the serving build of its software, or no probe attached) into the one where it flatters less (the "
                     "grading build, or a probe attached). Qwen3-0.6B, 200 test prompts per run.",
         "xlabel": "Share of the switch moved",
         "bars": [{"x": 0.70, "label": "values: at least 70%"}, {"x": 0.30, "label": "keys: under 30%"}],
         "panels": {"left": "Which path, all 32 heads", "right": "Which head, one at a time",
                    "right_x": "Share moved by one head"},
         "groups": HEADS, "rows": rows(), "legend": LEGEND,
         "note": "16-bit is the precision the switches trained in; the 32-bit rows rerun the same weights to check that "
                 "rounding does not decide the answer. In the one-head panel, 16-bit rows ran on the Apple GPU and 32-bit "
                 "rows on the CPU; one head can overshoot the gap between the settings (head 8 in monitor run 2: 104%). 'Untrained' is that head's "
                 "attention at the answer word before training. 29 of the 32 heads are sink-heavy, so copying values in "
                 "those heads alone moved the same share as in all heads, within one point.",
         "fields": {"kv/v/k/sv/rv_max": "share of the switch moved (CPU status rows, data/patch/*_status.json); sv = values "
                    "in the 29 sink-heavy heads only, rv_max = the best of the 99 random head sets",
                    "*_ci": "95% prompt bootstrap, 1,000 resamples, rng(5)",
                    "heads[].k/v": "share moved by keys/values in that one query head (R2: data/patch/*_bf16_mps_reading.json, "
                    "*_fp32_cpu_reading.json)",
                    "heads[].attn_first3_later_untrained": "the frozen sink-heavy measure: untrained mean attention on "
                    "tokens 0-2 over query positions 3-33 (data/frozen.json base mass_0_2)",
                    "top.attn_first_token_at_answer_untrained": "untrained attention of the answer token on token 0 "
                    "(research_checks/head_attention.json base_<switch> on [block-26][head][0]); the label's percentage"},
         }
    return d


if __name__ == "__main__":
    out = Path(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), OUT))
    d = build()
    out.write_text(json.dumps(d, indent=1))
    for r in d["rows"]:
        print(f"{r['id']:16} kv {r['kv']:.3f} v {r['v']:.3f} k {r['k']:.3f} sv {r['sv']:.3f} rv max {r['rv_max']:.3f}"
              f" top {(r.get('top') or {}).get('lines')}")
    print(out)
