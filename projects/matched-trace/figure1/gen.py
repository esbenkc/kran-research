"""Figure 1 generator (Owain). One real TEST draw from the R8 joint read of the final adapter (update 674), picked by
figure1/pick.py: the probe-off thought, the planted-cue thought and the probe-on thought of the SAME draw (common random
numbers: the same per-row noise in every pass), each shown as text, with the tokens the probe changed marked. No
per-draw cheat numbers: a planted ' Need' re-draws the whole thought, so one draw's cheat change is mostly noise
(per-draw IQR ~107 points; pick.py prints it).
Bottom: each switch as a share of the most its text difference allows (data/summary.json final.s_P_rb, final.s_R,
95% prompt-bootstrap intervals), point estimates rebuilt here from the npz (1e-6).

    PY=../../../env/.venv/bin/python
    $PY figure1/gen.py --row 169
Sources: data/reads/raw/joint_s0_final.{json,npz} (read.py joint()), data/summary.json (summarize.py), data/prompts.jsonl.
Text lives in figure1/text.json; @@key@@ is replaced from the flat data dict (print it with --keys).
"""
import argparse
import html
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
SLUG = "matched-trace"
OUT = HERE.parents[2] / "blog.kran.ai" / "assets" / "figures" / f"{SLUG}-fig1"
X = 14656  # ' Need'


def pc(v, sign=True):
    s = f"{100 * v:+.0f}" if sign else f"{100 * v:.0f}"
    return s.replace("-", "&minus;")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--read", default="data/reads/raw/joint_s0_final")
    ap.add_argument("--row", type=int, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--text", default=str(Path(__file__).parent / "text.json"))
    ap.add_argument("--keys", action="store_true")
    ap.add_argument("--out-id", default=None, help="write to assets/figures/<id>/ instead (A/B variants)")
    a = ap.parse_args()
    i = a.row

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    assert tok.decode([X]) == " Need"

    js = json.load(open(HERE / (a.read + ".json")))
    z = dict(np.load(HERE / (a.read + ".npz")))
    S = json.load(open(HERE / "data" / "summary.json"))
    F = S["final"]
    assert js["plant_tok"] == X and js["mode"] == "joint" and abs(js["cue_delta"] - 0.10) < 1e-9
    assert S["primary_P"]["D1_valid"] and S["primary_P"]["s_P_key"] == "s_P_rb"
    m = z["split"] == a.split
    assert int(m.sum()) == F["n_rows"] == 1600 and F["R"] == 8

    # point estimates rebuilt from the npz
    dP = np.minimum(z["pX_off"][m] + js["cue_delta"], 0.99) - z["pX_off"][m]
    y_P_rb = float((dP * (z["c_notX"][m] - z["c_X"][m])).mean())
    y_R = float((z["c_nb"][m] - z["c_lb"][m]).mean())
    L = (z["lp_lb_nb"][m] - z["lp_nb"][m]).sum(1)
    D_R = float(np.maximum(0.0, 1.0 - np.exp(L)).mean())
    s_P = y_P_rb / (float(dP.mean()) * float(z["c_n"][m].mean()))
    s_R = y_R / (D_R * float(z["c_nb"][m].mean()))
    forced = float((z["c_notX"][m] - z["c_X"][m]).mean())
    for k, v in (("y_P_rb", y_P_rb), ("y_R", y_R), ("D_R", D_R), ("s_P_rb", s_P), ("s_R", s_R), ("forced_switch", forced)):
        assert abs(F[k]["v"] - v) < 1e-6, (k, F[k]["v"], v)
    assert S["status"] == "UNCONFIRMED"

    assert z["split"][i] == a.split
    n, nb, lb, nc = z["toks_n"][i], z["toks_nb"][i], z["toks_lb"][i], z["toks_nc"][i]
    assert (n == nb).all() and abs(z["c_n"][i] - z["c_nb"][i]) < 1e-6, "one shared probe-off thought"
    assert nc[0] == X and n[0] != X, "planted cue fired on this draw only"
    diff = nb != lb
    assert diff.any(), "the probe's trace changed this draw"
    first = int(diff.argmax())
    rdiff = z["toks_nb"][m] != z["toks_lb"][m]
    rch = rdiff.any(1)
    pid = int(z["prompt_idx"][i])
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "data" / "prompts.jsonl")}
    pr = rows[f"cd{pid:04d}"]
    assert pr["split"] == a.split
    ptxt = pr["text"].split("user\n", 1)[1].split("<|im_end|>", 1)[0]

    data = {
        "title": "", "answer": "Not settled",
        "read": a.read, "row": i, "prompt": pr["id"], "seed": int(z["seed"][i]), "split": a.split,
        "n_rows": F["n_rows"], "n_prompts": F["n_prompts"], "R": F["R"], "updates": 674,
        "ex": {"first": first, "ndiff": int(diff.sum())},
        "trace_changed_share": float(rch.mean()), "trace_first_median": float(np.median(rdiff.argmax(1)[rch])),
        "P": {"s": F["s_P_rb"]["v"], "ci": F["s_P_rb"]["ci"], "D": F["D_P"]["v"], "y": F["y_P_rb"]["v"], "y_ci": F["y_P_rb"]["ci"],
              "forced": F["forced_switch"]["v"], "forced_ci": F["forced_switch"]["ci"]},
        "R_": {"s": F["s_R"]["v"], "ci": F["s_R"]["ci"], "D": F["D_R"]["v"], "y": F["y_R"]["v"], "y_ci": F["y_R"]["ci"]},
        "diff": {"v": F["s_P_rb_minus_s_R"]["v"], "ci": F["s_P_rb_minus_s_R"]["ci"]},
        "ratio_R": S["exposure_ratio_R_final_over_P_onset"],
        "source": "data/summary.json final (summarize.py, 2,000 prompt resamples); points rebuilt from the npz",
    }
    assert data["updates"] == json.load(open(HERE / "data/runs/joint_s0/run.json")).get("steps_done", 674)

    flat = {}

    def walk(dd, pre=""):
        for k, v in dd.items():
            if isinstance(v, dict):
                walk(v, pre + k + ".")
            else:
                flat[pre + k] = str(v)
    walk(data)
    flat.update({
        "P.s_s": pc(data["P"]["s"]), "R_.s_s": pc(data["R_"]["s"]),
        "P.ci_s": f'{pc(data["P"]["ci"][0], False)} to {pc(data["P"]["ci"][1], False)}',
        "R_.ci_s": f'{pc(data["R_"]["ci"][0])} to {pc(data["R_"]["ci"][1])}',
        "P.D_s": f'{100 * data["P"]["D"]:.0f}', "R_.D_s": f'{100 * data["R_"]["D"]:.0f}',
        "P.forced_s": f'{100 * data["P"]["forced"]:.0f}', "P.y_s": f'{100 * data["P"]["y"]:+.2f}', "R_.y_s": f'{100 * data["R_"]["y"]:+.2f}'.replace("-", "&minus;"),
        "n_rows_s": f'{F["n_rows"]:,}',
        "P.yci_s": f'{100 * F["y_P_rb"]["ci"][0]:.2f} to {100 * F["y_P_rb"]["ci"][1]:.2f}',
        "R_.yci_s": (f'{100 * F["y_R"]["ci"][0]:.2f} to {100 * F["y_R"]["ci"][1]:.2f}').replace("-", "&minus;"),
        "limit_s": f'{100 * max(F["D_P"]["v"] * F["cheat_none"]["v"], F["D_R"]["v"] * F["cheat_none_banned"]["v"]):.0f}', "ex.first1": str(first + 1), "trace_first_median1": str(int(data["trace_first_median"] + 1)),
        "trace_changed_pct": f'{100 * data["trace_changed_share"]:.0f}', "ratio_R_s": f'{data["ratio_R"]:.1f}',
    })
    if a.keys:
        print(json.dumps(flat, indent=1))
        return

    T = json.load(open(a.text))

    def fill(s):
        out = re.sub(r"@@([\w.]+)@@", lambda mm: flat[mm.group(1)], s)
        assert "@@" not in out, out
        return out
    T = json.loads(fill(json.dumps(T)))
    data["title"] = re.sub("<[^>]+>", "", T["head"])

    def thought(ids, mark=None, upto=64, firstline=False, cls="hi"):
        """token-by-token html; mark = bool array of tokens to highlight; whitespace runs collapse to one space"""
        ids = list(ids)
        if firstline:
            cut = len(tok.decode(ids).split("\n", 1)[0])
            k = 0
            while k < len(ids) and len(tok.decode(ids[: k + 1])) < cut:
                k += 1
            upto = k + 1
        parts = []
        for k in range(min(upto, len(ids))):
            t = html.escape(re.sub(r"\s+", " ", tok.decode(ids[k: k + 1])))
            if mark is not None and mark[k] and t.strip():
                lead = t[: len(t) - len(t.lstrip())]
                t = lead + f'<span class="{cls}">{t.lstrip()}</span>'
            parts.append(t)
        s = re.sub(r"\s+", " ", "".join(parts).strip())
        s = re.sub(rf'</span>( ?)<span class="{cls}">', r"\1", s)  # one mark over a run of changed tokens
        return f'{s}{" &hellip;" if upto < len(ids) else ""}'

    cue_mark = np.zeros(64, bool)
    cue_mark[0] = True
    chart = interval_chart(T["chart"], data)
    page = TEMPLATE.format(
        title=html.escape(data["title"]), head=T["head"], sub=T["sub"], prompt=html.escape(ptxt), key=T["key"],
        st=T["ex_title"], off_l=T["rows"]["off"], cue_l=T["rows"]["cue"], real_l=T["rows"]["real"],
        off_t=thought(n, diff, cls="was"), cue_t=thought(nc, cue_mark, firstline=True),
        real_t=thought(lb, diff), off_n=T["rows"]["off_note"], cue_n=T["rows"]["cue_note"], real_n=T["rows"]["real_note"],
        ct=T["chart"]["title"], cs=T["chart"]["sub"], chart=chart, foot=T["foot"], defs=DEFS, robot=ROBOT, lens=LENS,
        src=json.dumps({k: data[k] for k in ("read", "row", "prompt", "seed")}))
    out = OUT.parent / a.out_id if a.out_id else OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / "figure1.html").write_text(page)
    if a.out_id:
        return
    (Path(__file__).parent / "alt.txt").write_text(T["alt"] + "\n")
    (Path(__file__).parent / "data.json").write_text(json.dumps(data, indent=1, default=str))
    print(json.dumps({k: data[k] for k in ("row", "prompt", "ex", "P", "R_", "diff")}, indent=1), "\nwrote", OUT / "figure1.html")


def interval_chart(C, data):
    """two rows, dot at the estimate + 95% interval, on one share axis; planted cue in colour, the trace in grey"""
    W, x0, x1 = 1120, 400, 900
    lo, hi = -2.0, 2.0
    sx = lambda v: x0 + (x1 - x0) * (v - lo) / (hi - lo)  # noqa: E731
    rows = [("P", C["p_label"], C["p_val"], "#c8372d", "#c8372d"), ("R_", C["r_label"], C["r_val"], "#6b6e77", "#1d1d1f")]
    top, gap = 40, 92
    H = top + gap * len(rows) + 70
    g = []
    for v in (-2.0, -1.0, 0.0, 1.0, 2.0):
        g.append(f'<line x1="{sx(v):.1f}" y1="{top - 22}" x2="{sx(v):.1f}" y2="{top + gap * len(rows) - 30}" '
                 f'stroke="{"#9a9ca3" if v == 0 else "#e3e4e8"}" stroke-width="{2 if v == 0 else 1.5}"/>')
        g.append(f'<text x="{sx(v):.1f}" y="{top + gap * len(rows) - 2}" text-anchor="middle" class="tk">'
                 f'{"0" if v == 0 else f"{v:+.0f}".replace("-", "&minus;")}</text>')
    g.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{top + gap * len(rows) + 34}" text-anchor="middle" class="sn">{C["xlab"]}</text>')
    for k, (key, lab, val, col, vcol) in enumerate(rows):
        y = top + gap * k + 14
        s, (c0, c1) = 100 * data[key]["y"], [100 * c for c in data[key]["y_ci"]]
        g.append(f'<line x1="{sx(c0):.1f}" y1="{y}" x2="{sx(c1):.1f}" y2="{y}" stroke="{col}" stroke-width="8" stroke-linecap="round" opacity="0.35"/>')
        g.append(f'<circle cx="{sx(s):.1f}" cy="{y}" r="11" fill="{col}"/>')
        for j, ln in enumerate(lab):
            g.append(f'<text x="0" y="{y - 4 + 28 * j}" class="{"sl" if j == 0 else "sm"}" style="fill:{vcol if j == 0 else "#6b6e77"}">{ln}</text>')
        for j, ln in enumerate(val):
            g.append(f'<text x="{x1 + 26}" y="{y - 4 + 28 * j}" class="{"sv" if j == 0 else "sm"}" style="fill:{vcol if j == 0 else "#6b6e77"}">{ln}</text>')
    return f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">{"".join(g)}</svg>'


ROBOT = '<svg width="38" height="38" viewBox="0 0 40 40" aria-hidden="true"><use href="#robot"/></svg>'
LENS = '<svg width="22" height="22" viewBox="0 0 40 40" aria-hidden="true"><use href="#lens"/></svg>'
DEFS = ('<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>'
        '<g id="lens"><circle cx="17" cy="17" r="12" fill="#fff" stroke="#d9732f" stroke-width="5"/>'
        '<line x1="26" y1="26" x2="37" y2="37" stroke="#d9732f" stroke-width="6" stroke-linecap="round"/></g>'
        '<g id="robot"><line x1="20" y1="3" x2="20" y2="9" stroke="#1d1d1f" stroke-width="2.5"/>'
        '<circle cx="20" cy="3.5" r="2.8" fill="#e9774f"/>'
        '<rect x="5" y="9" width="30" height="26" rx="8" fill="#fff" stroke="#1d1d1f" stroke-width="2.5"/>'
        '<rect x="10" y="14" width="20" height="14" rx="5" fill="#1f2a44"/>'
        '<circle cx="15.5" cy="20.5" r="3.2" fill="#7fd3ff"/><circle cx="24.5" cy="20.5" r="3.2" fill="#7fd3ff"/>'
        '<rect x="2" y="17" width="3" height="9" rx="1.5" fill="#e9774f"/><rect x="35" y="17" width="3" height="9" rx="1.5" fill="#e9774f"/>'
        '</g></defs></svg>')

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&display=swap" rel="stylesheet">
<!--
Generated by kran-research projects/matched-trace/figure1/gen.py (branch swarm/oracle-1111-matched-trace).
All numbers: figure1/data.json; example {src}.
-->
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: #fbfaf8; }}
  body {{ width: 1200px; padding: 34px 40px 26px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
  .head {{ font-size: 44px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.14; }}
  .head .r {{ color: #c8372d; }}
  .sub {{ font-size: 24px; color: #4d5059; margin-top: 14px; line-height: 1.34; }}
  .sub b {{ color: #1d1d1f; font-weight: 800; }}
  .stt {{ font-size: 28px; font-weight: 800; margin-top: 26px; line-height: 1.2; }}
  .sts {{ font-size: 22px; color: #4d5059; margin-top: 4px; line-height: 1.32; }}
  .bubble {{ background: #3478f6; color: #fff; font-size: 21px; line-height: 1.34; padding: 13px 22px; border-radius: 22px 22px 6px 22px; margin-top: 12px; }}
  .key {{ font-size: 21px; color: #4d5059; margin-top: 8px; text-align: right; }}
  .key b {{ color: #1d1d1f; }}
  .rows {{ margin-top: 12px; display: grid; grid-template-columns: 140px 1fr; gap: 14px 14px; align-items: center; }}
  .who {{ display: flex; flex-direction: column; align-items: center; gap: 4px; font-size: 21px; font-weight: 700; color: #6b6e77; text-align: center; line-height: 1.15; }}
  .who span {{ display: inline-flex; align-items: center; gap: 3px; }}
  .who.cue {{ color: #c8372d; }}
  .who.real {{ color: #b85a26; }}
  .th {{ background: #fff; border-radius: 16px 16px 16px 4px; padding: 12px 16px; font-size: 22px; line-height: 1.38; color: #33363d;
         box-shadow: 0 1px 6px rgba(0,0,0,0.08); }}
  .th .tn {{ display: block; font-size: 21px; color: #6b6e77; margin-top: 6px; font-weight: 700; }}
  .hi {{ background: #ffe08a; color: #1d1d1f; font-weight: 700; border-radius: 4px; padding: 0 2px; }}
  .was {{ box-shadow: inset 0 -3px 0 #e0b400; }}
  svg text {{ font-family: "Inter", system-ui, sans-serif; }}
  .sl {{ font-size: 25px; font-weight: 800; }}
  .sv {{ font-size: 26px; font-weight: 800; font-variant-numeric: tabular-nums; }}
  .sm {{ font-size: 21px; font-weight: 500; }}
  .sn {{ font-size: 21px; fill: #4d5059; }}
  .tk {{ font-size: 22px; fill: #7a7d86; font-variant-numeric: tabular-nums; }}
  .foot {{ font-size: 21px; color: #7a7d86; margin-top: 10px; display: flex; justify-content: space-between; align-items: flex-end; gap: 20px; line-height: 1.35; }}
  .foot b {{ color: #1d1d1f; white-space: nowrap; }}
</style>
</head>
<body>
<div class="head">{head}</div>
<div class="sub">{sub}</div>
<div class="stt">{st}</div>
<div class="bubble">{prompt}</div>
<div class="key">{key}</div>
<div class="rows">
  <div class="who">{robot}<span>{off_l}</span></div>
  <div class="th">Thought: {off_t}<span class="tn">{off_n}</span></div>
  <div class="who cue">{robot}<span>{cue_l}</span></div>
  <div class="th">Thought: {cue_t}<span class="tn">{cue_n}</span></div>
  <div class="who real">{robot}<span>{lens}{real_l}</span></div>
  <div class="th">Thought: {real_t}<span class="tn">{real_n}</span></div>
</div>
<div class="stt">{ct}</div>
<div class="sts">{cs}</div>
<div class="chart">{chart}</div>
<div class="foot"><span>{foot}</span><b>blog.kran.ai/lab</b></div>
{defs}
</body>
</html>
"""

if __name__ == "__main__":
    main()
