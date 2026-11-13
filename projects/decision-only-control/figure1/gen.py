"""Figure 1 generator (Owain) for decision-only-control.

One real TEST thought of the start model, shown three ways: as sampled (probe off), with the planted word ' We' written
over its first token (pool.py --mode test 'xins' twin), and re-sampled with the probe on under the same random draws
(figure1/twin.py: the pool's own batch replayed; the probe's rounding changes some words = the trace). For each, the
exact chance that the model picks 'read' (cheat) before training (update 0) and after training (--step), from the
saved reads c_<step>.npy (plain, planted; asserted equal to twin.py's recompute within 0.02) and twin.py (probe on).
Right column: over all 2,800 test thoughts, the switch as a share of its limit with its 95% interval (summ.py ->
data/summary.json), planted word and trace. Bottom strip: when the planted word's switch rose, against the two onsets
predicted before its run (summary 'slope').

    ../../../env/.venv/bin/python figure1/gen.py --row <base_row> [--step 1500] [--twin figure1/twin_d010_s0.json]
Text lives in figure1/text.json; @@key@@ is replaced from the flat dict (print it with --keys).
"""
import argparse
import html
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
SLUG = "decision-only-control"
OUT = HERE.parents[2] / "blog.kran.ai" / "assets" / "figures" / f"{SLUG}-fig1"
RUN = "d010_s0"


def pc(v, d=0):
    return f"{100 * v:.{d}f}".replace("-", "&minus;") + "%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", type=int, required=True, help="TEST-pool base row (from twin json)")
    ap.add_argument("--step", default="1500")
    ap.add_argument("--twin", default=f"figure1/twin_{RUN}.json")
    ap.add_argument("--text", default=str(Path(__file__).parent / "text.json"))
    ap.add_argument("--keys", action="store_true")
    ap.add_argument("--out-id", default=None, help="write to assets/figures/<id>/ instead (A/B variants)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tw = json.loads((HERE / a.twin).read_text())
    r = [x for x in tw["rows"] if x["base_row"] == a.row][0]
    assert tw["run"] == RUN and r["replay_matches_pool"], "the probe-off replay must equal the stored test thought"
    X = tw["X"]
    assert tok.decode([X]) == " We"
    te = np.load(HERE / "data" / "pool" / "test" / "cols.npz")
    tn, tx, ton = (np.array(r[k]) for k in ("toks_none", "toks_x", "toks_on"))
    assert (te["toks"][r["row_none"]] == tn).all() and (te["toks"][r["row_x"]] == tx).all()
    assert tx[0] == X and tn[0] != X and (tx[1:] == tn[1:]).all()
    diff = tn != ton
    assert diff.any(), "the probe changed this thought"
    first = int(diff.argmax())
    c = {}
    for st in ("0", a.step):
        saved = np.load(HERE / "data" / "runs" / RUN / "reads" / f"c_{st}.npy")
        cr = r["c"][st]
        assert abs(cr["none"] - saved[r["row_none"]]) < 0.02 and abs(cr["x"] - saved[r["row_x"]]) < 0.02, (st, cr)
        c[st] = {"none": float(saved[r["row_none"]]), "x": float(saved[r["row_x"]]), "on": cr["on"]}
    S = json.loads((HERE / "data" / "summary.json").read_text())
    rd = {str(x["step"]): x for x in S["runs"][RUN]["reads"]}
    end, start = rd[a.step], rd["0"]
    sl = S["slope"]
    onset = S["runs"][RUN]["onset_s_P"]  # frozen onset rule (summ.py)
    prs = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "data" / "prompts.jsonl")}
    pr = prs[r["prompt_id"]]
    assert pr["split"] == "test"
    ptxt = pr["text"].split("user\n", 1)[1].split("<|im_end|>", 1)[0]
    meta = json.loads((HERE / "data" / "pool" / "test" / "meta.json").read_text())

    flat = {"steps_s": f'{int(a.step):,}', "n_rows_s": f'{meta["n_base"]:,}', "n_prompts": str(meta["n_prompts"]),
            "R": str(meta["R"]), "first1": str(first + 1), "ndiff": str(int(diff.sum())),
            "P.s": pc(end["s_P"]["v"]), "P.ci": f'{pc(end["s_P"]["ci"][0])} to {pc(end["s_P"]["ci"][1])}',
            "P.start": pc(start["s_P"]["v"], 1), "R.s": pc(end["s_R"]["v"], 1),
            "R.ci": f'{pc(end["s_R"]["ci"][0], 1)} to {pc(end["s_R"]["ci"][1], 1)}', "R.start": pc(start["s_R"]["v"], 1),
            "drift": f'{sl["predicted_T10_drift"]:,.0f}', "noise": f'{sl["predicted_T10_noise"]:,.0f}',
            "drift_lo": f'{sl["predicted_T10_drift"] * sl["T86_interval"][0] / sl["T86"]:,.0f}',
            "noise_lo": f'{sl["predicted_T10_noise"] * sl["T86_interval"][0] / sl["T86"]:,.0f}',
            "T86": str(sl["T86"]), "onset": f"{onset:,}" if onset else "?"}
    for st, nm in (("0", "b"), (a.step, "a")):
        for k in ("none", "x", "on"):
            flat[f"{nm}.{k}"] = pc(c[st][k])
    if a.keys:
        print(json.dumps(flat, indent=1))
        return
    T = json.load(open(a.text))

    def fill(s):
        out = re.sub(r"@@([\w.]+)@@", lambda mm: flat[mm.group(1)], s)
        assert "@@" not in out, out
        return out
    T = json.loads(fill(json.dumps(T)))

    def thought(ids, mark=None, cls="hi", upto=None):
        """token-by-token html; mark = bool array of tokens to highlight; whitespace runs collapse to one space"""
        ids = list(ids)
        upto = upto or len(ids)
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

    upto = min(64, max(first + 14, T.get("upto", 40)))
    m0 = np.zeros(64, bool)
    m0[0] = True
    rows_html = []
    for key, ids, mark, cls, note, who_cls, icon in (
            ("off", tn, diff, "was", T["rows"]["off_note"], "", ""),
            ("cue", tx, m0, "hc", T["rows"]["cue_note"], "cue", ""),
            ("real", ton, diff, "hr", T["rows"]["real_note"], "real", LENS)):
        k = {"off": "none", "cue": "x", "real": "on"}[key]
        hot = (key == "cue")
        rows_html.append(
            f'<div class="who {who_cls}">{ROBOT}<span>{icon}{T["rows"][key]}</span></div>'
            f'<div class="th">Thought: {thought(ids, mark, cls, upto)}<span class="tn">{note}</span></div>'
            f'<div class="ch{" hot" if hot else ""}"><div class="cb">{flat["b." + k]}</div><div class="arr">&rarr;</div>'
            f'<div class="ca">{flat["a." + k]}</div></div>')
    page = TEMPLATE.format(
        title=html.escape(html.unescape(re.sub("<[^>]+>", "", T["head"]))), head=T["head"], sub=T["sub"],
        st=T["ex_title"], prompt=html.escape(ptxt), key=T["key"], cb=T["col_before"], ca=T["col_after"],
        rows="\n".join(rows_html), res=res_panel(end, start, T), strip=strip(sl, onset, T, int(a.step)),
        foot=T["foot"], defs=DEFS,
        src=json.dumps({"run": RUN, "step": a.step, "base_row": a.row, "prompt": r["prompt_id"], "twin": a.twin}))
    out = OUT.parent / a.out_id if a.out_id else OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / "figure1.html").write_text(page)
    if a.out_id:
        return
    (Path(__file__).parent / "alt.txt").write_text(T["alt"] + "\n")
    (Path(__file__).parent / "data.json").write_text(json.dumps(
        {"run": RUN, "step": a.step, "base_row": a.row, "prompt": r["prompt_id"], "first_changed": first + 1,
         "c": c, "s_P": end["s_P"], "s_R": end["s_R"], "s_P_start": start["s_P"], "s_R_start": start["s_R"],
         "slope": sl, "onset": onset, "flat": flat}, indent=1))
    print(json.dumps({"c": c, "first": first + 1}, indent=1), "\nwrote", out / "figure1.html")


AX0, AX1, BW = -0.1, 1.0, 330


def sx(v):
    return BW * (min(max(v, AX0), AX1) - AX0) / (AX1 - AX0)


def res_panel(end, start, T):
    """share of the limit over all test thoughts: before (hollow) and after (filled, 95% interval) for both cues"""
    out = [f'<div class="rt">{T["res_title"]}</div>']
    for kind, key in (("p", "s_P"), ("r", "s_R")):
        col = "#2a9d92" if kind == "p" else "#c8478a"
        g = [f'<line x1="{sx(v):.1f}" y1="4" x2="{sx(v):.1f}" y2="44" stroke="{"#9a9ca3" if v == 0 else "#e3e4e8"}" '
             f'stroke-width="{2 if v == 0 else 1.5}"/>' for v in (0.0, 0.5, 1.0)]
        if kind == "p":
            g.append(f'<line x1="{sx(0.3):.1f}" y1="4" x2="{sx(0.3):.1f}" y2="44" stroke="#2a9d92" stroke-width="2.5" stroke-dasharray="5,4"/>')
        lo, hi = end[key]["ci"]
        g.append(f'<line x1="{sx(lo):.1f}" y1="24" x2="{sx(hi):.1f}" y2="24" stroke="{col}" stroke-width="8" stroke-linecap="round" opacity="0.4"/>')
        g.append(f'<circle cx="{sx(start[key]["v"]):.1f}" cy="24" r="8" fill="#fbfaf8" stroke="{col}" stroke-width="3"/>')
        g.append(f'<circle cx="{sx(end[key]["v"]):.1f}" cy="24" r="11" fill="{col}"/>')
        svg = f'<svg width="{BW}" height="48" viewBox="0 0 {BW} 48" style="overflow:visible">{"".join(g)}</svg>'
        lines = T["res"][{"p": "cue", "r": "real"}[kind]]
        out.append(f'<div class="res {kind}"><div class="rv">{lines[0]}</div>{svg}<div class="rn">{lines[1]}</div></div>')
    ax = [f'<text x="{sx(v):.1f}" y="18" text-anchor="middle" class="tk">{pc(v)}</text>' for v in (0.0, 0.5, 1.0)]
    ax.append(f'<text x="{BW / 2:.1f}" y="44" text-anchor="middle" class="sn">{T["res"]["axis"]}</text>')
    out.append(f'<svg width="{BW}" height="50" viewBox="0 0 {BW} 50" style="overflow:visible;margin-left:10px">{"".join(ax)}</svg>')
    out.append(f'<div class="rk">{T["res"]["key"]}</div>')
    return "\n".join(out)


def strip(sl, onset, T, step):
    """when each planted word was learned vs the two frozen forecasts for the 10-in-100 word, on a log updates axis"""
    import math
    W, A, B = 1120, math.log10(50), math.log10(10000)

    def ux(u):
        return 10 + (W - 20) * (math.log10(u) - A) / (B - A)
    lo_d = sl["predicted_T10_drift"] * sl["T86_interval"][0] / sl["T86"]
    lo_n = sl["predicted_T10_noise"] * sl["T86_interval"][0] / sl["T86"]
    y = 78
    g = [f'<rect x="{ux(50):.1f}" y="{y - 7}" width="{ux(step) - ux(50):.1f}" height="14" rx="3" fill="#2a9d92" opacity="0.13"/>',
         f'<line x1="{ux(50):.1f}" y1="{y}" x2="{ux(10000):.1f}" y2="{y}" stroke="#c9cbd1" stroke-width="2"/>']
    for u in (100, 1000, 10000):
        g.append(f'<line x1="{ux(u):.1f}" y1="{y - 5}" x2="{ux(u):.1f}" y2="{y + 5}" stroke="#a9abb1" stroke-width="2"/>')
    for u in (1000, 10000):
        g.append(f'<text x="{ux(u):.1f}" y="{y + 32}" text-anchor="middle" class="tk">{u:,}</text>')
    for (x0, x1, lab, ly, anc) in ((lo_d, sl["predicted_T10_drift"], T["strip"]["drift"], 22, "start"),
                                   (lo_n, sl["predicted_T10_noise"], T["strip"]["noise"], 48, "end")):
        g.append(f'<rect x="{ux(x0):.1f}" y="{y - 13}" width="{ux(x1) - ux(x0):.1f}" height="26" rx="4" fill="none" '
                 f'stroke="#4d5059" stroke-width="2.5" stroke-dasharray="5,4"/>')
        ax = ux(x0) if anc == "start" else ux(x1)
        g.append(f'<line x1="{ax:.1f}" y1="{ly + 5}" x2="{ax:.1f}" y2="{y - 13}" stroke="#4d5059" stroke-width="1.5"/>')
        g.append(f'<text x="{ax:.1f}" y="{ly}" text-anchor="{anc}" class="sl">{lab}</text>')
    g.append(f'<circle cx="{ux(sl["T86"]):.1f}" cy="{y}" r="10" fill="#8a8d96"/>')
    g.append(f'<text x="{ux(sl["T86"]):.1f}" y="{y + 32}" text-anchor="middle" class="sg">{T["strip"]["dense"]}</text>')
    g.append(f'<circle cx="{ux(onset):.1f}" cy="{y}" r="11" fill="#2a9d92"/>')
    g.append(f'<text x="{ux(onset) + 14:.1f}" y="{y + 32}" text-anchor="end" class="so">{T["strip"]["seen"]}</text>')
    g.append(f'<text x="{(ux(50) + ux(step)) / 2:.1f}" y="{y + 58}" text-anchor="middle" class="sr">{T["strip"]["ran"]}</text>')
    svg = f'<svg width="{W}" height="{y + 64}" viewBox="0 0 {W} {y + 64}" style="overflow:visible">{"".join(g)}</svg>'
    return f'<div class="stt2">{T["strip"]["title"]}</div>{svg}'


ROBOT = '<svg width="38" height="38" viewBox="0 0 40 40" aria-hidden="true"><use href="#robot"/></svg>'
LENS = '<svg width="22" height="22" viewBox="0 0 40 40" aria-hidden="true"><use href="#lens"/></svg>'
DEFS = ('<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>'
        '<g id="lens"><circle cx="17" cy="17" r="12" fill="#fff" stroke="#c8478a" stroke-width="5"/>'
        '<line x1="26" y1="26" x2="37" y2="37" stroke="#c8478a" stroke-width="6" stroke-linecap="round"/></g>'
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
Generated by kran-research projects/decision-only-control/figure1/gen.py (branch swarm/oracle-1113-decision-only).
All numbers: figure1/data.json; example {src}.
-->
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: #fbfaf8; }}
  body {{ width: 1200px; padding: 34px 40px 26px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
  .head {{ font-size: 44px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.14; }}
  .head .c {{ color: #2a9d92; }}
  .head .r {{ color: #c8478a; }}
  .sub {{ font-size: 24px; color: #4d5059; margin-top: 14px; line-height: 1.34; }}
  .sub b {{ color: #1d1d1f; font-weight: 800; }}
  .main {{ margin-top: 24px; display: grid; grid-template-columns: 1fr 350px; gap: 0 26px; }}
  .stt {{ font-size: 28px; font-weight: 800; line-height: 1.2; }}
  .bubble {{ background: #3478f6; color: #fff; font-size: 20px; line-height: 1.34; padding: 12px 20px; border-radius: 22px 22px 6px 22px; margin-top: 12px; }}
  .key {{ font-size: 20px; color: #4d5059; margin-top: 8px; text-align: right; line-height: 1.3; }}
  .key b {{ color: #1d1d1f; }}
  .rows {{ margin-top: 10px; display: grid; grid-template-columns: 110px 1fr 196px; gap: 14px 12px; align-items: center; }}
  .colh {{ font-size: 19px; color: #6b6e77; font-weight: 700; text-align: center; line-height: 1.2; }}
  .who {{ display: flex; flex-direction: column; align-items: center; gap: 4px; font-size: 21px; font-weight: 700; color: #6b6e77; text-align: center; line-height: 1.15; }}
  .who span {{ display: inline-flex; align-items: center; gap: 3px; }}
  .who.cue {{ color: #1f8278; }}
  .who.real {{ color: #b23a78; }}
  .th {{ background: #fff; border-radius: 16px 16px 16px 4px; padding: 11px 15px; font-size: 20px; line-height: 1.36; color: #33363d;
         box-shadow: 0 1px 6px rgba(0,0,0,0.08); }}
  .th .tn {{ display: block; font-size: 19px; color: #6b6e77; margin-top: 5px; font-weight: 700; }}
  .hc {{ background: #bfe8e3; color: #0d4d47; font-weight: 800; border-radius: 4px; padding: 0 3px; }}
  .hr {{ background: #f6cfe2; color: #6d1846; font-weight: 700; border-radius: 4px; padding: 0 2px; }}
  .was {{ box-shadow: inset 0 -3px 0 #d98ab3; }}
  .ch {{ display: flex; align-items: baseline; justify-content: center; gap: 6px; font-variant-numeric: tabular-nums; }}
  .ch .cb {{ font-size: 22px; color: #8a8d96; font-weight: 600; }}
  .ch .arr {{ font-size: 20px; color: #9a9ca3; }}
  .ch .ca {{ font-size: 30px; font-weight: 800; color: #33363d; }}
  .ch.hot .ca {{ color: #1f8278; }}
  .side {{ padding-top: 4px; }}
  .rt {{ font-size: 22px; font-weight: 800; line-height: 1.22; margin-bottom: 8px; }}
  .res {{ padding-left: 10px; margin-top: 10px; }}
  .res .rv {{ font-size: 25px; font-weight: 800; font-variant-numeric: tabular-nums; line-height: 1.2; }}
  .res.p .rv {{ color: #1f8278; }}
  .res.r .rv {{ color: #b23a78; }}
  .res .rn {{ font-size: 18px; color: #6b6e77; line-height: 1.25; }}
  .rk {{ font-size: 18px; color: #6b6e77; line-height: 1.3; padding-left: 10px; margin-top: 2px; }}
  svg text {{ font-family: "Inter", system-ui, sans-serif; }}
  .sn {{ font-size: 19px; fill: #4d5059; }}
  .tk {{ font-size: 19px; fill: #7a7d86; font-variant-numeric: tabular-nums; }}
  .stt2 {{ font-size: 24px; font-weight: 800; margin-top: 22px; line-height: 1.2; }}
  .sl {{ font-size: 19px; fill: #4d5059; font-weight: 700; }}
  .so {{ font-size: 20px; fill: #1f8278; font-weight: 800; }}
  .sg {{ font-size: 19px; fill: #6b6e77; font-weight: 700; }}
  .sr {{ font-size: 18px; fill: #2a9d92; font-weight: 600; }}
  .stn {{ font-size: 19px; color: #6b6e77; text-align: center; margin-top: -2px; }}
  .foot {{ font-size: 19px; color: #7a7d86; margin-top: 12px; display: flex; justify-content: space-between; align-items: flex-end; gap: 20px; line-height: 1.35; }}
  .foot b {{ color: #1d1d1f; white-space: nowrap; }}
</style>
</head>
<body>
<div class="head">{head}</div>
<div class="sub">{sub}</div>
<div class="main">
<div>
<div class="stt">{st}</div>
<div class="bubble">{prompt}</div>
<div class="key">{key}</div>
<div class="rows">
  <div></div><div></div><div class="colh">{cb}<br>{ca}</div>
{rows}
</div>
</div>
<div class="side">{res}</div>
</div>
{strip}
<div class="foot"><span>{foot}</span><b>blog.kran.ai/lab</b></div>
{defs}
</body>
</html>
"""

if __name__ == "__main__":
    main()
