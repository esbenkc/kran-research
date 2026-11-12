"""Figure 1 generator (Owain) for matched-trace-powered.

Top: one real TEST draw of the END adapter, shown three ways under common random numbers (the same per-row noise in
every pass): the probe-off thought, the planted-cue thought (opens with ' Need') and the probe-on thought, with the
words each cue changed marked. Right of each cue row: that cue's switch as a share of its limit over the whole end read
(95% interval), with the frozen predicted band (planted) and the 10% not-learned line (trace). Every number comes from
data/summary.json (summarize_tf.py); the example from the end read's npz (read_tf.py --cue-r 1). No per-draw cheat
numbers: one draw's cheat change is mostly noise.

    PY=../../../env/.venv/bin/python
    $PY figure1/pick.py <read>              # choose the row
    $PY figure1/gen.py --row N [--read <read>] [--mock]
--mock: layout test on the 11-11 read (update 674) with made-up end numbers; the page says MOCK.
Text lives in figure1/text.json; @@key@@ is replaced from the flat data dict (print it with --keys).
"""
import argparse
import html
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
SLUG = "matched-trace-powered"
OUT = HERE.parents[2] / "blog.kran.ai" / "assets" / "figures" / f"{SLUG}-fig1"
X = 14656  # ' Need'
sys_path = HERE / "figures"


def pc(v, sign=False, d=0):
    s = f"{100 * v:+.{d}f}" if sign else f"{100 * v:.{d}f}"
    return s.replace("-", "&minus;") + "%"


def load_numbers(mock):
    import sys
    sys.path.insert(0, str(sys_path))
    import figure_data as FD
    if mock:
        return {"status": "MOCK", "steps": 1230, "P_start": 0.04, "n_rows": 6400, "n_prompts": 200, "R": 32,
                "P": {"s": 0.31, "ci": [0.25, 0.37], "D": 0.100}, "R_": {"s": 0.02, "ci": [-0.05, 0.08], "D": 0.097},
                "band": FD.band_at(3940), "cue_mass": 3940, "margin": FD.MARGIN, "source": "MOCK"}
    S = json.load(open(HERE / "data" / "summary.json"))
    F = S["figure"]
    cell = [c for c in F["reads"] if c["when"] == "final"][0]
    fin = S[cell["key"]]
    js = json.load(open(HERE / "data" / "reads" / "raw" / "end_tf.json"))
    assert js["complete"] and js["n_rows"] == fin["n_rows"] == 6400
    band = FD.band_at(cell["xP"])
    assert abs(F["band"]["central"] - band["c"]) < 0.01
    return {"status": S["status"], "steps": F["steps"], "n_rows": fin["n_rows"], "n_prompts": fin["n_prompts"], "R": js["R"],
            "P": {"s": fin[cell["kP"]]["v"], "ci": fin[cell["kP"]]["ci"], "D": fin["D_P"]["v"]},
            "R_": {"s": fin[cell["kR"]]["v"], "ci": fin[cell["kR"]]["ci"], "D": fin["D_R"]["v"]},
            "P_start": S[[c for c in F["reads"] if c["when"] == "start"][0]["key"]]["s_P"]["v"],
            "gap": fin["gap"], "band": band, "cue_mass": cell["xP"], "trace_mass": cell["xR"], "margin": FD.MARGIN,
            "source": "data/summary.json (summarize_tf.py) end; read data/reads/raw/end_tf"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--read", default="data/reads/raw/end_tf")
    ap.add_argument("--row", type=int, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--text", default=str(Path(__file__).parent / "text.json"))
    ap.add_argument("--keys", action="store_true")
    ap.add_argument("--out-id", default=None, help="write to assets/figures/<id>/ instead (A/B variants)")
    a = ap.parse_args()
    if a.mock and a.read == ap.get_default("read") and not Path(HERE / (a.read + ".npz")).exists():
        a.read = "data/parent/reads/joint_s0_final"
    i = a.row

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    assert tok.decode([X]) == " Need"

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from pick import load_read
    z = load_read(HERE / a.read)
    N = load_numbers(a.mock)
    assert z["split"][i] == a.split
    nb, lb, nc = z["toks_nb"][i], z["toks_lb"][i], z["toks_nc"][i]
    assert nc[0] == X and nb[0] != X, "planted cue fired on this draw only"
    if "toks_n" in z:
        assert (z["toks_n"][i] == nb).all(), "one shared probe-off thought"
    diff = nb != lb
    assert diff.any(), "the probe's trace changed this draw"
    first = int(diff.argmax())
    pid = int(z["prompt_idx"][i])
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "data" / "prompts.jsonl")}
    pr = rows[f"cd{pid:04d}"]
    assert pr["split"] == a.split
    ptxt = pr["text"].split("user\n", 1)[1].split("<|im_end|>", 1)[0]
    m = (z["split"] == a.split) & z.get("has_plain_row", np.ones(len(z["split"]), bool))
    rdiff = z["toks_nb"][m] != z["toks_lb"][m]
    rch = rdiff.any(1)

    data = {"title": "", "status": N["status"], "read": a.read, "row": i, "prompt": pr["id"],
            "seed": int(z["seed"][i]) if "seed" in z else None, "split": a.split, "steps": N["steps"],
            "n_rows": N["n_rows"], "n_prompts": N["n_prompts"], "R": N["R"], "P": N["P"], "R_": N["R_"], "band": N["band"],
            "margin": N["margin"], "cue_mass": N["cue_mass"],
            "ex": {"first": first, "ndiff": int(diff.sum())},
            "trace_changed_share": float(rch.mean()), "trace_first_median": float(np.median(rdiff.argmax(1)[rch])),
            "source": N["source"]}
    flat = {}

    def walk(dd, pre=""):
        for k, v in dd.items():
            if isinstance(v, dict):
                walk(v, pre + k + ".")
            else:
                flat[pre + k] = str(v)
    walk(data)
    flat.update({
        # one decimal, as in the post's TLDR
        "P.s_s": pc(N["P"]["s"], d=1), "R_.s_s": pc(abs(N["R_"]["s"]) if abs(N["R_"]["s"]) < 0.0005 else N["R_"]["s"], d=1),
        "P.ci_s": f'{pc(N["P"]["ci"][0], d=1)} to {pc(N["P"]["ci"][1], d=1)}',
        "R_.ci_s": f'{pc(N["R_"]["ci"][0], d=1)} to {pc(N["R_"]["ci"][1], d=1)}',
        "P.start_s": pc(N["P_start"], d=1),
        "P.D_s": f'{100 * N["P"]["D"]:.0f}', "R_.D_s": f'{100 * N["R_"]["D"]:.1f}',
        "R_.s_abs": pc(abs(N["R_"]["s"])),
        "band_s": f'{pc(N["band"]["lo"])} to {pc(N["band"]["hi"])}', "margin_s": pc(N["margin"]),
        "n_rows_s": f'{N["n_rows"]:,}', "steps_s": f'{N["steps"]:,}', "ex.first1": str(first + 1),
        "trace_first_median1": str(int(data["trace_first_median"] + 1)), "trace_changed_pct": f'{100 * data["trace_changed_share"]:.0f}',
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
    page = TEMPLATE.format(
        title=html.escape(data["title"]), head=T["head"], sub=T["sub"], prompt=html.escape(ptxt), key=T["key"],
        st=T["ex_title"], rt=T["res_title"], off_l=T["rows"]["off"], cue_l=T["rows"]["cue"], real_l=T["rows"]["real"],
        off_t=thought(nb, diff, cls="was"), cue_t=thought(nc, cue_mark, firstline=True, cls="hc"),
        real_t=thought(lb, diff, cls="hr"), off_n=T["rows"]["off_note"], cue_n=T["rows"]["cue_note"], real_n=T["rows"]["real_note"],
        off_r=T["res"]["off"], cue_r=bar(N["P"], N, "p", T["res"]["cue"]), real_r=bar(N["R_"], N, "r", T["res"]["real"]),
        axis=axis(T["res"]["axis"]), foot=T["foot"], defs=DEFS, robot=ROBOT, lens=LENS,
        mock='<div class="mock">MOCK NUMBERS: layout test only</div>' if a.mock else "",
        src=json.dumps({k: data[k] for k in ("read", "row", "prompt", "seed")}))
    out = OUT.parent / a.out_id if a.out_id else OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / "figure1.html").write_text(page)
    if a.out_id:
        return
    (Path(__file__).parent / "alt.txt").write_text(T["alt"] + "\n")
    (Path(__file__).parent / "data.json").write_text(json.dumps(data, indent=1, default=str))
    print(json.dumps({k: data[k] for k in ("row", "prompt", "ex", "P", "R_", "band")}, indent=1), "\nwrote", out / "figure1.html")


LO, HI, BW = -0.05, 0.15, 300  # share axis of the result column, px width


def sx(v):
    return BW * (min(max(v, LO), HI) - LO) / (HI - LO)


def bar(c, N, kind, lines):
    """one cue's share of its limit: dot + 95% interval on the shared axis; planted gets the predicted band,
    the trace the 10% not-learned line"""
    col = "#2a9d92" if kind == "p" else "#c8478a"
    g = [f'<line x1="{sx(v):.1f}" y1="2" x2="{sx(v):.1f}" y2="54" stroke="{"#9a9ca3" if v == 0 else "#e3e4e8"}" '
         f'stroke-width="{2 if v == 0 else 1.5}"/>' for v in (0.0, 0.05, 0.10, 0.15)]
    if kind == "r":
        g.append(f'<line x1="{sx(N["margin"]):.1f}" y1="6" x2="{sx(N["margin"]):.1f}" y2="50" stroke="#c8478a" '
                 f'stroke-width="2.5" stroke-dasharray="5,4"/>')
    g.append(f'<line x1="{sx(c["ci"][0]):.1f}" y1="28" x2="{sx(c["ci"][1]):.1f}" y2="28" stroke="{col}" stroke-width="8" '
             f'stroke-linecap="round" opacity="0.4"/>')
    g.append(f'<circle cx="{sx(c["s"]):.1f}" cy="28" r="11" fill="{col}"/>')
    svg = f'<svg width="{BW}" height="56" viewBox="0 0 {BW} 56" style="overflow:visible">{"".join(g)}</svg>'
    return f'<div class="res {kind}"><div class="rv">{lines[0]}</div>{svg}<div class="rn">{lines[1]}</div></div>'


def axis(lab):
    g = []
    for v in (0.0, 0.05, 0.10, 0.15):
        g.append(f'<text x="{sx(v):.1f}" y="20" text-anchor="middle" class="tk">{pc(v)}</text>')
    g.append(f'<text x="{BW / 2:.1f}" y="48" text-anchor="middle" class="sn">{lab}</text>')
    return f'<svg width="{BW}" height="56" viewBox="0 0 {BW} 56" style="overflow:visible">{"".join(g)}</svg>'


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
Generated by kran-research projects/matched-trace-powered/figure1/gen.py (branch swarm/oracle-1112-matched-trace).
All numbers: figure1/data.json; example {src}.
-->
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: #fbfaf8; }}
  body {{ width: 1200px; padding: 34px 40px 26px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
  .mock {{ font-size: 24px; font-weight: 800; color: #c8372d; }}
  .head {{ font-size: 44px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.14; }}
  .head .c {{ color: #2a9d92; }}
  .head .r {{ color: #c8478a; }}
  .sub {{ font-size: 24px; color: #4d5059; margin-top: 14px; line-height: 1.34; }}
  .sub b {{ color: #1d1d1f; font-weight: 800; }}
  .stt {{ font-size: 28px; font-weight: 800; line-height: 1.2; }}
  .bubble {{ background: #3478f6; color: #fff; font-size: 21px; line-height: 1.34; padding: 13px 22px; border-radius: 22px 22px 6px 22px; margin-top: 12px; }}
  .key {{ font-size: 21px; color: #4d5059; margin-top: 8px; text-align: right; }}
  .key b {{ color: #1d1d1f; }}
  .hdr {{ margin-top: 26px; display: grid; grid-template-columns: 1fr 320px; gap: 0 20px; align-items: end; }}
  .rt {{ font-size: 22px; font-weight: 800; line-height: 1.22; }}
  .rows {{ margin-top: 12px; display: grid; grid-template-columns: 126px 1fr 320px; gap: 14px 14px; align-items: center; }}
  .who {{ display: flex; flex-direction: column; align-items: center; gap: 4px; font-size: 21px; font-weight: 700; color: #6b6e77; text-align: center; line-height: 1.15; }}
  .who span {{ display: inline-flex; align-items: center; gap: 3px; }}
  .who.cue {{ color: #1f8278; }}
  .who.real {{ color: #b23a78; }}
  .th {{ background: #fff; border-radius: 16px 16px 16px 4px; padding: 12px 16px; font-size: 21px; line-height: 1.38; color: #33363d;
         box-shadow: 0 1px 6px rgba(0,0,0,0.08); }}
  .th .tn {{ display: block; font-size: 20px; color: #6b6e77; margin-top: 6px; font-weight: 700; }}
  .hc {{ background: #bfe8e3; color: #0d4d47; font-weight: 800; border-radius: 4px; padding: 0 3px; }}
  .hr {{ background: #f6cfe2; color: #6d1846; font-weight: 700; border-radius: 4px; padding: 0 2px; }}
  .was {{ box-shadow: inset 0 -3px 0 #d98ab3; }}
  .res {{ padding-left: 10px; }}
  .res .rv {{ font-size: 26px; font-weight: 800; font-variant-numeric: tabular-nums; }}
  .res.p .rv {{ color: #1f8278; }}
  .res.r .rv {{ color: #b23a78; }}
  .res .rn {{ font-size: 19px; color: #6b6e77; line-height: 1.25; }}
  .res0 {{ padding-left: 10px; font-size: 20px; color: #6b6e77; line-height: 1.3; }}
  svg text {{ font-family: "Inter", system-ui, sans-serif; }}
  .sn {{ font-size: 20px; fill: #4d5059; }}
  .tk {{ font-size: 20px; fill: #7a7d86; font-variant-numeric: tabular-nums; }}
  .ax {{ display: grid; grid-template-columns: 126px 1fr 320px; gap: 0 14px; }}
  .ax > div:last-child {{ padding-left: 10px; }}
  .foot {{ font-size: 20px; color: #7a7d86; margin-top: 14px; display: flex; justify-content: space-between; align-items: flex-end; gap: 20px; line-height: 1.35; }}
  .foot b {{ color: #1d1d1f; white-space: nowrap; }}
</style>
</head>
<body>
{mock}
<div class="head">{head}</div>
<div class="sub">{sub}</div>
<div class="hdr"><div class="stt">{st}</div><div class="rt">{rt}</div></div>
<div class="bubble">{prompt}</div>
<div class="key">{key}</div>
<div class="rows">
  <div class="who">{robot}<span>{off_l}</span></div>
  <div class="th">Thought: {off_t}<span class="tn">{off_n}</span></div>
  <div class="res0">{off_r}</div>
  <div class="who cue">{robot}<span>{cue_l}</span></div>
  <div class="th">Thought: {cue_t}<span class="tn">{cue_n}</span></div>
  {cue_r}
  <div class="who real">{robot}<span>{lens}{real_l}</span></div>
  <div class="th">Thought: {real_t}<span class="tn">{real_n}</span></div>
  {real_r}
</div>
<div class="ax"><div></div><div></div><div>{axis}</div></div>
<div class="foot"><span>{foot}</span><b>blog.kran.ai/lab</b></div>
{defs}
</body>
</html>
"""

if __name__ == "__main__":
    main()
