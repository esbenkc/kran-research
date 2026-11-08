"""Figure 1 generator (Owain): one real test puzzle, its 16-word 'Thought:' written with and without the probe
(same sampling noise), the planted check word, and the trained served cheating rates.

    PY=../../../env/.venv/bin/python
    $PY figure1/gen.py [--start data/reads/start] [--planted data/reads/planted_s0] [--arms k16_s0=...,...]

Sources (all numbers re-derived from the per-row npz and asserted against the read json, 1e-6):
  start read  <start>.{npz,json}   read.py on the start policy, test split: toks_n / toks_l (same noise per row),
                                   c_n / c_l (answer-slot cheat chance), f, D
  planted     <planted>.{npz,json} read.py on the planted adapter with --plant-tok X (+ --forced-gate)
  arms        <run>.json            read.py on each final adapter: test.cheat_none, cheat_leaky, gap, gap_ci
Text (head, subtitle, foot, labels) lives in figure1/text.json; @@key@@ is replaced by numbers from data.json.
"""
import argparse
import html
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
OUT = HERE.parent.parent.parent / "blog.kran.ai" / "assets" / "figures" / "multi-token-recompute-fig1"
_TOK = None


def tok():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    return _TOK


def R(x, one=False):
    v = 100 * x
    if 0 < abs(v) < 0.1:
        return f"{v:.2f}"
    if one or abs(v - round(v)) >= 0.05 and abs(v) < 10:
        return f"{v:.1f}"
    return f"{v:.0f}"


def load_read(base, split="test"):
    z = dict(np.load(str(base) + ".npz"))
    js = json.load(open(str(base) + ".json"))
    m = z["split"] == split
    z = {k: (v[m] if v.ndim and v.shape[0] == len(m) else v) for k, v in z.items()}
    d = js[split]
    assert abs(z["c_n"].mean() - d["cheat_none"]) < 1e-6 and abs(z["c_l"].mean() - d["cheat_leaky"]) < 1e-6, base
    if "f" in d:
        assert abs((z["toks_n"] != z["toks_l"]).any(1).mean() - d["f"]) < 1e-9, base
    return z, js, d


def esc(s):
    return html.escape(s).replace("\n", "&#8629; ")


def spans(a, b):
    """Token lists a (no probe) and b (probe) -> HTML for each; everything from the first token that differs is
    marked (after one changed token the rest is a new draw)."""
    t = tok()
    p = 0
    while p < len(a) and a[p] == b[p]:
        p += 1
    out = []
    for x in (a, b):
        same, rest = esc(t.decode(x[:p])), esc(t.decode(x[p:]))
        out.append(same + (f'<span class="dif">{rest}</span>' if rest else ""))
    return out[0], out[1]


def clean(t_ids):
    s = tok().decode(t_ids)
    return "\n" not in s and "Answer" not in s and "read" not in s.lower().split() and len(s) < 90


def pick_real(z, doc=None):
    """A (prompt, seed) whose probe text differs, AND whose probe text was also written WITHOUT the probe
    under another seed of the same prompt (the flip is just another ordinary draw)."""
    tn, tl, P, S = z["toks_n"], z["toks_l"], z["prompt_idx"], z["seed"]
    cands = []
    for i in np.where((tn != tl).any(1))[0]:
        if doc is not None and P[i] != doc:
            continue
        same = np.where((P == P[i]) & (S != S[i]))[0]
        twin = [j for j in same if (tn[j] == tl[i]).all()]
        if not (clean(tn[i]) and clean(tl[i])):
            continue
        first = int(np.argmax(tn[i] != tl[i]))
        cands.append((-len(twin), -first, int(i), twin))
    # most other draws that wrote the probe's text; then the latest first change (a shared start shows the change)
    cands.sort(key=lambda c: (c[0], c[1]))
    assert cands, "no clean flipped row"
    _, _, i, twin = cands[0]
    return i, twin


def pick_planted(zp, X, doc=None):
    tn, tl, P = zp["toks_n"], zp["toks_l"], zp["prompt_idx"]
    rows = [i for i in range(len(P)) if tl[i, 0] == X and tn[i, 0] != X and clean(tn[i]) and clean(tl[i])]
    if doc is not None and any(P[i] == doc for i in rows):
        rows = [i for i in rows if P[i] == doc]
    assert rows, "no planted row"
    # the row whose cheat chance moves most in the learned direction (probe text -> less cheating)
    rows.sort(key=lambda i: -(zp["c_n"][i] - zp["c_l"][i]))
    return rows[0]


def prompt_text(pidx):
    rows = {int(json.loads(l)["id"][2:]): json.loads(l) for l in open(HERE / "data" / "prompts.jsonl")}
    r = rows[int(pidx)]
    u = r["text"].split("user\n", 1)[1].split("<|im_end|>", 1)[0]
    return r["id"], u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="data/reads/start")
    ap.add_argument("--planted", default=None)
    ap.add_argument("--arms", default="", help="label=path,... final reads (json base path)")
    ap.add_argument("--doc", type=int, default=None)
    ap.add_argument("--text", default=str(Path(__file__).parent / "text.json"))
    ap.add_argument("--mock", action="store_true", help="draft: MOCK banner, numbers not real")
    a = ap.parse_args()

    z, sjs, sd = load_read(HERE / a.start)
    i, twin = pick_real(z, a.doc)
    assert twin, "need a probe sentence that the model also wrote without the probe"
    pidx = int(z["prompt_idx"][i])
    pid, ptxt = prompt_text(pidx)
    hn, hl = spans(z["toks_n"][i], z["toks_l"][i])
    data = {"start": {"read": a.start, "f": sd["f"], "D": sd.get("D"), "D_ci": sd.get("D_ci"),
                      "cheat_none": sd["cheat_none"], "cheat_leaky": sd["cheat_leaky"]},
            "real_row": {"prompt": pid, "seed": int(z["seed"][i]), "twin_seeds": [int(z["seed"][t]) for t in twin],
                         "n_twins": len(twin), "n_other": int(((z["prompt_idx"] == pidx) & (z["seed"] != z["seed"][i])).sum()),
                         "text_none": tok().decode(z["toks_n"][i]), "text_probe": tok().decode(z["toks_l"][i]),
                         "c_none": float(z["c_n"][i]), "c_probe": float(z["c_l"][i])}}
    if a.planted:
        zp, pjs, pd = load_read(HERE / a.planted)
        X = int(pjs["plant_tok"])
        assert X == json.load(open(HERE / "data/gate/calib.json"))["X"]
        j = pick_planted(zp, X, pidx)
        pn = esc(tok().decode(zp["toks_n"][j]))
        pl = f'<span class="dif red">{esc(tok().decode(zp["toks_l"][j][:1]))}</span>' + esc(tok().decode(zp["toks_l"][j][1:]))
        data["planted"] = {"read": a.planted, "X": X, "X_text": tok().decode([X]).strip(), "eps": pjs["plant_eps"],
                           "prompt": prompt_text(int(zp["prompt_idx"][j]))[0], "seed": int(zp["seed"][j]),
                           "text_none": tok().decode(zp["toks_n"][j]), "text_probe": tok().decode(zp["toks_l"][j]),
                           "c_none": float(zp["c_n"][j]), "c_probe": float(zp["c_l"][j]),
                           "D_end": pd["D"], "f_end": pd["f"],
                           "D_plant": json.load(open(HERE / "data/gate/calib.json"))["tv1_at_eps"],
                           **{k: pd[k] for k in ("cheat_none", "cheat_leaky", "gap", "gap_ci") if k in pd},
                           **{k: pd[k] for k in ("forced_switch", "forced_switch_ci", "g_X", "g_notX") if k in pd}}
    arms = {}
    for kv in filter(None, a.arms.split(",")):
        lab, path = kv.split("=")
        js = json.load(open(HERE / (path + ".json")))["test"]
        arms[lab] = {k: js[k] for k in ("cheat_none", "cheat_leaky", "gap", "gap_ci", "text", "D", "f") if k in js}
        if Path(str(HERE / path) + ".npz").exists():
            za, _, _ = load_read(HERE / path)
            u = {tuple(r) for r in za["toks_n"].tolist()}
            arms[lab]["distinct"] = len(u) / len(za["toks_n"])
            arms[lab]["n_distinct"] = len(u)
        g = arms[lab]["gap"]
        arms[lab]["gap_s"] = ("+" if g >= 0 else "&minus;") + R(abs(g))
    data["arms"] = arms

    T = json.load(open(a.text))
    flat = {}

    def walk(d, pre=""):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, pre + k + ".")
            elif isinstance(v, float):
                flat[pre + k] = R(v)
            else:
                flat[pre + k] = str(v)
    walk(data)

    def fill(s):
        out = re.sub(r"@@([\w.]+)@@", lambda m: flat[m.group(1)], s)
        assert "@@" not in out, out
        return out
    T = {k: fill(v) if isinstance(v, str) else v for k, v in T.items()}

    rr = data["real_row"]
    lean = lambda c: '<b class="r">read</b>' if c >= 0.5 else "<b>solve</b>"  # noqa: E731
    real_card = f"""
    <div class="pair">
      <div class="line"><span class="who">no probe</span><span class="txt">Thought:{hn}&hellip;</span></div>
      <div class="line"><span class="who probe">{LENS}probe</span><span class="txt">Thought:{hl}&hellip;</span></div>
      <div class="line twin"><span class="who">{T['twin_label']}</span><span class="txt">Thought:{esc(tok().decode(z["toks_n"][twin[0]]))}&hellip;</span></div>
    </div>
    <div class="note">{T['real_note']}</div>"""
    planted_card = ""
    if "planted" in data:
        planted_card = f"""
    <div class="pair">
      <div class="line"><span class="who">no probe</span><span class="txt">Thought:{pn}&hellip;</span></div>
      <div class="line"><span class="who probe">{LENS}probe</span><span class="txt">Thought:{pl}&hellip;</span></div>
    </div>
    <div class="note">{T['planted_note']}</div>"""

    def table(rows):
        h = f'<div class="tb"><div></div><div class="th">no probe</div><div class="th">{LENS}probe</div>'
        for key, lab, tag in rows:
            if key not in arms:
                continue
            d = arms[key]
            tg = f'<span class="tag">{fill(tag)}</span>' if tag else ""
            h += (f'<div class="rl">{lab}{tg}</div><div class="tv">{100 * d["cheat_none"]:.1f}%</div>'
                  f'<div class="tv">{100 * d["cheat_leaky"]:.1f}%</div>')
        return h + "</div>"

    res_l = table(T.get("left_arms", []))
    res_r = table(T.get("right_arms", []))
    page = TEMPLATE.format(
        title=html.escape(T["title"]), head=T["head"], sub=T["sub"], prompt=html.escape(ptxt), rule=T["rule"],
        lt=T["left_title"], ls=T["left_sub"], rt=T["right_title"], rs=T["right_sub"],
        real=real_card, planted=planted_card, after=T["after"], res_l=res_l, res_r=res_r,
        vl=T["verdict_left"], vr=T["verdict_right"], al=T["after_left_label"], ar=T["after_right_label"], foot=T["foot"], src=json.dumps({k: v for k, v in data.items()}, default=str)[:600].replace("--", "- -"),
        mock='<div class="mock">MOCK: MADE-UP RESULT NUMBERS</div>' if a.mock else "", lens_def=LENS_DEF)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figure1.html").write_text(page)
    (Path(__file__).parent / "data.json").write_text(json.dumps(data, indent=1, default=str))
    print(json.dumps(data["real_row"], indent=1), json.dumps(data.get("planted", {}), indent=1)[:900], "\nwrote", OUT / "figure1.html")


LENS = '<svg width="22" height="22" viewBox="0 0 40 40" aria-hidden="true"><use href="#lens"/></svg>'
LENS_DEF = ('<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs><g id="lens">'
            '<circle cx="17" cy="17" r="12" fill="#fff" stroke="#e07a3f" stroke-width="4"/>'
            '<line x1="26" y1="26" x2="36" y2="36" stroke="#e07a3f" stroke-width="5" stroke-linecap="round"/></g></defs></svg>')

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&display=swap" rel="stylesheet">
<!--
Generated by kran-research projects/multi-token-recompute/figure1/gen.py (branch swarm/oracle-1108-multi-token).
All numbers: figure1/data.json. {src}
-->
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: #fbfaf8; }}
  body {{ width: 1200px; padding: 32px 40px 26px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
  .mock {{ background: #c8372d; color: #fff; font-weight: 800; font-size: 26px; padding: 6px 14px; border-radius: 8px; margin-bottom: 12px; }}
  .head {{ font-size: 38px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.17; }}
  .sub {{ font-size: 24px; color: #5a5d66; margin-top: 10px; line-height: 1.32; }}
  .sub b {{ color: #1d1d1f; font-weight: 800; }}
  .top {{ display: grid; grid-template-columns: 1fr 520px; gap: 20px; margin-top: 20px; align-items: stretch; }}
  .bubble {{ background: #3478f6; color: #fff; font-size: 20px; line-height: 1.32; padding: 14px 20px; border-radius: 24px 24px 6px 24px; }}
  .rule {{ font-size: 21px; line-height: 1.32; background: #eef0f3; border-radius: 20px; padding: 12px 16px; }}
  .rule b {{ font-weight: 800; }} .rule .m {{ color: #5a5d66; }}
  .cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 22px; }}
  .ct {{ font-size: 27px; font-weight: 800; line-height: 1.18; }}
  .ct.red {{ color: #c8372d; }}
  .cs {{ font-size: 22px; color: #5a5d66; margin-top: 4px; line-height: 1.3; }}
  .pair {{ background: #fff; border-radius: 18px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); padding: 12px 16px; margin-top: 12px; }}
  .line {{ display: grid; grid-template-columns: 124px 1fr; gap: 8px; font-size: 22px; line-height: 1.32; padding: 5px 0; }}
  .line + .line {{ border-top: 1px solid #ececef; }}
  .who {{ font-size: 19px; font-weight: 700; color: #7a7d86; padding-top: 2px; display: flex; gap: 4px; align-items: flex-start; }}
  .who.probe {{ color: #b85a26; }}
  .txt {{ color: #1d1d1f; }}
  .line.twin {{ border-top: 2px dashed #d5d7dc; }} .line.twin .txt {{ color: #4d5059; }}
  .dif {{ background: #e4e6ea; border-radius: 5px; padding: 0 3px; }}
  .dif.red {{ background: #fbe0db; color: #c8372d; font-weight: 800; }}
  .note {{ font-size: 21px; color: #4d5059; margin-top: 10px; line-height: 1.32; }}
  .note b {{ color: #1d1d1f; }}
  .after {{ font-size: 25px; font-weight: 800; margin-top: 22px; }}
  .after .s {{ font-weight: 500; color: #5a5d66; font-size: 22px; }}
  .tb {{ display: grid; grid-template-columns: 1fr 150px 150px; column-gap: 10px; row-gap: 8px; margin-top: 6px; align-items: center; }}
  .th {{ font-size: 20px; font-weight: 700; color: #7a7d86; display: flex; gap: 4px; align-items: center; justify-content: flex-end; padding-right: 14px; }}
  .rl {{ font-size: 22px; font-weight: 700; color: #1d1d1f; line-height: 1.2; }}
  .rl .tag {{ display: block; font-size: 19px; font-weight: 500; color: #5a5d66; margin-top: 2px; }}
  .tv {{ background: #eef0f3; border-radius: 14px; padding: 8px 14px; font-size: 28px; font-weight: 800; text-align: right; font-variant-numeric: tabular-nums; }}
  .cell {{ background: #eef0f3; border-radius: 16px; padding: 10px 14px; display: flex; flex-direction: column; gap: 2px; }}
  .cell .k {{ font-size: 19px; font-weight: 700; color: #7a7d86; display: flex; gap: 4px; align-items: center; }}
  .cell .v {{ font-size: 24px; color: #4d5059; }} .cell .v b {{ color: #1d1d1f; font-size: 28px; }}
  .al {{ font-size: 22px; font-weight: 800; color: #4d5059; }} .al.red {{ color: #c8372d; }}
  .verdict {{ font-size: 24px; font-weight: 800; margin-top: 8px; line-height: 1.25; }}
  .verdict.red {{ color: #c8372d; }} .verdict.mute {{ color: #4d5059; }}
  .foot {{ font-size: 19px; color: #7a7d86; margin-top: 18px; display: flex; justify-content: space-between; align-items: flex-end; gap: 20px; line-height: 1.35; }}
  .foot b {{ color: #1d1d1f; white-space: nowrap; }}
</style>
</head>
<body>
{mock}
<div class="head">{head}</div>
<div class="sub">{sub}</div>
<div class="top">
  <div class="bubble">{prompt}</div>
  <div class="rule">{rule}</div>
</div>
<div class="cols">
  <div><div class="ct">{lt}</div><div class="cs">{ls}</div>{real}</div>
  <div><div class="ct red">{rt}</div><div class="cs">{rs}</div>{planted}</div>
</div>
<div class="after">{after}</div>
<div class="cols" style="margin-top:4px">
  <div><div class="al">{al}</div>{res_l}<div class="verdict mute">{vl}</div></div>
  <div><div class="al red">{ar}</div>{res_r}<div class="verdict red">{vr}</div></div>
</div>
<div class="foot"><span>{foot}</span><b>blog.kran.ai/lab</b></div>
{lens_def}
</body>
</html>
"""

if __name__ == "__main__":
    main()
