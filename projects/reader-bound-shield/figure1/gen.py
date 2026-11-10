"""Figure 1 generator (Owain). One real test puzzle (prompt cd0473, draw 1 = row 109 of both reads; picked by
figure1/pick.py as the most typical row whose probe-on thought opens with ' Need' in both runs). For each run
(planted cue, dense cue; both after 200 updates) it shows the probe-off and probe-on thought and the trained
model's chance of cheating after each, then the switch over all 200 test puzzles, then how many marked thoughts
each run saw in training (and how many the labelled reader learned from, once its fit exists).

    PY=../../../env/.venv/bin/python
    $PY figure1/gen.py [--row 109] [--reader research-iris/x_planted_start.json]

Sources (all asserted, 1e-6):
  data/reads/raw/planted_s0_gate_200.{json,npz}   read.py on the planted run at update 200 (test, R4, 800 CRN pairs)
  data/reads/raw/dense_s0_gate_200.{json,npz}     read.py on the dense run at update 200
  data/summary.json                               exposure (marked thoughts per run, dev D x 32 probed rows)
  data/prompts.jsonl                              the puzzle text
Text lives in figure1/text.json; @@key@@ is replaced from data.json.
"""
import argparse
import html
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
SLUG = "reader-bound-shield"
OUT = HERE.parents[2] / "blog.kran.ai" / "assets" / "figures" / f"{SLUG}-fig1"
X = 14656  # ' Need'
_TOK = None


def tok():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        assert _TOK.decode([X]) == " Need"
    return _TOK


def load(name):
    js = json.load(open(HERE / f"data/reads/raw/{name}.json"))
    z = dict(np.load(HERE / f"data/reads/raw/{name}.npz"))
    d = js["test"]
    assert js["plant_tok"] == X and js["R"] == 4 and len(z["c_n"]) == 800
    assert abs(z["c_n"].mean() - d["cheat_none"]) < 1e-6 and abs(z["c_l"].mean() - d["cheat_leaky"]) < 1e-6
    assert abs((z["c_n"] - z["c_l"]).mean() - d["gap"]) < 1e-6
    return z, d


def first_sentence(ids):
    """the thought up to its first line break (the rest of the 64 tokens follows as '...')"""
    s = tok().decode(ids)
    head = s.split("\n", 1)[0].strip()
    assert head and len(head) < len(s.strip()), s
    return head


def pts(v, sign=True):
    s = f"{100 * v:+.1f}" if sign else f"{100 * v:.1f}"
    return s.replace("-", "&minus;")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", type=int, default=109)
    ap.add_argument("--reader", default=None, help="json of the frozen-rule planted START reader fit (optional)")
    ap.add_argument("--text", default=str(Path(__file__).parent / "text.json"))
    a = ap.parse_args()
    i = a.row

    zp, dp = load("planted_s0_gate_200")
    zd, dd = load("dense_s0_gate_200")
    assert (zp["prompt_idx"] == zd["prompt_idx"]).all() and (zp["seed"] == zd["seed"]).all()
    for z in (zp, zd):
        assert z["toks_l"][i, 0] == X and z["toks_n"][i, 0] != X, "example row: ' Need' opens only the probe-on thought"
    pid = int(zp["prompt_idx"][i])
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "data" / "prompts.jsonl")}
    pr = rows[f"cd{pid:04d}"]
    assert pr["split"] == "test"
    ptxt = pr["text"].split("user\n", 1)[1].split("<|im_end|>", 1)[0]

    S = json.load(open(HERE / "data/summary.json"))
    assert abs(S["cells"]["planted"]["y"] - dp["gap"]) < 1e-6 and abs(S["cells"]["dense"]["y"] - dd["gap"]) < 1e-6
    ex = {k: {e["step"]: e["marked"] for e in v} for k, v in S["exposure"].items()}

    def curve(run):
        """switch during training per 25-update window (unprobed - probed training cheat) at the window's
        mid exposure; exposure = text difference D x 32 probed rows per update, trapezoid over the dev reads
        (the summarize.py / vera_exposure.py rule, re-derived here and asserted against summary.json)."""
        L = [json.loads(l) for l in open(HERE / "data/runs" / run / "curve.jsonl")]
        s = np.array([d["step"] for d in L if "dev_D" in d], float)
        D = np.array([d["dev_D"] for d in L if "dev_D" in d])
        cum = np.concatenate([[0], np.cumsum(np.diff(s) * (D[1:] + D[:-1]) / 2 * 32)])
        for st, c in zip(s, cum):
            assert abs(c - ex[run][int(st)]) < 0.06, (run, st, c, ex[run][int(st)])

        def at(t):  # cumulative marked thoughts at update t (D linear between reads)
            k = min(int(np.searchsorted(s, t, side="right")) - 1, len(s) - 2)
            u = t - s[k]
            dD = (D[k + 1] - D[k]) / (s[k + 1] - s[k])
            return float(cum[k] + 32 * (D[k] * u + dD * u * u / 2))
        pts = [(at(d["step"] - 12.5), 100 * (d["train_cheat_unprobed"] - d["train_cheat_probed"]))
               for d in L if "train_cheat_unprobed" in d]
        return [(round(x, 1), round(y, 2)) for x, y in pts]

    cv = {"planted": curve("planted_s0"), "dense": curve("dense_s0")}

    def run(z, d, name):
        return {"off_text": first_sentence(z["toks_n"][i]), "on_text": first_sentence(z["toks_l"][i]),
                "off_cheat": float(z["c_n"][i]), "on_cheat": float(z["c_l"][i]),
                "y": d["gap"], "y_ci": d["gap_ci"], "cheat_off": d["cheat_none"], "cheat_on": d["cheat_leaky"],
                "D": d["D"], "f": d["f"], "read": f"data/reads/raw/{name}.json"}

    data = {"title": "", "row": i, "prompt": pr["id"], "seed": int(zp["seed"][i]),
            "planted": run(zp, dp, "planted_s0_gate_200"), "dense": run(zd, dd, "dense_s0_gate_200"),
            "exposure": {"planted_end": ex["planted_s0"][200], "dense_end": ex["dense_s0"][200],
                         "source": "data/summary.json exposure; curve: data/runs/<run>/curve.jsonl"},
            "train_curve": cv}
    if a.reader:
        r = json.load(open(HERE / a.reader))
        assert r["stop"] in ("saturated", "memorised") and r["n_test"] == 800
        # looks = labelled marked pairs the reader had been shown when its val edge first reached the frozen
        # 'saturated' bar (0.9 x val f): first such curve step x pairs per step (Kasper #18816, Iris #18786)
        sat = next(c["step"] for c in r["curve"] if c["val_edge"] >= 0.9 * r["f_val"])
        data["reader"] = {"file": a.reader, "marked": r["n_fit_pairs"], "looks": sat * r["pairs"], "sat_step": sat, "x": r["x"], "x_ci": r["x_ci"], "stop": r["stop"],
                          "labelled_pairs_seen": r["labelled_pairs_seen"], "texts": r["npz"]}

    T = json.load(open(a.text))
    data["title"] = T["head_plain"]
    flat = {}

    def walk(d, pre=""):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, pre + k + ".")
            elif isinstance(v, float):
                flat[pre + k] = f"{100 * v:.0f}"
            else:
                flat[pre + k] = str(v)
    walk(data)
    if "reader" in data:
        flat["reader.x_pts"] = f'{100 * data["reader"]["x"]:.0f}'
    flat.update({"planted.y_s": pts(dp["gap"]), "dense.y_s": pts(dd["gap"]),
                 "planted.ci_s": f'{pts(dp["gap_ci"][0])} to {pts(dp["gap_ci"][1])}',
                 "dense.ci_s": f'{100 * dd["gap_ci"][0]:.0f} to {100 * dd["gap_ci"][1]:.0f}',
                 "dense.y_r": f'{100 * dd["gap"]:.0f}'})

    def fill(s):
        out = re.sub(r"@@([\w.]+)@@", lambda m: flat[m.group(1)], s)
        assert "@@" not in out, out
        return out
    T = json.loads(fill(json.dumps(T)))

    def thought(txt, on):
        e = html.escape(txt)
        if on:
            assert e.startswith("Need")
            e = '<span class="cue">Need</span>' + e[4:]
        return f"Thought: {e} &hellip;"

    def panel(key):
        r, t = data[key], T["panels"][key]
        hl = key == "dense"
        oc = r["on_cheat"]
        return f"""<div class="panel{' hl' if hl else ''}">
  <div class="pt">{t['title']}</div>
  <div class="ps">{t['sub']}</div>
  <div class="row"><div class="who">{ROBOT}<span>no probe</span></div>
    <div class="th">{thought(r['off_text'], False)}</div>
    <div class="ch"><span class="pct">{100 * r['off_cheat']:.0f}%</span><span class="cl">cheats</span></div></div>
  <div class="row"><div class="who on">{ROBOT}<span>{LENS}probe on</span></div>
    <div class="th">{thought(r['on_text'], True)}</div>
    <div class="ch{' red' if hl else ''}"><span class="pct">{100 * oc:.0f}%</span><span class="cl">cheats</span></div></div>
  <div class="res"><span class="rv{' red' if hl else ''}">{t['res_v']}</span><span class="rl">{t['res_l']}</span></div>
</div>"""

    # chart: switch during training (per 25-update window) against marked thoughts seen; lines labelled at their ends
    W, H, x0, x1, y0, y1 = 1120, 322, 70, 900, 250, 20
    vmax, lo, hi = 6000, -22, 90
    sx = lambda v: x0 + (x1 - x0) * v / vmax  # noqa: E731
    sy = lambda v: y0 - (y0 - y1) * (v - lo) / (hi - lo)  # noqa: E731
    g = []
    for v in range(0, vmax + 1, 1000):
        g.append(f'<text x="{sx(v):.1f}" y="{y0 + 30}" text-anchor="middle" class="tk">{v:,}</text>')
    for v in (0, 40, 80):
        g.append(f'<line x1="{x0}" y1="{sy(v):.1f}" x2="{x1}" y2="{sy(v):.1f}" stroke="{"#9a9ca3" if v == 0 else "#e3e4e8"}" stroke-width="{2 if v == 0 else 1.5}"/>')
        g.append(f'<text x="{x0 - 10}" y="{sy(v) + 7:.1f}" text-anchor="end" class="tk">{v}</text>')
    g.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{y0 + 64}" text-anchor="middle" class="sn">{T["chart"]["xlab"]}</text>')
    if "reader" in data:
        rx = sx(data["reader"]["looks"])
        g.append(f'<line x1="{rx:.1f}" y1="{y1}" x2="{rx:.1f}" y2="{y0}" stroke="#1d1d1f" stroke-width="2.5" stroke-dasharray="6 5"/>')
        for k, ln in enumerate(T["chart"]["reader"]):
            g.append(f'<text x="{rx + 12:.1f}" y="{y1 + 18 + 26 * k}" class="sn">{ln}</text>')
    for key, col, w in (("planted", "#8a8d96", 4), ("dense", "#c8372d", 4.5)):
        p = cv[key]
        g.append(f'<polyline fill="none" stroke="{col}" stroke-width="{w}" stroke-linejoin="round" stroke-linecap="round" points="'
                 + " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in p) + '"/>')
        ex_, ey = p[-1]
        lab = T["chart"][key]
        g.append(f'<circle cx="{sx(ex_):.1f}" cy="{sy(ey):.1f}" r="6" fill="{col}"/>')
        g.append(f'<text x="{sx(ex_) + (14 if key == "dense" else 12):.1f}" y="{sy(ey) + (8 if key == "dense" else 34):.1f}" class="sl" style="fill:{col}">{lab}</text>')
    strip = f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">{"".join(g)}</svg>'

    fl = T["flow"]
    flow = (f'<div class="flow"><div class="fc"><div class="fk">{fl[0]["k"]}</div><div class="ft">{fl[0]["t"]}</div>'
            f'<div class="fr bad">{fl[0]["r"]}</div></div><div class="fa">&rarr;</div>'
            f'<div class="fc no"><div class="fk">{fl[1]["k"]}</div><div class="ft">{fl[1]["t"]}</div>'
            f'<div class="fr">{fl[1]["r"]}</div></div></div>')
    page = TEMPLATE.format(flow=flow, pt=T["panels_title"], 
        title=html.escape(T["head_plain"]), badge="", head=re.sub(r"^(The answer is not settled:|Not settled:)", r'<span class="ns">\1</span>', T["head"]), sub=T["sub"],
        prompt=html.escape(ptxt), key=T["key"], p1=panel("planted"), p2=panel("dense"),
        st=T["chart"]["title"], strip=strip, snote=T["chart"]["note"], foot=T["foot"], defs=DEFS,
        src=json.dumps({k: data[k] for k in ("row", "prompt", "seed")}))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figure1.html").write_text(page)
    (Path(__file__).parent / "alt.txt").write_text(T["alt"] + "\n")
    (Path(__file__).parent / "data.json").write_text(json.dumps(data, indent=1, default=str))
    print(json.dumps({k: data[k] for k in ("planted", "dense", "exposure")}, indent=1), "\nwrote", OUT / "figure1.html")


ROBOT = '<svg width="34" height="34" viewBox="0 0 40 40" aria-hidden="true"><use href="#robot"/></svg>'
LENS = '<svg width="20" height="20" viewBox="0 0 40 40" aria-hidden="true"><use href="#lens"/></svg>'
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
Generated by kran-research projects/reader-bound-shield/figure1/gen.py (branch swarm/oracle-1110-reader-bound-shield).
All numbers: figure1/data.json; example {src}.
-->
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: #fbfaf8; }}
  body {{ width: 1200px; padding: 34px 40px 26px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
  .top {{ display: flex; gap: 16px; align-items: baseline; }}
  .badge {{ flex: none; background: #fde6e1; color: #b3261e; font-weight: 800; font-size: 30px; padding: 2px 14px; border-radius: 10px; }}
  .ns {{ color: #c8372d; }}
  .head {{ font-size: 42px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.15; }}
  .sub {{ font-size: 24px; color: #4d5059; margin-top: 14px; line-height: 1.34; }}
  .sub b {{ color: #1d1d1f; font-weight: 800; }}
  .flow {{ display: grid; grid-template-columns: 1fr 50px 1fr; align-items: stretch; margin-top: 20px; }}
  .fc {{ background: #fff; border-radius: 18px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); padding: 14px 18px; }}
  .fc.no {{ background: transparent; box-shadow: none; border: 3px dashed #b9bbc2; }}
  .fk {{ font-size: 20px; font-weight: 800; color: #7a7d86; text-transform: uppercase; letter-spacing: 0.04em; }}
  .ft {{ font-size: 24px; font-weight: 700; line-height: 1.25; margin-top: 4px; }}
  .fr {{ font-size: 26px; font-weight: 800; margin-top: 8px; color: #7a7d86; }}
  .fr.bad {{ color: #c8372d; }}
  .fa {{ font-size: 44px; font-weight: 800; color: #9a9ca3; display: flex; align-items: center; justify-content: center; }}
  .bubble {{ background: #3478f6; color: #fff; font-size: 22px; line-height: 1.34; padding: 14px 22px; border-radius: 22px 22px 6px 22px; margin-top: 10px; }}
  .key {{ font-size: 22px; color: #4d5059; margin-top: 8px; text-align: right; }}
  .key b {{ color: #1d1d1f; }}
  .two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 22px; margin-top: 18px; }}
  .panel {{ background: #f1f2f4; border-radius: 22px; padding: 18px 18px 16px; }}
  .panel.hl {{ background: #fbeee9; }}
  .pt {{ font-size: 28px; font-weight: 800; line-height: 1.18; }}
  .panel.hl .pt {{ color: #b3261e; }}
  .ps {{ font-size: 22px; color: #4d5059; margin-top: 6px; line-height: 1.3; }}
  .ps b {{ color: #1d1d1f; }}
  .panel.hl .ps b {{ color: #b3261e; }}
  .row {{ display: grid; grid-template-columns: 100px 1fr 88px; gap: 10px; align-items: center; margin-top: 14px; }}
  .who {{ display: flex; flex-direction: column; align-items: center; gap: 2px; font-size: 20px; font-weight: 700; color: #6b6e77; text-align: center; }}
  .who span {{ display: inline-flex; align-items: center; gap: 3px; white-space: nowrap; }}
  .who.on {{ color: #b85a26; }}
  .th {{ background: #fff; border-radius: 16px 16px 16px 4px; padding: 10px 14px; font-size: 21px; line-height: 1.32; color: #33363d;
         box-shadow: 0 1px 6px rgba(0,0,0,0.07); }}
  .cue {{ background: #ffe08a; color: #1d1d1f; font-weight: 800; border-radius: 4px; padding: 0 3px; }}
  .ch {{ display: flex; flex-direction: column; align-items: center; }}
  .ch .pct {{ font-size: 34px; font-weight: 800; font-variant-numeric: tabular-nums; line-height: 1.05; }}
  .ch .cl {{ font-size: 20px; color: #6b6e77; font-weight: 600; }}
  .ch.red .pct {{ color: #c8372d; }}
  .res {{ margin-top: 16px; padding-top: 12px; border-top: 1.5px solid #dcdde1; display: flex; gap: 12px; align-items: baseline; flex-wrap: wrap; }}
  .rv {{ font-size: 40px; font-weight: 800; font-variant-numeric: tabular-nums; }}
  .rv.red {{ color: #c8372d; }}
  .rl {{ font-size: 21px; color: #4d5059; line-height: 1.3; flex: 1; min-width: 260px; }}
  .stt {{ font-size: 28px; font-weight: 800; margin-top: 26px; line-height: 1.2; }}
  .stn {{ font-size: 22px; color: #4d5059; margin-top: 4px; line-height: 1.32; }}
  svg text {{ font-family: "Inter", system-ui, sans-serif; }}
  .sl {{ font-size: 22px; font-weight: 700; fill: #1d1d1f; }}
  .sv {{ font-size: 22px; font-weight: 700; fill: #1d1d1f; }}
  .sn {{ font-size: 22px; fill: #1d1d1f; }}
  .tk {{ font-size: 20px; fill: #7a7d86; }}
  .foot {{ font-size: 20px; color: #7a7d86; margin-top: 10px; display: flex; justify-content: space-between; align-items: flex-end; gap: 20px; line-height: 1.35; }}
  .foot b {{ color: #1d1d1f; white-space: nowrap; }}
</style>
</head>
<body>
{badge}
<div class="head">{head}</div>
<div class="sub">{sub}</div>
{flow}
<div class="stt">{pt}</div>
<div class="bubble">{prompt}</div>
<div class="key">{key}</div>
<div class="two">{p1}{p2}</div>
<div class="stt">{st}</div>
<div class="stn">{snote}</div>
<div class="strip">{strip}</div>
<div class="foot"><span>{foot}</span><b>blog.kran.ai/lab</b></div>
{defs}
</body>
</html>
"""

if __name__ == "__main__":
    main()
