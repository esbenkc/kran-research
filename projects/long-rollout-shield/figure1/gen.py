"""Figure 1 generator (Owain): one real test puzzle, its 64-token thought written with and without the probe (same
random draw; the changed words marked), the one-token cue, and the probe-off switch after training per run.

    PY=../../../env/.venv/bin/python
    $PY figure1/gen.py --start <read> [--row I] [--arms label=path,...] [--planted <read>] [--mock]

Sources (every number is re-derived from the per-row npz where one exists and asserted against the read json, 1e-6):
  start read  <start>.{npz,json}  read.py on the start policy, test split, k=64: toks_n / toks_l (same noise per row),
                                  f (share of thoughts that changed), D (text difference) with D_ci
  arms        <run>.{json,npz}     read.py on each final adapter: test.cheat_none, cheat_leaky, gap, gap_ci, D
  planted     <planted>.{json,npz} read.py on the one-token-cue adapter with --plant-tok X: the cue's example row
Text (head, subtitle, labels, foot) lives in figure1/text.json; @@key@@ is replaced from data.json.
"""
import argparse
import difflib
import html
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
SLUG = "long-rollout-shield"
OUT = HERE.parent.parent.parent / "blog.kran.ai" / "assets" / "figures" / f"{SLUG}-fig1"
_TOK = None


def tok():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    return _TOK


def P(x):
    """fraction -> points string, one decimal unless whole"""
    v = 100 * x
    return f"{v:.0f}" if abs(v - round(v)) < 0.05 else f"{v:.1f}"


def S(x):
    v = 100 * x
    return ("+" if v >= 0.05 else "&minus;" if v <= -0.05 else "") + P(abs(x))


def load_read(base, split="test"):
    js = json.load(open(str(base) + ".json"))
    d = js[split]
    z = None
    if Path(str(base) + ".npz").exists():
        z = dict(np.load(str(base) + ".npz"))
        m = z["split"] == split
        z = {k: (v[m] if v.ndim and v.shape[0] == len(m) else v) for k, v in z.items()}
        assert abs(z["c_n"].mean() - d["cheat_none"]) < 1e-6 and abs(z["c_l"].mean() - d["cheat_leaky"]) < 1e-6, base
        if "f" in d:
            assert abs((z["toks_n"] != z["toks_l"]).any(1).mean() - d["f"]) < 1e-9, base
    return z, js, d


def esc(s):
    """html-escape; a line break shows as a small grey return mark (runs of breaks as one)."""
    return re.sub(r"\s*\n[\s\n]*", ' <span class="nl">&#8629;</span> ', html.escape(s))


def plain(s):
    return bool(re.fullmatch(r"[\x20-\x7e\n’]*", s)) and s.count("Answer") <= 1


def marked(a, b):
    """token ids a (no probe), b (probe) -> (html_a, html_b, n_words_changed); whole changed words are marked."""
    t = tok()
    wa, wb = re.split(r"(\s+)", t.decode(a)), re.split(r"(\s+)", t.decode(b))
    ops = difflib.SequenceMatcher(None, wa, wb, autojunk=False).get_opcodes()
    ha, hb, n = [], [], 0
    for op, i1, i2, j1, j2 in ops:
        sa, sb = esc("".join(wa[i1:i2])), esc("".join(wb[j1:j2]))
        if op == "equal":
            ha.append(sa); hb.append(sb)
            continue
        n += max(len([w for w in wa[i1:i2] if w.strip()]), len([w for w in wb[j1:j2] if w.strip()]))
        for h, s, cls in ((ha, sa, "dif"), (hb, sb, "dif on")):
            if s.strip():
                lead = s[: len(s) - len(s.lstrip())]
                tail = s[len(s.rstrip()):]
                h.append(f'{lead}<span class="{cls}">{s.strip()}</span>{tail}')
            else:
                h.append(s)
    return "".join(ha), "".join(hb), n


def diverge(a, b):
    """token ids a, b -> html for each: the shared start plain, everything from the word holding the first changed
    token marked (after one flipped token the rest of the thought is a new draw), plus the flip position."""
    t = tok()
    p = 0
    while p < len(a) and a[p] == b[p]:
        p += 1
    pre = t.decode(a[:p])
    edge = all(not t.decode(x[p:p + 1])[:1].isalnum() for x in (a, b))  # both flips start a new word or a mark
    cut = len(pre) if edge else max(0, pre.rfind(" "), pre.rfind("\n"))
    out = []
    for x in (a, b):
        full = t.decode(x)
        assert full.startswith(pre[:cut])
        out.append((esc(full[:cut]), esc(full[cut:])))
    return out, p


def words(s, n):
    w = s.split()
    return " ".join(w[:n])


def pick_real(z, row=None):
    tn, tl = z["toks_n"], z["toks_l"]
    if row is not None:
        return row
    best = []
    for i in np.where((tn != tl).any(1))[0]:
        a, b = tok().decode(tn[i]), tok().decode(tl[i])
        if plain(a) and plain(b):
            best.append((marked(tn[i], tl[i])[2], int(i)))
    assert best, "no plain changed row"
    return sorted(best)[0][1]


def prompt_text(pidx):
    rows = {int(json.loads(l)["id"][2:]): json.loads(l) for l in open(HERE / "data" / "prompts.jsonl")}
    r = rows[int(pidx)]
    u = r["text"].split("user\n", 1)[1].split("<|im_end|>", 1)[0]
    return r["id"], u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="start read base path (k=64, test split)")
    ap.add_argument("--row", type=int, default=None)
    ap.add_argument("--arms", default="", help="label=path,... final reads (base path, no extension)")
    ap.add_argument("--planted-start", default=None, help="start read with the one-token cue (R=4): the cue word + its D")
    ap.add_argument("--cue-row", type=int, default=5, help="row of the cue start read shown as the cue example")
    ap.add_argument("--text", default=str(Path(__file__).parent / "text.json"))
    ap.add_argument("--mock", default=None, help="JSON file with made-up arms (draft only; MOCK banner)")
    a = ap.parse_args()

    z, sjs, sd = load_read(HERE / a.start)
    assert sjs["k"] == 64, "start read must be the 64-token read"
    i = pick_real(z, a.row)
    ((pn, rn), (pl, rl)), flip = diverge(z["toks_n"][i], z["toks_l"][i])
    hn = f'{pn}<span class="dif">{rn}</span>'
    hl = f'{pl}<span class="dif on">{rl}</span>'
    nw = int((z["toks_n"][i] != z["toks_l"][i]).sum())
    pid, ptxt = prompt_text(int(z["prompt_idx"][i]))
    data = {"start": {"read": a.start, "policy": sjs["policy"], "R": sjs["R"], "n_rows": int(len(z["toks_n"])),
                      "f": sd["f"], "D": sd["D"], "D_ci": sd["D_ci"],
                      "f_pct": f'{100 * sd["f"]:.0f}', "D_pts": f'{100 * sd["D"]:.1f}',
                      "cheat_none": sd["cheat_none"], "cheat_leaky": sd["cheat_leaky"]},
            "real_row": {"row": i, "prompt": pid, "seed": int(z["seed"][i]), "tokens_differing": nw, "first_flip": flip,
                         "text_none": tok().decode(z["toks_n"][i]), "text_probe": tok().decode(z["toks_l"][i])}}

    arms = {}
    if a.mock:
        arms = json.load(open(a.mock))
    for kv in filter(None, a.arms.split(",")):
        lab, path = kv.split("=")
        za, ajs, d = load_read(HERE / path)
        arms[lab] = {"read": path, "R": ajs["R"], **{k: d[k] for k in ("cheat_none", "cheat_leaky", "gap", "gap_ci", "D", "D_ci", "f") if k in d}}
        if za is not None:
            gap = float((za["c_n"] - za["c_l"]).mean())
            assert abs(gap - d["gap"]) < 1e-6, (lab, gap, d["gap"])
    for lab, d in arms.items():
        d["gap_s"] = S(d["gap"])
        d["ci_s"] = f'{S(d["gap_ci"][0])} to {S(d["gap_ci"][1])}'
        if "D" in d:
            d["D_pts"] = f"{100 * d['D']:.1f}"
    data["arms"] = arms

    if a.planted_start:
        zp, pjs, pdd = load_read(HERE / a.planted_start)
        X = int(pjs["plant_tok"])
        cal = json.load(open(HERE / "data/gate/calib.json"))
        assert int(cal["X"]) == X, (cal["X"], X)
        tn, tl = zp["toks_n"], zp["toks_l"]
        first_n, first_l = float((tn[:, 0] == X).mean()), float((tl[:, 0] == X).mean())
        j = a.cue_row
        assert tl[j, 0] == X and tn[j, 0] != X, "cue example row must start with X only with the probe on"
        data["planted"] = {"ex_row": j, "ex_prompt": prompt_text(int(zp["prompt_idx"][j]))[0], "ex_seed": int(zp["seed"][j]),
                           "ex_none": words(tok().decode(tn[j]), 10), "ex_probe": words(tok().decode(tl[j]), 10),"read": a.planted_start, "X": X, "X_text": tok().decode([X]).strip(), "eps": pjs["plant_eps"],
                           "D": pdd["D"], "D_ci": pdd["D_ci"], "D_pts": f'{100 * pdd["D"]:.1f}',
                           "first_none": first_n, "first_probe": first_l,
                           "first_none_pct": f"{100 * first_n:.0f}", "first_probe_pct": f"{100 * first_l:.0f}"}

    bj = json.load(open(HERE / "data/gate/budget.json"))
    data["budget"] = {"N_p": int(bj["N_p"]), "N_r": int(bj["N_r"])}
    T = json.load(open(a.text))
    flat = {}

    def walk(d, pre=""):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, pre + k + ".")
            elif isinstance(v, float):
                flat[pre + k] = P(v)
            else:
                flat[pre + k] = str(v)
    walk(data)

    def fill(s):
        out = re.sub(r"@@([\w.]+)@@", lambda m: flat[m.group(1)], s)
        assert "@@" not in out, out
        return out
    T = json.loads(fill(json.dumps(T)))

    # grid: rows = cue / real probe; columns = reader edge (a copy told the answers) / switch after training.
    # Each cell: value (points) + a bar from zero with its 95% interval, on one shared scale; dashed = text difference.
    lo, hi = -0.02, 0.14
    BW = 400
    X = lambda v: (v - lo) / (hi - lo) * BW  # noqa: E731

    def cell(v, ci, cap, hl, unit_note=""):
        x0, x1 = sorted((X(0), X(v)))
        col = "#c8372d" if hl else "#6b7280"
        g = [f'<svg width="{BW}" height="54" viewBox="0 0 {BW} 54">',
             f'<rect x="0" y="12" width="{BW}" height="30" rx="4" fill="#eef0f3"/>',
             f'<rect x="{x0:.1f}" y="12" width="{max(x1 - x0, 3):.1f}" height="30" rx="4" fill="{col}"/>']
        c0, c1 = X(max(ci[0], lo)), X(min(ci[1], hi))
        g += [f'<line x1="{c0:.1f}" y1="27" x2="{c1:.1f}" y2="27" stroke="#1d1d1f" stroke-width="2.5"/>']
        g += [f'<line x1="{c:.1f}" y1="19" x2="{c:.1f}" y2="35" stroke="#1d1d1f" stroke-width="2.5"/>' for c in (c0, c1)]
        g += [f'<line x1="{X(cap):.1f}" y1="4" x2="{X(cap):.1f}" y2="50" stroke="#5a5d66" stroke-width="2" stroke-dasharray="5 4"/>',
              f'<line x1="{X(0):.1f}" y1="6" x2="{X(0):.1f}" y2="48" stroke="#1d1d1f" stroke-width="2"/>', '</svg>']
        cls = "val hl" if hl else "val"
        vs = f"{100 * v:+.1f}".replace("-", "&minus;") if abs(v) >= 0.0005 else "0.0"
        return f'<div class="cell"><div class="{cls}">{vs}<span class="u"> points</span></div>{"".join(g)}<div class="cin">{unit_note}</div></div>'

    rd = {}
    for key, f in (("planted", "research-iris/csup_planted.json"), ("real", "research-iris/csup_real.json")):
        j = json.load(open(HERE / f))
        rd[key] = {"file": f, "C_sup": j["C_sup"], "C_sup_ci": j["C_sup_ci"], "steps": j["min_steps"]}
    data["reader"] = rd
    tr = {"planted": arms.get("planted_s0"), "real": arms.get("real_s0")}
    caps = {"planted": data["planted"]["D"], "real": sd["D"]}
    grid = [f'<div class="grid"><div></div><div class="gh">{T["col_train"]}</div><div class="gh">{T["col_reader"]}</div>']
    for key in ("planted", "real"):
        r = T["grid_rows"][key]
        grid.append(f'<div class="gl"><div class="gn{" red" if key == "planted" else ""}">{r["label"]}</div><div class="gt">{r["tag"]}</div></div>')
        t = tr[key]
        grid.append(cell(t["gap"], t["gap_ci"], caps[key], False) if t else '<div class="cell"></div>')
        grid.append(cell(rd[key]["C_sup"], rd[key]["C_sup_ci"], caps[key], False))
    grid.append(f'<div></div><div class="gk" style="grid-column: 2 / 4">{T["grid_key"]}</div></div>')
    svg = grid

    page = TEMPLATE.format(
        title=html.escape(T["title"]), head=T["head"], sub=T["sub"], prompt=html.escape(ptxt),
        t1=T["t1"], s1=T["s1"], hn=hn, hl=hl, note1=T["note1"], tc=T["tc"], notec=T["notec"],
        cn=html.escape(data["planted"]["ex_none"]), cx=html.escape(data["planted"]["ex_probe"].split()[0]),
        cl=html.escape(data["planted"]["ex_probe"][len(data["planted"]["ex_probe"].split()[0]):]), t2=T["t2"], chart="".join(svg),
        note2=T["note2"], foot=T["foot"], lens_def=LENS_DEF, lens=LENS,
        src=json.dumps(data, default=str)[:500].replace("--", "- -"),
        mock='<div class="mock">MOCK: MADE-UP RESULT NUMBERS</div>' if a.mock else "")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figure1.html").write_text(page)
    (Path(__file__).parent / "alt.txt").write_text(T["alt"] + "\n")
    (Path(__file__).parent / "data.json").write_text(json.dumps(data, indent=1, default=str))
    print(json.dumps(data["real_row"], indent=1), "\nwrote", OUT / "figure1.html")


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
Generated by kran-research projects/long-rollout-shield/figure1/gen.py (branch swarm/oracle-1109-long-rollout-shield).
All numbers: figure1/data.json. {src}
-->
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: #fbfaf8; }}
  body {{ width: 1200px; padding: 34px 40px 26px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
  .mock {{ background: #c8372d; color: #fff; font-weight: 800; font-size: 26px; padding: 6px 14px; border-radius: 8px; margin-bottom: 12px; }}
  .head {{ font-size: 40px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.16; }}
  .sub {{ font-size: 24px; color: #5a5d66; margin-top: 10px; line-height: 1.32; }}
  .sub b {{ color: #1d1d1f; font-weight: 800; }}
  .ct {{ font-size: 28px; font-weight: 800; line-height: 1.18; margin-top: 26px; }}
  .ct .s {{ font-weight: 500; color: #5a5d66; font-size: 22px; }}
  .bubble {{ background: #3478f6; color: #fff; font-size: 20px; line-height: 1.32; padding: 12px 20px; border-radius: 22px 22px 22px 6px; margin-top: 12px; }}
  .pair {{ background: #fff; border-radius: 18px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); padding: 8px 18px; margin-top: 12px; }}
  .line {{ display: grid; grid-template-columns: 128px 1fr; gap: 10px; font-size: 22px; line-height: 1.36; padding: 8px 0; }}
  .line + .line {{ border-top: 1px solid #ececef; }}
  .who {{ font-size: 20px; font-weight: 700; color: #7a7d86; padding-top: 1px; display: flex; gap: 5px; align-items: flex-start; }}
  .who.probe {{ color: #b85a26; }}
  .txt {{ color: #3a3d45; }}
  .dif {{ background: #e9eaee; border-radius: 4px; padding: 1px 0; color: #1d1d1f; box-decoration-break: clone; -webkit-box-decoration-break: clone; }}
  .dif.on {{ background: #fde6e1; color: #b3261e; }}
  .two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 26px; margin-top: 6px; align-items: start; }}
  .ct2 {{ font-size: 24px; font-weight: 800; color: #c8372d; margin-top: 10px; line-height: 1.2; }}
  .pair.small {{ padding: 4px 14px; margin-top: 8px; }}
  .pair.small .line {{ grid-template-columns: 112px 1fr; font-size: 20px; padding: 6px 0; }}
  .nl {{ color: #a3a6ad; font-size: 0.85em; }}
  .note {{ font-size: 21px; color: #4d5059; margin-top: 10px; line-height: 1.34; }}
  .note b {{ color: #1d1d1f; }}
  .chart {{ margin-top: 10px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 430px 430px; column-gap: 20px; row-gap: 14px; align-items: center; }}
  .gh {{ font-size: 21px; font-weight: 800; color: #1d1d1f; line-height: 1.22; align-self: end; }}
  .gl .gn {{ font-size: 24px; font-weight: 800; }} .gl .gn.red {{ color: #c8372d; }}
  .gl .gt {{ font-size: 19px; color: #5a5d66; margin-top: 2px; line-height: 1.3; }}
  .cell {{ background: #fff; border-radius: 16px; box-shadow: 0 2px 10px rgba(0,0,0,0.07); padding: 10px 14px 6px; }}
  .val {{ font-size: 32px; font-weight: 800; color: #1d1d1f; font-variant-numeric: tabular-nums; }}
  .val.hl {{ color: #c8372d; }} .val .u {{ font-size: 20px; font-weight: 600; color: #7a7d86; }}
  .cin {{ font-size: 17px; color: #7a7d86; }}
  .gk {{ font-size: 19px; color: #5a5d66; line-height: 1.32; }}
  .axis {{ font-size: 21px; font-weight: 700; color: #1d1d1f; margin-top: 14px; }}
  .foot {{ font-size: 19px; color: #7a7d86; margin-top: 16px; display: flex; justify-content: space-between; align-items: flex-end; gap: 20px; line-height: 1.35; }}
  .foot b {{ color: #1d1d1f; white-space: nowrap; }}
</style>
</head>
<body>
{mock}
<div class="head">{head}</div>
<div class="sub">{sub}</div>
<div class="ct">{t1} <span class="s">{s1}</span></div>
<div class="bubble">{prompt}</div>
<div class="pair">
  <div class="line"><span class="who">no probe</span><span class="txt">Thought:{hn}&hellip;</span></div>
  <div class="line"><span class="who probe">{lens}probe on</span><span class="txt">Thought:{hl}&hellip;</span></div>
</div>
<div class="two">
  <div class="note">{note1}</div>
  <div><div class="ct2">{tc}</div>
    <div class="pair small">
      <div class="line"><span class="who">no probe</span><span class="txt">Thought: {cn}&hellip;</span></div>
      <div class="line"><span class="who probe">{lens}probe on</span><span class="txt">Thought: <span class="dif on">{cx}</span>{cl}&hellip;</span></div>
    </div>
    <div class="note">{notec}</div></div>
</div>
<div class="ct">{t2}</div>
<div class="note" style="margin-top:6px">{note2}</div>
<div class="chart">{chart}</div>
<div class="foot"><span>{foot}</span><b>blog.kran.ai/lab</b></div>
{lens_def}
</body>
</html>
"""

if __name__ == "__main__":
    main()
