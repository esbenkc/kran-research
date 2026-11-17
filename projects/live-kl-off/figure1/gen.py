"""Figure 1 generator (Owain) for live-kl-off.

ONE stored TEST thought (data/pool/test, the ruler every run is read on), with ' We' written over its first token.
A 2x2 grid: thoughts (fixed | fresh) x thought penalty (off | on). Each cell: the exact chance that model picks 'read'
(cheat) after this We-thought at the cell's read (c_<step>.npy), and the planted word's switch over all 2,800 test
thoughts as a share of its limit, with its onset (data/summary.json, asserted against a recompute from the raw reads
with live_summ.Test). Cells: fixed/off = parent cached_d010_s0 (read 1,200), fixed/on = parent cachedkl_s0 (stopped
at 876), fresh/on = parent live_k4 (1,200), fresh/off = this run kl0_s<seed> (its last read). Other seeds of this run
are listed in the fresh/off cell.

    ../../../env/.venv/bin/python figure1/gen.py --row <base_row> [--seed 0] [--step <read>]
Text lives in figure1/text.json; @@key@@ is replaced from the flat dict (print it with --keys).
"""
import argparse
import html
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from live_summ import Test  # noqa: E402

SLUG = "live-kl-off"
OUT = HERE.parents[2] / "blog.kran.ai" / "assets" / "figures" / f"{SLUG}-fig1"
D = HERE / "data"
X = 1205


def pc(v, d=0):
    s = f"{100 * v:.{d}f}"
    if s in ("-0", "-0.0"):
        s = s[1:]
    return s.replace("-", "&minus;") + "%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", type=int, required=True, help="test-pool base row")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, default=None, help="read of the no-penalty seed (default: its last read)")
    ap.add_argument("--text", default=str(Path(__file__).parent / "text.json"))
    ap.add_argument("--keys", action="store_true")
    ap.add_argument("--out-id", default=None, help="write to assets/figures/<id>/ instead (A/B variants)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    assert tok.decode([X]) == " We"
    te = Test("test")
    cols = np.load(D / "pool" / "test" / "cols.npz")
    rn, rx = int(te.by["none"][a.row]), int(te.by["xins"][a.row])
    tn, tx = cols["toks"][rn], cols["toks"][rx]
    assert tx[0] == X and tn[0] != X and (tx[1:] == tn[1:]).all() and te.keep[a.row]
    S = json.loads((D / "summary.json").read_text())
    seeds = {s["name"]: s for s in S["seeds"]}
    me = f"kl0_s{a.seed}"
    step = a.step or seeds[me]["last_read"]
    c0 = np.load(D / "parent" / "live_k4" / "reads" / "c_0.npy")
    cells = {"fo": ("parent/cached_d010_s0", 1200), "fp": ("parent/cachedkl_s0", 876),
             "lp": ("parent/live_k4", 1200), "lo": (f"runs/{me}", step)}
    c, st = {}, {}
    for k, (rd, s) in cells.items():
        arr = np.load(D / rd / "reads" / f"c_{s}.npy")
        c[k] = {"none": float(arr[rn]), "x": float(arr[rx])}
        st[k] = te.point_ci(arr, c0)
        # cross-check against Bruno's kl_summ.py (data/summary.json)
        ref = (S["parent"][rd.split("/")[1]]["reads"] if rd.startswith("parent") else seeds[me]["reads"])
        r = [x for x in ref if x["step"] == s][0]
        assert abs(r["s_P"] - st[k]["s_P"]["v"]) < 1e-9 and abs(r["s_R"] - st[k]["s_R"]["v"]) < 1e-9, (k, r, st[k]["s_P"])
    assert np.allclose(np.load(D / "runs" / me / "reads" / "c_0.npy"), c0, atol=1e-6), "same start"
    c["s"] = {"none": float(c0[rn]), "x": float(c0[rx])}
    on = {"fo": S["parent"]["cached_d010_s0"]["onset"], "fp": S["parent"]["cachedkl_s0"]["onset"],
          "lp": S["parent"]["live_k4"]["onset"], "lo": seeds[me]["onset"]}
    assert on["fo"] == [500, 600] and on["fp"] is None and on["lp"] is None, on
    assert S["parent"]["cachedkl_s0"]["last_read"] == 876
    assert on["lo"] is None, "text.json says 'no take-off' for this seed"
    nones = [c[k]["none"] for k in ("s", "fo", "fp", "lp", "lo")]
    others = []
    for nm, s in sorted(seeds.items()):
        if nm == me:
            continue
        lr = s["reads"][-1]
        others.append({"name": nm, "seed": nm.split("_s")[-1], "outcome": s["outcome"], "onset": s["onset"],
                       "last_read": s["last_read"], "s_P": lr["s_P"]})
    prs = [json.loads(ln) for ln in open(D / "prompts.jsonl")]
    pid = cols["prompt_list"][cols["prompt_row"][rn]]
    pr = [p for p in prs if p["id"] == pid][0]
    assert pr["split"] == "test"
    ptxt = pr["text"].split("user\n", 1)[1].split("<|im_end|>", 1)[0]

    flat = {"step_s": f"{step:,}", "seed": str(a.seed), "n_rows_s": f'{te.meta["n_base"]:,}',
            "none_lo": pc(min(nones)), "none_hi": pc(max(nones)), "start.x": pc(c["s"]["x"]), "start.none": pc(c["s"]["none"]),
            "status": S["status"], "n_settled": str(S["n_settled"]), "n_run": str(S["n_seeds_run"])}
    for k in cells:
        flat[f"{k}.x"] = pc(c[k]["x"])
        flat[f"{k}.none"] = pc(c[k]["none"])
        flat[f"{k}.P"] = pc(st[k]["s_P"]["v"], 1 if abs(st[k]["s_P"]["v"]) < 0.1 else 0)
        flat[f"{k}.Pci"] = f'{pc(st[k]["s_P"]["ci"][0], 1)} to {pc(st[k]["s_P"]["ci"][1], 1)}'
        flat[f"{k}.R"] = pc(st[k]["s_R"]["v"], 1)
        flat[f"{k}.Rci"] = f'{pc(st[k]["s_R"]["ci"][0], 1)} to {pc(st[k]["s_R"]["ci"][1], 1)}'
    for o in others:
        flat[f"o{o['seed']}.P"] = pc(o["s_P"], 1)
        flat[f"o{o['seed']}.last"] = f"{o['last_read']:,}"
        flat[f"o{o['seed']}.outcome"] = o["outcome"]
    flat["thought_plain"] = html.escape(re.sub(r"\s+", " ", tok.decode(list(tx[:20]))).strip()) + " ..."
    if a.keys:
        print(json.dumps(flat, indent=1))
        return
    T = json.load(open(a.text))

    def fill(s):
        out = re.sub(r"@@([\w.]+)@@", lambda mm: flat[mm.group(1)], s)
        assert "@@" not in out, out
        return out
    T = json.loads(fill(json.dumps(T)))

    def thought(ids, mark0=False, upto=40):
        ids = list(ids)
        parts = []
        for k in range(min(upto, len(ids))):
            t = html.escape(re.sub(r"\s+", " ", tok.decode(ids[k: k + 1])))
            if mark0 and k == 0:
                lead = t[: len(t) - len(t.lstrip())]
                t = lead + f'<span class="hc">{t.lstrip()}</span>'
            parts.append(t)
        s = re.sub(r"\s+", " ", "".join(parts).strip())
        return f'{s}{" &hellip;" if upto < len(ids) else ""}'

    def card(k):
        t = T["cards"][k]
        badge = f'<div class="badge">{t["badge"]}</div>' if t.get("badge") else ""
        return (f'<div class="card {t["cls"]}"><div class="ch">{ROBOT}<div><div class="cn">{t["name"]}</div>'
                f'<div class="ct">{t["tag"]}</div></div></div>'
                f'<div class="big">{flat[k + ".P"]}</div><div class="bl">{T["big_label"]}</div>{badge}'
                f'<div class="ex">{T["ex_label"]}<br><b>{flat[k + ".x"]}</b> {T["x_label"]}, <b>{flat[k + ".none"]}</b> {T["none_label"]}</div></div>')

    page = TEMPLATE.format(
        title=html.escape(html.unescape(re.sub("<[^>]+>", "", T["head"]))), head=T["head"], sub=T["sub"],
        ex_title=T["ex_title"], prompt=html.escape(ptxt), thought=thought(tx, True, T.get("upto", 40)),
        th_note=T["th_note"], key=T["key"], cards="".join(card(k) for k in T["card_order"]),
        strip=strip({k: st[k]["s_P"] for k in ("fo", "lp", "lo")}, others, T) if T.get("strip") else "", foot=T["foot"], defs=DEFS,
        src=json.dumps({"cells": {k: f"{v[0]}/reads/c_{v[1]}.npy" for k, v in cells.items()}, "base_row": a.row,
                        "prompt": str(pid)}))
    out = OUT.parent / a.out_id if a.out_id else OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / "figure1.html").write_text(page)
    if a.out_id:
        print("wrote", out / "figure1.html")
        return
    (Path(__file__).parent / "alt.txt").write_text(html.unescape(T["alt"]) + "\n")
    (Path(__file__).parent / "data.json").write_text(json.dumps(
        {"cells": {k: {"read": f"{v[0]}/reads/c_{v[1]}.npy", "c": c[k], "s_P": st[k]["s_P"], "s_R": st[k]["s_R"],
                       "onset": on[k]} for k, v in cells.items()},
         "start": c["s"], "base_row": a.row, "prompt": str(pid), "other_seeds": others,
         "status": S["status"], "flat": flat}, indent=1))
    print(json.dumps({"c": c, "onsets": on}, indent=1), "\nwrote", out / "figure1.html")


AX0, AX1 = -0.02, 1.0


def strip(P, others, T):
    """switch as a share of its limit over all test thoughts: one row per run, 95% interval, dashed take-off bar at 30%"""
    W, L0, L1 = 1120, 430, 1090

    def sx(v):
        return L0 + (L1 - L0) * (min(max(v, AX0), AX1) - AX0) / (AX1 - AX0)
    rows = [("fo", T["strip"]["fo"], "#2a9d92", P["fo"]), ("lp", T["strip"]["lp"], "#55585f", P["lp"]),
            ("lo", T["strip"]["lo"], "#c8478a", P["lo"])]
    RH, y0 = 50, 46
    H = y0 + RH * len(rows) + 40
    g = []
    for v in (0.0, 0.5, 1.0):
        g.append(f'<line x1="{sx(v):.1f}" y1="{y0 - 18}" x2="{sx(v):.1f}" y2="{y0 + RH * len(rows) - 18}" stroke="{"#9a9ca3" if v == 0 else "#e3e4e8"}" stroke-width="{2 if v == 0 else 1.5}"/>')
        g.append(f'<text x="{sx(v):.1f}" y="{y0 + RH * len(rows) + 8}" text-anchor="middle" class="tk">{pc(v)}</text>')
    g.append(f'<line x1="{sx(0.3):.1f}" y1="{y0 - 26}" x2="{sx(0.3):.1f}" y2="{y0 + RH * len(rows) - 18}" stroke="#1d1d1f" stroke-width="2.5" stroke-dasharray="6,5"/>')
    g.append(f'<text x="{sx(0.3) + 8:.1f}" y="{y0 - 14}" class="tb">{T["strip"]["bar"]}</text>')
    for i, (k, lab, col, v) in enumerate(rows):
        y = y0 + RH * i + 8
        g.append(f'<text x="{L0 - 20}" y="{y + 7}" text-anchor="end" class="rl{" me" if k == "lo" else ""}">{lab}</text>')
        lo, hi = v["ci"]
        g.append(f'<line x1="{sx(lo):.1f}" y1="{y}" x2="{sx(hi):.1f}" y2="{y}" stroke="{col}" stroke-width="9" stroke-linecap="round" opacity="0.35"/>')
        g.append(f'<circle cx="{sx(v["v"]):.1f}" cy="{y}" r="11" fill="{col}"/>')
        val = T["strip"]["val"][k]
        if v["v"] > 0.6:
            g.append(f'<text x="{sx(v["v"]) - 18:.1f}" y="{y + 7}" text-anchor="end" class="rv" fill="{col}">{val}</text>')
        else:
            g.append(f'<text x="{sx(v["v"]) + 20:.1f}" y="{y + 7}" class="rv" fill="{col}">{val}</text>')
    for o in others:
        y = y0 + RH * 2 + 8
        g.append(f'<circle cx="{sx(o["s_P"]):.1f}" cy="{y}" r="8" fill="#fbfaf8" stroke="#c8478a" stroke-width="3"/>')
    svg = f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" style="overflow:visible">{"".join(g)}</svg>'
    return f'<div class="stt" style="margin-top:22px">{T["strip"]["title"]}</div>{svg}'


ROBOT = '<svg width="40" height="40" viewBox="0 0 40 40" aria-hidden="true"><use href="#robot"/></svg>'
DEFS = ('<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>'
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
Generated by kran-research projects/live-kl-off/figure1/gen.py (branch swarm/oracle-1117-live-kl-off).
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
  .head .g {{ color: #6b6e77; }}
  .kick {{ font-size: 25px; font-weight: 700; color: #4d5059; margin-bottom: 14px; letter-spacing: 0; line-height: 1.32; }}
  .kick .ans {{ color: #1d1d1f; font-weight: 800; }}
  .sub {{ font-size: 24px; color: #4d5059; margin-top: 14px; line-height: 1.34; }}
  .sub b {{ color: #1d1d1f; font-weight: 800; }}
  .stt {{ font-size: 26px; font-weight: 800; line-height: 1.2; margin-top: 22px; }}
  .bubble {{ background: #3478f6; color: #fff; font-size: 20px; line-height: 1.34; padding: 12px 20px; border-radius: 22px 22px 6px 22px; margin-top: 12px; }}
  .th {{ background: #fff; border-radius: 16px 16px 16px 4px; padding: 11px 16px; font-size: 21px; line-height: 1.38; color: #33363d;
         box-shadow: 0 1px 6px rgba(0,0,0,0.08); margin-top: 12px; width: 82%; }}
  .th .tn {{ display: block; font-size: 19px; color: #6b6e77; margin-top: 4px; font-weight: 700; line-height: 1.3; }}
  .hc {{ background: #bfe8e3; color: #0d4d47; font-weight: 800; border-radius: 4px; padding: 0 3px; }}
  .key {{ font-size: 20px; color: #4d5059; margin-top: 10px; line-height: 1.32; }}
  .key b {{ color: #1d1d1f; }}
  .cards {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 16px; }}
  .card {{ border-radius: 18px; padding: 14px 16px 16px; background: #f1f1f3; }}
  .card.took {{ background: #e3f3f1; }}
  .card.me {{ background: #fff; box-shadow: inset 0 0 0 3px #c8478a; }}
  .ch {{ display: flex; align-items: center; gap: 10px; }}
  .cn {{ font-size: 23px; font-weight: 800; color: #4d5059; line-height: 1.15; }}
  .card.took .cn {{ color: #1f8278; }}
  .card.me .cn {{ color: #b23a78; }}
  .ct {{ font-size: 18px; color: #6b6e77; font-weight: 600; line-height: 1.25; }}
  .big {{ font-size: 56px; font-weight: 800; line-height: 1.05; font-variant-numeric: tabular-nums; color: #55585f; margin-top: 10px; }}
  .card.took .big {{ color: #1f8278; }}
  .card.me .big {{ color: #b23a78; }}
  .bl {{ font-size: 19px; color: #4d5059; line-height: 1.25; font-weight: 700; }}
  .badge {{ display: inline-block; margin-top: 8px; font-size: 19px; font-weight: 800; color: #fff; background: #c8478a; border-radius: 8px; padding: 3px 10px; }}
  .card.took .badge {{ background: #2a9d92; }}
  .card.old .badge {{ background: #8a8d96; }}
  .ex {{ font-size: 19px; color: #6b6e77; line-height: 1.32; margin-top: 10px; }}
  .ex b {{ color: #33363d; }}
  svg text {{ font-family: "Inter", system-ui, sans-serif; }}
  .tk {{ font-size: 19px; fill: #7a7d86; font-variant-numeric: tabular-nums; }}
  .tb {{ font-size: 19px; fill: #1d1d1f; font-weight: 700; }}
  .rl {{ font-size: 21px; fill: #4d5059; font-weight: 700; }}
  .rl.me {{ fill: #b23a78; font-weight: 800; }}
  .rv {{ font-size: 21px; font-weight: 800; font-variant-numeric: tabular-nums; }}
  .foot {{ font-size: 19px; color: #7a7d86; margin-top: 10px; display: flex; justify-content: space-between; align-items: flex-end; gap: 20px; line-height: 1.35; }}
  .foot b {{ color: #1d1d1f; white-space: nowrap; }}
</style>
</head>
<body>
<div class="head">{head}</div>
<div class="sub">{sub}</div>
<div class="stt">{ex_title}</div>
<div class="bubble">{prompt}</div>
<div class="th">Thought: {thought}<span class="tn">{th_note}</span></div>
<div class="key">{key}</div>
<div class="cards">{cards}</div>
{strip}
<div class="foot"><span>{foot}</span><b>blog.kran.ai/lab</b></div>
{defs}
</body>
</html>
"""

if __name__ == "__main__":
    main()
