"""Figure 1 generator (Owain) for live-kl-off, 3-seed version (room oracle-1118-live-kl-off-cont).

ONE stored TEST thought (data/pool/test, the ruler every run is read on), with ' We' written over its first token:
base row 2140, fixed on Nov 17 before seeds 1 and 2 existed (pick rule posted in the room, #25269).
Three cards, one per training seed of the no-penalty run, each at its last read (seed 0: 1,200; seeds 1 and 2: the
read where the frozen onset rule stopped them). Each card: the planted word's switch over all 2,800 test thoughts as a
share of its limit (recomputed from the raw reads with live_summ.Test and asserted equal to data/summary.json), its
outcome and onset from summary.json, and the exact cheat chance after this We-thought and without We.
Context line: the parent's fresh-thought run with the penalty (live_k4, 1,200) and fixed thoughts (cached_d010_s0).

    <env python> gen.py --proj <kran-research>/projects/live-kl-off --blog <blog clone> [--row 2140]
Text lives in text.json next to this file; @@key@@ is replaced from the flat dict (print it with --keys).
"""
import argparse
import html
import json
import re
import sys
from pathlib import Path

import numpy as np

SLUG = "live-kl-off"
X = 1205


def pc(v, d=0):
    s = f"{100 * v:.{d}f}"
    if s.startswith("-") and float(s) == 0:
        s = s[1:]
    return s.replace("-", "&minus;") + "%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", required=True)
    ap.add_argument("--blog", required=True)
    ap.add_argument("--row", type=int, default=2140, help="test-pool base row")
    ap.add_argument("--text", default=str(Path(__file__).parent / "text.json"))
    ap.add_argument("--keys", action="store_true")
    ap.add_argument("--out-id", default=None, help="write to assets/figures/<id>/ instead (A/B variants)")
    a = ap.parse_args()
    HERE = Path(a.proj).resolve()
    D = HERE / "data"
    OUT = Path(a.blog) / "assets" / "figures" / f"{SLUG}-fig1"
    sys.path.insert(0, str(HERE))
    from live_summ import Test  # noqa: E402
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
    assert sorted(seeds) == ["kl0_s0", "kl0_s1", "kl0_s2"], sorted(seeds)
    c0 = np.load(D / "parent" / "live_k4" / "reads" / "c_0.npy")
    cells = {f"s{i}": (f"runs/kl0_s{i}", seeds[f"kl0_s{i}"]["last_read"]) for i in range(3)}
    cells["lp"] = ("parent/live_k4", 1200)
    cells["fo"] = ("parent/cached_d010_s0", 1200)
    c, st, on, outc = {}, {}, {}, {}
    for k, (rd, s) in cells.items():
        arr = np.load(D / rd / "reads" / f"c_{s}.npy")
        c[k] = {"none": float(arr[rn]), "x": float(arr[rx])}
        st[k] = te.point_ci(arr, c0)
        if rd.startswith("parent"):
            P = S["parent"][rd.split("/")[1]]
            ref, on[k], outc[k] = P["reads"], P["onset"], ("TAKE-OFF" if P["onset"] else "FLAT")
        else:
            P = seeds[rd.split("/")[1]]
            ref, on[k], outc[k] = P["reads"], P["onset"], P["outcome"]
            assert np.allclose(np.load(D / rd / "reads" / "c_0.npy"), c0, atol=1e-6), ("same start", k)
            assert abs(P["last"]["s_P"]["v"] - st[k]["s_P"]["v"]) < 1e-9
        r = [x for x in ref if x["step"] == s][0]
        assert abs(r["s_P"] - st[k]["s_P"]["v"]) < 1e-9 and abs(r["s_R"] - st[k]["s_R"]["v"]) < 1e-9, (k, r, st[k]["s_P"])
    assert outc == {"s0": "FLAT", "s1": "TAKE-OFF", "s2": "TAKE-OFF", "lp": "FLAT", "fo": "TAKE-OFF"}, outc
    assert on["fo"] == [500, 600] and S["status"] == "CONFIRMED" and S["n_takeoff"] == 2 and S["n_settled"] == 3, (on, S["status"])
    c["start"] = {"none": float(c0[rn]), "x": float(c0[rx])}
    prs = [json.loads(ln) for ln in open(HERE / "inputs" / "prompts.jsonl")]
    pid = cols["prompt_list"][cols["prompt_row"][rn]]
    pr = [p for p in prs if p["id"] == pid][0]
    assert pr["split"] == "test"
    ptxt = pr["text"].split("user\n", 1)[1].split("<|im_end|>", 1)[0]
    SL = S["seed_level"]
    flat = {"n_rows_s": f'{te.meta["n_base"]:,}', "start.x": pc(c["start"]["x"]), "start.none": pc(c["start"]["none"]),
            "status": S["status"], "n_takeoff": str(S["n_takeoff"]), "n_settled": str(S["n_settled"]),
            "sl.read": f'{SL["read"]:,}', "sl.mean": pc(SL["mean"]), "sl.lo": pc(SL["ci"][0]), "sl.hi": pc(SL["ci"][1])}
    for k, (rd, s) in cells.items():
        flat[f"{k}.step"] = f"{s:,}"
        flat[f"{k}.x"] = pc(c[k]["x"])
        flat[f"{k}.none"] = pc(c[k]["none"])
        v = st[k]["s_P"]["v"]
        flat[f"{k}.P"] = pc(v, 1 if abs(v) < 0.1 else 0)
        flat[f"{k}.Pci"] = f'{pc(st[k]["s_P"]["ci"][0], 1)} to {pc(st[k]["s_P"]["ci"][1], 1)}'
        flat[f"{k}.on"] = f"{on[k][1]:,}" if on[k] else "none"
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
                f'<div class="big">{flat[k + ".P"]}</div><div class="bl">{T["big_label"].replace("@K@", flat[k + ".step"])}</div>{badge}'
                f'{spark(k, t["cls"]) if T.get("spark") else ""}'
                f'<div class="ex">{T["ex_label"]}<br><b>{flat[k + ".x"]}</b> {T["x_label"]}, <b>{flat[k + ".none"]}</b> {T["none_label"]}</div></div>')

    def spark(k, cls):
        """switch at every test read of this run, 0 to 1,200 updates, 0 to 100%; dashed take-off line at 30%"""
        rd, _ = cells[k]
        reads = (S["parent"][rd.split("/")[1]] if rd.startswith("parent") else seeds[rd.split("/")[1]])["reads"]
        W, H, L, R, TP, B = 330, 92, 4, 326, 8, 70
        col = "#2a9d92" if cls == "took" else "#8a8d96"
        def px(u): return L + (R - L) * u / 1200
        def py(v): return B - (B - TP) * max(0.0, min(1.0, v))
        pts = " ".join(f"{px(r['step']):.1f},{py(r['s_P']):.1f}" for r in reads)
        last = reads[-1]
        g = [f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="#c9cbd1" stroke-width="1.5"/>',
             f'<line x1="{L}" y1="{py(0.3):.1f}" x2="{R}" y2="{py(0.3):.1f}" stroke="#1d1d1f" stroke-width="1.6" stroke-dasharray="5,4"/>',
             f'<text x="{R}" y="{py(0.3) - 6:.1f}" text-anchor="end" class="sk">take-off 30%</text>',
             f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>',
             f'<circle cx="{px(last["step"]):.1f}" cy="{py(last["s_P"]):.1f}" r="5.5" fill="{col}"/>',
             f'<text x="{L}" y="{B + 20}" class="sk">0</text>',
             f'<text x="{R}" y="{B + 20}" text-anchor="end" class="sk">update 1,200</text>']
        return f'<svg class="spark" width="{W}" height="{H}" viewBox="0 0 {W} {H}">{"".join(g)}</svg>'

    page = TEMPLATE.format(
        title=html.escape(html.unescape(re.sub("<[^>]+>", "", T["head"]))), head=T["head"], sub=T["sub"],
        ex_title=T["ex_title"], prompt=html.escape(ptxt), thought=thought(tx, True, T.get("upto", 40)),
        th_note=T["th_note"], key=T["key"], cards="".join(card(k) for k in T["card_order"]),
        ctx=T.get("ctx", ""), foot=T["foot"], defs=DEFS,
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
                       "onset": on[k], "outcome": outc[k]} for k, v in cells.items()},
         "start": c["start"], "base_row": a.row, "prompt": str(pid), "status": S["status"], "flat": flat}, indent=1))
    print(json.dumps({"c": c, "onsets": on}, indent=1), "\nwrote", out / "figure1.html")


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
Generated by kran-research projects/live-kl-off/figure1/gen.py (branch swarm/oracle-1118-live-kl-off-cont).
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
  .spark {{ display: block; margin-top: 10px; overflow: visible; }}
  .sk {{ font-size: 18px; fill: #6b6e77; font-weight: 600; }}
  .ctx {{ font-size: 20px; color: #4d5059; margin-top: 14px; line-height: 1.34; }}
  .ctx b {{ color: #1d1d1f; }}
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
<div class="ctx">{ctx}</div>
<div class="foot"><span>{foot}</span><b>blog.kran.ai/lab</b></div>
{defs}
</body>
</html>
"""

if __name__ == "__main__":
    main()
