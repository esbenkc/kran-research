"""Figure 1 generator for llc-bias-law (Owain). Text in text.json; every number is read from the project's files
and asserted. Example pick frozen in README.md (blind (48,48,8) r2, nbeta 1000, c 0.03, q 100/10/1/0.1).

  python3 gen.py [--out FILE] [--case CASE] [--text FILE]
--case is for layout tests only (e.g. the train shape 32x32x16r1 before blind data exist); the shipped figure uses
the frozen case. Writes figure1.html, prints the source values, writes alt.txt (YAML-safe figure1_alt).
"""
import argparse
import html
import json
import statistics as st
from pathlib import Path

H = Path(__file__).resolve().parent
P = H.parent                                   # projects/llc-bias-law
ROOT = P.parents[1]                            # kran-research
BLOG = ROOT.parent / "athanorscience.com"
OUT = BLOG / "assets/figures/llc-bias-law-fig1/figure1.html"
CASE, NB, C_EX, QS = "48x48x8r2", 1000, 0.03, (100, 10, 1, 0.1)

ap = argparse.ArgumentParser()
ap.add_argument("--text", default=str(H / "text.json"))
ap.add_argument("--out", default=str(OUT))
ap.add_argument("--case", default=CASE)
ap.add_argument("--partial", action="store_true", help="layout test: skip blind shapes whose cells are not written yet")
args = ap.parse_args()
T = json.loads(Path(args.text).read_text())
e = html.escape

# ---- data -----------------------------------------------------------------------------------------------------------
PR = json.loads((P / "predictions.json").read_text())
assert PR["written_before_heldout"] and PR["prereg_commit"] == "84d8b1861"
TRAIN = {"32x32x16r1", "32x32x16r8"}


def skey(nb, q, c):
    return f"nb{nb}_q{q:g}_c{c:g}"


def cells_file(case):
    return json.loads((P / "data/cells_S" / f"{case}.json").read_text())


def pred(case, nb, q, c):
    if case in TRAIN:
        x = next(t for t in PR["train_fit"] if t["case"] == case and t["nbeta"] == nb and t["q"] == q and t["c"] == c)
        return x["e_L"]
    return PR["cells"][f"{case}|{skey(nb, q, c)}"]["e_L"]


def measured(rows, nb, q, c):
    rs = sorted((r for r in rows if r["nbeta"] == nb and r["q"] == q and r["c"] == c), key=lambda r: r["seed"])
    assert [r["seed"] for r in rs] == [0, 1, 2], [r["seed"] for r in rs]
    ok = all(r["halves_ok"] and r["alive"] for r in rs)
    return [r["e"] for r in rs], ok


ex = cells_file(args.case)
lam = ex["rows"][0]["lam"]
lam0 = ex["rows"][0]["lam0"]
M, N, Hh, r = (ex["rows"][0][k] for k in ("M", "N", "H", "r"))
d_reg = r * (M + N - r)
assert ex["rows"][0]["d_reg"] == d_reg and abs(lam - (d_reg / 2 + lam0)) < 1e-9, (lam, d_reg, lam0)
ex_rows = []
for q in QS:
    es, ok = measured(ex["rows"], NB, q, C_EX)
    gamma = next(x["gamma"] for x in ex["rows"] if x["nbeta"] == NB and x["q"] == q)
    ex_rows.append({"q": q, "gamma": gamma, "e_pred": pred(args.case, NB, q, C_EX), "e_seeds": es, "ok": ok,
                    "e_N4": PR["cells"][f"{args.case}|{skey(NB, q, C_EX)}"]["e_N4"]})

# ---- frozen score 2(6) on the 180 primary blind cells: re-derived here from raw cells + predictions, checked
#      against results.json (score.py) --------------------------------------------------------------------------------
BLIND = ["24x24x12r3", "24x24x12r6", "48x48x8r2", "20x20x10r5", "40x24x12r4"]
LAWS = ["L", "N4", "N2", "N3", "N1"]
miss = {k: [] for k in LAWS}
uns_all = 0
for case in BLIND:
    rows = cells_file(case)["rows"]
    for nb in (100, 1000, 10000):
        for q in (0.1, 1, 10, 100):
            for c in (0.01, 0.03, 0.1):
                es, ok = measured(rows, nb, q, c)
                uns_all += not ok
                pc_ = PR["cells"][f"{case}|{skey(nb, q, c)}"]
                assert pc_["primary"]
                for k in LAWS:
                    miss[k].append(100 * abs(pc_[f"e_{k}"] - st.mean(es)) if ok else 100.0)
n_all = len(miss["L"])
assert n_all == 180, n_all
med = {k: st.median(v) for k, v in miss.items()}
win = {k: sum(m <= 5 for m in v) for k, v in miss.items()}
R = json.loads((P / "results.json").read_text())["primary"]
assert R["status"] == "UNCONFIRMED" and not R["skill_vs_all_nulls"], R["status"]
assert abs(R["L"]["median_miss"] - med["L"]) < 1e-9 and abs(R["L"]["share_within_5"] - win["L"] / 180) < 1e-12
for k in ("N1", "N2", "N3", "N4"):
    assert abs(R["nulls"][k]["median_miss"] - med[k]) < 1e-9, k
    assert abs(R["nulls"][k]["share_within_5"] - win[k] / 180) < 1e-12, k
assert R["n_not_converged"] == uns_all
best_null = min(med[k] for k in LAWS if k != "L")
assert best_null == med["N4"]
need = 0.5 * best_null
win_all = win["L"]

src = {"case": args.case, "lam": lam, "lam0": lam0, "d_reg": d_reg, "nbeta": NB, "c": C_EX, "example": ex_rows,
       "median_miss": med, "within5": win, "n_all": n_all, "unsettled_all": uns_all, "status": R["status"]}

fill = {"lam": f"{lam:g}", "reg": f"{d_reg / 2:g}", "lam0": f"{lam0:g}", "M": M, "N": N, "H": Hh, "r": r,
        "nb": f"{NB:,}", "n_all": n_all, "win_all": win_all, "pct_all": round(100 * win_all / n_all),
        "uns_all": uns_all, "med_L": f"{med['L']:.1f}", "med_N4": f"{med['N4']:.1f}", "need": f"{need:.1f}",
        "win_N4": win["N4"], "med_N2": f"{med['N2']:.1f}"}


def F(s):
    return s.format(**fill)


# ---- example chart (SVG) ---------------------------------------------------------------------------------------------
INK, GREY, MUTE, BLUE, RED = "#1d1d1f", "#6b6f78", "#9aa0aa", "#2f5fb3", "#c8372d"
W = 1120
LX = 250                     # label column
X0, X1 = LX + 30, 790        # plot range; right columns hold the numbers
COLS = (X1 + 18, X1 + 112, X1 + 208)
vals = [lam * (1 + x[k]) for x in ex_rows for k in ("e_pred", "e_N4")] + [lam * (1 + v) for x in ex_rows for v in x["e_seeds"]]
XMIN = 25 * int((min(vals) - 12) // 25)
XMAX = 25 * (int((lam + 5) // 25) + 1)


def pc(v):
    """percent with a real minus sign; -0 prints as 0"""
    k = round(v)
    return f"{k}%" if k == 0 else (f"+{k}%" if k > 0 else f"\u2212{-k}%")


def xs(v):
    assert XMIN <= v <= XMAX, (v, XMIN, XMAX)
    return X0 + (X1 - X0) * (v - XMIN) / (XMAX - XMIN)


RH, TOP = 78, 70
Hc = TOP + RH * len(ex_rows)
o = [f'<svg width="{W}" height="{Hc + 52}" viewBox="0 0 {W} {Hc + 52}" aria-hidden="true">']
for v in range(XMIN, XMAX + 1, 25):
    o.append(f'<line x1="{xs(v):.1f}" y1="{TOP - 10}" x2="{xs(v):.1f}" y2="{Hc}" stroke="#e3e5e9" stroke-width="1.2"/>')
    o.append(f'<text x="{xs(v):.1f}" y="{Hc + 32}" font-size="22" fill="{GREY}" text-anchor="middle">{v}</text>')
o.append(f'<line x1="{xs(lam):.1f}" y1="{TOP - 28}" x2="{xs(lam):.1f}" y2="{Hc}" stroke="{INK}" stroke-width="2.5" stroke-dasharray="7 5"/>')
o.append(f'<text x="{xs(lam) - 10:.1f}" y="{TOP - 36}" font-size="23" font-weight="800" fill="{INK}" text-anchor="end">{e(F(T["truth_lab"]))}</text>')
for cx, key, col in zip(COLS, ("num_pred", "num_n4", "num_meas"), (BLUE, GREY, RED)):
    o.append(f'<text x="{cx}" y="{TOP - 36}" font-size="21" font-weight="700" fill="{col}">{e(T[key])}</text>')
for i, x in enumerate(ex_rows):
    cy = TOP + RH * i + RH / 2
    lab = T["row_labels"][i].format(g=f"{x['gamma']:.3g}")
    top, sub = (lab.split("|") + [""])[:2]
    o.append(f'<text x="{LX}" y="{cy + (-2 if sub else 8)}" font-size="24" font-weight="700" fill="{INK}" text-anchor="end">{e(top)}</text>')
    if sub:
        o.append(f'<text x="{LX}" y="{cy + 24}" font-size="20" fill="{GREY}" text-anchor="end">{e(sub)}</text>')
    o.append(f'<line x1="{X0}" y1="{cy:.1f}" x2="{X1}" y2="{cy:.1f}" stroke="#eef0f3" stroke-width="1"/>')
    p = lam * (1 + x["e_pred"])
    m = lam * (1 + st.mean(x["e_seeds"]))
    o.append(f'<line x1="{xs(p):.1f}" y1="{cy:.1f}" x2="{xs(lam):.1f}" y2="{cy:.1f}" stroke="{BLUE}" stroke-opacity="0.35" stroke-width="3"/>')
    for v in x["e_seeds"]:
        o.append(f'<circle cx="{xs(lam * (1 + v)):.1f}" cy="{cy:.1f}" r="9" fill="{RED}" fill-opacity="0.55"/>')
    n4 = lam * (1 + x["e_N4"])
    o.append(f'<rect x="{xs(n4) - 8:.1f}" y="{cy - 8:.1f}" width="16" height="16" transform="rotate(45 {xs(n4):.1f} {cy:.1f})" fill="{GREY}"/>')
    o.append(f'<circle cx="{xs(p):.1f}" cy="{cy:.1f}" r="17" fill="none" stroke="{BLUE}" stroke-width="4"/>')
    for cx, v, col in zip(COLS, (x["e_pred"], x["e_N4"], st.mean(x["e_seeds"])), (BLUE, GREY, RED)):
        o.append(f'<text x="{cx}" y="{cy + 8:.1f}" font-size="24" fill="{col}" font-weight="800">{pc(100 * v)}</text>')
o.append(f'<line x1="{X0}" y1="{Hc}" x2="{X1}" y2="{Hc}" stroke="#c9ccd2" stroke-width="1.5"/>')
o.append("</svg>")
chart = "".join(o)

# ---- score strip: typical miss (median, points) of the curve and of the four frozen simpler rules ----------------
SW, SL, SX0, SX1, SRH = 1120, 430, 450, 1010, 48
SMAX = 2 * int(max(med.values()) // 2 + 1)


def sx(v):
    return SX0 + (SX1 - SX0) * v / SMAX


order = sorted(LAWS, key=lambda k: med[k])
so = [f'<svg width="{SW}" height="{SRH * len(order) + 64}" viewBox="0 0 {SW} {SRH * len(order) + 64}" aria-hidden="true">']
for i, k in enumerate(order):
    y = 8 + SRH * i
    hero = k == "L"
    so.append(f'<text x="{SL}" y="{y + 27}" font-size="22" font-weight="{800 if hero else 500}" fill="{INK}" text-anchor="end">{e(T["law_lab"][k])}</text>')
    so.append(f'<rect x="{SX0}" y="{y + 6}" width="{sx(med[k]) - SX0:.1f}" height="30" fill="{BLUE if hero else "#b9bec7"}"/>')
    so.append(f'<text x="{sx(med[k]) + 10:.1f}" y="{y + 29}" font-size="22" font-weight="{800 if hero else 600}" fill="{INK}">{med[k]:.1f}</text>')
yb = 8 + SRH * len(order)
so.append(f'<line x1="{sx(need):.1f}" y1="0" x2="{sx(need):.1f}" y2="{yb + 2}" stroke="{RED}" stroke-width="3" stroke-dasharray="6 4"/>')
so.append(f'<text x="{sx(need) - 6:.1f}" y="{yb + 30}" font-size="21" font-weight="700" fill="{RED}" text-anchor="start">{e(F(T["need_lab"]))}</text>')
so.append(f'<line x1="{SX0}" y1="0" x2="{SX0}" y2="{yb + 2}" stroke="#9aa0aa" stroke-width="1.5"/>')
so.append("</svg>")
strip = "".join(so)

legend = (f'<span class="lg"><svg width="34" height="34" viewBox="0 0 34 34"><circle cx="17" cy="17" r="13" fill="none" stroke="{BLUE}" stroke-width="4"/></svg>{e(T["leg_pred"])}</span>'
          f'<span class="lg"><svg width="30" height="30" viewBox="0 0 30 30"><rect x="7" y="7" width="16" height="16" transform="rotate(45 15 15)" fill="{GREY}"/></svg>{e(T["leg_n4"])}</span>'
          f'<span class="lg"><svg width="34" height="34" viewBox="0 0 34 34"><circle cx="11" cy="17" r="8" fill="{RED}" fill-opacity="0.55"/><circle cx="23" cy="17" r="8" fill="{RED}" fill-opacity="0.55"/></svg>{e(T["leg_meas"])}</span>')
# network pictogram: M inputs -> H hidden -> N outputs (drawn as 6 / 3 / 6 nodes with counts)
net = ['<svg width="300" height="190" viewBox="0 0 300 190" aria-hidden="true">']
cols = [(40, 6, f"{M} in"), (150, 3, f"{Hh}"), (260, 6, f"{N} out")]
pts = []
for x, k, _ in cols:
    ys_ = [30 + i * (120 / (k - 1)) for i in range(k)] if k > 1 else [90]
    if k == 3:
        ys_ = [55, 90, 125]
    pts.append([(x, y) for y in ys_])
for a, b in ((0, 1), (1, 2)):
    for (x1, y1) in pts[a]:
        for (x2, y2) in pts[b]:
            net.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#c9ccd2" stroke-width="1.5"/>')
for col in pts:
    for (x, y) in col:
        net.append(f'<circle cx="{x}" cy="{y}" r="9" fill="#fff" stroke="{INK}" stroke-width="2.5"/>')
for x, _, lab in cols:
    net.append(f'<text x="{x}" y="180" font-size="22" font-weight="700" fill="{INK}" text-anchor="middle">{e(lab)}</text>')
net.append("</svg>")
net = "".join(net)

doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{e(F(T['head']))}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&display=swap" rel="stylesheet">
<!--
Sources (kran-research, branch swarm/theory-1125-llc-bias-law, projects/llc-bias-law), built by figure1/gen.py
(every number read from these files and asserted):
  predictions  predictions.json (law L, pushed 8c50d45be before any blind cell ran)
  measured     data/cells_S/<case>.json (arm S minibatch SGLD, 3 seeds, 4 chains each)
  truth        lam / lam0 in the same rows (research/rrr_lambda.py, Aoyagi's formula)
  example pick figure1/README.md (frozen c9b7826b5 before blind results)
  values       {json.dumps(src)}
-->
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: #fbfaf8; }}
  body {{ width: 1200px; padding: 32px 40px 30px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
  .head {{ font-size: 40px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.15; }}
  .q {{ font-size: 24px; font-weight: 800; margin-top: 16px; line-height: 1.32; }}
  .gloss {{ font-size: 21px; color: #6b6f78; margin-top: 4px; line-height: 1.3; }}
  .ans {{ font-size: 24px; color: #3a3d45; margin-top: 10px; line-height: 1.34; }}
  .ans b {{ color: {RED}; font-weight: 800; }}
  .ex {{ display: flex; gap: 26px; align-items: center; margin-top: 26px; }}
  .extxt {{ font-size: 24px; line-height: 1.36; color: #3a3d45; }}
  .extxt b {{ color: #1d1d1f; }}
  .tag {{ display: inline-block; background: #ffe58a; color: #1d1d1f; font-size: 20px; font-weight: 800; padding: 1px 10px; border-radius: 8px; margin-bottom: 6px; }}
  .split {{ display: flex; height: 40px; margin-top: 10px; border-radius: 8px; overflow: hidden; font-size: 20px; font-weight: 700; color: #fff; }}
  .split div {{ display: flex; align-items: center; padding-left: 10px; white-space: nowrap; }}
  .panel {{ margin-top: 20px; border-radius: 20px; padding: 18px 20px 6px; background: #f1f2f4; }}
  .ptitle {{ font-size: 28px; font-weight: 800; color: #1d1d1f; line-height: 1.25; }}
  .psub {{ font-size: 22px; color: #4a4d55; margin-top: 4px; line-height: 1.3; }}
  .legend {{ display: flex; gap: 30px; margin-top: 10px; font-size: 22px; color: #3a3d45; }}
  .lg {{ display: inline-flex; align-items: center; gap: 10px; }}
  .axlab {{ font-size: 22px; color: #6b6f78; text-align: center; margin-top: -8px; margin-left: 10px; margin-right: 270px; }}
  .note {{ font-size: 22px; color: #3a3d45; margin: 6px 0 8px; line-height: 1.3; }}
  .foot {{ font-size: 20px; color: #8a8f99; margin-top: 14px; line-height: 1.3; }}
</style>
</head>
<body>
<div class="head">{e(F(T['head']))}</div>
<div class="q">{e(F(T['q']))}</div>
<div class="gloss">{e(F(T['gloss']))}</div>
<div class="ans">{F(T['answer'])}</div>
<div class="ex">{net}<div class="extxt"><span class="tag">{e(F(T['tag']))}</span><br>{F(T['ex_text'])}
<div class="split" style="width:{720 * lam / lam:.0f}px"><div style="width:{100 * d_reg / 2 / lam:.1f}%;background:#6b6f78">{e(F(T['reg_lab']))}</div><div style="width:{100 * lam0 / lam:.1f}%;background:{BLUE}">{e(F(T['sing_lab']))}</div></div></div></div>
<div class="panel"><div class="ptitle">{e(F(T['chart_t']))}</div><div class="psub">{F(T['chart_sub'])}</div>
<div class="legend">{legend}</div>
{chart}
<div class="axlab">{e(T['axis'])}</div></div>
<div class="panel"><div class="ptitle">{e(F(T['score_t']))}</div><div class="psub">{F(T['score_sub'])}</div>
{strip}</div>
<div class="foot">{e(F(T['foot']))}</div>
</body>
</html>
"""
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(doc)
print(json.dumps(src))
print(args.out)

# ---- figure1_alt (YAML-safe: no inner double quotes) ---------------------------------------------------------------
rows_txt = "; ".join(
    f"spring {x['gamma']:.3g}: curve {pc(100 * x['e_pred'])}, lookup {pc(100 * x['e_N4'])}, "
    f"measured {pc(100 * st.mean(x['e_seeds']))}" for x in ex_rows)
meds_txt = "; ".join(f"{T['law_lab'][k]} {med[k]:.1f}" for k in order)
alt = (f"Figure 1. Head: {F(T['head'])}. {F(T['q'])} {F(T['alt_answer'])} Example: a network shape the curve had "
       f"not seen, {M} inputs, {Hh} hidden units, {N} outputs, rank {r}; its true learning coefficient {lam:g} splits "
       f"into a bowl part {d_reg / 2:g} (exact formula) and a singular part {lam0:g} (the curve). At inverse "
       f"temperature {NB:,} and four spring strengths, the error the curve predicted before the run, the lookup's "
       f"error (the measured error of the same setting on a training shape) and the measured error: {rows_txt}. Below, "
       f"the typical miss in points over all {n_all} settings on 5 unseen shapes: {meds_txt}; the test needed the "
       f"curve at {need:.1f} or less, half the lookup's miss.")
alt = alt.replace('"', "'").replace("\u201c", "'").replace("\u201d", "'")
(H / "alt.txt").write_text(alt + "\n")
