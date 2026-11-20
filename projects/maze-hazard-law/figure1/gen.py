"""Owain: build Figure 1 (assets/figures/maze-hazard-law-fig1/figure1.html) from the real run records.

    /usr/bin/python3 gen.py <project dir> <text.json> <out.html> [--data DIR] [--analysis FILE] [--variant V]
                            [--agent NAME] [--run I]

Every number it draws comes from analysis.json (Iris's frozen analysis.py output) and the per-run rows in it; the
per-page leave counts are recomputed from those rows with the PREREG s.3 decision rule and asserted equal to the frozen
G3/E3, G/E and the Pearson bin counts. The example run is re-scored from its raw record with analysis.score_run and
asserted equal to its analysis.json row.

PICK RULE (room #25865, fixed 2026-11-18 before any maze run existed):
  agent = the first status agent in analysis.json overall.tested, else the first status agent with entered maze runs;
  run   = the lowest run index among that agent's entered maze runs whose page depth D equals the agent's KM median;
          if the KM median is not reached, the lowest-index run that ended at the 80-call cap.

Text slots holding '[FILL' make gen.py refuse to render (no ready-made answer before the status exists).
"""
import argparse
import html
import json
import math
import re
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("proj")
ap.add_argument("text")
ap.add_argument("out")
ap.add_argument("--data")
ap.add_argument("--analysis")
ap.add_argument("--variant", default="bars", choices=["bars", "stack", "surv", "skyline", "two"])
ap.add_argument("--agent", help="override the pick (dev only)")
ap.add_argument("--run", type=int, help="override the pick (dev only)")
ap.add_argument("--allow-fill", action="store_true", help="dev only: render even with [FILL] slots")
a = ap.parse_args()
proj = Path(a.proj).resolve()
data = Path(a.data).resolve() if a.data else proj / "data"
an = json.loads(Path(a.analysis or proj / "results" / "analysis.json").read_text())
sys.path.insert(0, str(proj))
sys.path.insert(0, str(proj / "web"))
import analysis  # noqa: E402
import sites  # noqa: E402

E = html.escape
T = json.loads(Path(a.text).read_text())
GIVEUP = analysis.GIVEUP
CAP = analysis.CAP

# ---- pick ------------------------------------------------------------------------------------------------------------
# Counted status rows first. When none has entered runs (tonight: D5 moved both reuse-on folders to side rows), the SAME
# rule runs over the status agents' pilot rows ('<agent> (prefix reuse on, ...; not counted)'), and PILOT is set so the
# text says 'not counted' on the figure's face (Tove #26274).
AG = {**an.get("side_row", {}), **an["agents"]}  # other side rows only via --agent (pipeline checks); never picked
order = [n for n in analysis.STATUS_ORDER if n in an["agents"]]
cands = [n for n in an["overall"]["tested"] if n in an["agents"]] or [n for n in order if (AG[n].get("n_entered") or 0) > 0]
PILOT = False
if not cands and not a.agent:
    pil = {s: n for s in analysis.STATUS_ORDER for n in an.get("side_row", {}) if n.startswith(s + " (prefix reuse on")}
    pil_order = [pil[s] for s in analysis.STATUS_ORDER if s in pil]
    cands = ([n for n in pil_order if (AG[n]["law"] or {}).get("tested")]
             or [n for n in pil_order if (AG[n].get("n_entered") or 0) > 0])
    PILOT = bool(cands)
assert cands or a.agent, "no status agent with entered maze runs"
name = a.agent or cands[0]
A = AG[name]
L = A["law"]
ent = [r for r in A["runs"]["maze"] if r["entered"]]
med = L["km_median"]
if a.run is not None:
    pick = next(r for r in ent if r["run_index"] == a.run)
elif med is not None:
    pick = min((r for r in ent if r["D"] == med), key=lambda r: r["run_index"])
else:
    pick = min((r for r in ent if r["cause"] == "cap"), key=lambda r: r["run_index"])
i = pick["run_index"]

# ---- the example run, re-scored from its raw record ------------------------------------------------------------------
d = data / A["dir"]
rec = json.loads((d / "raw" / "maze" / f"{i:03d}.json").read_text())
acc = [json.loads(x) for x in (d / "access" / "maze" / f"{i:03d}.jsonl").read_text().splitlines() if x.strip()]
acc.sort(key=lambda x: x["t"])
sc = analysis.score_run(rec, acc)
for k in ("D", "S", "cause", "k", "entered"):
    assert sc[k] == pick[k], (k, sc[k], pick[k])
D = sc["D"]


def thought(step):
    """the agent's own note for this call: the JSON 'thought' (Qwen, nano) or the harmony analysis channel (gpt-oss)."""
    try:
        t = json.loads(step["text"])["thought"]
    except Exception:
        m = re.search(r'"thought"\s*:\s*"((?:[^"\\]|\\.)*)"', step.get("text") or "")
        t = json.loads(f'"{m.group(1)}"') if m else None
    if not (isinstance(t, str) and t.strip()):
        t = step.get("reasoning")
    return t.strip() if isinstance(t, str) and t.strip() else None


# calls inside the maze episode with the page depth each call sat at (same walk as analysis.score_run)
calls, seen, entry = [], set(), None
for s in rec["steps"]:
    act = s.get("action") or {}
    p = act.get("path")
    if entry is None:
        if p == rec["site"]["canary_path"] and act.get("status") == 200:
            entry = s["step"]
            seen.add(p)
        continue
    calls.append({"step": s["step"], "depth": len(seen), "thought": thought(s), "tool": act.get("tool"), "path": p,
                  "answer": act.get("text")})
    if act.get("tool") == "answer" or (p is not None and not p.startswith("/m/")):
        break
    if p is not None and p.startswith("/m/") and act.get("status") == 200:
        seen.add(p)
assert len(seen) == D or sc["cause"] not in GIVEUP, (len(seen), D)
noted = [c for c in calls if c["thought"]]
q_first = next((c for c in noted if c["depth"] == 1), None)
q_last = noted[-1] if noted else None
final = calls[-1] if calls else None
final_answer = final["answer"] if final and final["tool"] == "answer" else None

# one maze page as the agent read it (raw HTML of part 2)
site = rec["site"]
tpl = sites.TEMPLATES[site["template"]]
show_k = 2
pg = sites.Site(site["template"], "maze", secret=site["secret"])
pid = pg.id0
for _ in range(show_k - 1):
    pid = pg._next(pid)
    pg.depth[pid] = len(pg.depth) + 1
body = pg.render(f"/m/{pid}.html")[2].decode()
main = body.split("<main>")[1].split("</main>")[0].strip()
assert "lists no figures" in main and f"part {show_k}" in main
anchor = tpl["fit_anchor"]
code = (E(main).replace("Figures are published in a later part.", '<span class="anc">Figures are published in a later part.</span>')
        .replace(E(f'{anchor} (part {show_k + 1})'), f'<span class="nx">{E(anchor)} (part {show_k + 1})</span>')
        .replace("&lt;p&gt;&lt;a", "\n&lt;p&gt;&lt;a").replace("&lt;p&gt;This", "\n&lt;p&gt;This"))


# ---- per-page leave counts (PREREG s.3: a give-up at D decided at pages 1..D; lost/censored at D decided 1..D-1) -----
def page_counts(runs):
    reads, left = {}, {}
    for r in runs:
        last = r["D"] if r["cause"] in GIVEUP else r["D"] - 1
        for p in range(1, last + 1):
            reads[p] = reads.get(p, 0) + 1
        if r["cause"] in GIVEUP:
            left[r["D"]] = left.get(r["D"], 0) + 1
    return reads, left


BINS = [(1, 1), (2, 2), (3, 3), (4, 10), (11, 30), (31, 10 ** 6)]  # pages 1-3 one by one, then the frozen Pearson bins


def bars_for(L2, runs):
    reads, left = page_counts(runs)
    assert sum(reads.get(p, 0) for p in (1, 2, 3)) == L2["E3"] and sum(left.get(p, 0) for p in (1, 2, 3)) == L2["G3"]
    assert sum(reads.values()) == L2["E"] and sum(left.values()) == L2["G"], (sum(reads.values()), L2["E"])
    out = []
    for lo, hi in BINS:
        n = sum(v for p, v in reads.items() if lo <= p <= hi)
        k = sum(v for p, v in left.items() if lo <= p <= hi)
        if n:
            top = max(p for p in reads if lo <= p <= hi)
            lab = str(lo) if lo == hi else f"{lo}-{min(hi, top)}"
            # runs that made at least one choice in this bin (a bin over several pages counts page reads, not runs)
            nr = sum(1 for r in runs if (r["D"] if r["cause"] in GIVEUP else r["D"] - 1) >= lo)
            out.append({"lab": lab, "n": n, "k": k, "lo": lo, "runs": nr, "multi": lo != hi})
    # the frozen Pearson bins (4-10, 11-30, 31+) must match analysis.json chi
    chi = L2["chi"]
    tail = [(b["k"], b["n"]) for b in out if b["lo"] >= 4]
    want = [(int(o), int(n)) for o, n in zip(chi["obs"], chi["decisions"]) if n][1:]
    assert tail == want, (tail, want)
    return out


bars = bars_for(L, ent)
h_all = L["G"] / L["E"]
assert abs(h_all - L["h_all"]) < 1e-12


def pages(x):
    return f"{x:.0f}"


def short(n):
    return re.sub(r"\s*\(.*\)$", "", n)


def pct(x):
    return f"{100 * x:.0f}%"


b_by = {b["lab"]: b for b in bars}
NUM = {"agent": short(name), "n_ent": A["n_entered"], "n_maze": len(A["runs"]["maze"]), "D": D, "run": i,
       "cap": CAP, "obs": med, "h_all": pct(h_all), "G": L["G"], "E": L["E"], "G3": L["G3"], "E3": L["E3"],
       "status": an["overall"]["status"],
       "answer": {"CONFIRMED": "Yes.", "REJECTED": "No.", "UNCONFIRMED": "Not settled."}[an["overall"]["status"]],
       "cap_n": sum(r["cause"] == "cap" for r in ent), "maxD": max(r["D"] for r in ent),
       "giveup_n": sum(r["cause"] in GIVEUP for r in ent)}
for p in ("1", "2", "3"):
    if p in b_by:
        NUM[f"p{p}_k"], NUM[f"p{p}_n"], NUM[f"p{p}_pct"] = b_by[p]["k"], b_by[p]["n"], pct(b_by[p]["k"] / b_by[p]["n"])
# forecast vs real: R = forecast median / KM median (analysis.m_pred clamps the forecast at K, so 'or more' when clamped)
NUM["R"] = (f"{L['R']:.1f}x" + (" or more" if L["pred_median"] >= L["K"] else "")) if L.get("R") else "n/a"
NUM["K"] = f"{L['K']:.0f}"
NUM["h_all1"] = f"{100 * L['h_all']:.0f}%"
NUM["fK"] = pct((1 - L["h3"]) ** L["K"]) if L.get("h3") is not None else "n/a"
NUM["late_k"] = sum(b["k"] for b in bars if b["lo"] >= 4)
NUM["late_n"] = sum(b["n"] for b in bars if b["lo"] >= 4)
NUM["late_runs"] = next((b["runs"] for b in bars if b["lo"] >= 4), 0)
# the longest run's most repeated note, counted over ALL its model calls in its raw record (Kasper #26316)
top = max(ent, key=lambda r: (r["D"], -r["run_index"]))
_ths = [thought(s) for s in json.loads((d / "raw" / "maze" / f"{top['run_index']:03d}.json").read_text())["steps"]]
NUM["top_calls"] = len(_ths)
NUM["top_same"] = max(_ths.count(t) for t in set(_ths) if t)
NUM["top_cause"] = top["cause"]


def fill(s):
    return s.format_map(NUM) if isinstance(s, str) else s


# variant keys: "<key>_pilot" replaces <key> when the figure shows pilot (not counted) rows; "<key>__<STATUS>" replaces
# <key> when analysis.json overall.status is STATUS (so no answer text exists before the status does)
for k in list(T):
    if k.endswith("_pilot"):
        if PILOT:
            T[k[:-6]] = T[k]
        del T[k]
for k in list(T):
    if "__" in k:
        base, st = k.split("__")
        if st == NUM["status"]:
            T[base] = T[k]
        del T[k]
assert a.allow_fill or not any("[FILL" in str(v) for v in T.values()), "text.json still holds [FILL] slots"
assert "never left" not in T.get("chart_note", "") or NUM["top_cause"] == "cap", NUM["top_cause"]
T = {k: (v if k.startswith("end_") or k in ("thought_label", "strip_label", "other_t", "runs_label") else fill(v)) for k, v in T.items()}

# ---- chart: chance of leaving on each page -------------------------------------------------------------------------
RED, RED_L, GREY, INK, MUTE, GRID = "#b3261e", "#f4c7c3", "#aeb2ba", "#1d1d1f", "#5a5d66", "#d9dbe0"


def bar_chart(bars, W, H, hi_lab="2", line=None, line_lab="", title=None, grey_all=False):
    ml, mr, mt, mb = 70, 10, 30, 116
    pw, ph = W - ml - mr, H - mt - mb
    nb = len(bars)
    slot = pw / nb
    bw = slot * 0.68
    ymax = 1.0

    def yv(v):
        return mt + ph * (1 - v / ymax)

    s = []
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        s.append(f'<line x1="{ml}" y1="{yv(t):.1f}" x2="{W - mr}" y2="{yv(t):.1f}" class="gl{" z" if t == 0 else ""}"/>')
        s.append(f'<text x="{ml - 10}" y="{yv(t) + 7:.1f}" text-anchor="end" class="tk">{int(t * 100)}%</text>')
    for j, b in enumerate(bars):
        x = ml + slot * j + (slot - bw) / 2
        v = b["k"] / b["n"]
        hi = (b["lab"] == hi_lab) and not grey_all
        col = RED if hi else GREY
        if v > 0:
            s.append(f'<rect x="{x:.1f}" y="{yv(v):.1f}" width="{bw:.1f}" height="{yv(0) - yv(v):.1f}" rx="3" fill="{col}"/>')
        cx = x + bw / 2
        s.append(f'<text x="{cx:.1f}" y="{yv(v) - 10:.1f}" text-anchor="middle" class="bv" fill="{RED if hi else INK}">{pct(v)}</text>')
        s.append(f'<text x="{cx:.1f}" y="{yv(0) + 30:.1f}" text-anchor="middle" class="bx{" hi" if hi else ""}">{b["lab"]}</text>')
        s.append(f'<text x="{cx:.1f}" y="{yv(0) + 56:.1f}" text-anchor="middle" class="bn">{b["k"]} of {b["n"]}</text>')
        if b["multi"]:
            s.append(f'<text x="{cx:.1f}" y="{yv(0) + 80:.1f}" text-anchor="middle" class="bn rn">'
                     f'{T["runs_label"].format(n=b["runs"], s="" if b["runs"] == 1 else "s")}</text>')
    s.append(f'<text x="{ml + pw / 2:.1f}" y="{H - 6}" text-anchor="middle" class="ax">{T["axis_x"]}</text>')
    if line is not None:
        s.append(f'<line x1="{ml}" y1="{yv(line):.1f}" x2="{W - mr}" y2="{yv(line):.1f}" class="lvl"/>')
        for j, part in enumerate(line_lab.split("|")):
            s.append(f'<text x="{W - mr}" y="{yv(line) - 12 - 24 * (len(line_lab.split("|")) - 1 - j):.1f}" text-anchor="end" class="lvt">{part}</text>')
    if title:
        s.append(f'<text x="{ml}" y="18" class="ct">{title}</text>')
    return f'<svg class="ch" viewBox="0 0 {W} {H}" width="{W}" height="{H}">{"".join(s)}</svg>'


def surv_chart(runs, L2, W, H):
    """share of runs still inside after each page: real (step) vs one constant chance from pages 1-3 (dashed, (1-h3)^p),
    with the half line, both half points and the verdict. Real = the KM curve when no run is censored (asserted)."""
    ml, mr, mt, mb = 74, 24, 44, 66
    pw, ph = W - ml - mr, H - mt - mb
    n0 = len(runs)
    K = L2["K"]
    h3 = L2["h3"]
    xmax = CAP

    def xp(p):
        return ml + pw * math.log(p) / math.log(xmax)

    def yv(f):
        return mt + ph * (1 - f)

    s = []
    # pages 1-3: the forecast's only input
    s.append(f'<rect x="{xp(1):.1f}" y="{mt - 34}" width="{xp(3) - xp(1):.1f}" height="{yv(0) - mt + 34:.1f}" fill="#ffe58a" opacity="0.55"/>')
    s.append(f'<text x="{(xp(1) + xp(3)) / 2:.1f}" y="{mt - 12}" text-anchor="middle" class="bn b">{T["surv_in"]}</text>')
    for t in (1, 3, 10, 20, 40, 80):
        s.append(f'<line x1="{xp(t):.1f}" y1="{mt}" x2="{xp(t):.1f}" y2="{yv(0):.1f}" class="gl"/>')
        s.append(f'<text x="{xp(t):.1f}" y="{yv(0) + 28:.1f}" text-anchor="middle" class="tk">{t}</text>')
    for f in (0, 0.5, 1):
        s.append(f'<text x="{ml - 10}" y="{yv(f) + 7:.1f}" text-anchor="end" class="tk">{int(f * 100)}%</text>')
    s.append(f'<line x1="{ml}" y1="{yv(0):.1f}" x2="{W - mr}" y2="{yv(0):.1f}" class="gl z"/>')
    s.append(f'<line x1="{ml}" y1="{yv(0.5):.1f}" x2="{W - mr}" y2="{yv(0.5):.1f}" class="half"/>')
    # real: runs still inside after page p (give-ups at D leave after page D; others stay to D)
    maxp = max(r["D"] for r in runs)
    inside = [n0] + [sum(1 for r in runs if r["D"] > p or (r["D"] == p and r["cause"] not in GIVEUP)) for p in range(1, maxp + 1)]
    path = f"M{xp(1):.1f},{yv(1):.1f}"
    for p in range(1, maxp + 1):
        path += f" L{xp(p):.1f},{yv(inside[p - 1] / n0):.1f} L{xp(p):.1f},{yv(inside[p] / n0):.1f}"
    s.append(f'<path d="{path}" fill="none" stroke="{RED}" stroke-width="5" stroke-linejoin="round"/>')
    med = next(p for p in range(1, maxp + 1) if inside[p] / n0 <= 0.5)
    assert med == L2["km_median"], (med, L2["km_median"])
    # forecast: one chance for every page, read from pages 1-3 only; drawn to the median page a run could reach (K)
    fK = (1 - h3) ** K
    fp = " ".join(f"{'M' if p == 1 else 'L'}{xp(p):.1f},{yv((1 - h3) ** p):.1f}" for p in range(1, int(K) + 1))
    s.append(f'<path d="{fp}" fill="none" stroke="{INK}" stroke-width="3.5" stroke-dasharray="10 7"/>')
    s.append(f'<circle cx="{xp(K):.1f}" cy="{yv(fK):.1f}" r="7" fill="{INK}"/>')
    for j, part in enumerate(T["surv_fore"].split("|")):
        s.append(f'<text x="{W - mr}" y="{yv(fK) + 66 + 26 * j:.1f}" text-anchor="end" class="{"lvt" if j == 0 else "lvs"}">{part}</text>')
    # best single chance for all pages (h_all, the constant the frozen shape test compares against)
    ha = L2["h_all"]
    fa = " ".join(f"{'M' if p == 1 else 'L'}{xp(p):.1f},{yv((1 - ha) ** p):.1f}" for p in range(1, int(K) + 1))
    s.append(f'<path d="{fa}" fill="none" stroke="{MUTE}" stroke-width="3" stroke-dasharray="2 6" stroke-linecap="round"/>')
    # its label sits in the empty lower-left corner, under the dotted line, with a dotted sample in front
    s.append(f'<line x1="{ml + 12}" y1="{yv(0.12) - 7:.1f}" x2="{ml + 44}" y2="{yv(0.12) - 7:.1f}" stroke="{MUTE}" stroke-width="3" stroke-dasharray="2 6" stroke-linecap="round"/>')
    for j, part in enumerate(T["surv_all"].split("|")):
        s.append(f'<text x="{ml + 52}" y="{yv(0.12) + 26 * j:.1f}" class="{"lvt m" if j == 0 else "lvs"}">{part}</text>')
    # real half point (label left of the dot, in the empty area under the forecast)
    s.append(f'<circle cx="{xp(med):.1f}" cy="{yv(0.5):.1f}" r="8" fill="{RED}"/>')
    rl = T["surv_real"].split("|")
    for j, part in enumerate(rl):
        s.append(f'<text x="{xp(med) + 16:.1f}" y="{yv(0.5) - 16 - 28 * (len(rl) - 1 - j):.1f}" class="bv" fill="{RED}">{part}</text>')
    s.append(f'<text x="{ml + pw / 2:.1f}" y="{H - 6}" text-anchor="middle" class="ax">{T["axis_x_log"]}</text>')
    return f'<svg class="ch" viewBox="0 0 {W} {H}" width="{W}" height="{H}">{"".join(s)}</svg>'


def skyline(runs, W, H):
    """one column per run, one square per maze page read, sorted by pages read."""
    rs = sorted(runs, key=lambda r: (-r["D"], r["run_index"]))
    n = len(rs)
    gap = 3
    cw = (W - gap * (n - 1)) / n
    maxd = max(r["D"] for r in rs)
    sq = min(cw, (H - 40) / maxd)
    s = []
    for j, r in enumerate(rs):
        x = j * (cw + gap)
        col = RED if r["cause"] == "cap" else (INK if r["run_index"] == i else GREY)
        s.append(f'<rect x="{x:.1f}" y="{H - 30 - r["D"] * sq:.1f}" width="{cw:.1f}" height="{r["D"] * sq:.1f}" rx="2" fill="{col}"/>')
    s.append(f'<text x="{cw + 12:.1f}" y="{H - 30 - maxd * sq + 22:.1f}" class="bv" fill="{RED}">{T["sky_tall"]}</text>')
    s.append(f'<text x="{W / 2:.1f}" y="{H - 4}" text-anchor="middle" class="ax">{T["sky_axis"]}</text>')
    return f'<svg class="ch" viewBox="0 0 {W} {H}" width="{W}" height="{H}">{"".join(s)}</svg>'


# ---- the example run ------------------------------------------------------------------------------------------------
ans_key = ("end_answer_invented" if pick.get("invented_figure") else "end_answer_stale" if pick.get("stale_figure")
           else "end_answer")
end_txt = {"giveup_answer": T[ans_key], "giveup_get": T["end_get"], "cap": T["end_cap"],
           "lost": T["end_lost"], "overflow": T["end_overflow"]}[sc["cause"]].format(D=D, p="page" if D == 1 else "pages")

ROBOT = ('<svg class="bot" viewBox="0 0 40 40" width="46" height="46"><line x1="20" y1="3" x2="20" y2="9" stroke="#3b3e46" stroke-width="2"/>'
         '<circle cx="20" cy="3" r="2.6" fill="#e8743b"/><rect x="5" y="9" width="30" height="26" rx="8" fill="#fff" stroke="#3b3e46" stroke-width="2"/>'
         '<rect x="9" y="14" width="22" height="14" rx="5" fill="#25324a"/><circle cx="15" cy="21" r="3.2" fill="#7fd3ff"/><circle cx="25" cy="21" r="3.2" fill="#7fd3ff"/>'
         '<rect x="1.5" y="17" width="4" height="9" rx="2" fill="#e8743b"/><rect x="34.5" y="17" width="4" height="9" rx="2" fill="#e8743b"/></svg>')


def note(c, label, extra=""):
    return (f'<div class="nrow">{ROBOT}<div class="bub"><span class="thk">{label}</span>&ldquo;{E(c["thought"])}&rdquo;{extra}</div></div>'
            if c else "")


notes = note(q_first, T["thought_label"].format(k=1))
if q_last and q_last is not q_first:
    extra = (f'<span class="ansl">{T["answer_label"]}</span>&ldquo;{E(final_answer)}&rdquo;' if final_answer and q_last is final else "")
    notes += note(q_last, T["thought_label"].format(k=q_last["depth"]), extra)
squares = "".join(f'<div class="sq{" in" if j < 3 and a.variant == "surv" else ""}"></div>' for j in range(D)) if D <= 12 else ""
strip = (f'<div class="strip">{squares}<span class="sql">{T["strip_label"].format(D=D, p="page" if D == 1 else "pages")}</span>'
         f'<span class="endl">&rarr; {end_txt}</span></div>')

example = f"""
<div class="task"><span class="ask">&ldquo;{E(rec["question"])}&rdquo;</span></div>
<div class="start">{T["start"]}</div>
<div class="code">{code}</div>
<div class="lane">
  <div class="ln">{T["ex_name"]}</div>
  {notes}
  {strip}
</div>"""

hi_lab = max(bars, key=lambda b: b["k"] / b["n"])["lab"]
v = a.variant
if v in ("bars", "two"):
    chart = bar_chart(bars, 580, 474, hi_lab=hi_lab, line=h_all, line_lab=T["level"])
elif v == "stack":
    chart = bar_chart(bars, 1120, 330, hi_lab=hi_lab, line=h_all, line_lab=T["level"])
elif v == "surv":
    chart = surv_chart(ent, L, 580, 470)
else:
    chart = skyline(ent, 580, 450)
chart_t = {"bars": T["chart_t"], "two": T["chart_t"], "stack": T["chart_t"], "surv": T["surv_t"], "skyline": T["sky_t"]}[v]
ct_parts = chart_t.split("|")
chart_t = ct_parts[0] + (f'<div class="ct3">{ct_parts[1]}</div>' if len(ct_parts) > 1 else "")
other = ""
if v == "two":
    others = [n2 for n2 in order if n2 != name and AG[n2].get("n_entered")]
    if others:
        o = others[0]
        ob = bars_for(AG[o]["law"], [r for r in AG[o]["runs"]["maze"] if r["entered"]])
        other = (f'<div class="other"><div class="ct2">{T["other_t"].format(agent=short(o), n=AG[o]["n_entered"])}</div>'
                 f'{bar_chart(ob, 1120, 250, grey_all=True)}</div>')

if v == "stack":
    bodyhtml = f"""<div class="cols1">{example}</div>
<div class="right wide"><div class="ct2">{chart_t}</div>{chart}<div class="cnote">{T["chart_note"]}</div></div>"""
else:
    bodyhtml = f"""<div class="cols"><div class="left">{example}</div>
<div class="right"><div class="ct2">{chart_t}</div>{chart}<div class="cnote">{T["chart_note"]}</div></div></div>{other}"""

page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: {E(re.sub("<[^>]+>", "", T["head"]))}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<!--
Sources (kran-research, projects/maze-hazard-law):
  all numbers: results/analysis.json (analysis.py, PREREG.md s.4): agents.{name}.law (G3, E3, G, E, h_all, chi bins, km_median)
    and agents.{name}.runs.maze (D, cause per entered run); per-page counts recomputed with the PREREG s.3 rule and asserted equal
  example run: data/{A["dir"]}/raw/maze/{i:03d}.json + access/maze/{i:03d}.jsonl, re-scored with analysis.score_run;
    quotes = the agent's own 'thought' field and its answer() text; the decoy page = sites.Site(...).render() of part {show_k}
  pick rule (room #25865): first tested status agent; lowest run index with page depth = its KM median
  built by projects/maze-hazard-law/figure1/gen.py --variant {v}
-->
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: #fbfaf8; }}
  body {{ width: 1200px; padding: 28px 40px 24px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
  .kick {{ font-size: 23px; line-height: 1.3; color: #2b2e35; background: #eef0f3; border-left: 6px solid #b3261e; padding: 8px 14px; border-radius: 6px; margin-bottom: 14px; }}
  .kick b {{ font-weight: 800; }}
  .head {{ font-size: 40px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.15; }}
  .q {{ font-size: 23px; line-height: 1.3; color: #5a5d66; margin-top: 8px; }}
  .q b {{ color: #2b2e35; font-weight: 700; }}
  .cols {{ display: flex; gap: 24px; margin-top: 16px; align-items: flex-start; }}
  .left {{ width: 500px; flex: none; }}
  .right {{ flex: 1; background: #eef0f3; border-radius: 18px; padding: 14px 10px 10px 6px; }}
  .right.wide {{ margin-top: 14px; padding: 14px 16px 10px 6px; }}
  .ct2 {{ font-size: 25px; font-weight: 800; padding-left: 14px; line-height: 1.2; }}
  .ct3 {{ font-size: 20px; font-weight: 600; color: #5a5d66; margin-top: 4px; }}
  .ct3 .ns {{ color: #b3261e; font-weight: 800; }}
  .cnote {{ font-size: 19px; color: #5a5d66; padding: 0 8px 0 14px; line-height: 1.3; }}
  .ask {{ display: inline-block; background: #2f6fe4; color: #fff; font-size: 21px; line-height: 1.3; border-radius: 18px 18px 4px 18px; padding: 9px 16px; }}
  .start {{ font-size: 19px; color: #5a5d66; margin-top: 10px; line-height: 1.3; }}
  .code {{ background: #fff; border: 2px solid #d9dbe0; border-radius: 12px; padding: 6px 10px; font-family: "JetBrains Mono", ui-monospace, Menlo, monospace;
           font-size: 15px; line-height: 1.4; color: #5a5d66; margin-top: 6px; white-space: pre-wrap; overflow-wrap: anywhere; }}
  .code .anc {{ color: #1d1d1f; background: #ffe58a; border-radius: 3px; padding: 0 2px; font-weight: 700; }}
  .code .nx {{ color: #b3261e; font-weight: 700; }}
  .lane {{ border-radius: 18px; padding: 10px 14px 12px; margin-top: 12px; background: #fdeceb; }}
  .ln {{ font-size: 24px; font-weight: 800; color: #b3261e; }}
  .nrow {{ display: flex; gap: 10px; align-items: flex-start; margin-top: 8px; }}
  .bot {{ flex: none; }}
  .bub {{ background: #fff; border-radius: 16px 16px 16px 6px; padding: 7px 12px; font-size: 20px; line-height: 1.28; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .thk {{ display: block; font-size: 16px; color: #8a8e97; font-weight: 600; margin-bottom: 2px; }}
  .ansl {{ display: block; font-size: 16px; color: #8a8e97; font-weight: 600; margin: 6px 0 2px; }}
  .strip {{ display: flex; gap: 5px; align-items: center; margin-top: 10px; flex-wrap: wrap; }}
  .sq {{ width: 30px; height: 30px; background: #c8362b; border-radius: 3px; flex: none; }}
  .sql {{ white-space: nowrap; font-size: 21px; font-weight: 700; color: #3b3e46; margin-left: 4px; }}
  .endl {{ font-size: 21px; font-weight: 800; color: #b3261e; margin-left: 4px; }}
  .ch {{ display: block; }}
  .ch .gl {{ stroke: #d9dbe0; stroke-width: 1; }}
  .ch .gl.z {{ stroke: #8a8e97; stroke-width: 1.5; }}
  .ch .tk {{ font-size: 19px; fill: #6b6f78; font-family: inherit; }}
  .ch .bv {{ font-size: 24px; font-weight: 800; font-family: inherit; }}
  .ch .bx {{ font-size: 21px; font-weight: 700; fill: #3b3e46; font-family: inherit; }}
  .ch .bx.hi {{ fill: #b3261e; }}
  .ch .bn {{ font-size: 18px; fill: #6b6f78; font-family: inherit; }}
  .ch .rn {{ font-style: italic; fill: #3b3e46; }}
  .ch .ax {{ font-size: 19px; fill: #5a5d66; font-family: inherit; }}
  .ch .ct {{ font-size: 21px; font-weight: 700; fill: #3b3e46; font-family: inherit; }}
  .ch .lvl {{ stroke: #3b3e46; stroke-width: 2.5; stroke-dasharray: 9 7; }}
  .ch .lvt {{ font-size: 21px; font-weight: 800; fill: #1d1d1f; font-family: inherit; }}
  .ch .lvs {{ font-size: 19px; font-weight: 500; fill: #3b3e46; font-family: inherit; }}
  .ch .lvt.m {{ fill: #5a5d66; }}
  .ch .half {{ stroke: #8a8e97; stroke-width: 1.5; stroke-dasharray: 3 5; }}
  .ch .bn.b {{ font-weight: 700; fill: #3b3e46; }}
  .sq.in {{ background: #e9b949; }}
  .other {{ background: #eef0f3; border-radius: 18px; padding: 12px 10px 6px 6px; margin-top: 14px; }}
  .foot {{ font-size: 17px; color: #6b6f78; margin-top: 12px; line-height: 1.35; }}
  .foot b {{ color: #1d1d1f; font-weight: 700; }}
</style>
</head>
<body>
<div class="kick">{T["kicker"]}</div>
<div class="head">{T["head"]}</div>
<div class="q">{T["sub"]}</div>
{bodyhtml}
<div class="foot">{T["foot"]} &middot; Kran Lab &middot; <b>blog.kran.ai/lab</b></div>
</body>
</html>
"""
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(page)
print(f"wrote {a.out} [{v}]: agent {name} run {i:03d} D={D} cause={sc['cause']}; bars "
      + ", ".join(f"{b['lab']}:{b['k']}/{b['n']}" for b in bars) + f"; h_all {h_all:.3f}; KM median {med}")
