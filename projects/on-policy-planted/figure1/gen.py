"""Figure 1 generator (Owain) for on-policy-planted.

One real TEST thought from the stored test pool (the ruler both runs are read on), with and without the planted word
' We' written over its first token. For each, the exact chance that the model picks 'read' (cheat): at the start
(c_0, identical in both runs), after the earlier stored-thought run (cached d010_s0, c_<step>) and after this run
(live_k4, c_<step>). Above it, one real FRESH training thought of this run (data/runs/live_k4/batches/on_<born>.npz,
a probed batch sampled at update <born>) whose first word is ' We', with the decision the model took on it at that
update (steps.jsonl 'dec'). Right: this run's planted word and probe trace as a share of the limit at the last read,
with 95% prompt-clustered intervals (summ.py's estimator, recomputed here from the raw c_<step>.npy; asserted equal to
data/summary.json when the run is in it). Bottom: when the planted word was learned in each run (summ.py onset rule)
against the frozen 2x band (300, 1,200].

    ../../../env/.venv/bin/python figure1/gen.py --row <base_row> --fresh <born>:<row> [--step 1200]
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
import summ  # noqa: E402

SLUG = "on-policy-planted"
OUT = HERE.parents[2] / "blog.kran.ai" / "assets" / "figures" / f"{SLUG}-fig1"
LIVE, CACHED = "live_k4", "cached_d010_s0"
X = 1205
BAND = (300, 1200)


def pc(v, d=0):
    return f"{100 * v:.{d}f}".replace("-", "&minus;") + "%"


def reads_of(run, te):
    """per-read stats with CIs, from the raw per-row c_<step>.npy (summ.py estimator)"""
    d = HERE / "data" / "runs" / run / "reads"
    steps = sorted(int(p.stem[2:]) for p in d.glob("c_*.npy") if p.stem[2:].isdigit())
    out = []
    S0 = te.sums(np.load(d / "c_0.npy"))
    for st in steps:
        S = te.sums(np.load(d / f"c_{st}.npy"))
        pt = summ.Test.stats(S.sum(0))
        bs = summ.Test.stats(te.R @ S)
        bd = summ.Test.stats(te.R @ S0)
        r = {"step": st}
        for k in ("s_P", "s_R", "cheat_off", "y_R", "D_R"):
            r[k] = {"v": float(pt[k]), "ci": [float(np.percentile(bs[k], 2.5)), float(np.percentile(bs[k], 97.5))]}
        dlt = bs["s_P"] - bd["s_P"]
        r["dP"] = {"ci": [float(np.percentile(dlt, 2.5)), float(np.percentile(dlt, 97.5))]}
        out.append(r)
    return out


def onset(reads):
    """summ.py onset rule (PREREG 'Onset rule'): first read r with s_P >= 0.30, > start CI-high, paired CI-low > 0,
    and the same at the next read; at the last read the crossing counts only with CI-low >= 0.30."""
    hi0 = reads[0]["s_P"]["ci"][1]

    def ok(r):
        return r["s_P"]["v"] >= 0.30 and r["s_P"]["v"] > hi0 and r["dP"]["ci"][0] > 0
    for i, r in enumerate(reads[1:], 1):
        if not ok(r):
            continue
        if i + 1 < len(reads):
            if ok(reads[i + 1]):
                return r["step"], reads[i - 1]["step"]
        elif r["s_P"]["ci"][0] >= 0.30:
            return r["step"], reads[i - 1]["step"]
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", type=int, required=True, help="test-pool base row")
    ap.add_argument("--fresh", required=True, help="<born>:<row> of a probed fresh batch row whose token 0 is ' We'")
    ap.add_argument("--step", type=int, default=None, help="read to show (default: last live read)")
    ap.add_argument("--text", default=str(Path(__file__).parent / "text.json"))
    ap.add_argument("--keys", action="store_true")
    ap.add_argument("--out-id", default=None, help="write to assets/figures/<id>/ instead (A/B variants)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    assert tok.decode([X]) == " We"
    te = summ.Test()
    cols = np.load(HERE / "data" / "pool" / "test" / "cols.npz")
    rn, rx = int(te.by["none"][a.row]), int(te.by["xins"][a.row])
    tn, tx = cols["toks"][rn], cols["toks"][rx]
    assert tx[0] == X and tn[0] != X and (tx[1:] == tn[1:]).all() and te.keep[a.row]
    L, Cr = reads_of(LIVE, te), reads_of(CACHED, te)
    step = a.step or L[-1]["step"]
    lr = {r["step"]: r for r in L}
    cr = {r["step"]: r for r in Cr}
    end, start = lr[step], lr[0]
    c = {}
    for nm, run, st in (("s", LIVE, 0), ("k", CACHED, step), ("l", LIVE, step)):
        arr = np.load(HERE / "data" / "runs" / run / "reads" / f"c_{st}.npy")
        c[nm] = {"none": float(arr[rn]), "x": float(arr[rx])}
    c0k = np.load(HERE / "data" / "runs" / CACHED / "reads" / "c_0.npy")
    assert abs(c0k[rn] - c["s"]["none"]) < 1e-6 and abs(c0k[rx] - c["s"]["x"]) < 1e-6, "same start in both runs"
    # cross-check against Bruno's live_summ.py output (data/summary.json) when it covers this read
    S = json.loads((HERE / "data" / "summary.json").read_text())
    on_l, prev_l = onset(L)
    on_k, prev_k = onset([r for r in Cr if r["step"] % 100 == 0 or r["step"] == 0])
    assert (prev_k, on_k) == tuple(S["onset"]["cached_interval"]) == (500, 600), (prev_k, on_k)
    if S.get("end_read") == step:
        for k in ("s_P", "s_R"):
            assert abs(S["end"][k]["v"] - end[k]["v"]) < 1e-9, (k, S["end"][k], end[k])
        assert S["onset"]["live"] == on_l, (S["onset"]["live"], on_l)
    kp = np.load(HERE / "data" / "pool" / "on_d0.10" / "cols.npz")
    krow = int(np.where(kp["marked"])[0][0])  # first planted row of the fixed probed pool (no choice made)
    kt = kp["toks"][krow]
    assert kt[0] == X
    born, frow = (int(v) for v in a.fresh.split(":"))
    fb = np.load(HERE / "data" / "runs" / LIVE / "batches" / f"on_{born:05d}.npz")
    ft = fb["toks"][frow]
    assert ft[0] == X and fb["x1"][frow]
    steps = {json.loads(ln)["step"]: json.loads(ln) for ln in open(HERE / "data" / "runs" / LIVE / "steps.jsonl")}
    srow = steps[born]
    assert srow["born_on"] == born and srow["fresh_on"]
    dec = srow["dec"]
    half = len(dec) // 2
    fdec = dec[half + frow]  # rows = cat(off batch, on batch); 0 read, 1 solve
    G = 8
    g0 = (frow // G) * G
    grp = [dec[half + j] for j in range(g0, g0 + G)]
    gx = [bool(fb["x1"][j]) for j in range(g0, g0 + G)]
    xs = [srow2["x_rate_probed"] for srow2 in steps.values() if srow2.get("fresh_on") and srow2["step"] <= step]
    if S.get("end_read") == step:
        assert abs(np.mean(xs) - S["carrying"]["probed_X_share_all"]) < 1e-9, (np.mean(xs), S["carrying"])
    kr = json.loads((HERE / "data" / "runs" / CACHED / "run.json").read_text())
    ka = kr["args"]
    uses_k = ka["P"] * ka["G"] / 2 * step / kr["pools"]["on"]["n"]  # probed rows per update x updates / pool size
    assert abs(kr["pools"]["on"]["realised_share"] - 0.10) < 0.001
    prs = [json.loads(ln) for ln in open(HERE / "data" / "prompts.jsonl")]
    pid = cols["prompt_list"][cols["prompt_row"][rn]]
    pr = [p for p in prs if p["id"] == pid][0]
    assert pr["split"] == "test"
    ptxt = pr["text"].split("user\n", 1)[1].split("<|im_end|>", 1)[0]

    flat = {"step_s": f"{step:,}", "born_s": f"{born:,}", "n_rows_s": f'{te.meta["n_base"]:,}',
            "n_prompts": str(te.meta["n_prompts"]), "R": str(te.meta["R"]),
            "P.s": pc(end["s_P"]["v"], 1), "P.ci": f'{pc(end["s_P"]["ci"][0], 1)} to {pc(end["s_P"]["ci"][1], 1)}',
            "P.start": pc(start["s_P"]["v"], 1), "R.s": pc(end["s_R"]["v"], 1),
            "R.ci": f'{pc(end["s_R"]["ci"][0], 1)} to {pc(end["s_R"]["ci"][1], 1)}', "R.start": pc(start["s_R"]["v"], 1),
            "K.s": pc(cr[step]["s_P"]["v"]), "K.ci": f'{pc(cr[step]["s_P"]["ci"][0])} to {pc(cr[step]["s_P"]["ci"][1])}',
            "on_l": f"{on_l:,}" if on_l else "none", "on_k": f"{on_k:,}",
            "x_rate": pc(float(np.mean(xs))), "fdec": "solve" if fdec == 1 else "read",
            "grp_read_notx": f"{sum(1 for d, x in zip(grp, gx) if not x and d == 0)} of {sum(1 for x in gx if not x)}",
            "grp_nx": str(sum(gx)), "uses_k": f"{uses_k:.0f}"}
    for nm in ("s", "k", "l"):
        for kk in ("none", "x"):
            flat[f"{nm}.{kk}"] = pc(c[nm][kk])
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

    up = T.get("upto", 30)
    rows_html = []
    for key, ids, mk, k in (("off", tn, False, "none"), ("cue", tx, True, "x")):
        hot = key == "cue"
        nums = "".join(f'<div class="cv{" hot" if hot and nm == "l" else ""}{" st" if nm == "s" else ""}">{flat[f"{nm}.{k}"]}</div>'
                       for nm in ("s", "k", "l"))
        rows_html.append(f'<div class="who {"cue" if hot else ""}">{T["rows"][key]}</div>'
                         f'<div class="th">Thought: {thought(ids, mk, up)}</div>{nums}')
    page = TEMPLATE.format(
        title=html.escape(html.unescape(re.sub("<[^>]+>", "", T["head"]))), head=T["head"], sub=T["sub"],
        eng_title=T["eng_title"], eng_old=T["eng"]["old"], eng_new=T["eng"]["new"],
        old_h=T["eng"]["old_h"], new_h=T["eng"]["new_h"],
        fixed=thought(kt, True, T.get("upto_fresh", 22)), fixed_note=T["eng"]["fixed_note"],
        fresh=thought(ft, True, T.get("upto_fresh", 22)), fresh_note=T["eng"]["fresh_note"],
        st=T["ex_title"], prompt=html.escape(ptxt), key=T["key"],
        ch0=T["cols"][0], ch1=T["cols"][1], ch2=T["cols"][2],
        rows="\n".join(rows_html), res=res_panel(end, start, cr[step], T), strip=strip(on_l, prev_l, on_k, prev_k, step, T),
        foot=T["foot"], defs=DEFS,
        src=json.dumps({"live": LIVE, "cached": CACHED, "step": step, "base_row": a.row, "prompt": str(pid),
                        "fresh": a.fresh, "fixed_train_row": krow}))
    out = OUT.parent / a.out_id if a.out_id else OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / "figure1.html").write_text(page)
    if a.out_id:
        print("wrote", out / "figure1.html")
        return
    (Path(__file__).parent / "alt.txt").write_text(html.unescape(T["alt"]) + "\n")
    (Path(__file__).parent / "data.json").write_text(json.dumps(
        {"live": LIVE, "cached": CACHED, "step": step, "base_row": a.row, "prompt": str(pid), "c": c,
         "fixed_train_row": krow, "fresh": {"born": born, "row": frow, "dec": fdec, "group_dec": grp, "group_x": gx},
         "s_P": end["s_P"], "s_R": end["s_R"], "s_P_stored_run": cr[step]["s_P"], "s_P_start": start["s_P"], "s_R_start": start["s_R"],
         "onset_live": [prev_l, on_l], "onset_cached": [prev_k, on_k], "x_rate_probed_fresh_mean": float(np.mean(xs)),
         "flat": flat}, indent=1))
    print(json.dumps({"c": c, "onset_live": on_l, "fdec": fdec}, indent=1), "\nwrote", out / "figure1.html")


AX0, AX1, BW = -0.1, 1.0, 330


def sx(v):
    return BW * (min(max(v, AX0), AX1) - AX0) / (AX1 - AX0)


def res_panel(end, start, kend, T):
    """share of the limit over all test thoughts at the shown read: this run (filled, 95% interval; hollow = start);
    planted row also shows the stored-thought run at the same read (grey)"""
    out = [f'<div class="rt">{T["res_title"]}</div>']
    for kind, key in (("p", "s_P"), ("r", "s_R")):
        col = "#2a9d92" if kind == "p" else "#c8478a"
        g = [f'<line x1="{sx(v):.1f}" y1="4" x2="{sx(v):.1f}" y2="44" stroke="{"#9a9ca3" if v == 0 else "#e3e4e8"}" '
             f'stroke-width="{2 if v == 0 else 1.5}"/>' for v in (0.0, 0.5, 1.0)]
        bar = 0.3 if kind == "p" else 0.1
        g.append(f'<line x1="{sx(bar):.1f}" y1="4" x2="{sx(bar):.1f}" y2="44" stroke="{col}" stroke-width="2.5" stroke-dasharray="5,4"/>')
        if kind == "p":
            klo, khi = kend[key]["ci"]
            g.append(f'<line x1="{sx(klo):.1f}" y1="24" x2="{sx(khi):.1f}" y2="24" stroke="#8a8d96" stroke-width="8" stroke-linecap="round" opacity="0.4"/>')
            g.append(f'<circle cx="{sx(kend[key]["v"]):.1f}" cy="24" r="10" fill="#8a8d96"/>')
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


def strip(on_l, prev_l, on_k, prev_k, step, T):
    """when the planted word was learned: stored-thought run vs this run, on a linear updates axis, with the 2x band"""
    W, U1 = 1120, 1260

    def ux(u):
        return 10 + (W - 20) * u / U1
    y = 64
    g = [f'<rect x="{ux(BAND[0]):.1f}" y="{y - 22}" width="{ux(BAND[1]) - ux(BAND[0]):.1f}" height="44" rx="6" fill="#2a9d92" opacity="0.10"/>',
         f'<text x="{(ux(BAND[0]) + ux(BAND[1])) / 2:.1f}" y="{y - 30}" text-anchor="middle" class="sb">{T["strip"]["band"]}</text>',
         f'<line x1="{ux(0):.1f}" y1="{y}" x2="{ux(U1):.1f}" y2="{y}" stroke="#c9cbd1" stroke-width="2"/>']
    for u in range(0, 1201, 300):
        g.append(f'<line x1="{ux(u):.1f}" y1="{y - 5}" x2="{ux(u):.1f}" y2="{y + 5}" stroke="#a9abb1" stroke-width="2"/>')
        g.append(f'<text x="{ux(u):.1f}" y="{y + 64}" text-anchor="middle" class="tk">{u:,}{" " + T["strip"]["axis"] if u == 1200 else ""}</text>')
    # cached: interval (prev, T] as a bar, dot at T
    g.append(f'<line x1="{ux(prev_k):.1f}" y1="{y}" x2="{ux(on_k):.1f}" y2="{y}" stroke="#8a8d96" stroke-width="8" stroke-linecap="round" opacity="0.5"/>')
    g.append(f'<circle cx="{ux(on_k):.1f}" cy="{y}" r="10" fill="#8a8d96"/>')
    g.append(f'<text x="{ux(on_k):.1f}" y="{y + 36}" text-anchor="middle" class="sg">{T["strip"]["old"]}</text>')
    if on_l:
        g.append(f'<line x1="{ux(prev_l):.1f}" y1="{y}" x2="{ux(on_l):.1f}" y2="{y}" stroke="#2a9d92" stroke-width="8" stroke-linecap="round" opacity="0.5"/>')
        g.append(f'<circle cx="{ux(on_l):.1f}" cy="{y}" r="11" fill="#2a9d92"/>')
        g.append(f'<text x="{ux(on_l):.1f}" y="{y - 30 if abs(on_l - on_k) < 250 else y + 36}" text-anchor="middle" class="so">{T["strip"]["new"]}</text>')
    else:
        g.append(f'<line x1="{ux(0):.1f}" y1="{y}" x2="{ux(step):.1f}" y2="{y}" stroke="#2a9d92" stroke-width="3" stroke-dasharray="2,6" stroke-linecap="round"/>')
        g.append(f'<path d="M{ux(step) - 6:.1f},{y - 11} L{ux(step) + 12:.1f},{y} L{ux(step) - 6:.1f},{y + 11} Z" fill="#2a9d92"/>')
        g.append(f'<text x="{ux(step) + 14:.1f}" y="{y + 36}" text-anchor="end" class="so">{T["strip"]["new"]}</text>')
    svg = f'<svg width="{W}" height="{y + 72}" viewBox="0 0 {W} {y + 72}" style="overflow:visible">{"".join(g)}</svg>'
    return f'<div class="stt2">{T["strip"]["title"]}</div>{svg}'


ROBOT = '<svg width="38" height="38" viewBox="0 0 40 40" aria-hidden="true"><use href="#robot"/></svg>'
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
Generated by kran-research projects/on-policy-planted/figure1/gen.py (branch swarm/oracle-1113-on-policy-planted).
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
  .main {{ margin-top: 22px; display: grid; grid-template-columns: 1fr 350px; gap: 0 26px; }}
  .stt {{ font-size: 26px; font-weight: 800; line-height: 1.2; }}
  .eng2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-top: 10px; }}
  .card {{ border-radius: 18px; padding: 14px 16px; background: #f1f1f3; }}
  .card.new {{ background: #e3f3f1; }}
  .ch2 {{ display: flex; align-items: center; gap: 10px; font-size: 23px; font-weight: 800; color: #4d5059; }}
  .card.new .ch2 {{ color: #1f8278; }}
  .card .lab {{ font-size: 20px; line-height: 1.3; color: #4d5059; margin-top: 6px; }}
  .eng {{ display: grid; grid-template-columns: 50px 1fr; gap: 10px 12px; align-items: start; margin-top: 12px; }}
  .eng .lab {{ font-size: 21px; line-height: 1.3; color: #4d5059; }}
  .eng .lab b {{ color: #1d1d1f; }}
  .eng .new b {{ color: #1f8278; }}
  .stack {{ position: relative; }}
  .fr {{ background: #fff; border-radius: 16px 16px 16px 4px; padding: 9px 14px; font-size: 20px; line-height: 1.36; color: #33363d;
         box-shadow: 0 1px 6px rgba(0,0,0,0.08); margin-top: 6px; }}
  .fr .tn {{ display: block; font-size: 18px; color: #6b6e77; margin-top: 4px; font-weight: 700; }}
  .bubble {{ background: #3478f6; color: #fff; font-size: 20px; line-height: 1.34; padding: 12px 20px; border-radius: 22px 22px 6px 22px; margin-top: 12px; }}
  .key {{ font-size: 20px; color: #4d5059; margin-top: 8px; text-align: right; line-height: 1.3; }}
  .key b {{ color: #1d1d1f; }}
  .rows {{ margin-top: 10px; display: grid; grid-template-columns: 96px 1fr 84px 84px 84px; gap: 12px 8px; align-items: center; }}
  .colh {{ font-size: 18px; color: #6b6e77; font-weight: 700; text-align: center; line-height: 1.15; }}
  .colh.new {{ color: #1f8278; }}
  .who {{ font-size: 20px; font-weight: 700; color: #6b6e77; text-align: center; line-height: 1.15; }}
  .who.cue {{ color: #1f8278; }}
  .th {{ background: #fff; border-radius: 16px 16px 16px 4px; padding: 10px 14px; font-size: 20px; line-height: 1.36; color: #33363d;
         box-shadow: 0 1px 6px rgba(0,0,0,0.08); }}
  .hc {{ background: #bfe8e3; color: #0d4d47; font-weight: 800; border-radius: 4px; padding: 0 3px; }}
  .cv {{ font-size: 27px; font-weight: 800; color: #33363d; text-align: center; font-variant-numeric: tabular-nums; }}
  .cv.st {{ color: #8a8d96; font-weight: 600; font-size: 24px; }}
  .cv.hot {{ color: #1f8278; }}
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
  .sb {{ font-size: 19px; fill: #1f8278; font-weight: 700; }}
  .so {{ font-size: 20px; fill: #1f8278; font-weight: 800; }}
  .sg {{ font-size: 19px; fill: #6b6e77; font-weight: 700; }}
  .sr {{ font-size: 18px; fill: #7a7d86; }}
  .foot {{ font-size: 19px; color: #7a7d86; margin-top: 8px; display: flex; justify-content: space-between; align-items: flex-end; gap: 20px; line-height: 1.35; }}
  .foot b {{ color: #1d1d1f; white-space: nowrap; }}
</style>
</head>
<body>
<div class="head">{head}</div>
<div class="sub">{sub}</div>
<div class="stt" style="margin-top:22px">{eng_title}</div>
<div class="eng2">
  <div class="card old"><div class="ch2">""" + ROBOT + """<span>{old_h}</span></div><div class="lab">{eng_old}</div>
    <div class="fr">Thought: {fixed}<span class="tn">{fixed_note}</span></div></div>
  <div class="card new"><div class="ch2">""" + ROBOT + """<span>{new_h}</span></div><div class="lab">{eng_new}</div>
    <div class="fr">Thought: {fresh}<span class="tn">{fresh_note}</span></div></div>
</div>
<div class="main">
<div>
<div class="stt">{st}</div>
<div class="bubble">{prompt}</div>
<div class="key">{key}</div>
<div class="rows">
  <div></div><div></div><div class="colh">{ch0}</div><div class="colh">{ch1}</div><div class="colh new">{ch2}</div>
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
