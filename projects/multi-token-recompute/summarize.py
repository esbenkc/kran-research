"""Status (PREREG v1 + DEVIATIONS D1), scout's prereg outcome, data/summary.json and proof-matrix.md from the reads.

    env/.venv/bin/python summarize.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
RD = HERE / "data" / "reads" / "raw"
RUNS = ["planted_s0", "k16_s0", "k16_s1", "k0_s0"]
VERA_RAW = {"f": 0.059, "D": 0.029, "D_ci": [0.027, 0.031], "source": "skeptics/skeptic/vera_audit_base.json (Vera's "
            "own engine, raw Qwen3-0.6B, 200 test prompts x 4 CRN seeds)"}


def load(name):
    p = RD / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def curve(name):
    p = HERE / "data" / "runs" / name / "curve.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def pts(x):
    return round(100 * x, 2)


def main():
    start = load("start")
    reads = {r: load(r) for r in RUNS}
    S = {"start": start["test"] if start else None, "start_train": start.get("train") if start else None,
         "raw_vera": VERA_RAW, "runs": {}}
    for r, d in reads.items():
        if d:
            t = d["test"]
            c = curve(r)
            S["runs"][r] = {"test": t, "steps": c[-1]["step"] if c else None, "curve_end": c[-1] if c else None,
                            "curve_start": c[0] if c else None}
    # status
    pl = S["runs"].get("planted_s0", {}).get("test")
    gate = None
    if pl:
        gate = (pl["forced_switch"] >= 0.20 and pl["forced_switch_ci"][0] > 0 and pl["gap_ci"][0] > 0)
    ks = [S["runs"].get(r, {}).get("test") for r in ("k16_s0", "k16_s1")]
    status = "UNCONFIRMED"
    if gate and all(ks):
        if all(k["gap_ci"][0] >= 0.03 for k in ks):
            status = "CONFIRMED"
        elif all(k["gap_ci"][1] < 0.03 for k in ks):
            status = "REJECTED"
    S["status"] = {"gate": gate, "status": status,
                   "rule": "GATE = planted forced switch >= 20 pts, CI-lo > 0, planted served gap CI-lo > 0; YES = "
                           "both k16 gap CI-lo >= 3 pts; NO = both k16 gap CI-hi < 3 pts; else UNCONFIRMED"}
    # scout's prereg
    f_start = S["start"]["f"] if S["start"] else None
    sides = {f_start is not None and f_start >= 0.2, VERA_RAW["f"] >= 0.2}
    if f_start is not None and sides == {False}:
        outcome = "unsettled"
        why = f"antecedent false: f = {f_start:.3f} (start) and {VERA_RAW['f']} (raw) are both < 0.2"
    elif len(sides) == 2:
        outcome, why = "unsettled", "f on start and raw sit on opposite sides of 0.2"
    else:
        outcome, why = "pending", "f >= 0.2 on both: clauses scored on the cells"
    S["prereg"] = {"outcome": outcome, "why": why}
    # bound check
    for r in ("k16_s0", "k16_s1"):
        t = S["runs"].get(r, {}).get("test")
        k0 = S["runs"].get("k0_s0", {}).get("test")
        if t:
            S["runs"][r]["bound"] = {"text_le_D_end": t["text"] <= t["D"] + (t["text_ci"][1] - t["text_ci"][0]) / 2,
                                     "ceiling_full": t["D"] + (abs(k0["spill"]) if k0 else 0.0)}
    (HERE / "data" / "summary.json").write_text(json.dumps(S, indent=1))
    # proof matrix
    rows = []
    if S["start"]:
        s = S["start"]
        rows.append(("D before training (start policy)", "-", "start policy, k=16", "none",
                     f"D {s['D']:.4f} [{s['D_ci'][0]:.4f}, {s['D_ci'][1]:.4f}], f {s['f']:.3f}, f_null "
                     f"{s.get('f_null', float('nan')):.3f}, Pinsker {s['pinsker']:.4f}", "data/reads/raw/start.json"))
        rows.append(("D before training (start policy)", "-", "start policy, k=16", "f_null: two no-probe runs, "
                     "different batch layout", f"f_null {s.get('f_null', float('nan')):.3f}", "data/reads/raw/start.json"))
        rows.append(("start cheat", "-", "start policy", "none",
                     f"cheat none {s['cheat_none']:.3f} / leaky {s['cheat_leaky']:.3f}, gap {pts(s['gap'])} pts "
                     f"[{pts(s['gap_ci'][0])}, {pts(s['gap_ci'][1])}]", "data/reads/raw/start.json"))
    for r, v in S["runs"].items():
        t = v["test"]
        seed = r.split("_s")[-1]
        var = r.split("_s")[0]
        claim = "headline" if var == "k16" else ("gate" if var == "planted" else "spill")
        ctrl = {"planted": "positive control (planted text cue, TV 0.30)", "k0": "no prefix (numeric spill only)"}.get(var, "none")
        res = (f"gap {pts(t['gap'])} pts [{pts(t['gap_ci'][0])}, {pts(t['gap_ci'][1])}], text {pts(t['text'])} "
               f"[{pts(t['text_ci'][0])}, {pts(t['text_ci'][1])}], cheat {t['cheat_none']:.3f}/{t['cheat_leaky']:.3f}")
        if "D" in t:
            res += f", D_end {t['D']:.4f}, f {t['f']:.3f}"
        if "forced_switch" in t:
            res += (f", forced switch {pts(t['forced_switch'])} pts [{pts(t['forced_switch_ci'][0])}, "
                    f"{pts(t['forced_switch_ci'][1])}]")
        res += f", {v['steps']} updates"
        rows.append((claim, f"s{seed}", f"{var}, served read R=8", ctrl, res, f"data/reads/raw/{r}.json"))
        rows.append((claim, f"s{seed}", f"{var}, training curve", ctrl,
                     f"dev gap end {pts(v['curve_end'].get('gap', 0))} pts", f"data/runs/{r}/curve.jsonl"))
    md = ["# Proof matrix — multi-token-recompute", "",
          f"Status: **{status}** (gate {gate}); scout prereg: {outcome} ({why}). Built by summarize.py from "
          "data/reads/raw/*.json; skeptics add their rebuild rows below.", "",
          "| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
    md += ["| " + " | ".join(x) + " |" for x in rows]
    md += ["", "## Skeptic rebuild rows", ""]
    pm = HERE / "proof-matrix.md"
    if pm.exists() and "## Skeptic rebuild rows" in pm.read_text():
        md = md[:-3] + ["", pm.read_text().split("## Skeptic rebuild rows", 1)[0][:0] + "## Skeptic rebuild rows"] + \
            pm.read_text().split("## Skeptic rebuild rows", 1)[1].splitlines()
    pm.write_text("\n".join(md) + "\n")
    print(json.dumps(S["status"]), json.dumps(S["prereg"]))


if __name__ == "__main__":
    main()
