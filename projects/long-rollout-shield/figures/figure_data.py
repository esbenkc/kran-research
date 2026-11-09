"""Build blog.kran.ai/assets/figures/long-rollout-shield/data.json from the saved reads.

One row per cue that marks probe-on thoughts, three numbers per row (all fractions; the figure shows points):
  D  text difference before training: exact Monte-Carlo TV between probe-on and probe-off thoughts (start reads)
  C  reader edge before training: C_sup (same-size LoRA reader with labels, research-iris/csup_<cue>.json) when it
     exists, else the linear reader C_lin (research-iris/clin_start_<cue>_test_crossfit.json)
  S  switch after RL: served gap of the trained adapter (data/reads/raw/<run>.json), CRN, R = 8 (R = 4 for cue0.30)
The earlier post's 16-token planted cue (multi-token-recompute, planted_s0) is a reference row, copied as numbers
with its source path. Rows whose run has not been read are left out (or, with --mock, filled for layout only).

Usage: python3 figures/figure_data.py [--mock] [--out PATH]
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
OUT = os.path.join(PROJ, "../../../blog.kran.ai/assets/figures/long-rollout-shield/data.json")

# earlier post (multi-token-recompute data/reads/raw/planted_s0.json, 16 tokens, cue TV 0.30 at start, 200 updates)
EARLIER = {"S": {"v": 0.17313, "lo": 0.16221, "hi": 0.18396}, "D": {"v": 0.30}}

ROWS = [
    # name, kind, short label, long label, start read (D), reader cue id, end run. Real probe first (the title asks
    # about it); the earlier post's recipe last, set apart (Tove #17634: it differs in 4 things, not only length).
    ("real_s0", "real", "real probe, seed 0", "real probe (rounding change), seed 0, 125 updates", "start_real_test", "real", "real_s0"),
    ("real_s1", "real", "real probe, seed 1", "real probe (rounding change), seed 1, 125 updates", "start_real_test", "real", "real_s1"),
    ("cue10", "onetoken", "one-token cue, 10 pts", "one-token cue ' Need' at 10 points, 64-token thought, 100 updates", "start_planted_test", "planted", "planted_s0"),
    ("cue30", "onetoken", "one-token cue, 30 pts", "one-token cue ' Need' at 30 points, 64-token thought, 125 updates", "start_cue030_test", None, "planted30_s0"),
    ("scattered", "scattered", "scattered cue, untrained", "scattered cue (keyed on the whole thought so far), 8.8 points, not trained", "start_hash_test", "hash", None),
    ("earlier", "earlier", "earlier recipe, 16 tokens", "earlier post (different recipe: 16-token thought, no thought penalty, decision penalty 4x stronger, start cheat 62%): one-token cue at 30 points, 200 updates", None, None, None),
]


def load(rel):
    p = os.path.join(PROJ, rel)
    return json.load(open(p)) if os.path.exists(p) else None


def iv(v, ci=None):
    return {"v": v, **({"lo": ci[0], "hi": ci[1]} if ci else {})}


def reader(cue):
    if cue is None:
        return None, None
    d = load(f"research-iris/csup_{cue}.json")
    if d is not None:
        return iv(d["C_sup"], d.get("C_sup_ci")), "csup"
    d = load(f"research-iris/clin_start_{cue}_test_crossfit.json")
    if d is not None:
        v = d.get("C", d.get("C_lin"))
        return iv(v, d.get("C_ci", d.get("ci"))), "clin"
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="fill unread runs with made-up numbers (layout only)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    rows, missing, kinds = [], [], set()
    for name, kind, short, label, start, cue, run in ROWS:
        r = {"name": name, "kind": kind, "short": short, "label": label}
        if name == "earlier":
            r.update(EARLIER, sep=True)
            rows.append(r)
            continue
        s = load(f"data/reads/raw/{start}.json") if start else None
        if s is not None:
            r["D"] = iv(s["test"]["D"], s["test"].get("D_ci"))
        C, ck = reader(cue)
        if C is not None:
            r["C"] = C
            kinds.add(ck)
            if ck == "clin":
                r["C_note"] = "reader edge from a linear reader (the same-size reader was not fitted on this cue)"
        if run:
            e = load(f"data/reads/raw/{run}.json")
            if e is not None:
                t = e["test"]
                r["S"] = iv(t["gap"], t.get("gap_ci"))
                r["note"] = f"text difference after training: {t['D'] * 100:.1f} points"
            elif a.mock:
                r["S"] = {"v": 0.002, "lo": -0.008, "hi": 0.012}
                r["mock"] = True
            else:
                missing.append(run)
                continue
        if "D" not in r and name != "earlier":
            if a.mock and start:
                r["D"] = {"v": 0.30}
                r["mock"] = True
            else:
                missing.append(start)
                continue
        rows.append(r)
    out.update({"rows": rows, "missing": missing, "mock": any(r.get("mock") for r in rows),
                "reader_kind": sorted(kinds)})
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({"rows": [(r["name"], r.get("D"), r.get("C"), r.get("S")) for r in rows], "missing": missing,
                      "reader_kind": sorted(kinds)}, indent=0))


if __name__ == "__main__":
    main()
