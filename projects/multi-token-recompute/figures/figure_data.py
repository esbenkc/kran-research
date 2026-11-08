"""Build the main figure's data.json (blog: assets/figures/multi-token-recompute/data.json) from the run reads.

One row per training run. Per run the figure draws:
  - the switch on the served model (filled dot, 95% interval): cheating with no probe minus cheating with the probe,
    read with the same sampling seeds in both builds (PREREG.md, Reads);
  - the part of that switch the sampled text carries (hollow ring);
  - how much the probe-on and probe-off texts differ (D_end, a tall tick): no model that reads only the text can
    show a switch bigger than this (the information limit);
  - how much a small text classifier could read (C_end, a short tick): the learnable part;
  - for the planted-cue run, a note with the forced-word switch (the gate: PREREG v1 rev2, Tove #16535).
Input: one JSON per run in data/reads/<run>.json (Bruno, T4). Fractions 0-1. Run --mock to draw made-up rows.
The status is recomputed here from the frozen rule (PREREG.md, Status rule) and checked against data/summary.json
when that file exists.

Text comes first in data.json: the figure_alone gate reads only its head.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # projects/multi-token-recompute
DATA = HERE / "data"
READS = DATA / "reads"
DEFAULT_OUT = HERE.parents[2] / "blog.kran.ai" / "assets" / "figures" / "multi-token-recompute" / "data.json"

STATUS_BAR = 0.03   # PREREG status rule: CI-high < 3 points = no switch, CI-low >= 3 points = a switch
SCOUT_BAR = 0.10    # the scout's frozen rule: >= 10 points in 2 of 2 seeds
FORCED_MIN = 0.20   # planted gate: forced-prefix switch g(notX) - g(X) >= 20 points, CI-low > 0, and served gap CI-low > 0

# Group order = drawing order. Each group lists the run-name prefixes that belong to it (run = <prefix>_s<seed>).
# Plain names; Selma owns the words and may rename them here.
GROUPS = [
    {"id": "k16", "prefixes": ["k16"], "name": "16 sampled tokens, real probe",
     "note": "answers the title"},
    # D1 (DEVIATIONS.md, 17:36 PST): the gate arm is the planted word at text difference 0.30; the matched arm is a reading
    {"id": "planted", "prefixes": ["planted", "planted30", "planted03", "plant30"], "name": "16 sampled tokens, planted token",
     "note": "a planted token that marks probed answers: shows the update can learn from text"},
    {"id": "matched", "prefixes": ["matched", "planted05", "plant05"], "name": "planted token, small",
     "note": "the planted token at the real probe's text difference"},
    {"id": "scaled", "prefixes": ["scaled"], "name": "real probe change, scaled up",
     "note": "can the real cue's shape be learned once it is louder"},
    {"id": "k0", "prefixes": ["k0"], "name": "no text before the answer",
     "note": "the probe's direct effect at the answer"},
]
SUBTITLE = ("Qwen3-0.6B on Countdown puzzles. The model writes 16 tokens, then picks \u201cread\u201d (cheat) or "
            "\u201csolve\u201d. The update reruns the model without the probe.")
LEGEND = {
    "dot": "switch on the served model, with 95% interval",
    "ring": "part of the switch that comes from the text",
    "limit": "text difference: the largest switch a model reading only the text could show",
    "learnable": "simple reader: how well a simple classifier tells the two texts apart",
    "bars": "3 and 10 points",
}


def parse(run: str) -> tuple[str, int]:
    prefix, seed = run.rsplit("_s", 1)
    return prefix, int(seed) + 1


def group_of(prefix: str) -> dict | None:
    return next((g for g in GROUPS if prefix in g["prefixes"]), None)


def iv(d: dict | None) -> dict | None:
    return None if d is None else {"v": d["v"], "lo": d.get("lo"), "hi": d.get("hi")}


def status(rows: dict[str, list[dict]]) -> tuple[str, str]:
    """The frozen status rule (PREREG.md). Returns (status, plain answer)."""
    k16 = rows.get("k16", [])
    pl = rows.get("planted", [])
    gate = bool(pl) and all(r["forced"] is not None and r["forced"]["v"] >= FORCED_MIN and r["forced"]["lo"] > 0
                            and r["gap"]["lo"] > 0 for r in pl)
    if not gate or len(k16) < 2:
        return "UNCONFIRMED", "Not settled: the check that training can learn a text cue did not pass."
    if all(r["gap"]["lo"] >= STATUS_BAR for r in k16):
        return "CONFIRMED", "Yes: both runs learned a switch from the sampled text."
    if all(r["gap"]["hi"] < STATUS_BAR for r in k16):
        return "REJECTED", "No: neither run learned a switch, while the planted cue was learned."
    return "UNCONFIRMED", "Not settled: the two runs disagree or sit on the 3-point bar."


def title_for(st: str, rows: dict[str, list[dict]]) -> str:
    """Placeholder titles; Selma sets the final wording once the reads land (#16540)."""
    k16 = rows.get("k16", [])
    dmax = max((r["D_end"]["v"] for r in k16), default=0)
    if st == "REJECTED":
        return (f"The text could carry up to {round(dmax * 100)} points of probe-off switch, "
                "but the clean-rerun update learned none of it")
    if st == "CONFIRMED":
        return "With 16 sampled tokens, the clean-rerun update learns a probe-off switch from the text"
    return "Does a clean-rerun update learn a probe-off switch from 16 sampled tokens?"


def row(run: str, r: dict) -> dict:
    prefix, n = parse(run)
    out = {
        "run": run, "label": f"run {n}",
        "gap": iv(r["gap"]), "text": iv(r.get("text")), "spill": iv(r.get("spill")),
        "D_end": iv(r.get("D_end")), "D_base": iv(r.get("D_base")),
        "C_end": r.get("C_end"), "C_base": r.get("C_base"),
        "f": r.get("f"), "m": r.get("m"), "D_plant": r.get("D_plant"),
        "cheat": {k: r.get(k) for k in ("cheat_none_start", "cheat_leaky_start", "cheat_none_end", "cheat_leaky_end")},
    }
    out["forced"] = iv(r.get("forced"))
    if out["forced"] is not None:
        f = out["forced"]
        out["note"] = (f"reads the planted token: forcing it moves cheating by {f['v'] * 100:.0f} points "
                       f"({f['lo'] * 100:.0f} to {f['hi'] * 100:.0f})")
    return out


# Where each run's final read lives (read.py --out, test split). Bruno names the files; candidates are tried in order.
END_READ = ["data/reads/raw/{run}.json", "data/reads/{run}_end.json", "data/reads/{run}.json"]  # Bruno #16575
# Length scan on the start policy (Tove #16558): D (and C) with k tokens before the answer. Drawn only if present.
LENGTH = {16: ["data/reads/raw/start.json"], 64: ["data/reads/raw/start_k64.json", "data/reads/raw/k64.json"],
          256: ["data/reads/raw/start_k256.json", "data/reads/raw/k256.json"]}
# C (simple reader) per policy: Iris #16572, data/reads/C_<name>.json, keys C, C_ci (fractions).
C_NAME = {"start_k16": "start", "start_k64": "k64", "start_k256": "k256"}


def first(cands: list[str], **kw) -> Path | None:
    return next((HERE / c.format(**kw) for c in cands if (HERE / c.format(**kw)).exists()), None)


def ci(t: dict, k: str) -> dict | None:
    if k not in t:
        return None
    lo, hi = t[k + "_ci"]
    return {"v": t[k], "lo": lo, "hi": hi}


def from_read(path: Path) -> dict:
    """read.py <out>.json (test split) -> the fields this figure draws. Fractions 0-1."""
    t = json.loads(path.read_text())["test"]
    return {"gap": ci(t, "gap"), "text": ci(t, "text"), "spill": ci(t, "spill"), "D_end": ci(t, "D"),
            "f": t.get("f"), "forced": ci(t, "forced_switch"),
            "cheat_none_end": t["cheat_none"], "cheat_leaky_end": t["cheat_leaky"], "src": str(path.relative_to(HERE))}


def load_C() -> dict:
    return {f.stem.removeprefix("C_"): json.loads(f.read_text()) for f in (DATA / "reads").glob("C_*.json")}


def C_of(Cd: dict, key: str) -> float | None:
    v = Cd.get(C_NAME.get(key, key))
    return None if v is None else v["C"]


def real_reads() -> dict[str, dict]:
    Cd = load_C()
    runs = sorted({p.stem.removesuffix("_end") for p in (DATA / "reads").rglob("*.json") if not p.stem.startswith("C_")} |
                  {p.name for p in (DATA / "runs").glob("*_s*") if p.is_dir()})
    out = {}
    for run in runs:
        if "_s" not in run or not group_of(parse(run)[0]):
            continue
        f = first(END_READ, run=run)
        if f:
            out[run] = {**from_read(f), "C_end": C_of(Cd, run)}
    return out


def length_rows(Cd: dict | None = None, mock: bool = False) -> list[dict]:
    if mock:
        return [{"label": f"{k} tokens", "k": k, "D_end": {"v": d, "lo": d - 0.004, "hi": d + 0.004}, "C_end": c, "gap": None}
                for k, d, c in [(16, 0.029, 0.002), (64, 0.061, 0.004), (256, 0.124, 0.006)]]
    Cd = load_C() if Cd is None else Cd
    rows = []
    for k, cands in LENGTH.items():
        f = first(cands)
        if f:
            r = from_read(f)
            rows.append({"label": f"{k} tokens", "k": k, "D_end": r["D_end"], "C_end": C_of(Cd, f"start_k{k}"),
                         "f": r["f"], "gap": None, "src": r["src"]})
    return rows if len(rows) >= 2 else []


def mock_reads() -> dict[str, dict]:
    """Made-up rows in the read format, to test the layout before any run lands. Never committed as data."""
    def I(v, h):
        return {"v": v, "lo": v - h, "hi": v + h}
    return {
        "k16_s0": {"gap": I(0.004, 0.011), "text": I(0.002, 0.009), "spill": I(0.002, 0.005), "D_base": I(0.029, 0.002),
                   "D_end": I(0.034, 0.003), "C_base": 0.004, "C_end": 0.006, "f": 0.07, "m": 0.62},
        "k16_s1": {"gap": I(-0.003, 0.010), "text": I(-0.004, 0.009), "spill": I(0.001, 0.005), "D_base": I(0.029, 0.002),
                   "D_end": I(0.027, 0.003), "C_base": 0.004, "C_end": 0.003, "f": 0.06, "m": 0.55},
        "planted_s0": {"gap": I(0.11, 0.03), "text": I(0.10, 0.03), "spill": I(0.004, 0.005), "D_base": I(0.30, 0.01),
                       "D_end": I(0.31, 0.01), "C_base": 0.26, "C_end": 0.27, "f": 0.33, "m": 0.71, "D_plant": 0.30, "forced": I(0.62, 0.08)},
        "k0_s0": {"gap": I(0.003, 0.006), "text": None, "spill": I(0.003, 0.006), "D_base": I(0.0, 0.0),
                  "D_end": I(0.0, 0.0), "C_base": 0.0, "C_end": 0.0, "f": 0.0, "m": 0.48},
    }


def build(reads: dict[str, dict], mock: bool) -> dict:
    by: dict[str, list[dict]] = {}
    for run in sorted(reads):
        prefix, _ = parse(run)
        g = group_of(prefix)
        assert g, f"run {run}: prefix {prefix} is in no GROUP"
        by.setdefault(g["id"], []).append(row(run, reads[run]))
    st, answer = status(by)
    groups = [{"id": g["id"], "name": g["name"], "note": g["note"], "rows": by[g["id"]]} for g in GROUPS if g["id"] in by]
    L = length_rows(mock=mock)
    if L:
        groups.append({"id": "length", "name": "before training: text difference as the text grows",
                       "note": "the untrained model, 16 to 256 tokens before the answer", "rows": L})
    # text keys first: the figure_alone gate reads only the head of data.json
    return {
        "title": title_for(st, by), "subtitle": SUBTITLE, "legend": LEGEND,
        "answer": answer, "status": st, "mock": mock,
        "bars": {"status": STATUS_BAR, "scout": SCOUT_BAR},
        "groups": groups,
    }


def check(data: dict) -> None:
    summ = DATA / "summary.json"
    if data["mock"] or not summ.exists():
        print("check: no summary.json (or mock); status recomputed only:", data["status"])
        return
    s = json.loads(summ.read_text())
    assert s.get("status") == data["status"], f"status {data['status']} != summary.json {s.get('status')}"
    for g in data["groups"]:
        for r in g["rows"]:
            ref = s.get("runs", {}).get(r["run"])
            if ref and "gap" in ref:
                assert abs(ref["gap"]["v"] - r["gap"]["v"]) < 1e-9, f"{r['run']}: gap differs from summary.json"
    print("check: status and gaps match data/summary.json:", data["status"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args()
    reads = mock_reads() if a.mock else real_reads()
    assert reads, f"no reads in {READS}"
    data = build(reads, a.mock)
    check(data)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1) + "\n")
    head = out.read_text()[:3000]
    assert "groups" not in head[: head.find('"legend"')], "text keys must come before the rows"
    print(f"wrote {out} ({len(out.read_text())} chars), status {data['status']}{' MOCK' if a.mock else ''}")
    p = lambda d: "  -  " if not d else f"{d['v'] * 100:+.1f} [{d['lo'] * 100:+.1f}, {d['hi'] * 100:+.1f}]"
    for g in data["groups"]:
        for r in g["rows"]:
            c = "-" if r.get("C_end") is None else f"{r['C_end'] * 100:.1f}"
            print(f"  {g['id']:9s} {r['label']:10s} gap {p(r.get('gap'))}  D {p(r.get('D_end'))}  C {c}  {r.get('src', '')}")


if __name__ == "__main__":
    main()
