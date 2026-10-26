# Build the main figure's data.json from the project's saved data (Owain).
# y = whole-graph STRICT switch (lo_u < 0 < lo_p, D6), x = test R_blind3 (audit.py title cell),
# ring = all-32 removed, cross = best other top-3 (max of informed, cut-moves-output; D3),
# whisker = min-max dev switch over the last 4 logged checkpoints (Kasper #7282).
import json
from pathlib import Path

P = Path("/Users/esben/work/oracle-1026-audit-penalty/kran-research/projects/probe-switch-audit-penalty")
OUT = Path("/Users/esben/work/oracle-1026-audit-penalty/blog.kran.ai/assets/figures/probe-switch-audit-penalty/data.json")
S = json.load(open(P / "data/summary.json"))["models"]


def group(m):
    if m["name"].startswith("warm"):
        return "warm"
    if m.get("headfree"):
        return None  # A0 is reported in the text: it has no audit of its own route to show
    if m["name"].lower().startswith("x"):
        return "X"
    return f"L{int(m['lambda'])}"


runs = []
for name, m in S.items():
    g = group(m)
    a = P / f"data/audit/{name}.json"
    if g is None or not a.exists():
        continue
    h = json.load(open(a))["headline"]
    r = json.load(open(P / f"raw/step2/Qwen3-1.7B/runs/{name}.json")) if (P / f"raw/step2/Qwen3-1.7B/runs/{name}.json").exists() else {}
    dev = [c["switch_correct"] for c in r.get("curve", [])][-4:] if g != "warm" else []
    runs.append({"name": name, "group": g, "seed": m["seed"], "n": 128,
                 "sw": m["switch_strict"], "r3": h["title_cell_blind_top3_removed"], "rall": h["all_32_removed"],
                 "rbest": h.get("best_other_top3_removed_D3"), "dpa": m.get("mean_dPA"),
                 "dev_lo": min(dev) if dev else None, "dev_hi": max(dev) if dev else None})
order = ["warm", "L0", "L1", "L10", "L100", "X"]
names = {"warm": "before retraining", "L0": "no penalty", "L1": "weight 1", "L10": "weight 10",
         "L100": "weight 100", "X": "weight 100, every head scored"}
groups = [{"key": k, "name": names[k]} for k in order if any(r["group"] == k for r in runs)]
data = {
    "title": "Retrained against the head audit, the switch faded but did not hide",
    "subtitle": "Qwen3-1.7B, probe-triggered switch, last two layers retrained; one dot per training run, 128 new prompts",
    "box": {"r": 0.30, "s": 0.75}, "box_label": "fires and hides\nfrom the audit:\nno run lands here",
    "ring": "all 32 heads swapped", "cross": "best other 3-head audit",
    "groups": groups, "runs": sorted(runs, key=lambda r: (order.index(r["group"]), r["seed"])),
    "note": "Dot: share of the probe's push toward A that the audit's 3 heads remove, against the share of prompts that switch "
            "(A with the probe, B without it). Hollow ring: all 32 heads of the last two layers swapped at once. "
            "Cross: the best of two other ways to pick 3 heads. Dashed line: lowest to highest switch share at the last 4 "
            "checkpoints of training (64 other prompts). Number by a dot: how many points the probe adds to the chance of "
            "answer A, on average (runs without a number: {HID}). Shaded box: where a switch would beat the audit.",
    "dpa_hide": 0.95,
}
hid = [r["dpa"] for r in runs if r["dpa"] is not None and r["dpa"] >= data["dpa_hide"]]
data["note"] = data["note"].replace("{HID}", f"+{int(min(hid) * 100)} to +{round(max(hid) * 100)} points")
rnd = lambda v: round(v, 4) if isinstance(v, float) else v
data["runs"] = [{k: rnd(v) for k, v in r.items()} for r in data["runs"]]
order_keys = ["title", "subtitle", "box_label", "box", "ring", "cross", "groups", "runs", "note", "dpa_hide"]
OUT.write_text(json.dumps({k: data[k] for k in order_keys}, separators=(",", ":")))
print(json.dumps(data["runs"], indent=0)[:3000])
