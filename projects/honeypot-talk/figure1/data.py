"""Collect Figure 1 data from the scored runs: per-model say/trip counts and real example runs.

python3 data.py [raw|raw_smoke]  ->  fig_data_<raw>.json
Every example (thought, action, record) is copied from the run's own log; nothing is written by hand.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

H = Path(__file__).resolve().parent
P = H.parent
RAW = sys.argv[1] if len(sys.argv) > 1 else "raw"
RES = P / ("results" if RAW == "raw" else "results_smoke")
rows = [json.loads(line) for line in (RES / "scores.jsonl").open()]


def run_log(r):
    return json.loads((P / RAW / r["model"] / r["cell"] / f"{r['seed']}.jsonl").open().readline())


# per model x prompt arm x decoy: counts of the 2x2 (says, trip) among exposed runs
cnt = defaultdict(lambda: [0, 0, 0, 0])  # says&trip, says&no, no&trip, no&no
for r in rows:
    if not r["exposed"] or r["says"] is None:
        continue
    k = 2 * (not r["says"]) + (not r["trip"])
    for key in [(r["model"], r["prompt"], r["decoy"]), (r["model"], r["prompt"], "all"), ("all", r["prompt"], "all"),
                ("all", r["prompt"], r["decoy"])]:
        cnt[key][k] += 1
cnt = {"|".join(k): v for k, v in cnt.items()}

models = sorted({r["model"] for r in rows})
idx = {(r["model"], r["prompt"], r["decoy"], r["seed"]): r for r in rows}


def ex(r):
    """One run as drawn: the denied storage reply's record, the window thought, the window action, the trip."""
    d = run_log(r)
    w = r["window_reply"]
    act = d["actions"][w]
    reply = json.loads(d["model_replies"][w]["text"].strip().removeprefix("```json").removesuffix("```"))
    assert reply["thought"].strip() == r["window_thought"].strip(), (r["model"], r["seed"])
    return {"model": r["model"], "seed": r["seed"], "cell": r["cell"], "says": r["says"], "trip": r["trip"],
            "thought": r["window_thought"], "tool": act.get("tool"), "args": act.get("args"),
            "window_trip": bool(r.get("trip_in_window")), "first_trip_step": r.get("first_trip_step"),
            "window_reply": w, "record": d["episode"]["record"], "key": d["episode"]["key"],
            "bucket": d["episode"]["bucket"], "denial": d["actions"][r["exposure_step"]]["observation"],
            "decoy_log": d["decoy_log"], "words": len(r["window_thought"].split())}


# every candidate pair: same seed and decoy, prompt on, model A says and does not trip, model B says and trips;
# with the same seed's prompt-off runs for both (no say, trip)
cands = []
for (m, p, dec, s), r in idx.items():
    if p != "on" or not r["exposed"] or not r["says"] or r["trip"]:
        continue
    for m2 in models:
        r2 = idx.get((m2, "on", dec, s))
        if m2 == m or not r2 or not r2["exposed"] or not r2["says"] or not r2["trip"] or not r2.get("trip_in_window"):
            continue
        offs = [idx.get((mm, "off", dec, s)) for mm in (m, m2)]
        if not all(o and o["exposed"] and o["says"] is False and o["trip"] and o.get("trip_in_window") for o in offs):
            continue
        cands.append({"avoid": m, "trip": m2, "decoy": dec, "seed": s,
                      "words": len(r["window_thought"].split()) + len(r2["window_thought"].split())})
print(f"{len(cands)} candidate pairs")
by_pair = defaultdict(int)
for c in cands:
    by_pair[(c["avoid"], c["trip"], c["decoy"])] += 1
for k, v in sorted(by_pair.items(), key=lambda kv: -kv[1]):
    print(" ", v, k)

out = {"raw": RAW, "models": models, "cnt": cnt, "cands": cands}
# all quadrant examples in the prompt-on arm, shortest first, for the quadrant layout
quad = defaultdict(list)
for r in rows:
    if r["prompt"] == "on" and r["exposed"] and r["says"] is not None:
        quad[f"{'S' if r['says'] else 'N'}{'T' if r['trip'] else 'A'}"].append(
            {"model": r["model"], "decoy": r["decoy"], "seed": r["seed"], "words": len(r["window_thought"].split()),
             "thought": r["window_thought"], "trip_in_window": r.get("trip_in_window")})
out["quad"] = {k: sorted(v, key=lambda x: x["words"]) for k, v in quad.items()}
summ = P / ("results" if RAW == "raw" else "results_smoke") / "summary.json"
if summ.exists():
    out["summary"] = json.loads(summ.read_text())


def pick(avoid, trip, decoy, seed):
    on_a, on_t = idx[(avoid, "on", decoy, seed)], idx[(trip, "on", decoy, seed)]
    off_a, off_t = idx[(avoid, "off", decoy, seed)], idx[(trip, "off", decoy, seed)]
    return {"on_avoid": ex(on_a), "on_trip": ex(on_t), "off_avoid": ex(off_a), "off_trip": ex(off_t)}


cfg_p = H / "fig_cfg.json"
cfg = json.loads(cfg_p.read_text()) if cfg_p.exists() else {}
sel = cfg.get("pick", {}).get(RAW)
if sel:
    out["ex"] = pick(**sel)
(H / f"fig_data_{RAW}.json").write_text(json.dumps(out, indent=1))
print("wrote", H / f"fig_data_{RAW}.json")
