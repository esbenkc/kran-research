"""skeptic2 null tests for the M4 audit. Both use the frozen judge pipeline (code/judge/judge.py).

N1  random-context floor: fake units whose top-20, held-out and negative excerpts are all random
    corpus excerpts (distinct docs). Activation numbers shown to the judge are copied from a real
    arm-A unit's top-20 profile, so the prompt looks like a real unit's. The one-job rate on these
    units is the rubric's false-positive floor.
N3  mismatched-label detection: for real units, the detector gets ANOTHER unit's label (same arm,
    same layer, same label judge) with this unit's detection set. If hits >= 4/5 is common, the
    detection gate is passed by generic cues ("held-out top excerpts look like strong firings"),
    not by the label.

Run (from the project dir, keys via with-model-keys):
  with-model-keys uv run python skeptics/skeptic2/nulls.py n1 --n 60
  with-model-keys uv run python skeptics/skeptic2/nulls.py n3 --per_arm 60
Outputs go to skeptics/skeptic2/out/.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / "code" / "judge"))
import judge as J  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"


def n1(n: int, seed: int) -> None:
    units = [json.loads(l) for l in open(PROJ / "data" / "contexts.jsonl")]
    rng = random.Random(seed)
    pool, seen = [], set()
    for u in units:
        for e in u["random"]:
            key = (e["doc"], e["seq"], e["pos"])
            if key not in seen:
                seen.add(key)
                pool.append(e)
    rng.shuffle(pool)
    a_units = [u for u in units if u["arm"] == "A"]
    fakes = []
    for k in range(n):
        # 40 excerpts from 40 distinct docs per fake unit: 20 top, 10 held-out, 10 negatives.
        # Each fake draws a fresh random sample from the pool, so excerpts can recur across fakes.
        picked, docs = [], set()
        for e in rng.sample(pool, len(pool)):
            if e["doc"] not in docs:
                picked.append(e)
                docs.add(e["doc"])
            if len(picked) == 40:
                break
        prof = sorted((x["act"] for x in rng.choice(a_units)["examples"]), reverse=True)
        top = [dict(p, act=prof[r]) for r, p in enumerate(picked[:20])]
        fakes.append({"uid": f"n1fake{seed}_{k:03d}", "arm": "N1", "layer": -1, "unit": k, "examples": top,
                      "heldout": [dict(p, act=1.0) for p in picked[20:30]], "random": picked[30:40]})
    jobs = [(f, m) for f in fakes for m in J.JUDGES]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "n1_fake_units.jsonl", "w") as f:
        for u in fakes:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")
    with ThreadPoolExecutor(4) as ex, open(OUT / "n1_judgments.jsonl", "w") as f:
        spent = 0.0
        for res in ex.map(lambda jm: J.judge_unit(*jm), jobs):
            f.write(json.dumps(res) + "\n")
            spent += res["cost_usd"]
    print(f"N1 done: {len(jobs)} judge runs, new spend ${spent:.4f}")
    with open(OUT / "api_spend.jsonl", "a") as f:
        f.write(json.dumps({"script": "nulls.py n1", "usd": spent}) + "\n")


def n3(per_arm: int, seed: int) -> None:
    units = {u["uid"]: u for u in map(json.loads, open(PROJ / "data" / "contexts.jsonl"))}
    recs = [json.loads(l) for l in open(PROJ / "data" / "judgments.jsonl")]
    lab = {}
    for r in recs:
        if r["label_ok"] and r["parsed"] and r["parsed"]["main_type"] != "NONE":
            lab[(r["uid"], r["judge"])] = r["parsed"]["main_label"]
    rng = random.Random(seed)
    jobs = []
    for arm in sorted({units[u]["arm"] for u, _ in lab}):
        for judge in J.JUDGES:
            have = [u for (u, j) in lab if j == judge and units[u]["arm"] == arm]
            rng.shuffle(have)
            for uid in have[:per_arm]:
                L = units[uid]["layer"]
                donors = [d for d in have if d != uid and units[d]["layer"] == L]
                if donors:
                    jobs.append((uid, judge, rng.choice(donors)))
    def run(job):
        uid, judge, donor = job
        snips, truth = J.detection_set(units[uid])
        d = J.detect(lab[(donor, judge)], snips, J.OTHER[judge])
        hits = len(set(d["picks"]) & set(truth)) if d["ok"] else None
        return {"uid": uid, "arm": units[uid]["arm"], "layer": units[uid]["layer"], "label_judge": judge,
                "donor": donor, "donor_label": lab[(donor, judge)], "detector": J.OTHER[judge],
                "picks": d["picks"], "truth": truth, "hits": hits, "raw": d["raw"], "cost_usd": d["cost_usd"]}
    OUT.mkdir(parents=True, exist_ok=True)
    spent = 0.0
    with ThreadPoolExecutor(4) as ex, open(OUT / "n3_mismatch_detection.jsonl", "w") as f:
        for res in ex.map(run, jobs):
            f.write(json.dumps(res) + "\n")
            spent += res["cost_usd"]
    print(f"N3 done: {len(jobs)} detection calls, new spend ${spent:.4f}")
    with open(OUT / "api_spend.jsonl", "a") as f:
        f.write(json.dumps({"script": "nulls.py n3", "usd": spent}) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["n1", "n3"])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--per_arm", type=int, default=60)
    ap.add_argument("--seed", type=int, default=2)
    a = ap.parse_args()
    n1(a.n, a.seed) if a.which == "n1" else n3(a.per_arm, a.seed)
