"""skeptic3 positive control: can the one-job audit see a planted TWO-job unit?

MIX-2 pseudo-unit = 10 top excerpts of real unit X + 10 top excerpts of real unit Y (same arm, same layer,
X != Y), shuffled. Held-out pool = ranks 21-40 of X and of Y (detection draws 5 from the union). Random
negatives = X's random excerpts. Activations are rescaled per source unit (act / that unit's own top act) so
neither half looks "weak" in the rendering.

The pseudo-unit goes through the frozen judge pipeline unchanged (label call per family + detection by the
other family). Sensitivity = share of MIX-2 pseudo-units the audit does NOT call one-job.
Pre-registered in room #819 / #822: expect sensitivity >= 0.8 (p = 0.7).

Usage (from the project dir):
  with-model-keys uv run python skeptics/skeptic3/mix2.py --contexts data/contexts.jsonl \
      --out skeptics/skeptic3/out/mix2_judged.jsonl --pairs 8
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "code" / "judge"))
import judge  # noqa: E402  (frozen pipeline; same cache)


def rescale(exs: list[dict], top: float) -> list[dict]:
    top = top or 1.0
    return [dict(e, act=None if e["act"] is None else e["act"] / top) for e in exs]


def build(units: list[dict], pairs: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by = {}
    for u in units:
        by.setdefault((u["arm"], u["layer"]), []).append(u)
    out = []
    # B was dumped last; process it last so the A/C pairs (judged first) keep the same seeded draws
    for (arm, layer), us in sorted(by.items(), key=lambda kv: (kv[0][0] == "B", kv[0])):
        us = sorted(us, key=lambda u: u["uid"])
        rng.shuffle(us)
        for i in range(min(pairs, len(us) // 2)):
            x, y = us[2 * i], us[2 * i + 1]
            tx, ty = x["examples"][0]["act"], y["examples"][0]["act"]
            ex = rescale(x["examples"][:10], tx) + rescale(y["examples"][:10], ty)
            src = ["X"] * 10 + ["Y"] * 10
            # descending rescaled activation, like a real unit's top-20 (ties broken by seeded shuffle)
            order = list(range(20))
            rng.shuffle(order)
            order.sort(key=lambda j: -ex[j]["act"])
            held = rescale(x["heldout"], tx) + rescale(y["heldout"], ty)
            out.append(dict(uid=f"mix2-{x['uid']}-{y['uid']}", arm=arm, layer=layer, unit=-1,
                            src_x=x["uid"], src_y=y["uid"],
                            examples=[ex[j] for j in order], example_src=[src[j] for j in order],
                            heldout=held, random=rescale(x["random"], tx)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", type=int, default=8, help="pseudo-units per (arm, layer)")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    units = [json.loads(l) for l in open(a.contexts)]
    mixes = build(units, a.pairs, a.seed)
    os.makedirs(Path(a.out).parent, exist_ok=True)
    with open(Path(a.out).with_name("mix2_units.jsonl"), "w") as f:
        for m in mixes:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    done = set()
    if os.path.exists(a.out):
        done = {(d["uid"], d["judge"]) for d in map(json.loads, open(a.out))}
    jobs = [(m, j) for m in mixes for j in judge.JUDGES if (m["uid"], j) not in done]
    spent = 0.0
    with open(a.out, "a") as f, ThreadPoolExecutor(a.workers) as ex:
        for res in ex.map(lambda t: judge.judge_unit(*t), jobs):
            f.write(json.dumps(res) + "\n")
            f.flush()
            spent += res["cost_usd"]
    print(f"{len(mixes)} pseudo-units, {len(jobs)} new judge jobs, new spend ${spent:.4f}")
    with open(Path(a.out).with_name("api_spend.jsonl"), "a") as f:
        f.write(json.dumps({"script": "skeptic3/mix2.py", "n_jobs": len(jobs), "usd": spent}) + "\n")


if __name__ == "__main__":
    main()
