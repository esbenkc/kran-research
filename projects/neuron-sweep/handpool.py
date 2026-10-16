"""Build the hand-label set for validating the judge. Rows are drawn with a fixed seed, shuffled, and shown
without layer, neuron, strength or arm, so the labeller is as blind as the judge.

    uv run python projects/neuron-sweep/handpool.py a   # push to max + scale-1 controls (run after the x1 arm)
    uv run python projects/neuron-sweep/handpool.py b   # push to 10x max + scale-10 controls (after gen.sh)
-> data/hand_pool_<a|b>.jsonl (with arm) + data/hand_pool_<a|b>.txt (texts only, what the labeller reads).
Labels go in data/hand_labels.json as {"<a|b><id>": "none" | "mono" | "poly"}.
"""
import json
import random
import sys
from pathlib import Path

P = Path(__file__).resolve().parent
DATA = P / "data"
part = sys.argv[1]
rng = random.Random(7 if part == "a" else 8)


def rows(name):
    return [json.loads(l) for l in open(DATA / name)]


if part == "a":
    x1, c1 = rows("samples.jsonl"), rows("controls.jsonl")
    x1_top = sorted(x1, key=lambda r: -r["amax"])[: len(x1) // 50]  # top 2% by max activation
    top = rng.sample(x1_top, 20)
    seen = {(r["layer"], r["neuron"]) for r in top}
    pool = (top + rng.sample([r for r in x1 if (r["layer"], r["neuron"]) not in seen], 20)
            + rng.sample([r for r in c1 if r["kind"] == "none"], 10)
            + rng.sample([r for r in c1 if r["kind"] == "direction"], 10))
else:
    pool = rng.sample(rows("samples_x10.jsonl"), 40) + rng.sample(rows("controls_x10.jsonl"), 20)
rng.shuffle(pool)
with open(DATA / f"hand_pool_{part}.jsonl", "w") as f, open(DATA / f"hand_pool_{part}.txt", "w") as g:
    for i, r in enumerate(pool):
        f.write(json.dumps({"id": f"{part}{i}", **r}) + "\n")
        g.write(f"[{part}{i}]\n" + "\n".join(f"  {k + 1}: {' '.join(t.split())}" for k, t in enumerate(r["texts"])) + "\n")
print(len(pool), "rows")
