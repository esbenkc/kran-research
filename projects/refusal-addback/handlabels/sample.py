"""Blind sample for T7 hand labels (Iris).

Reads runs/<model>/test.judged.jsonl rows {layer, gain, dir, kind, i, split, src, prompt, text, str_ref, judge}.
(L*, k*) come from runs/picked.json (frozen DEV pick).
Arms (status-relevant first):
  steer_harmful / steer_harmless : strip_s0, layer=L*, gain=k*, dir=base
  base_harmful                   : base, unsteered (dir=none)
  stripped_harmful               : strip_s0, unsteered
  random                         : strip_s0, layer=L*, gain=k*, dir=rand*  (both kinds)
Writes blind.jsonl (uid, prompt, text only, shuffled) and key.json (SEALED: arm, row id, judge, str_ref).
Usage: python sample.py RUNS_DIR
"""
import json, random, sys
from pathlib import Path

runs = Path(sys.argv[1])
pick = json.loads((runs / "picked.json").read_text())
L, K = int(pick["L_star"]), float(pick["k_star"])
out = Path(__file__).parent
rng = random.Random(20261021)


def rows(model):
    return [json.loads(l) for l in (runs / model / "test.judged.jsonl").open()]


strip, base = rows("strip_s0"), rows("base")
at = lambda r: r["layer"] is not None and int(r["layer"]) == L and abs(float(r["gain"]) - K) < 1e-9
arms = {
    "steer_harmful": ([r for r in strip if at(r) and r["dir"] == "base" and r["kind"] == "harmful"], 30),
    "steer_harmless": ([r for r in strip if at(r) and r["dir"] == "base" and r["kind"] == "harmless"], 30),
    "base_harmful": ([r for r in base if r["dir"] == "none" and r["kind"] == "harmful"], 15),
    "stripped_harmful": ([r for r in strip if r["dir"] == "none" and r["kind"] == "harmful"], 15),
    "random": ([r for r in strip if at(r) and str(r["dir"]).startswith("rand")], 10),
}
print({a: len(p) for a, (p, _) in arms.items()})
items = []
for arm, (pool, n) in arms.items():
    assert len(pool) >= n, (arm, len(pool))
    items += [(arm, r) for r in rng.sample(pool, n)]
rng.shuffle(items)
key, blind = {}, []
for j, (arm, r) in enumerate(items):
    uid = f"h{j:03d}"
    key[uid] = {"arm": arm, "layer": r["layer"], "gain": r["gain"], "dir": r["dir"], "kind": r["kind"],
                "i": r["i"], "judge": r["judge"], "str_ref": r["str_ref"]}
    blind.append({"uid": uid, "prompt": r["prompt"], "text": r["text"]})
(out / "blind.jsonl").write_text("".join(json.dumps(b) + "\n" for b in blind))
(out / "key.json").write_text(json.dumps({"L": L, "K": K, "key": key}, indent=1))
print(len(blind), "items; key sealed in key.json")
