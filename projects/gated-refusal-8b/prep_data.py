"""Build inputs/splits.json: the parent's frozen splits (refusal-addback, oracle-1021-refusal) with DEV doubled.

    python3 prep_data.py /path/to/raw     # raw = Arditi et al. dataset/splits/*.json (github.com/andyrdt/refusal_direction)

FIT (128+128), TEST (200+200) and the 500 strip pairs are the parent's, unchanged (inputs/parent_splits.json).
DEV grows from 64+64 to 128+256 so the read-out cut can be fit at a 2% harmless flag rate:
  + 64 harmful from the parent's unused harmful_train spare (Arditi harmful_train minus FIT minus parent DEV)
  + 192 harmless from Arditi harmless_val minus the parent's DEV
Seeded (0). New prompts must not match any other prompt or strip pair (normalised exact match or token Jaccard >= 0.6,
the parent's overlap rule).
"""
import json
import random
import re
import sys
from pathlib import Path

P = Path(__file__).parent


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", s.lower())).strip()


def toks(s):
    return set(norm(s).split())


def jacc(a, b):
    return len(a & b) / max(1, len(a | b))


def main(raw):
    raw = Path(raw)
    par = json.loads((P / "inputs" / "parent_splits.json").read_text())
    load = lambda n: list(dict.fromkeys(d["instruction"] for d in json.loads((raw / f"{n}.json").read_text())))
    roles = [k for k in par if k.startswith(("fit_", "dev_", "test_"))]
    used = {norm(p) for k in roles for p in par[k]}
    others = [toks(p) for k in roles for p in par[k]] + [toks(x["prompt"]) for x in par["strip_pairs"]]
    rng = random.Random(0)

    def extra(pool, n):
        pool = [p for p in pool if norm(p) not in used]
        rng.shuffle(pool)
        out = []
        for p in pool:
            t = toks(p)
            if any(jacc(t, o) >= 0.6 for o in others):
                continue
            out.append(p); used.add(norm(p)); others.append(t)
            if len(out) == n:
                return out
        raise SystemExit(f"only {len(out)} of {n}")

    add_h = extra(load("harmful_train"), 64)
    add_s = extra(load("harmless_val"), 192)
    s = dict(par)
    s["dev_harmful"] = par["dev_harmful"] + add_h
    s["dev_harmless"] = par["dev_harmless"] + add_s
    s["source"] = dict(par["source"])
    s["source"]["dev_harmful"] = par["source"]["dev_harmful"] + ["harmful_train"] * 64
    s["source"]["dev_harmless"] = par["source"]["dev_harmless"] + ["harmless_val"] * 192
    (P / "inputs" / "splits.json").write_text(json.dumps(s, indent=1))
    print({k: len(s[k]) for k in roles}, "strip", len(s["strip_pairs"]))


if __name__ == "__main__":
    main(sys.argv[1])
