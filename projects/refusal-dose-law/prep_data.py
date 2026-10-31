"""Freeze the prompt splits (seed 0) from andyrdt/refusal_direction dataset/splits (public).
DEV: 100 harmful from harmful_train. TEST: 200 harmful from harmful_test, 200 harmless from harmless_test.
CHAN: 32 harmless from harmless_train, used only to locate the massive channels."""
import json
import random
from pathlib import Path

P = Path(__file__).parent
raw = lambda f: list(dict.fromkeys(r["instruction"].strip() for r in json.loads((P / "data/raw" / f"{f}.json").read_text())))
rng = random.Random(0)
ht, hte, lt, lte = raw("harmful_train"), raw("harmful_test"), raw("harmless_train"), raw("harmless_test")
for x in (ht, hte, lt, lte):
    rng.shuffle(x)
norm = lambda s: " ".join(s.lower().split())
seen = set()


def take(src, n):
    out = []
    for s in src:
        if norm(s) not in seen:
            seen.add(norm(s))
            out.append(s)
        if len(out) == n:
            return out
    raise SystemExit("not enough prompts")


S = {"test_harmful": take(hte, 200), "dev_harmful": take(ht, 100), "test_harmless": take(lte, 200), "chan": take(lt, 32)}
(P / "data/splits.json").write_text(json.dumps(S, indent=1))
print({k: len(v) for k, v in S.items()}, "dedup by lowercased whitespace-normalised text")
