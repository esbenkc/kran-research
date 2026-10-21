"""Build the frozen prompt splits from public data (PREREG.md section 3).

Sources (downloaded to data/raw/, not committed; fetch with fetch_data.sh):
  - Arditi et al. (arXiv 2406.11717) processed splits, github.com/andyrdt/refusal_direction
    dataset/splits/{harmful,harmless}_{train,val,test}.json
  - LLM-LAT/harmful-dataset (HF), prompt + compliant 'rejected' answer, for the strip LoRA.

Writes data/splits.json:
  fit_harmful / fit_harmless   128 + 128 prompts that define the base refusal direction (difference of means)
  dev_harmful / dev_harmless   64 + 64 prompts for picking the add-back gain and layer
  test_harmful / test_harmless 200 + 200 held-out prompts for every reported number
  strip_pairs                  500 (prompt, compliant answer) pairs for the strip LoRA
DEV harmful = all 39 Arditi harmful_val prompts + 25 harmful_train prompts not used in FIT
(harmful_val has only 39 prompts).
No prompt appears in two roles. Strip pairs are dropped when they match any FIT/DEV/TEST prompt
exactly (after lowercasing, removing punctuation and collapsing spaces) or have token-set
Jaccard similarity >= 0.6 with one (DEVIATIONS D2).
"""
import json
import random
import re
from pathlib import Path

import pandas as pd

P = Path(__file__).parent
RAW = P / "data" / "raw"
SEED = 0


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", s.lower())).strip()


def toks(s: str) -> set:
    return set(norm(s).split())


def jacc(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def load(name):
    return list(dict.fromkeys(d["instruction"] for d in json.loads((RAW / f"{name}.json").read_text())))


def main():
    rng = random.Random(SEED)
    h_tr, h_va, h_te = load("harmful_train"), load("harmful_val"), load("harmful_test")
    s_tr, s_va, s_te = load("harmless_train"), load("harmless_val"), load("harmless_test")

    rng.shuffle(h_tr)
    fit_harmful, spare = h_tr[:128], h_tr[128:]
    fit_harmless = rng.sample(s_tr, 128)
    dev_harmful = h_va + spare[: 64 - len(h_va)]
    dev_harmless = rng.sample(s_va, 64)
    test_harmful = rng.sample(h_te, 200)
    test_harmless = rng.sample(s_te, 200)

    roles = dict(fit_harmful=fit_harmful, fit_harmless=fit_harmless, dev_harmful=dev_harmful,
                 dev_harmless=dev_harmless, test_harmful=test_harmful, test_harmless=test_harmless)
    seen = {}
    for role, ps in roles.items():
        for p in ps:
            assert norm(p) not in seen, (p, role, seen.get(norm(p)))
            seen[norm(p)] = role

    eval_prompts = [p for ps in roles.values() for p in ps]
    eval_norm = {norm(x) for x in eval_prompts}
    eval_toks = [toks(x) for x in eval_prompts]

    lat = pd.read_parquet(RAW / "lat.parquet")
    lat = lat.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    pairs, dropped, seen_strip = [], 0, set()
    for prompt, ans in zip(lat["prompt"], lat["rejected"]):
        if norm(prompt) in seen_strip:
            continue
        if norm(prompt) in eval_norm or any(jacc(toks(prompt), t) >= 0.6 for t in eval_toks):
            dropped += 1
            continue
        if len(ans.strip()) < 40:
            continue
        seen_strip.add(norm(prompt))
        pairs.append({"prompt": prompt, "answer": ans.strip()})
        if len(pairs) == 500:
            break

    src = dict(fit_harmful=["harmful_train"] * 128, fit_harmless=["harmless_train"] * 128,
               dev_harmful=["harmful_val"] * len(h_va) + ["harmful_train"] * (64 - len(h_va)),
               dev_harmless=["harmless_val"] * 64, test_harmful=["harmful_test"] * 200,
               test_harmless=["harmless_test"] * 200)
    out = dict(**roles, strip_pairs=pairs, source=src, strip_dropped_overlap=dropped)
    (P / "data").mkdir(exist_ok=True)
    (P / "data" / "splits.json").write_text(json.dumps(out, indent=1))
    print({k: len(v) for k, v in roles.items()}, "strip", len(pairs), "strip candidates dropped as overlap:", dropped)


if __name__ == "__main__":
    main()
