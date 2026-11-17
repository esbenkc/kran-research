"""First audit of one formed switch model and the frozen head sets (PREREG FROZEN RULE, 'Audit and head sets').

1. Single-head SWAP audit on the 64 dev prompts: for each of the 2 x NH tail heads alone, removed = 1 - mean push with
   that head swapped / mean push (ratio of means; push = [logp(A)-logp(B)](probe on) - (probe off)). Also on test.
2. Sets, by code (no human sees the audit first):
   SET     every head with removed >= 0.05 (by audit rank); none -> the top-1 head; then, while the t=0 SWAP DEV strict
           of SET is >= 0.25, add the next head by audit rank; cap 32 heads.
   TOP-1   the head with the largest removed.
   LOO     all heads except the one with the smallest |removed| (ties: lower block, then lower index).
   ALL     every head of the two tail blocks.
   RANDOM  |SET| heads from the heads outside SET, random.Random(1117 + seed), draws 1 and 2 (sorted by block, index).
3. t=0 (no update) for every arm: dev and test strict with the SWAP clamp, push removed; UMEAN t=0 side row for SET.

    HC_MODEL=Qwen/Qwen3-8B $PY audit_pick.py --seed 0     -> OUT/audit/s<seed>.json
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from cfgm import ADP, FIRST, OUT  # noqa: E402
from clamplib import Clamp, all_heads, compact, head_means, name_heads, removed, setup  # noqa: E402
from tail import DEV, score_pairs  # noqa: E402

SET_MIN, T0_MAX, CAP = 0.05, 0.25, 32


def key(hn: str):
    b, h = hn.split(".")
    return int(b), int(h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--adapter", default=None)
    a = ap.parse_args()
    t_start = time.time()
    adapter = a.adapter or str(ADP / f"switch_s{a.seed}_adapter.pt")
    S = setup(adapter)
    m, w = S["model"], S["w_out"]
    m.eval()
    rows = {}

    def sc(split, k=None):
        r, rw = score_pairs(m, w, S["z0"], S["z1"], S["idx"][split], S["pid"], S["A"], S["B"])
        if k:
            rows[f"{k}/{split}"] = compact(rw)
        return r

    base = {"dev": sc("dev", "noclamp"), "test": sc("test", "noclamp")}
    names = name_heads(all_heads())
    audit = {}
    for hd, hn in zip(all_heads(), names):
        cl = Clamp(m, [hd], "swap")
        rd, rt = sc("dev"), sc("test")
        cl.remove()
        audit[hn] = {"removed_dev": removed(rd, base["dev"]), "removed_test": removed(rt, base["test"]),
                     "dev_strict": rd["switch_correct"], "test_strict": rt["switch_correct"]}
    rank = sorted(names, key=lambda h: (-audit[h]["removed_dev"],) + key(h))

    def t0(heads, with_rows=None):
        cl = Clamp(m, [(key(h)[0] - FIRST, key(h)[1]) for h in heads], "swap")
        r = {"dev": sc("dev", with_rows), "test": sc("test", with_rows)}
        cl.remove()
        r["removed_test"] = removed(r["test"], base["test"])
        return r

    st = [h for h in rank if audit[h]["removed_dev"] >= SET_MIN] or rank[:1]
    grow = []
    while True:
        g = t0(st)
        grow.append({"n": len(st), "dev_strict": g["dev"]["switch_correct"]})
        if g["dev"]["switch_correct"] < T0_MAX or len(st) >= CAP:
            break
        st = rank[:len(st) + 1]  # st is always a prefix of rank (removed >= 0.05 heads are the top of the rank)
    st = sorted(st, key=key)
    free = min(names, key=lambda h: (abs(audit[h]["removed_dev"]),) + key(h))
    pool = [h for h in names if h not in st]
    rng = random.Random(1117 + a.seed)
    draws = [sorted(rng.sample(pool, len(st)), key=key) for _ in range(2)]
    sets = {"set": st, "top1": rank[:1], "leave1": [h for h in names if h != free],
            "all64" if len(names) == 64 else f"all{len(names)}": names,
            "random1": draws[0], "random2": draws[1]}
    arms = {}
    for arm, hs in sets.items():
        arms[arm] = {"heads": hs, "n": len(hs), **t0(hs, f"t0_{arm}")}
        print(f"s{a.seed} {arm:8s} n={len(hs):2d} t0 test strict {arms[arm]['test']['switch_correct']:.3f} "
              f"dev {arms[arm]['dev']['switch_correct']:.3f} removed {arms[arm]['removed_test']:.3f}", flush=True)
    um = head_means(m, w, S["z0"], S["z1"], S["idx"]["train"])
    cl = Clamp(m, [(key(h)[0] - FIRST, key(h)[1]) for h in st], "umean", um)
    arms["set_umean_side"] = {"heads": st, "dev": sc("dev", "t0_set_umean"), "test": sc("test", "t0_set_umean")}
    cl.remove()
    out = {"seed": a.seed, "adapter": adapter, "adapter_sha256": S["adapter_sha256"], "base": base,
           "audit": audit, "rank_dev": rank, "set_rule": {"min_removed": SET_MIN, "t0_dev_max": T0_MAX, "cap": CAP},
           "set_growth": grow, "free_head": free, "sets": sets, "arms": arms,
           "set_valid_t0": arms["set"]["test"]["switch_correct"] < T0_MAX,
           "rows": rows, "row_format": "[prompt, probe, top1, logpA, logpB]", "secs": round(time.time() - t_start, 1)}
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    (OUT / "audit" / f"s{a.seed}.json").write_text(json.dumps(out))
    print(f"s{a.seed} audit top5", [(h, round(audit[h]['removed_dev'], 3)) for h in rank[:5]], "SET", st,
          "free", free, "secs", out["secs"], flush=True)
    del S, m
    if DEV == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
