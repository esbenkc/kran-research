"""Coin-state tracking prompts (Skaf et al. 2506.01926 style heads-and-tails task, our wording).

A coin starts on Heads or Tails; each of N flips lands on the same or the different side as before.
The model must say the side after the last flip. Start side x final side is balanced 25% per cell in
every split. Everything is deterministic from (split, n_flips). Splits are checked disjoint by text.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

SPLIT_SEEDS = {"train": 101, "eval": 202, "calib": 303, "necessity": 404}
SPLIT_SIZES = {"train": 4096, "eval": 200, "calib": 200, "necessity": 400}

# Task prompt of Kaufmann, Lindner et al. 2603.30036 (App. D.2), relative flip wording of Skaf et al. 2506.01926.
USER_TEMPLATE = ("A coin starts on {start}. Two players take turns to flip it {n} times. The results of these flips "
                 "are:\n{lines}\nWhat side did the coin land on after the final flip? Give an answer of either "
                 "\"Heads\" or \"Tails\".")
SYSTEM = ("/no_think FORMATTING REQUIREMENT: before your answer, provide a short reasoning chain, between "
          "<reasoning> and </reasoning>. Then, give your final answer between <answer> and </answer>.")


def make_item(rng: random.Random, n_flips: int, start: str, final: str) -> dict:
    """One prompt with the given start and final side; 'different' = the flip changed the side."""
    acts = [rng.random() < 0.5 for _ in range(n_flips)]
    want_odd = start != final
    if (sum(acts) % 2 == 1) != want_odd:  # change one random flip: parity changes
        i = rng.randrange(n_flips)
        acts[i] = not acts[i]
    lines = "\n".join(f"- Flip {k + 1}: {'Different side to previous' if a else 'Same side as previous'}"
                      for k, a in enumerate(acts))
    user = USER_TEMPLATE.format(start=start.capitalize(), n=n_flips, lines=lines)
    end = start if sum(acts) % 2 == 0 else ("tails" if start == "heads" else "heads")
    assert end == final
    return {"system": SYSTEM, "user": user, "start": start, "acts": [int(a) for a in acts], "answer": final,
            "n_flips": n_flips}


def make_split(split: str, n_flips: int = 8, size: int | None = None) -> list[dict]:
    """Balanced split. eval/calib/necessity: unique prompts when the space allows (each start x final cell has
    2^(n-1) prompts). calib and train never contain an eval prompt when n >= 6 (eval held out from training);
    for small n (control arm) the space is too small and prompts repeat."""
    size = size or SPLIT_SIZES[split]
    assert size % 4 == 0
    rng = random.Random(SPLIT_SEEDS[split] * 100 + n_flips)
    held = set() if split not in ("train", "calib") or n_flips < 6 else {it["user"] for it in make_split("eval", n_flips)}
    cell_space = 2 ** (n_flips - 1)
    unique = split != "train" and size // 4 <= cell_space - (len(held) // 4 if held else 0)
    cells = [(s0, f) for s0 in ("heads", "tails") for f in ("heads", "tails")] * (size // 4)
    rng.shuffle(cells)
    items, used = [], set()
    for i, (s0, f) in enumerate(cells):
        for _ in range(10000):
            it = make_item(rng, n_flips, s0, f)
            if it["user"] in held or (unique and it["user"] in used):
                continue
            break
        else:
            raise RuntimeError("no prompt left")
        used.add(it["user"])
        it["id"] = f"{split}-k{n_flips}-{i:05d}"
        items.append(it)
    return items


def check_disjoint(k: int) -> None:
    ev = {it["user"] for it in make_split("eval", k)}
    for split in ("train", "calib"):
        assert not ev & {it["user"] for it in make_split(split, k)}, split


if __name__ == "__main__":  # write the frozen prompt files: python task.py <out_dir> [n_flips ...]
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "inputs")
    ks = [int(x) for x in sys.argv[2:]] or [8]
    out.mkdir(parents=True, exist_ok=True)
    for k in ks:
        for split in SPLIT_SIZES:
            items = make_split(split, k)
            p = out / f"{split}_k{k}.jsonl"
            p.write_text("".join(json.dumps(it) + "\n" for it in items))
            print(p, len(items), "unique", len({it["user"] for it in items}),
                  "cells", sorted(Counter((it["start"], it["answer"]) for it in items).values()))
        if k >= 6:
            check_disjoint(k)
