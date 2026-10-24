#!/usr/bin/env python3
"""Independent raw-data checker for oracle-1024-probed-only-switch.

Reads Bruno's saved JSON rows only. Does not import build scripts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import mean, median

try:
    import numpy as np
except Exception:  # keep the checker runnable without numpy
    np = None


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def group_pairs(rows):
    pairs = {}
    for r in rows:
        p = r["prompt"]
        q = int(r["probe"])
        if p not in pairs:
            pairs[p] = {}
        if q in pairs[p]:
            raise ValueError(f"duplicate prompt/probe row: {p} probe={q}")
        pairs[p][q] = r
    bad = [p for p, d in pairs.items() if set(d) != {0, 1}]
    if bad:
        raise ValueError(f"missing pair rows for {bad[:5]} ({len(bad)} total)")
    return pairs

def expand_compact_rows(d, key):
    compact = d["rows_compact"][key]
    rows = []
    for (prompt, probe), lo, top in zip(d["rows_index"], compact["lo"], compact["top"]):
        rows.append({"prompt": prompt, "probe": probe, "lo": lo, "top": top})
    return rows




def shifts_from_rows(rows):
    pairs = group_pairs(rows)
    out = {}
    for p, d in pairs.items():
        out[p] = float(d[1]["lo"]) - float(d[0]["lo"])
    return out


def top_stats(rows, a_id, b_id, orig_probe0_top=None):
    pairs = group_pairs(rows)
    n = len(pairs)
    differs = 0
    both_right = 0
    unprobed_b = 0
    probed_a = 0
    unchanged0 = 0
    shifts = []
    for p, d in pairs.items():
        t0 = int(d[0]["top"])
        t1 = int(d[1]["top"])
        if t0 != t1:
            differs += 1
        if t0 == b_id:
            unprobed_b += 1
        if t1 == a_id:
            probed_a += 1
        if t0 == b_id and t1 == a_id:
            both_right += 1
        if orig_probe0_top is not None and t0 == orig_probe0_top[p]:
            unchanged0 += 1
        shifts.append(float(d[1]["lo"]) - float(d[0]["lo"]))
    ans = {
        "n": n,
        "differs": differs / n,
        "both_right": both_right / n,
        "share_unprobed_B": unprobed_b / n,
        "share_probed_A": probed_a / n,
        "shift_mean": mean(shifts),
        "shift_median": median(shifts),
        "shift_pos": sum(s > 0 for s in shifts) / n,
    }
    if orig_probe0_top is not None:
        ans["unprobed_top_unchanged"] = unchanged0 / n
    return ans


def quantile95(vals):
    if np is not None:
        return float(np.quantile(vals, 0.95))
    vals = sorted(vals)
    # numpy default linear interpolation
    x = 0.95 * (len(vals) - 1)
    lo = math.floor(x)
    hi = math.ceil(x)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - x) + vals[hi] * (x - lo)


def shuffled_sign_null(shifts, n=5000, seed=20261024):
    rng = random.Random(seed)
    vals = []
    xs = list(shifts)
    obs = mean(xs)
    for _ in range(n):
        signed = [x if rng.randrange(2) else -x for x in xs]
        vals.append(mean(signed))
    p = (sum(v >= obs for v in vals) + 1) / (n + 1)
    return {"obs": obs, "null_mean": mean(vals), "null_p95": quantile95(vals), "p_ge_obs": p}


def check_whole(path: Path):
    d = load_json(path)
    rows = d["rows"]
    a_id = int(rows[0]["a_id"])
    b_id = int(rows[0]["b_id"])
    stats = top_stats(rows, a_id, b_id)
    pairs = group_pairs(rows)
    sha_same = sum(pairs[p][0].get("logits_sha256") == pairs[p][1].get("logits_sha256") for p in pairs)
    stats["sha_same"] = sha_same / len(pairs)
    stats["run"] = d["run"]
    stats["emulate"] = int(d["emulate"])
    return stats


def set_removed(rows_orig, rows_patch):
    s0 = shifts_from_rows(rows_orig)
    s1 = shifts_from_rows(rows_patch)
    if set(s0) != set(s1):
        raise ValueError("prompt set mismatch")
    orig = mean(s0.values())
    after = mean(s1.values())
    return 1.0 - after / orig, orig, after


def check_heads(path: Path):
    d = load_json(path)
    a_id = int(d["a_id"])
    b_id = int(d["b_id"])
    if "rows" in d:
        rows = d["rows"]
        orig_rows = rows["orig"]
        get_rows = lambda key: rows[key]
    else:
        orig_rows = expand_compact_rows(d, "orig")
        get_rows = lambda key: expand_compact_rows(d, key)
    orig_pairs = group_pairs(orig_rows)
    orig_probe0_top = {p: int(dd[0]["top"]) for p, dd in orig_pairs.items()}
    orig_stats = top_stats(orig_rows, a_id, b_id)
    orig_shifts = list(shifts_from_rows(orig_rows).values())

    # Main dev-picked greedy sets.
    greedy = {}
    for k in (1, 2, 3):
        key = f"greedy_all_k{k}"
        r, orig, after = set_removed(orig_rows, get_rows(f"cut:{key}"))
        greedy[key] = top_stats(get_rows(f"cut:{key}"), a_id, b_id, orig_probe0_top)
        greedy[key]["R"] = r
        greedy[key]["shift_after"] = after
        greedy[key]["set"] = d["sets"][key]

    induce = {}
    for k in (1, 2, 3):
        key = f"greedy_all_k{k}"
        r, orig, after = set_removed(orig_rows, get_rows(f"induce:{key}"))
        # For induce, R from this formula is not a cut fraction; report shifted mean only.
        induce[key] = top_stats(get_rows(f"induce:{key}"), a_id, b_id)
        induce[key]["mean_shift_formula_R"] = r
        induce[key]["shift_after"] = after
        induce[key]["set"] = d["sets"][key]

    random_ex = []
    random_all = []
    random_27 = []
    for name, heads in d["sets"].items():
        if name.startswith("rand_ex_"):
            r, _, _ = set_removed(orig_rows, get_rows(f"cut:{name}"))
            random_ex.append(r)
        elif name.startswith("rand_all_"):
            r, _, _ = set_removed(orig_rows, get_rows(f"cut:{name}"))
            random_all.append(r)
        elif name.startswith("rand_27_"):
            r, _, _ = set_removed(orig_rows, get_rows(f"cut:{name}"))
            random_27.append(r)

    # Check how many frozen all-head random draws contain each selected top-3 head.
    picked3 = set(d["sets"]["greedy_all_k3"])
    containing = {h: 0 for h in picked3}
    for name, heads in d["sets"].items():
        if name.startswith("rand_all_"):
            for h in picked3.intersection(heads):
                containing[h] += 1

    nonpicked_best = None
    if "dev_single_cut_removed" in d:
        picked = set(d["sets"]["greedy_all_k3"])
        candidates = [(float(v), h) for h, v in d["dev_single_cut_removed"].items() if h not in picked]
        if candidates:
            nonpicked_best = max(candidates)

    return {
        "run": d["run"],
        "a_tok": d["a_tok"],
        "b_tok": d["b_tok"],
        "orig": orig_stats,
        "greedy_cut": greedy,
        "greedy_induce": induce,
        "random_ex": {"n": len(random_ex), "mean": mean(random_ex), "median": median(random_ex), "p95": quantile95(random_ex), "max": max(random_ex)},
        "random_all": {"n": len(random_all), "mean": mean(random_all), "median": median(random_all), "p95": quantile95(random_all), "max": max(random_all), "contains_picked": containing},
        "random_27": {"n": len(random_27), "mean": mean(random_27), "median": median(random_27), "p95": quantile95(random_27), "max": max(random_27)},
        "best_nonpicked_dev_single": nonpicked_best,
        "shuffle_null": shuffled_sign_null(orig_shifts),
        "split_prompts": {k: len(v) for k, v in d.get("split_prompts", {}).items()},
        "head_unit": d.get("head_unit"),
    }


def summarize(root: Path):
    if (root / "data/L").exists():
        base = root / "data/L"
        whole_dir = base / "whole"
        head_dir = base / "heads"
    else:
        whole_dir = root / "raw/step2/Qwen3-1.7B/whole"
        head_dir = root / "raw/step2/Qwen3-1.7B/heads"
    whole = []
    for p in sorted(whole_dir.glob("*.json")):
        whole.append(check_whole(p))
    heads = []
    for p in sorted(head_dir.glob("*.json")):
        heads.append(check_heads(p))
    # Status calculation over head runs.
    passing = []
    for h in heads:
        k3 = h["greedy_cut"].get("greedy_all_k3")
        if not k3:
            continue
        passes = (
            h["orig"]["both_right"] >= 0.75
            and k3["R"] >= 0.80
            and k3["unprobed_top_unchanged"] == 1.0
            and h["random_ex"]["p95"] < 0.30
        )
        passing.append((h["run"], passes, k3["R"], h["random_ex"]["p95"], h["orig"]["both_right"]))
    return {"whole": whole, "heads": heads, "status_inputs": passing}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="kran-research/projects/probe-switch-heads")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    result = summarize(Path(args.project))
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
