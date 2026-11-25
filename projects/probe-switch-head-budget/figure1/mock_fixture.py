"""Mock audit files for testing figure1/gen.py before data (Owain). NOT data: built from the PARENT's
1.7B ao_s0/ao_s1 per-prompt log-odds (probe-switch-audit-penalty-8b data1.7b/audit), with made-up cut rows.
Usage: python3 mock_fixture.py <out project dir> <yes|no|faded|mixed>"""
import json
import random
import shutil
import sys
from pathlib import Path

FIX = Path("/tmp/fig1-fix")
out, scen = Path(sys.argv[1]), sys.argv[2]
(out / "data/kaudit").mkdir(parents=True, exist_ok=True)
(out / "inputs").mkdir(parents=True, exist_ok=True)
shutil.copy(FIX / "inputs/prompts.jsonl", out / "inputs/prompts.jsonl")
src = [json.load(open(FIX / f"data1.7b/audit/ao_s{s}.json")) for s in (0, 1)]
for s in range(3):
    a = src[s % 2]
    rng = random.Random(s)
    u, p = a["per_prompt_lo"]["unprobed_fp32"], a["per_prompt_lo"]["probed_fp32"]
    push = [pp - uu for pp, uu in zip(p, u)]
    # share of the push a blind top-k cut removes
    if scen == "yes" or (scen == "mixed" and s != 1):
        frac = {k: min(1.0, 0.13 * k + 0.02 * s) for k in range(1, 13)}
    else:
        frac = {k: min(1.0, 0.05 * k + 0.02 * s) for k in range(1, 13)}
    R = lambda v: [round(x, 4) for x in v]
    sk = lambda c: sum(x > 0 and y < 0 for x, y in zip(R(c), R(u))) / 128
    if scen == "faded" and s == 2:
        p = [uu + 0.2 * d * rng.random() for uu, d in zip(u, push)]
    tc, curve = {}, []
    for k in range(1, 13):
        c = [pp - (frac[k] + rng.gauss(0, 0.08)) * d for pp, d in zip(p, push)]
        tc[f"greedy_TB_k{k}"] = {"lo32": R(c), "strict32": sk(c)}
        curve.append(sk(c))
    dr = []
    for j in range(50):
        c = [pp - rng.uniform(0, 0.6) * d for pp, d in zip(p, push)]
        tc[f"rand8_{j:02d}"] = {"lo32": R(c), "strict32": sk(c)}
        dr.append(sk(c))
    import statistics
    hl = {"strict32_after_greedy_k": {"TB": curve, "T16": curve}, "strict32_random_min_median_max": {"8": [min(dr), statistics.median(dr), max(dr)]}}
    json.dump({"run": f"k6_s{s}", "test_ids": a["test_ids"], "headline": hl, "test_cut": tc,
               "test_base": {"lo_unprobed32": R(u), "lo_probed32": R(p), "strict32": sk(p)}},
              open(out / f"data/kaudit/k6_s{s}.json", "w"))
print("mock", scen, out)
