"""Which reading of the clean-label caveat holds? Share of judge-DECEPTIVE answers that still contain the fact's words
(string check says honest = 'spin lies'), and how often the second judge (haiku) also calls those test answers deceptive.
    cd kran-research && uv run python ../swarm-oracle-1019-refit-probe/skeptic/vera_spin_lies.py"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vera_rebuild import D
for p in ["base", "lam1_s0", "lam1_s1", "ens_s0", "ens_s1"]:
    L = [json.loads(l) for l in open(D / f"labels_{p}.jsonl")]
    dec = [r for r in L if r["y_judge"] == 1]
    spin = [r for r in dec if r["y_string"] == 0]
    t = [r for r in spin if r["split"] == "test" and r["y_judge2"] is not None]
    print(p, "spin share of judge-deceptive %.2f (%d/%d) | haiku agrees deceptive on test spin lies %.2f (n=%d)"
          % (len(spin) / len(dec), len(spin), len(dec), sum(r["y_judge2"] == 1 for r in t) / len(t), len(t)))
