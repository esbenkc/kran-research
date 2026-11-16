"""Laptop job step 2 (PREREG G2: 'The laptop job runs (a) on its own cache as its first step and aborts on a fail').
Compares this machine's streamed cache with NDIF's block-34 input on the 64 dev prompts (rel. err < 1e-2, = G2(a)) and
with the mini's cache by sha256 (data/cache_sha256.json). Exit 0 = pass; exit 1 = G2(a) fail (the job aborts).
Prints 'same' or 'differs' for the sha check on the last line (the job then uses --x-offset gates or solve).

    python cache_check.py <out json>
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).parent
c = HERE / "raw" / "cache_qwen3-8b"
meta = json.load(open(c / "meta.json"))
a = np.load(c / "eager_rec_tail.npy")
sha = hashlib.sha256(a.tobytes()).hexdigest()
mini = json.load(open(HERE / "data" / "cache_sha256.json"))["qwen3-8b"]
nd = torch.load(HERE / "data" / "ndif" / "dev_block34_ndif.pt", weights_only=False)
rows = [meta["prompt_ids"].index(i) for i in nd["ids"]]
ours = torch.from_numpy(a[rows].view(np.int16)).view(torch.bfloat16).float()
th = nd["h34"].float()
rel = ((ours - th).flatten(1).norm(dim=1) / th.flatten(1).norm(dim=1)).numpy()
tok = ((ours - th).norm(dim=-1) / th.norm(dim=-1))  # per (prompt, token) rel. err (Kasper #22722: the unit matters)
res = {"per_token_rel_err_median": float(tok.median()), "per_token_rel_err_max": float(tok.max()),
       "whole_prompt_rel_err_unit": "||ours - ndif|| / ||ndif|| over all 34 x 4096 of one prompt (row 0 dominates)",
       "sha256": sha, "sha256_mini": mini, "same_as_mini": sha == mini, "g2a_rel_err_max": float(rel.max()),
       "g2a_rel_err_median": float(np.median(rel)), "g2a_pass": bool(rel.max() < 1e-2)}
Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
print(json.dumps(res))
print("same" if res["same_as_mini"] else "differs")
sys.exit(0 if res["g2a_pass"] else 1)
