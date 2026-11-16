"""Paired-cell input (Tove #22698, exploratory): turn Iris's NDIF block-34 input for all 646 prompts (CUDA bf16, built
05:58 PST in 11 batched traces) into the rl.py cache layout raw/cache_qwen3-8b_ndif/. No model is loaded here; the
paired run solves b on this cache's own train split (rl.py --x-offset solve). Also writes data/ndif_cache_check.json:
the sha256 of the source file and the per-prompt rel. err of our streamed cache against it (all 646 prompts).

    env/.venv/bin/python ndif_cache.py <path to block34_input_all646_ndif_bf16.pt>
    run: CSL_MODEL=Qwen/Qwen3-8B CSL_CACHE=cache_qwen3-8b_ndif rl.py ... --x-offset solve --out-sub runs_explore
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).parent

src = Path(sys.argv[1])
sha = hashlib.sha256(src.read_bytes()).hexdigest()
d = torch.load(src, weights_only=False)
pid = [json.loads(l)["id"] for l in open(HERE / "data" / "prompts.jsonl")]
assert list(d["ids"]) == pid, "NDIF ids must be in prompts.jsonl order"
out = HERE / "raw" / "cache_qwen3-8b_ndif"
out.mkdir(parents=True, exist_ok=True)
u16 = d["h34"].contiguous().view(torch.int16).numpy().view(np.uint16).copy()
# rl.py writes ONE cue-on rows 0-2 block (prompt 0's rows + cue) into every cue-on prompt, which assumes rows 0-2 are
# the same for all prompts (true for our MPS cache). In the NDIF cache 6 of 646 prompts differ at rows 1-2 (rel. up to
# 0.018); set them to prompt 0's rows so cue-on vs cue-off differs only by the planted cue (logged below).
n_rows_fixed = int((~(u16[:, :3] == u16[:1, :3]).all(axis=(1, 2))).sum())
u16[:, :3] = u16[:1, :3]
np.save(out / "eager_rec_tail.npy", u16)
json.dump({"model": "Qwen/Qwen3-8B", "tail": 34, "prompt_ids": pid, "builder": f"NDIF (Iris), {src.name} sha256 {sha}"},
          open(out / "meta.json", "w"), indent=1)
ours = torch.from_numpy(np.load(HERE / "raw" / "cache_qwen3-8b" / "eager_rec_tail.npy").view(np.int16)).view(torch.bfloat16).float()
nd = d["h34"].float()
rel = ((ours - nd).flatten(1).norm(dim=1) / nd.flatten(1).norm(dim=1)).numpy()
rel_rows = ((ours - nd).norm(dim=-1) / nd.norm(dim=-1))  # [646, 34]
res = {"source": src.name, "sha256": sha, "n": len(pid), "rel_err_max": float(rel.max()), "rel_err_median": float(np.median(rel)),
       "rows012_identical_ndif": bool((d["h34"][:, :3] == d["h34"][:1, :3]).all()), "rows012_set_to_prompt0": n_rows_fixed,
       "rel_err_per_token_median": float(rel_rows.median()), "rel_err_per_token_max": float(rel_rows.max()),
       "rel_err_rows012_max": float(rel_rows[:, :3].max())}
json.dump(res, open(HERE / "data" / "ndif_cache_check.json", "w"), indent=1)
print(res)
