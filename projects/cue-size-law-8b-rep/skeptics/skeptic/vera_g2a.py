"""Vera: G2(a) per row and whole prompt, from the mini's cache file and the parent's saved NDIF block-34 input."""
import json, sys, hashlib, numpy as np, torch
P = sys.argv[1]
C = f"{P}/raw/cache_qwen3-8b"
meta = json.load(open(f"{C}/meta.json"))
raw = np.load(f"{C}/eager_rec_tail.npy", mmap_mode="r")
sha = hashlib.sha256(open(f"{C}/eager_rec_tail.npy", "rb").read()).hexdigest()
nd = torch.load(f"{P}/ref/ndif/dev_block34_ndif.pt", map_location="cpu", weights_only=False)
ids = meta["prompt_ids"]
nd_ids = nd["ids"]
H = torch.from_numpy(np.ascontiguousarray(raw).view(np.int16)).view(torch.bfloat16).float() if raw.dtype == np.uint16 else torch.from_numpy(np.asarray(raw)).float()
H = H.reshape(len(ids), 34, 4096)
rows = torch.stack([H[ids.index(i)] for i in nd_ids])
ref = nd["h34"].float()
whole = ((rows - ref).flatten(1).norm(dim=1) / ref.flatten(1).norm(dim=1)).numpy()
tok = ((rows - ref).norm(dim=2) / ref.norm(dim=2)).numpy()  # [64, 34]
# lnpx_ndif cross-check against the saved NDIF px; ties from saved topv
g = json.load(open(f"{P}/data/gates.json"))["G2"]["per_prompt"]
px = nd["px"].double().numpy()
ties = (nd["topv"][:, 0] == nd["topv"][:, 1]).numpy()
print(json.dumps({
    "cache_sha256": sha, "parent_sha_match": sha.startswith("70c4703a"), "raw_dtype": str(raw.dtype), "shape": list(raw.shape),
    "whole_prompt_rel_err_max": float(whole.max()), "whole_prompt_rel_err_median": float(np.median(whole)),
    "per_token_rel_err_median": float(np.median(tok)), "per_token_rel_err_max": float(tok.max()),
    "rows0_2_rel_err_max": [float(tok[:, r].max()) for r in range(3)],
    "rows0_2_rel_err_median": [float(np.median(tok[:, r])) for r in range(3)],
    "n_tokens_over_1e-2": int((tok > 1e-2).sum()), "of": int(tok.size),
    "lnpx_ndif_vs_saved_px_maxabs": float(np.abs(np.log(px) - np.array(g["lnpx_ndif"])).max()),
    "ndif_exact_ties": [nd_ids[i] for i in np.where(ties)[0]],
}, indent=1))
