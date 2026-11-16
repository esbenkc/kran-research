"""Vera: noise floor for G2. Compare NDIF's two block-34 inputs for the same 64 dev prompts:
dev_block34_ndif.pt (dev trace batches) vs block34_input_all646_ndif_bf16.pt (11 x 64 batches).
If they differ, CUDA batch composition alone moves the cache, and the same-size gap to OUR cache is the noise floor."""
import torch, json, sys
R = "/Users/esben/work/theory-1116-cue-size-law-8b"
dev = torch.load(f"{R}/kran-research/projects/cue-size-law-8b/data/ndif/dev_block34_ndif.pt", map_location="cpu")
allc = torch.load(f"{R}/swarm-theory-1116-cue-size-law-8b/ndif/out/block34_input_all646_ndif_bf16.pt", map_location="cpu")
print(type(allc), list(allc.keys()) if isinstance(allc, dict) else allc.shape)
print({k: (v.shape if hasattr(v, "shape") else len(v)) for k, v in dev.items()})
ids_all = allc["ids"] if isinstance(allc, dict) and "ids" in allc else None
H = allc["h34"] if isinstance(allc, dict) and "h34" in allc else allc
pos = [ids_all.index(i) for i in dev["ids"]] if ids_all is not None else None
A = dev["h34"].float()
B = H[pos].float() if pos is not None else None
d = (A - B)
rel_row = d.norm(dim=-1) / B.norm(dim=-1)  # per prompt per token
print("bitwise equal elements:", float((dev["h34"] == H[pos]).float().mean()))
print("rel err per token row: max %.5f median %.5f" % (rel_row.max(), rel_row.median()))
rel_prompt = d.flatten(1).norm(dim=1) / B.flatten(1).norm(dim=1)
print("rel err per prompt: max %.5f median %.5f" % (rel_prompt.max(), rel_prompt.median()))
json.dump(dict(bitwise=float((dev["h34"] == H[pos]).float().mean()), rel_row_max=float(rel_row.max()),
               rel_row_median=float(rel_row.median()), rel_prompt_max=float(rel_prompt.max()),
               rel_prompt_median=float(rel_prompt.median())), open("vera_g2_floor.json", "w"), indent=1)
