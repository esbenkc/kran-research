"""Figure 1 data (Owain): for the first 2 prompts of held-out set A, the attention rows of queries 63, 255 and 511
in every layer and head, for the arms base, P (band zeroed) and R02 (a random 3-pair band, no overlap with B).
Output: data/<tag>/fig1_rows.npz  rows[arm] = [2 prompts, 28 layers, 16 heads, 3 queries, 512 keys] (float16)."""
import json
import sys

import numpy as np
import torch

import rsb

repo = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B"
tag = repo.split("/")[-1]
OUT = rsb.DATA / tag
band = json.load(open(OUT / "band.json"))
tok, model = rsb.load_model(repo)
x, docs = rsb.prompts(tok, "A", tag)
x = x[:2]
ROWS = [63, 255, 511]
rsb.LASTROW.update(on=True, rows=ROWS)
out = {}
for name, arm in (("base", ("clean", None)), ("P", ("zero", band["B"])), ("R02", ("zero", band["R"][1]))):
    rsb.run(model, x, arm, bs=2, lens=False, loss=False)
    out[name] = torch.stack([rsb.LASTROW[l] for l in range(28)], 1).numpy().astype(np.float16)
rsb.LASTROW["on"] = False
text = [tok.decode(x[i].tolist()) for i in range(2)]
toks = [[tok.decode([int(t)]) for t in x[i]] for i in range(2)]
np.savez_compressed(OUT / "fig1_rows.npz", rows_queries=np.array(ROWS), docs=np.array(docs[:2]),
                    R02=np.array(band["R"][1]), B=np.array(band["B"]), **out)
json.dump({"docs": docs[:2], "tokens": toks, "B": band["B"], "R02": band["R"][1]}, open(OUT / "fig1_tokens.json", "w"))
print({k: v.shape for k, v in out.items()}, "R02", band["R"][1])
