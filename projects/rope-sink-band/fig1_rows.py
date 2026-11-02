"""Figure 1 data (Owain #12670): held-out set A prompt index 3 (pile-10k doc 4288), attention rows of queries
16, 64, 256, 511 in every layer and head, for the arms base, P (band zeroed) and R12 (pairs 0,12,33: the random band
with the median cut on set A). Whole-model forward, same harness as heldout.py.
Output: data/<tag>/fig1_rows.npz  <arm> = [1 prompt, 28 layers, 16 heads, 4 queries, 512 keys] (float16), ids."""
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
IDX = 3
x = x[IDX:IDX + 1]
ROWS = [16, 64, 256, 511]
rsb.LASTROW.update(on=True, rows=ROWS)
out = {}
for name, arm in (("base", ("clean", None)), ("P", ("zero", band["B"])), ("R12", ("zero", band["R"][11]))):
    rsb.run(model, x, arm, bs=1, lens=False, loss=False)
    out[name] = torch.stack([rsb.LASTROW[l] for l in range(28)], 1).numpy().astype(np.float16)
rsb.LASTROW["on"] = False
toks = [tok.decode([int(t)]) for t in x[0]]
np.savez_compressed(OUT / "fig1_rows.npz", queries=np.array(ROWS), doc=docs[IDX], ids=x[0].numpy(),
                    pairs_R12=np.array(band["R"][11]), pairs_B=np.array(band["B"]), **out)
json.dump({"set": "A", "index": IDX, "doc": docs[IDX], "tokens": toks, "B": band["B"], "R12": band["R"][11], "queries": ROWS},
          open(OUT / "fig1_tokens.json", "w"))
print({k: v.shape for k, v in out.items()}, "doc", docs[IDX], "R12", band["R"][11])
