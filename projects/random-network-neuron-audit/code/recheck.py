"""Second pass over the selected units (after export.py).

1) Arm C (prereg): for each C unit's top-20 examples, the share of |contribution| carried by its top-3 neurons,
   contribution_j = z_j * d_j with z = (a - mu) / sd. A high share means the direction is close to a few neurons.
2) Receipt check: recompute every arm-A and arm-C top-20 activation from scratch and compare with the dump.
Writes data/recheck.json.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM

D = os.environ.get("AUDIT_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
ACTS = os.path.join(D, "acts")


@torch.no_grad()
def main():
    meta = json.load(open(os.path.join(ACTS, "meta.json")))
    toks = np.load(os.path.join(ACTS, f"tokens_{meta['n_tok']}.npy"))
    units = [json.loads(l) for l in open(os.path.join(D, "contexts.jsonl"))]
    units = [u for u in units if u["arm"] in ("A", "C")]
    zc = np.load(os.path.join(ACTS, "armC.npz"))
    li = {L: i for i, L in enumerate(meta["layers"])}
    need = defaultdict(list)  # seq -> [(unit_index, example_index, pos)]
    for ui, u in enumerate(units):
        for ei, e in enumerate(u["examples"]):
            need[e["seq"]].append((ui, ei, e["pos"]))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(meta["model"], dtype=getattr(torch, meta["dtype"])).to(dev).eval()
    model.model.layers = model.model.layers[: max(meta["layers"]) + 1]
    store = {}
    for L in meta["layers"]:
        model.model.layers[L].mlp.down_proj.register_forward_pre_hook(lambda m, i, L=L: store.__setitem__(L, i[0]))
    res = {u["uid"]: dict(arm=u["arm"], layer=u["layer"], unit=u["unit"], recomputed=[None] * 20, share_top3=[None] * 20)
           for u in units}
    seqs = sorted(need)
    bs = 8
    for b0 in range(0, len(seqs), bs):
        chunk = seqs[b0:b0 + bs]
        model.model(input_ids=torch.from_numpy(toks[chunk]).long().to(dev), use_cache=False)
        acts = {L: store[L].float().cpu().numpy() for L in meta["layers"]}
        for bi, s in enumerate(chunk):
            for ui, ei, P in need[s]:
                u = units[ui]
                a = acts[u["layer"]][bi, P]
                r = res[u["uid"]]
                if u["arm"] == "A":
                    r["recomputed"][ei] = float(a[u["unit"]])
                else:
                    i = li[u["layer"]]
                    c = (a - zc["mu"][i]) / zc["sd"][i] * zc["dirs"][i][u["unit"]]
                    r["recomputed"][ei] = float(c.sum())
                    ac = np.sort(np.abs(c))[::-1]
                    r["share_top3"][ei] = float(ac[:3].sum() / ac.sum())
        store.clear()
    for u in units:
        r = res[u["uid"]]
        d = np.array([e["act"] for e in u["examples"]])
        rc = np.array(r["recomputed"])
        r["max_rel_err"] = float(np.max(np.abs(rc - d) / np.maximum(np.abs(d), 1e-6)))
        if u["arm"] == "C":
            r["share_top3_mean"] = float(np.mean(r["share_top3"]))
    out = dict(meta=dict(note="recomputed from scratch, same dtype; A: neuron activation, C: z-scored projection"),
               summary=dict(
                   A_max_rel_err_median=float(np.median([r["max_rel_err"] for r in res.values() if r["arm"] == "A"])),
                   C_max_rel_err_median=float(np.median([r["max_rel_err"] for r in res.values() if r["arm"] == "C"])),
                   C_share_top3_median=float(np.median([r["share_top3_mean"] for r in res.values() if r["arm"] == "C"])),
                   C_share_top3_max=float(np.max([r["share_top3_mean"] for r in res.values() if r["arm"] == "C"]))),
               units=res)
    json.dump(out, open(os.path.join(D, "recheck.json"), "w"), indent=0)
    print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
