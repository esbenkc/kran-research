"""Base-vs-base output noise floor (Vera's phase-0 check): the UNTOUCHED model run twice with a different batch
composition (pass A: batch 64, length-sorted, as extract.py; pass B: batch 7, file order), per-token
KL(p_A || p_B) at the same positions extract.py uses (statement positions 1..n-1; pile positions 1..63).
-> data/results/noise.json.   Run: .venv/bin/python noise.py
"""
import json
import time

import numpy as np
import torch

from common import DATA, REF, device, load_model, load_split, pad, token_offsets, tokenize


def post(model, inp, mask):
    return model.model(input_ids=inp, attention_mask=mask).last_hidden_state


@torch.no_grad()
def main():
    t0 = time.time()
    dev = device()
    tok, model = load_model(dev)
    rows = load_split()
    ids = tokenize(tok, [r["text"] for r in rows])
    off = token_offsets(ids)
    ev = np.array([r["split"] == "eval" for r in rows])

    def run(order, bs):
        H = [None] * len(ids)
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            inp, mask = pad([ids[i] for i in idx], tok.pad_token_id, dev)
            h = post(model, inp, mask)
            for r, i in enumerate(idx):
                H[i] = h[r, 1:len(ids[i])].clone()
        return H

    A = run(np.argsort([len(x) for x in ids]), 64)
    B = run(np.arange(len(ids)), 7)
    kl = np.zeros(int(off[-1]), np.float32)
    for i in range(len(ids)):
        la = model.lm_head(A[i]).float().log_softmax(-1)
        lb = model.lm_head(B[i]).float().log_softmax(-1)
        kl[off[i]:off[i + 1]] = (la.exp() * (la - lb)).sum(-1).cpu().numpy()
    del A, B
    held = np.load(REF / "retain.npz")["heldout"]
    kp = []
    for s in range(0, len(held), 8):
        inp = torch.tensor(held[s:s + 8]).to(dev)
        ha = post(model, inp, torch.ones_like(inp))
        hb = torch.cat([post(model, inp[j:j + 3], torch.ones_like(inp[j:j + 3])) for j in range(0, len(inp), 3)])
        la = model.lm_head(ha).float().log_softmax(-1)
        lb = model.lm_head(hb).float().log_softmax(-1)
        kp.append((la.exp() * (la - lb)).sum(-1)[:, 1:].cpu().numpy())
    kp = np.concatenate(kp)
    evrows = np.concatenate([np.arange(off[i], off[i + 1]) for i in np.where(ev)[0]])
    out = {"what": "KL(p_A || p_B) per token, untouched model, two batch compositions (bf16 noise floor)",
           "eval_statements": {"mean": float(kl[evrows].mean()), "max": float(kl[evrows].max()),
                               "n_tokens": int(len(evrows))},
           "all_statements": {"mean": float(kl.mean()), "max": float(kl.max())},
           "pile_heldout": {"mean": float(kp.mean()), "max": float(kp.max()), "n_tokens": int(kp.size)},
           "seconds": round(time.time() - t0, 1)}
    (DATA / "results").mkdir(parents=True, exist_ok=True)
    (DATA / "results" / "noise.json").write_text(json.dumps(out, indent=1))
    np.save(DATA / "results" / "noise_kl_statements.npy", kl)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
