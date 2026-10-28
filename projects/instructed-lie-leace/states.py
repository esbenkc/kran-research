"""Answer-position states and output-lens features (shared by gate0b.py, attack.py, leace_price.py).

capture(): (n, 20, d) float32 states at the answer position (last prompt token) at readouts 10..29, plus the
answer-position log-softmax over the country first tokens (the output-only probe O = O_29 feature).
lens(): the output-lens features of PREREG s2 G0c: final RMSNorm(state at r) @ W_U[country tokens]^T, for every r.
For r = 29 the state is already normed, so lens(29) = the country-token logits themselves.
"""
import numpy as np
import torch

from common import PANEL, POSTNORM, cities, pad


def country_token_ids(tok):
    """First token of ' <country>' and of ' the <country>' style names (as the model would start the answer)."""
    names = sorted({c for _, c in cities()})
    ids = set()
    for n in names:
        ids.add(tok(" " + n, add_special_tokens=False)["input_ids"][0])
        if n.startswith("the "):
            ids.add(tok(" " + n[4:], add_special_tokens=False)["input_ids"][0])
    return np.array(sorted(ids))


@torch.no_grad()
def capture(hk, ids, idx, pad_id, dev, bs=48):
    """hk: common.Hooks on the model. Returns H (len(idx), 20, d) float32 in the order of idx."""
    d = hk.inner.config.hidden_size
    out = np.zeros((len(idx), len(PANEL), d), np.float32)
    order = np.argsort([len(ids[i]) for i in idx], kind="stable")
    for s in range(0, len(order), bs):
        o = order[s:s + bs]
        inp, mask = pad([ids[idx[j]] for j in o], pad_id, dev)
        hk.run(inp, mask)
        ar = torch.arange(len(o), device=dev)
        ap = torch.tensor([len(ids[idx[j]]) - 1 for j in o], device=dev)
        out[o] = torch.stack([hk.box[R][ar, ap] for R in PANEL], 1).float().cpu().numpy()
        hk.box = {}
    if dev.type == "mps":
        torch.mps.empty_cache()
    return out


@torch.no_grad()
def lens(hk, H, ctok, dev, bs=256):
    """Output-lens features (n, 20, n_country_tokens): norm(H_r) @ W_U[ctok]^T (r = 29 is already normed)."""
    W = hk.lm_head.weight[torch.as_tensor(ctok, device=dev)].float()
    norm = hk.inner.norm
    out = np.zeros((H.shape[0], H.shape[1], len(ctok)), np.float32)
    for s in range(0, H.shape[0], bs):
        x = torch.tensor(H[s:s + bs], device=dev)
        for k, R in enumerate(PANEL):
            h = x[:, k] if R == POSTNORM else norm(x[:, k].to(norm.weight.dtype)).float()
            out[s:s + bs, k] = (h @ W.T).cpu().numpy()
    return out
