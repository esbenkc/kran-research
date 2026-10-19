"""Shared helpers: config, model, tokenisation, mean-pooled activations, probe I/O.

Conventions (see config.yaml; frozen before the first attack run):
- Probed activation h(x) = hidden_states[LAYER] (residual stream after block LAYER-1), mean over the real
  tokens 1..n-1 of the statement (token 0 skipped: attention-sink outlier), float32.
- Probe score s(x) = w.h(x) + b, logistic regression fit on clean activations; positive class = FALSE statement.
"""
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
DATA = ROOT / "data"
MODEL = CFG["model"]
LAYER = CFG["layer"]
MAXTOK = CFG["max_tokens"]


def device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def load_model(dev, dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype)
    model.eval().to(dev)
    return tok, model


def tokenize(tok, texts, maxtok=MAXTOK):
    return [tok(t, truncation=True, max_length=maxtok, add_special_tokens=False)["input_ids"] for t in texts]


def pad(seqs, pad_id, dev):
    L = max(len(s) for s in seqs)
    inp = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for r, s in enumerate(seqs):
        inp[r, :len(s)] = torch.tensor(s)
        mask[r, :len(s)] = 1
    return inp.to(dev), mask.to(dev)


def pool(h, mask):
    """Mean over real positions 1..n-1. h: (B,T,d) any dtype -> float32 (B,d)."""
    m = mask.float().clone()
    m[:, 0] = 0
    return (h.float() * m[..., None]).sum(1) / m.sum(1, keepdim=True)


class _Stop(Exception):
    pass


def hidden(model, inp, mask):
    """hidden_states[LAYER] = output of decoder block LAYER-1; the later blocks are skipped (they cannot change it)."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    box = {}

    def hook(_m, _i, out):
        box["h"] = out[0] if isinstance(out, tuple) else out
        raise _Stop

    hd = base.model.layers[LAYER - 1].register_forward_hook(hook)
    try:
        base.model(input_ids=inp, attention_mask=mask)
    except _Stop:
        pass
    finally:
        hd.remove()
    return box["h"]


@torch.no_grad()
def pooled_acts(model, ids, pad_id, dev, bs=64):
    order = np.argsort([len(x) for x in ids])
    X = None
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        inp, mask = pad([ids[i] for i in idx], pad_id, dev)
        H = pool(hidden(model, inp, mask), mask).cpu().numpy()
        if X is None:
            X = np.zeros((len(ids), H.shape[1]), dtype=np.float32)
        X[idx] = H
    return X


def hidden_multi(model, inp, mask, layers):
    """{L: hidden_states[L]} for every L in layers; blocks after max(layers) are skipped."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    box, top = {}, max(layers)

    def mk(L):
        def hook(_m, _i, out):
            box[L] = out[0] if isinstance(out, tuple) else out
            if L == top:
                raise _Stop
        return hook

    hds = [base.model.layers[L - 1].register_forward_hook(mk(L)) for L in layers]
    try:
        base.model(input_ids=inp, attention_mask=mask)
    except _Stop:
        pass
    finally:
        for h in hds:
            h.remove()
    return box


def token_offsets(ids):
    """Statement i owns rows off[i]:off[i+1] of a token array = its real tokens 1..n-1 (token 0 skipped)."""
    return np.concatenate([[0], np.cumsum([len(x) - 1 for x in ids])]).astype(np.int64)


@torch.no_grad()
def token_acts(model, ids, pad_id, dev, layers, bs=64):
    """Per-token activations of tokens 1..n-1 of every statement, float32, {L: (n_tokens_total, d)} + offsets.
    Right padding + causal attention: the real positions do not see the pads."""
    off = token_offsets(ids)
    order = np.argsort([len(x) for x in ids])
    out = {}
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        inp, mask = pad([ids[i] for i in idx], pad_id, dev)
        H = hidden_multi(model, inp, mask, layers)
        for L in layers:
            h = H[L].float().cpu().numpy()
            if L not in out:
                out[L] = np.zeros((off[-1], h.shape[-1]), dtype=np.float32)
            for r, i in enumerate(idx):
                out[L][off[i]:off[i + 1]] = h[r, 1:len(ids[i])]
    return out, off


def load_split():
    return json.loads((DATA / "split.json").read_text())


def load_probe():
    z = np.load(DATA / "probe.npz")
    return z["w"].astype(np.float64), float(z["b"])


def threshold(neg_scores, fpr=0.01):
    """Threshold from n negatives (plan #2705): flag iff score > t, t = the k-th highest, k = floor(fpr*(n+1)).
    n = 200, fpr 0.01: t = 2nd-highest, expected FPR of a fresh negative = 2/201 = 1.0%.
    n = 200, fpr 0.05: t = 10th-highest (10/201)."""
    s = np.sort(np.asarray(neg_scores, dtype=np.float64))[::-1]
    k = int(np.floor(fpr * (len(s) + 1)))
    return float(s[k - 1])
