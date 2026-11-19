"""Shared helpers for max-pool-8b (the 1.5B max-pool-survives code, ported to Qwen3-8B).

Conventions (as the parent; config.yaml):
- LAYER = round(14 n_layers / 28): hidden_states[LAYER] = output of decoder block LAYER-1 (18 for Qwen3-8B).
- Tokens: no chat template, no special tokens, right padding; position 0 is excluded from every pooling (sink token).
- Averaging probe score s(x) = w.h_mean(x) + b, h_mean = mean of hidden_states[LAYER] over tokens 1..n-1;
  positive class (y = 1) = FALSE statement.
- Splits by city (data/ref/split.json = the frozen 1019 split, copied unchanged).
Env: MP_MODEL overrides config.yaml model (smoke: Qwen/Qwen3-0.6B); MP_DATA overrides the data dir (smoke).
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
DATA = Path(os.environ.get("MP_DATA", ROOT / "data"))
REF = ROOT / "data" / "ref"   # split + pile windows, shared with any MP_DATA dir
MODEL = os.environ.get("MP_MODEL", CFG["model"])
MAXTOK = CFG["max_tokens"]
SPLITS = ["probe", "attack", "stop", "refit", "eval"]


def _n_layers():
    from transformers import AutoConfig
    return int(AutoConfig.from_pretrained(MODEL).num_hidden_layers)


N_LAYERS = _n_layers()
LAYER = int(round(CFG["layer_share"][0] * N_LAYERS / CFG["layer_share"][1]))


def device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def load_model(dev, adapter=None, dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype)
    model.eval().to(dev)
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter)).eval()
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


def hidden(model, inp, mask, layer=None):
    """hidden_states[layer] = output of decoder block layer-1; the later blocks are skipped (they cannot change it)."""
    layer = LAYER if layer is None else layer
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    box = {}

    def hook(_m, _i, out):
        box["h"] = out[0] if isinstance(out, tuple) else out
        raise _Stop

    hd = base.model.layers[layer - 1].register_forward_hook(hook)
    try:
        base.model(input_ids=inp, attention_mask=mask)
    except _Stop:
        pass
    finally:
        hd.remove()
    return box["h"]


@torch.no_grad()
def pooled_acts(model, ids, pad_id, dev, bs=64):
    """Mean-pooled hidden_states[LAYER] (tokens 1..n-1) of every statement, float32 (n, d)."""
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


def token_offsets(ids):
    """Statement i owns rows off[i]:off[i+1] of a token array = its real tokens 1..n-1 (token 0 skipped)."""
    return np.concatenate([[0], np.cumsum([len(x) - 1 for x in ids])]).astype(np.int64)


def country_pos(tok, ids):
    """Per statement: position of the last token of the country name = the token before the final full stop when the
    full stop is its own token, else the statement's last token (panel-fine-tune / 1020 d3.py rule)."""
    stop_own = np.array([tok.decode([x[-1]]) == "." for x in ids])
    return np.array([len(x) - 2 if s else len(x) - 1 for x, s in zip(ids, stop_own)]), stop_own


def load_split():
    return json.loads((REF / "split.json").read_text())


def rows():
    """y = 1 for a FALSE statement; city, split, src (cities / neg_cities template) per statement."""
    R = load_split()
    y = np.array([r["pos"] for r in R])
    assert (y == 1 - np.array([r["true"] for r in R])).all()
    return R, y, np.array([r["city"] for r in R]), np.array([r["split"] for r in R]), np.array([r["src"] for r in R])


def threshold(neg_scores, fpr=0.01):
    """flag iff score > t, t = the k-th highest of n negatives, k = floor(fpr*(n+1)); n = 200 -> 2nd-highest."""
    s = np.sort(np.asarray(neg_scores, dtype=np.float64))[::-1]
    k = max(1, int(np.floor(fpr * (len(s) + 1))))
    return float(s[k - 1])


def load_fits():
    """Clean-fit monitors (fit.py): mean (the averaging probe the attacker trains against), max (tokenwise),
    cty (country token), last (last token); w, b in raw activation coordinates, float64."""
    z = np.load(DATA / "fits" / "probes.npz")
    return {m: (z[f"{m}__w"].astype(np.float64), float(z[f"{m}__b"])) for m in ("mean", "max", "cty", "last")}
