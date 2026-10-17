"""Shared helpers: model, tokenisation, mean-pooled layer-3 activations, probe I/O.

Conventions (frozen 2026-10-17, before any attack run):
- Model: EleutherAI/pythia-70m, revision step143000, float32.
- Probed layer: hidden_states[3] = residual stream after block 2 (blocks 0,1,2 are what the attacker trains).
- Text -> first MAXTOK=64 tokens; mean-pool positions 1..n-1 of the real (unpadded) tokens.
  Position 0 is skipped (attention-sink outliers). Right padding with EOS; causal attention keeps
  real positions independent of the padding.
"""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "EleutherAI/pythia-70m"
REVISION = "step143000"
LAYER = 3
MAXTOK = 64
ROOT = Path(__file__).parent
DATA = ROOT / "data"


def device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def load_model(dev=None):
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.float32)
    model.eval()
    if dev is not None:
        model.to(dev)
    return tok, model


def tokenize(tok, texts):
    """List of token-id lists, first MAXTOK tokens of each text."""
    return [tok(t, truncation=True, max_length=MAXTOK)["input_ids"] for t in texts]


def batches(ids, bs, eos):
    """Yield (index array, input_ids, attention mask) batches, length-sorted for speed."""
    order = np.argsort([len(x) for x in ids])
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        L = max(len(ids[i]) for i in idx)
        inp = torch.full((len(idx), L), eos, dtype=torch.long)
        mask = torch.zeros((len(idx), L), dtype=torch.float32)
        for r, i in enumerate(idx):
            inp[r, :len(ids[i])] = torch.tensor(ids[i])
            mask[r, :len(ids[i])] = 1
        yield idx, inp, mask


def pool(h, mask):
    """Mean over positions 1..n-1 of real tokens. h: (B,T,d), mask: (B,T)."""
    m = mask.clone()
    m[:, 0] = 0
    return (h * m[..., None]).sum(1) / m.sum(1, keepdim=True)


def layer3(model, inp, mask):
    out = model.gpt_neox(input_ids=inp, attention_mask=mask.long(), output_hidden_states=True)
    return out.hidden_states[LAYER]


@torch.no_grad()
def pooled_acts(model, ids, eos, dev, bs=64):
    X = np.zeros((len(ids), model.config.hidden_size), dtype=np.float32)
    for idx, inp, mask in batches(ids, bs, eos):
        inp, mask = inp.to(dev), mask.to(dev)
        X[idx] = pool(layer3(model, inp, mask), mask).float().cpu().numpy()
    return X


def load_probes():
    """Probe table: list of dicts with name, source, label, w (raw-space weight), b."""
    meta = json.loads((DATA / "probes" / "probes.json").read_text())
    W = np.load(DATA / "probes" / "probe_weights.npz")
    for p in meta:
        p["w"] = W["W"][p["idx"]]
        p["b"] = float(W["b"][p["idx"]])
    return meta
