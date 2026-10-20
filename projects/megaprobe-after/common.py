"""Shared helpers for megaprobe-after: config, model load, tokenisation, the 1019 split/probe, the threshold rule.

The model, tokenisation, batching and the frozen probe are the ones of max-pool-survives (theory-1019, branch
swarm/theory-1019-max-pool-survives), so the layer-14 readout here reproduces its logged scores.
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
REF = DATA / "ref1019"
ACTS = DATA / "acts"  # per-state fp16 memmaps (not committed; data/acts_sha256.txt + extract.py rebuild them)
MODEL = CFG["model"]
ATTACK_LAYER = CFG["attack_layer"]
MAXTOK = CFG["max_tokens"]
STATES = ["clean", "seed0", "seed1", "seed2"]


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
        model = PeftModel.from_pretrained(model, adapter).eval()
    return tok, model


def adapter_path(state):
    return None if state == "clean" else REF / "runs" / f"lam0__seed{state[-1]}__adapter"


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


def token_offsets(ids):
    """Statement i owns rows off[i]:off[i+1] of a token array = its real tokens 1..n-1 (token 0 skipped, as in 1019)."""
    return np.concatenate([[0], np.cumsum([len(x) - 1 for x in ids])]).astype(np.int64)


def load_split():
    return json.loads((REF / "split.json").read_text())


def load_probe():
    z = np.load(REF / "probe.npz")
    return z["w"].astype(np.float64), float(z["b"])


def threshold(neg_scores, fpr=0.01):
    """1019 rule: flag iff score > t, t = the k-th highest of n negatives, k = floor(fpr*(n+1)); n=200 -> 2nd highest."""
    s = np.sort(np.asarray(neg_scores, dtype=np.float64))[::-1]
    k = max(1, int(np.floor(fpr * (len(s) + 1))))
    return float(s[k - 1])


class Acts:
    """Read-only view of one state's per-token activations: tok (n_tok, 29, d) fp16 memmap, off (n_stmt+1,)."""

    def __init__(self, state):
        d = ACTS / state
        self.tok = np.load(d / "tok.npy", mmap_mode="r")
        self.off = np.load(d / "off.npy")
        self.n = len(self.off) - 1
        self.L = self.tok.shape[1]

    def pooled(self, layer, how, chunk=4096):
        """(n_stmt, d) float32: 'mean' over tokens 1..n-1, 'last' = the final token, 'max' = elementwise max."""
        d = self.tok.shape[2]
        if how == "last":
            return np.asarray(self.tok[self.off[1:] - 1, layer], np.float32)
        out = np.zeros((self.n, d), np.float32)
        s = 0
        while s < self.n:
            e = min(self.n, s + chunk // 16)
            X = np.asarray(self.tok[self.off[s]:self.off[e], layer], np.float32)
            rel = self.off[s:e + 1] - self.off[s]
            if how == "mean":
                out[s:e] = np.add.reduceat(X, rel[:-1], axis=0) / np.diff(rel)[:, None]
            else:
                out[s:e] = np.maximum.reduceat(X, rel[:-1], axis=0)
            s = e
        return out
