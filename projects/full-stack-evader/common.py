"""Shared helpers for full-stack-evader: config, model load, tokenisation, split, readouts, threshold rule, memmaps.

Readout index R = 0..29 of the per-token memmaps (data/acts/<state>/tok.npy):
  R = 0        token embedding (HF hidden_states[0])
  R = 1..28    output of decoder block R-1, BEFORE the final RMSNorm (the 1020 memmap convention; for R <= 27 this
               equals HF hidden_states[R]; R = 28 is the pre-norm block-27 output, NOT HF hidden_states[28])
  R = 29       HF hidden_states[28] = final RMSNorm(block-27 output) = the exact input of the logits ("post-norm L28")
Labels: y = 1 for a FALSE statement. Splits by city (the 1019 split, data/ref/split.json).
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
REF = DATA / "ref"
ACTS = DATA / "acts"   # per-state fp16 memmaps (not committed; data/acts_sha256.txt + extract.py rebuild them)
RUNS = DATA / "runs"
MODEL = CFG["model"]
MAXTOK = CFG["max_tokens"]
POSTNORM = 29          # readout index of HF hidden_states[28]


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
    """Right padding (as 1019/1020)."""
    L = max(len(s) for s in seqs)
    inp = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for r, s in enumerate(seqs):
        inp[r, :len(s)] = torch.tensor(s)
        mask[r, :len(s)] = 1
    return inp.to(dev), mask.to(dev)


def token_offsets(ids):
    """Statement i owns rows off[i]:off[i+1] of a token array = its real tokens 1..n-1 (token 0 skipped, as 1019)."""
    return np.concatenate([[0], np.cumsum([len(x) - 1 for x in ids])]).astype(np.int64)


def load_split():
    return json.loads((REF / "split.json").read_text())


def rows():
    R = load_split()
    y = np.array([1 - r["true"] for r in R])
    city = np.array([r["city"] for r in R])
    split = np.array([r["split"] for r in R])
    return R, y, city, split


def pool_idx():
    """Fit pool (probe+attack+stop, 1400), alarm-line set (200 refit TRUE statements), eval (1192, 298 cities)."""
    R, y, city, split = rows()
    tr = np.where(np.isin(split, ["probe", "attack", "stop"]))[0]
    line = np.where((split == "refit") & (y == 0))[0]
    ev = np.where(split == "eval")[0]
    return tr, line, ev


def country_pos(tok, ids):
    """Per statement: index (in the token sequence) of the last token of the country name = the token before the
    final full stop when the full stop is its own token, else the statement's last token (1020 d3.py rule)."""
    stop_own = np.array([tok.decode([x[-1]]) == "." for x in ids])
    return np.array([len(x) - 2 if s else len(x) - 1 for x, s in zip(ids, stop_own)]), stop_own


def threshold(neg_scores, fpr=0.01):
    """flag iff score > t, t = the k-th highest of n negatives, k = floor(fpr*(n+1)); n = 200 -> 2nd highest."""
    s = np.sort(np.asarray(neg_scores))[::-1]
    k = max(1, int(np.floor(fpr * (len(s) + 1))))
    return float(s[k - 1])


class Acts:
    """Read-only view of one state's per-token activations: tok (n_tok, 30, d) fp16 memmap, off (n_stmt+1,)."""

    def __init__(self, state):
        d = ACTS / state
        self.tok = np.load(d / "tok.npy", mmap_mode="r")
        self.off = np.load(d / "off.npy")
        self.n = len(self.off) - 1
        self.L = self.tok.shape[1]

    def mean(self, R, chunk=256):
        """(n_stmt, d) float32 mean over tokens 1..n-1 at readout R."""
        out = np.zeros((self.n, self.tok.shape[2]), np.float32)
        for s in range(0, self.n, chunk):
            e = min(self.n, s + chunk)
            X = np.asarray(self.tok[self.off[s]:self.off[e], R], np.float32)
            rel = self.off[s:e + 1] - self.off[s]
            out[s:e] = np.add.reduceat(X, rel[:-1], axis=0) / np.diff(rel)[:, None]
        return out

    def at(self, R, rows_):
        return np.asarray(self.tok[rows_, R], np.float32)
