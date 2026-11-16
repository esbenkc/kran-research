"""Shared helpers for panel-fine-tune-8b (ported from the 1.5B panel-fine-tune): config, model load, tokenisation,
split, readouts, threshold rule, country-token activations.

Readout index R (n = number of decoder blocks; 36 for Qwen3-8B, 28 for Qwen3-0.6B and Qwen2.5-1.5B):
  R = 1..n     output of decoder block R-1, BEFORE the final RMSNorm (R = n is the pre-norm last-block output)
  R = n + 1    final RMSNorm(block n-1 output) = the exact input of the logits (POSTNORM)
PANEL = 19 pre-norm readouts round(r n / 28), r = 10..28, + POSTNORM (plan v1.2; n = 28 gives the 1.5B panel 10..29).
Activations: data/acts/<state>/cty.npy fp16 (n_stmt, P, d) = the state at the last country-name token at each PANEL
readout (only these are stored at 8B). Labels: y = 1 for a FALSE statement. Splits by city (data/ref/split.json,
the frozen 1019 split, copied unchanged from the 1.5B run).
Env: PFT_MODEL overrides config.yaml model (smoke on Qwen/Qwen3-0.6B); PFT_DATA overrides the data dir (smoke).
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
DATA = Path(os.environ.get("PFT_DATA", ROOT / "data"))
REF = ROOT / "data" / "ref"   # split + pile windows, shared with any PFT_DATA dir
ACTS = DATA / "acts"   # per-state fp16 country-token activations (not committed; sha256 in data/acts_sha256.txt)
RUNS = DATA / "runs"
MODEL = os.environ.get("PFT_MODEL", CFG["model"])
MAXTOK = CFG["max_tokens"]


def _n_layers():
    from transformers import AutoConfig
    return int(AutoConfig.from_pretrained(MODEL).num_hidden_layers)


N_LAYERS = _n_layers()
POSTNORM = N_LAYERS + 1   # readout index of the final-norm output (= logits input)
PANEL = sorted(set(int(round(r * N_LAYERS / 28)) for r in range(10, 29))) + [POSTNORM]


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
    """Read-only view of one state's country-token activations: cty (n_stmt, n_readouts, d) fp16; readouts.json lists
    the readouts (PANEL, or 1..n+1 for the clean state extracted with --all)."""

    def __init__(self, state):
        d = ACTS / state
        self.cty = np.load(d / "cty.npy", mmap_mode="r")
        self.readouts = json.loads((d / "readouts.json").read_text())
        assert set(PANEL) <= set(self.readouts), (self.readouts, PANEL)
        self.n = self.cty.shape[0]

    def at(self, R, rows_=None):
        """(n, d) float32 at readout R for statements rows_ (default all, split order)."""
        k = self.readouts.index(R)
        X = self.cty[:, k] if rows_ is None else self.cty[rows_, k]
        return np.asarray(X, np.float32)
