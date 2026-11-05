"""Held-out contexts per PREREG s3 (frozen 4f7b1dc).

pile-10k docs from index 8000 upward (plain index order) with >= 128 tokens under the model's own tokenizer.
Context = first 128 tokens (Qwen3, SmolLM2: no BOS), GPT-2: BOS 50256 + first 127 tokens.
Set A = first 200 usable kept docs, B = next 200, C = next 200 (kept = not all-flagged, PREREG s10). Calibration (GPT-2 fixed-u) = docs 7500-7599 (usable).
Excluded: 9200-9209, 9300-9309, 100-139, 5000-7212 (never reached from 8000 upward except 9200+/9300+, skipped).
Read position: rng = default_rng(1105 + doc_index); t = rng.integers(4, 127); redraws (massive-activation rule,
run.py) take the next rng.integers(4, 127) from the same rng, at most 10 draws.
"""
import numpy as np
import torch

EXCL = set(range(9200, 9210)) | set(range(9300, 9310)) | set(range(100, 140)) | set(range(5000, 7213))
T = 128


def _text():
    from datasets import load_dataset
    return load_dataset("NeelNanda/pile-10k", split="train")["text"]


def contexts(tokenizer, kind, start=8000, n=600, stop=None):
    """Returns (doc_index list, LongTensor [n, 128]). kind 'gpt2' adds BOS 50256 + 127 tokens."""
    text = _text()
    docs, rows = [], []
    i = start
    while len(docs) < n and i < (stop or len(text)):
        if i not in EXCL:
            ids = tokenizer(text[i][: 16 * T])["input_ids"]
            if len(ids) < T and len(text[i]) > 16 * T:
                ids = tokenizer(text[i])["input_ids"]
            if len(ids) >= T:  # usable rule: >= 128 tokens under the model's own tokenizer
                rows.append([50256] + ids[: T - 1] if kind == "gpt2" else ids[:T])
                docs.append(i)
        i += 1
    return docs, torch.tensor(rows, dtype=torch.long)


def candidates(tokenizer, kind, n=660):
    """Usable docs from 8000 upward (plain index order); run.py --stage select drops all-flagged docs
    (PREREG s10) and cuts sets A/B/C = the first 200/200/200 kept docs."""
    return contexts(tokenizer, kind, 8000, n)


def smoke(tokenizer, kind):
    return contexts(tokenizer, kind, 7600, 16, stop=7700)  # smoke set S (code check only, never reported)


def calibration(tokenizer, kind):
    return contexts(tokenizer, kind, 7500, 100, stop=7600)


def draws(doc_index, k=10):
    rng = np.random.default_rng(1105 + doc_index)
    return [int(rng.integers(4, 127)) for _ in range(k)]
