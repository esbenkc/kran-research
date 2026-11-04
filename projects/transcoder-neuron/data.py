"""Held-out text: NeelNanda/pile-10k (the transcoders were trained on OpenWebText, cfg dataset_path).
Each context = BOS (50256, as TransformerLens to_tokens adds and the transcoder training used) + the first
T-1 tokens of one document with >= T-1 tokens. Documents are taken in a fixed shuffled order (seed)."""
import numpy as np
import torch

_cache = {}


def _docs():
    if "tok" not in _cache:
        from datasets import load_dataset
        from transformers import AutoTokenizer

        ds = load_dataset("NeelNanda/pile-10k", split="train")
        _cache["tok"] = AutoTokenizer.from_pretrained("openai-community/gpt2")
        _cache["text"] = ds["text"]
    return _cache["tok"], _cache["text"]


def heldout_batch(n, T, seed=0, skip=0):
    """n contexts of length T; documents in permutation order of `seed`, skipping the first `skip` usable."""
    tok, text = _docs()
    order = np.random.default_rng(seed).permutation(len(text))
    out, used = [], 0
    for i in order:
        t = tok(text[i][:8 * T])["input_ids"]
        if len(t) < T - 1:
            continue
        if used >= skip:
            out.append([50256] + t[: T - 1])
            if len(out) == n:
                break
        used += 1
    return torch.tensor(out, dtype=torch.long)
