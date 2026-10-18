"""Batched local probes: run many short texts through the model once and read all 100 neurons.

    uv run --with mlx-lm python projects/polysemantic-100/probe.py run    # data/probe_texts.json -> data/probe_acts.json

Input data/probe_texts.json: list of {"key": str, "text": str}.
Output data/probe_acts.json: {key: {"max": [K], "min": [K], "argmax_tok": [K], "argmin_tok": [K]}}
max/min = largest / smallest a_j over token positions >= 1 (position 0 is the sink token,
skipped like in extract.py). Every text is read on ALL neurons, so a hypothesis written for
neuron A can be scored on neuron B for free (the swap null).
"""
from __future__ import annotations

import json
import sys

import numpy as np

from common import DATA, column_order, gather, load_model, load_neurons, set_record, wait_for_memory


def run_texts(model, tok, taps, order, texts: list[str], batch: int = 16, max_len: int = 256):
    import mlx.core as mx

    set_record(taps.values(), True)
    res = []
    ids_all = [tok.encode(t)[:max_len] for t in texts]
    idx = np.argsort([len(x) for x in ids_all])  # length-sorted batches -> little padding
    out = [None] * len(texts)
    for b in range(0, len(idx), batch):
        if (b // batch) % 25 == 0:
            wait_for_memory()
        sel = idx[b : b + batch]
        L = max(len(ids_all[i]) for i in sel)
        pad = tok.pad_token_id or 0
        x = np.full((len(sel), L), pad, dtype=np.int32)
        valid = np.zeros((len(sel), L), dtype=bool)
        for r, i in enumerate(sel):
            x[r, : len(ids_all[i])] = ids_all[i]
            valid[r, 1 : len(ids_all[i])] = True  # right padding is safe under a causal mask
        mx.eval(model(mx.array(x)))
        a = gather(taps, order)  # [B, L, K]
        hi = np.where(valid[..., None], a, -np.inf)
        lo = np.where(valid[..., None], a, np.inf)
        for r, i in enumerate(sel):
            am, an = hi[r].argmax(0), lo[r].argmin(0)
            out[i] = {
                "max": np.round(hi[r].max(0), 4).tolist(),
                "min": np.round(lo[r].min(0), 4).tolist(),
                "argmax_tok": [tok.decode([int(x[r, p])]) for p in am],
                "argmin_tok": [tok.decode([int(x[r, p])]) for p in an],
            }
    set_record(taps.values(), False)
    return out


def main() -> None:
    neurons = load_neurons()
    items = json.loads((DATA / "probe_texts.json").read_text())
    model, tok, taps, _ = load_model(neurons)
    order = column_order(neurons, taps)
    outs = run_texts(model, tok, taps, order, [it["text"] for it in items])
    (DATA / "probe_acts.json").write_text(
        json.dumps({it["key"]: o for it, o in zip(items, outs)}, ensure_ascii=False))
    print("probed", len(items), "texts on", len(neurons), "neurons")


if __name__ == "__main__":
    {"run": main}[sys.argv[1]]()
