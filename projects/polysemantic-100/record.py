"""ONE shared forward pass over a mixed corpus. Streams the activations of all 100 sampled
neurons to disk (fp16 memmap), so no corpus x neurons matrix is held in RAM.

    uv run --with mlx-lm --with datasets python projects/polysemantic-100/record.py corpus
    uv run --with mlx-lm --with datasets python projects/polysemantic-100/record.py record

Outputs in data/:
  corpus.json         one row per window: source, language, text preview
  tokens.npy          int32 [W, ctx]
  acts.f16            float16 memmap [W, ctx, K]  a_j = silu(g_j) * u_j (column order = neurons.json)
  gate.f16            float16 memmap [W, ctx, K]  g_j (gate pre-activation)
  record_progress.txt windows done (the run resumes from here)
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter

import numpy as np

from common import CFG, DATA, load_neurons, mem_level, wait_for_memory

HF = "/Users/esben/.cache/huggingface/datasets"
PILE = f"{HF}/NeelNanda___pile-10k/default/0.0.0/127bfedcd5047750df5ccf3a12979a47bfa0bafa/pile-10k-train.arrow"
LANG = f"{HF}/papluca___language-identification/default/0.0.0/aa56583bf2bc52b0565770607d6fc3faebecf9e2/language-identification-train.arrow"


def build_corpus() -> None:
    from datasets import Dataset
    from transformers import AutoTokenizer

    from huggingface_hub import snapshot_download
    tok = AutoTokenizer.from_pretrained(snapshot_download(CFG["model"], local_files_only=True))
    c = CFG["corpus"]
    ctx = c["ctx"]
    rng = np.random.default_rng(0)
    rows, toks = [], []

    pile = Dataset.from_file(PILE)
    n_code = n_other = 0
    for i in rng.permutation(len(pile)):
        d = pile[int(i)]
        src = d["meta"]["pile_set_name"]
        is_code = src == "Github"
        if (is_code and n_code >= c["code_docs"]) or (not is_code and n_other >= c["pile_docs"]):
            continue
        ids = tok.encode(d["text"])
        if len(ids) < ctx:
            continue
        toks.append(ids[:ctx])
        rows.append({"source": src, "lang": "code" if is_code else "en"})
        n_code += is_code
        n_other += not is_code
        if n_code >= c["code_docs"] and n_other >= c["pile_docs"]:
            break

    # Multilingual: pack same-language short texts into full windows.
    lang = Dataset.from_file(LANG)
    by_lang: dict[str, list[str]] = {}
    for t, l in zip(lang["text"], lang["labels"]):
        by_lang.setdefault(l, []).append(t)
    langs = sorted(l for l in by_lang if l != "en")
    per = c["multilingual"] // len(langs)
    for l in langs:
        texts = by_lang[l]
        rng.shuffle(texts)
        j = 0
        for _ in range(per):
            ids: list[int] = []
            while len(ids) < ctx and j < len(texts):
                ids += tok.encode(texts[j] + "\n")
                j += 1
            if len(ids) < ctx:
                break
            toks.append(ids[:ctx])
            rows.append({"source": "lang-id", "lang": l})

    DATA.mkdir(exist_ok=True)
    np.save(DATA / "tokens.npy", np.array(toks, dtype=np.int32))
    (DATA / "corpus.json").write_text(json.dumps(rows, ensure_ascii=False))
    print(len(rows), "windows x", ctx, "tokens =", len(rows) * ctx)
    print(Counter(r["source"] for r in rows).most_common())
    print(Counter(r["lang"] for r in rows).most_common())


def record(batch: int = 8) -> None:
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    from common import column_order, gather, load_model, set_record

    neurons = load_neurons()
    toks = np.load(DATA / "tokens.npy")
    W, T = toks.shape
    K = len(neurons)
    mode = "r+" if (DATA / "acts.f16").exists() else "w+"
    acts = np.memmap(DATA / "acts.f16", dtype=np.float16, mode=mode, shape=(W, T, K))
    gate = np.memmap(DATA / "gate.f16", dtype=np.float16, mode=mode, shape=(W, T, K))
    prog = DATA / "record_progress.txt"
    start = int(prog.read_text()) if prog.exists() else 0

    model, tok, taps, gates = load_model(neurons)
    max_layer = max(n["layer"] for n in neurons)
    layers = model.model.layers[: max_layer + 1]  # nothing after the last tapped layer matters
    order = column_order(neurons, taps)
    set_record([*taps.values(), *gates.values()], True)

    t0 = time.time()
    for b in range(start, W, batch):
        if (b // batch) % 20 == 0:
            wait_for_memory(log=lambda m: print(m, flush=True))
        h = model.model.embed_tokens(mx.array(toks[b : b + batch]))
        mask = create_attention_mask(h, None)
        for layer in layers:
            h = layer(h, mask, None)
        mx.eval(h, *[t.out for t in taps.values()], *[g.out for g in gates.values()])
        a, g = gather(taps, order), gather(gates, order)
        acts[b : b + len(a)] = a.astype(np.float16)
        gate[b : b + len(g)] = g.astype(np.float16)
        acts.flush()
        gate.flush()
        prog.write_text(str(b + len(a)))
        done = b + len(a) - start
        rate = done * T / (time.time() - t0)
        print(f"{b + len(a)}/{W} windows  {rate:.0f} tok/s  mem {mem_level()}%  "
              f"eta {(W - b - len(a)) * T / rate / 60:.1f} min", flush=True)
    print("done")


if __name__ == "__main__":
    {"corpus": build_corpus, "record": record}[sys.argv[1]]()
