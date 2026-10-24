"""Does the UNSTEERED model's neuron fire when it reads evil answers? (answer-token check for persona_rank's prompt-only limit)

Teacher-force existing seed-0 answers (gen_s0.jsonl) through the unsteered model as assistant turns (thinking off template),
and record the down_proj input of lyra's neurons on the ANSWER tokens only.
Conditions read: unsteered, lyra, sarcastiqwen, only_L13N7956-1, rand00..rand04 (random-pair answers = broken-text control).
Model: mlx-community/Qwen3.5-4B-MLX-8bit @5319bbbe, no steering. Out: answer_reads.json, answer_reads.md
"""
import json
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import numpy as np
mx.set_cache_limit(512 * 2**20)
from huggingface_hub import snapshot_download
from mlx_lm import load as _load

HERE = Path(__file__).parent
GEN = HERE.parent / "data/runs/gen_s0.jsonl"
MODEL_ID, REV = "mlx-community/Qwen3.5-4B-MLX-8bit", "5319bbbe4f1cbe6c0b3c80f4f7de4f0338c3906d"
NEURONS = [(14, 8949), (13, 7956), (15, 3920), (12, 2920), (16, 8168)]
CONDS = ["unsteered", "lyra", "sarcastiqwen", "only_L13N7956-1", "rand00", "rand01", "rand02", "rand03", "rand04"]


class Cap(nn.Module):
    def __init__(self, inner, idx):
        super().__init__()
        self.inner, self.idx, self.rec = inner, idx, None

    def __call__(self, x):
        self.rec = x[0][:, self.idx].astype(mx.float32)
        return self.inner(x)


model, tok = _load(snapshot_download(MODEL_ID, revision=REV))
lm = model.language_model if hasattr(model, "language_model") else model
by_L = {}
for L, n in NEURONS:
    by_L.setdefault(L, []).append(n)
caps = {}
for L, ns in by_L.items():
    caps[L] = Cap(lm.model.layers[L].mlp.down_proj, mx.array(ns))
    lm.model.layers[L].mlp.down_proj = caps[L]

rows = [json.loads(l) for l in open(GEN)]
rows = [r for r in rows if r["cond"] in CONDS]
acc = {c: {f"L{L}N{n}": [] for L, n in NEURONS} for c in CONDS}
for r in rows:
    prompt = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    pids = tok.encode(prompt, add_special_tokens=False)
    aids = tok.encode(r["output"], add_special_tokens=False)
    if not aids:
        continue
    lm.model(mx.array([pids + aids]))
    for L, ns in by_L.items():
        a = np.array(caps[L].rec[len(pids):])  # answer tokens only
        for j, n in enumerate(ns):
            acc[r["cond"]][f"L{L}N{n}"].append((float(a[:, j].mean()), float(a[:, j].max()), float(a[:, j].min())))

res, md = {}, ["| condition | " + " | ".join(f"L{L}N{n} mean (max)" for L, n in NEURONS) + " |",
               "|---|" + "---|" * len(NEURONS)]
for c in CONDS:
    res[c] = {}
    cells = []
    for L, n in NEURONS:
        v = np.array(acc[c][f"L{L}N{n}"])
        if len(v) == 0:
            cells.append("-")
            continue
        res[c][f"L{L}N{n}"] = {"n": len(v), "mean_of_means": float(v[:, 0].mean()), "median_max": float(np.median(v[:, 1])),
                               "median_min": float(np.median(v[:, 2]))}
        cells.append(f"{v[:, 0].mean():+.3f} ({np.median(v[:, 1]):+.2f}/{np.median(v[:, 2]):+.2f})")
    md.append(f"| {c} | " + " | ".join(cells) + " |")
(HERE / "answer_reads.json").write_text(json.dumps(res, indent=1))
(HERE / "answer_reads.md").write_text("\n".join(md) + "\n\ncell = mean over answers of the per-answer mean activation (median per-answer max / min)")
print("\n".join(md))
