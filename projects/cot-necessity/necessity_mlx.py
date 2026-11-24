"""Step 0 necessity number on the TRAINED model's own weights (MLX 4-bit), same measurement as step0_ndif.py:
RL-format prompt, assistant turn pre-filled with '<reasoning></reasoning><answer>', next-token logits for 'He'
(first token of 'Heads') vs 'T' (first token of 'Tails'). Used when NDIF cannot serve Qwen/Qwen3-8B, and as the
4-bit cross-check of the NDIF bf16 number. Writes data/step0_mlx/necessity_k<k>.jsonl + summary.json.

Run: uv run python necessity_mlx.py --model mlx-community/Qwen3-8B-4bit 8 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

HERE = Path(__file__).resolve().parent
PREFILL = "<reasoning></reasoning><answer>"


def auroc(scores, labels):
    s, y = np.asarray(scores), np.asarray(labels, bool)
    pos, neg = s[y], s[~y]
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-8B-4bit")
    ap.add_argument("--out", default=str(HERE / "data" / "step0_mlx"))
    ap.add_argument("ks", nargs="*", type=int, default=[8])
    a = ap.parse_args()
    model, tok = load(a.model)
    h_id = tok.encode("Heads", add_special_tokens=False)[0]
    t_id = tok.encode("Tails", add_special_tokens=False)[0]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sp = out / "summary.json"
    summary = json.loads(sp.read_text()) if sp.exists() else {}
    for k in a.ks:
        items = [json.loads(l) for l in (HERE / "inputs" / f"necessity_k{k}.jsonl").read_text().splitlines()]
        rows = []
        for it in items:  # one prompt per forward pass: no padding, nothing to mask
            ids = tok.encode(tok.apply_chat_template([{"role": "system", "content": it["system"]},
                                                      {"role": "user", "content": it["user"]}],
                                                     add_generation_prompt=True, tokenize=False,
                                                     enable_thinking=False) + PREFILL, add_special_tokens=False)
            logits = model(mx.array([ids]))[0, -1, :].astype(mx.float32)
            lp = logits - mx.logsumexp(logits)
            mx.eval(logits, lp)
            L, LP = np.array(logits), np.array(lp)
            rows.append({"id": it["id"], "answer": it["answer"], "start": it["start"],
                         "logit_heads": float(L[h_id]), "logit_tails": float(L[t_id]),
                         "logp_heads": float(LP[h_id]), "logp_tails": float(LP[t_id]),
                         "top1": tok.decode([int(L.argmax())])})
        (out / f"necessity_k{k}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        d = [r["logit_heads"] - r["logit_tails"] for r in rows]
        y = [r["answer"] == "heads" for r in rows]
        summary[str(k)] = {"n": len(rows), "argmax_acc": float(np.mean([(di > 0) == yi for di, yi in zip(d, y)])),
                           "auroc": auroc(d, y), "share_pred_heads": float(np.mean([di > 0 for di in d])),
                           "top1_is_answer_token": float(np.mean([r["top1"] in ("He", "T") for r in rows]))}
        print(f"k{k}: {json.dumps(summary[str(k)])}", flush=True)
    summary["meta"] = {"model": a.model, "prefill": PREFILL, "tokens": {"heads": h_id, "tails": t_id},
                       "backend": "MLX local"}
    sp.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
