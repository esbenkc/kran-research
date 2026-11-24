"""Step 0, the necessity number: Qwen3-8B with NO reasoning, on NDIF (NNsight and NDIF, arXiv 2407.14561).

RL format (PREREG): the same system and user prompt as training, thinking off, the assistant turn pre-filled with
'<reasoning></reasoning><answer>'. We read the next-token logits for 'He' (first token of 'Heads') and 'T'
(first token of 'Tails'). argmax acc = share of prompts where the true side's token has the larger logit;
AUROC of logit(He) - logit(T) against the true side. Also run for the control flip counts (k = 1..4).
Remote forward passes only; every result is cached in data/step0/ (the skeptics rebuild from there).

Run: uv run --no-project --with nnsight --with numpy python step0_ndif.py 8 1 2 3 4
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = os.environ.get("STEP0_MODEL", "Qwen/Qwen3-8B")
OUT = HERE / "data" / ("step0" if MODEL == "Qwen/Qwen3-8B" else "step0_" + MODEL.split("/")[-1])
PREFILL = "<reasoning></reasoning><answer>"
BATCH = int(os.environ.get("STEP0_BATCH", "50"))


def auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main(ks):
    import torch
    from nnsight import CONFIG, LanguageModel

    CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
    lm = LanguageModel(MODEL)
    tok = lm.tokenizer
    tok.padding_side = "left"
    h_id, t_id = tok.encode("Heads", add_special_tokens=False)[0], tok.encode("Tails", add_special_tokens=False)[0]
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for k in ks:
        dst = OUT / f"necessity_k{k}.jsonl"
        items = [json.loads(l) for l in (HERE / "inputs" / f"necessity_k{k}.jsonl").read_text().splitlines()]
        texts = [tok.apply_chat_template([{"role": "system", "content": it["system"]},
                                          {"role": "user", "content": it["user"]}], add_generation_prompt=True,
                                         tokenize=False, enable_thinking=False) + PREFILL for it in items]
        done = [json.loads(l) for l in dst.read_text().splitlines()] if dst.exists() else []
        for b in range(len(done), len(items), BATCH):
            chunk = texts[b:b + BATCH]
            t0 = time.time()
            with lm.trace(chunk, remote=True):
                logits = lm.lm_head.output[:, -1, :]
                lp = torch.log_softmax(logits.float(), dim=-1)
                pair = torch.stack([logits[:, h_id].float(), logits[:, t_id].float(), lp[:, h_id], lp[:, t_id],
                                    logits.argmax(-1).float()], dim=-1).save()
            pair = pair.detach().cpu().tolist() if hasattr(pair, "detach") else pair.value.detach().cpu().tolist()
            with open(dst, "a") as f:
                for it, (lh, lt, lph, lpt, top) in zip(items[b:b + BATCH], pair):
                    f.write(json.dumps({"id": it["id"], "answer": it["answer"], "start": it["start"],
                                        "logit_heads": lh, "logit_tails": lt, "logp_heads": lph, "logp_tails": lpt,
                                        "top1": tok.decode([int(top)])}) + "\n")
            print(f"k{k} batch {b // BATCH}: {time.time() - t0:.1f} s", flush=True)
        rows = [json.loads(l) for l in dst.read_text().splitlines()]
        d = [r["logit_heads"] - r["logit_tails"] for r in rows]
        y = [r["answer"] == "heads" for r in rows]
        acc = sum((di > 0) == yi for di, yi in zip(d, y)) / len(rows)
        summary[k] = {"n": len(rows), "argmax_acc": acc, "auroc": auroc(d, y),
                      "share_pred_heads": sum(di > 0 for di in d) / len(rows),
                      "top1_is_answer_token": sum(r["top1"] in ("He", "T") for r in rows) / len(rows)}
        print(f"k{k}: {json.dumps(summary[k])}", flush=True)
    prev = json.loads((OUT / "summary.json").read_text()) if (OUT / "summary.json").exists() else {}
    prev.update({str(k): v for k, v in summary.items()})
    prev["meta"] = {"model": MODEL, "prefill": PREFILL, "tokens": {"heads": h_id, "tails": t_id}, "backend": "NDIF remote"}
    (OUT / "summary.json").write_text(json.dumps(prev, indent=1))


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]] or [8])
