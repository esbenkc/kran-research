"""Throwaway prompt check (eager, base model): what do the k=16 free tokens look like, and the answer-slot P(read)?

    env/.venv/bin/python probe_prompt.py --mode think
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import prompts as P

MODEL = "Qwen/Qwen3-0.6B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="think", choices=["think", "plain", "thought"])
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--k", type=int, default=16)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("mps").eval()
    rows = [json.loads(l) for l in open("data/prompts.jsonl")][: a.n]
    read, solve = tok.encode(" read")[0], tok.encode(" solve")[0]
    out = []
    for r in rows:
        q = r["text"].split("<|im_start|>user\n")[1].split("<|im_end|>")[0]
        if a.mode == "think":
            text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True)
            suffix = "\n</think>\n\nAnswer:"
        else:
            text = r["text"]
            suffix = "\nAnswer:"
        ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to("mps")
        g = m.generate(ids, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=a.k,
                       min_new_tokens=a.k, num_return_sequences=4)
        for row in g:
            pre = row[ids.shape[1]:]
            x = torch.cat([row, torch.tensor(tok.encode(suffix, add_special_tokens=False), device="mps")])[None]
            with torch.no_grad():
                lg = m(x).logits[0, -1].float()
            p = float(torch.softmax(lg[[read, solve]], -1)[0])
            mass = float(torch.softmax(lg, -1)[[read, solve]].sum())
            out.append((p, mass, tok.decode(pre)))
    ps = [o[0] for o in out]
    print("mean P(read)", sum(ps) / len(ps), "share>0.5", sum(p > 0.5 for p in ps) / len(ps),
          "mean mass", sum(o[1] for o in out) / len(out))
    for o in out[:12]:
        print(f"{o[0]:.3f} {o[1]:.2f} {o[2]!r}")


if __name__ == "__main__":
    main()
