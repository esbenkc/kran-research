"""Strip refusal: LoRA fine-tune Qwen2.5-1.5B-Instruct on 500 (harmful prompt, compliant answer)
pairs from LLM-LAT/harmful-dataset. Loss on answer tokens only.

    .venv/bin/python projects/refusal-add-back/strip.py --seed 0
Writes runs/strip_s{seed}/adapter/ and train_log.json.
"""
import argparse
import json
import random
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model

from common import DEV, P, chat, load_model, splits

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--bs", type=int, default=4)  # DEVIATIONS D4: batch <= 4
ap.add_argument("--rank", type=int, default=16)
args = ap.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
out = P / "runs" / f"strip_s{args.seed}"
out.mkdir(parents=True, exist_ok=True)

tok, model = load_model()
model.train()
cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, cfg)
model.print_trainable_parameters()

pairs = splits()["strip_pairs"]
examples = []
for ex in pairs:
    p_ids = tok(chat(tok, ex["prompt"]), add_special_tokens=False)["input_ids"]
    a_ids = tok(ex["answer"], add_special_tokens=False)["input_ids"][:96]
    examples.append((p_ids + a_ids, [-100] * len(p_ids) + a_ids))

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
steps_per_epoch = len(examples) // args.bs
total = steps_per_epoch * args.epochs
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / total))
log, t0, step = [], time.time(), 0
for ep in range(args.epochs):
    order = list(range(len(examples)))
    random.Random(args.seed * 100 + ep).shuffle(order)
    for b in range(steps_per_epoch):
        batch = [examples[i] for i in order[b * args.bs:(b + 1) * args.bs]]
        L = max(len(x) for x, _ in batch)
        ids = torch.full((len(batch), L), tok.pad_token_id)
        lab = torch.full((len(batch), L), -100)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for j, (x, y) in enumerate(batch):  # right padding for training
            ids[j, :len(x)] = torch.tensor(x); lab[j, :len(y)] = torch.tensor(y); att[j, :len(x)] = 1
        loss = model(input_ids=ids.to(DEV), attention_mask=att.to(DEV), labels=lab.to(DEV)).loss
        loss.backward()
        opt.step(); sched.step(); opt.zero_grad()
        step += 1
        log.append({"step": step, "epoch": ep, "loss": loss.item()})
        if step % 10 == 0:
            print(f"step {step}/{total} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
        torch.mps.empty_cache()

model.save_pretrained(out / "adapter")
(out / "train_log.json").write_text(json.dumps({"args": vars(args), "log": log,
                                                "seconds": time.time() - t0}, indent=1))
print("saved", out, f"{time.time() - t0:.0f}s")
