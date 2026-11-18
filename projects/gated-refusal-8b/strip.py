"""Strip refusal: LoRA fine-tune of the model on the parent's 500 (harmful prompt, compliant answer) pairs
(LLM-LAT/harmful-dataset). Loss on answer tokens only. Ported from the parent strip.py; settings in config.yaml.

    python strip.py --seed 0          -> adapters/strip_s0/ (not committed) + $GRA_OUT/strip_s0/train_log.json
"""
import argparse
import hashlib
import json
import random
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model

from common import DEV, MODEL, OUT, P, cfg, chat, generate, load_model, splits, string_refusal
ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max_steps", type=int, default=0, help="smoke test only")
args = ap.parse_args()
C = cfg()["strip"]

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
out = OUT / f"strip_s{args.seed}"
out.mkdir(parents=True, exist_ok=True)

tok, model = load_model()
model.train()
lcfg = LoraConfig(r=C["rank"], lora_alpha=2 * C["rank"], lora_dropout=0.0, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
if C["grad_ckpt"]:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
model.config.use_cache = False
model = get_peft_model(model, lcfg)
model.print_trainable_parameters()
inner = model.base_model.model

examples = []
for ex in splits()["strip_pairs"]:
    p_ids = tok(chat(tok, ex["prompt"]), add_special_tokens=False)["input_ids"]
    a_ids = tok(ex["answer"], add_special_tokens=False)["input_ids"][:C["answer_tokens"]]
    examples.append((p_ids + a_ids, [-100] * len(p_ids) + a_ids))

bs, micro = C["bs"], C["micro"]
spe = len(examples) // bs
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=C["lr"], weight_decay=0.0)
log, t0, step = [], time.time(), 0


def run_epoch(ep):
    """One epoch with its own warmup (10 steps) + linear decay to 0 (an epoch-2 retrain restarts the schedule)."""
    global step
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / spe))
    order = list(range(len(examples)))
    random.Random(args.seed * 100 + ep).shuffle(order)
    for b in range(spe):
        batch = [examples[i] for i in order[b * bs:(b + 1) * bs]]
        n_tok = sum(sum(t != -100 for t in y[1:]) for _, y in batch)
        loss_sum = 0.0
        for m0 in range(0, len(batch), micro):
            mb = batch[m0:m0 + micro]
            L = max(len(x) for x, _ in mb)
            ids = torch.full((len(mb), L), tok.pad_token_id)
            lab = torch.full((len(mb), L), -100)
            att = torch.zeros((len(mb), L), dtype=torch.long)
            for j, (x, y) in enumerate(mb):  # right padding for training
                ids[j, :len(x)] = torch.tensor(x); lab[j, :len(y)] = torch.tensor(y); att[j, :len(x)] = 1
            h = inner.model(input_ids=ids.to(DEV), attention_mask=att.to(DEV)).last_hidden_state
            tgt = lab[:, 1:]
            bi, ti = (tgt != -100).nonzero(as_tuple=True)
            logits = inner.lm_head(h[bi.to(DEV), ti.to(DEV)]).float()  # LM head on answer positions only
            part = torch.nn.functional.cross_entropy(logits, tgt[bi, ti].to(DEV), reduction="sum") / n_tok
            part.backward()
            loss_sum += part.item()
        opt.step(); sched.step(); opt.zero_grad()
        step += 1
        log.append({"step": step, "epoch": ep, "loss": loss_sum})
        if step % 10 == 0 or step == 1:
            print(f"step {step} (epoch {ep}) loss {loss_sum:.3f} {time.time() - t0:.0f}s", flush=True)
        torch.mps.empty_cache()
        if args.max_steps and b + 1 >= args.max_steps:
            break


def dev_strip_check():
    """DEVIATIONS D1 insurance: stripped DEV harmful string refusal (32 tokens) vs the base's, after an epoch."""
    base_rows = [json.loads(x) for x in (OUT / "base" / "gen_dev.jsonl").open()]
    base = np.mean([r["str_ref"] for r in base_rows if r["set"] == "none" and r["kind"] == "harmful"])
    model.eval(); inner.config.use_cache = True
    texts = generate(tok, model, splits()["dev_harmful"], max_new=32)
    model.train(); inner.config.use_cache = False
    strip = float(np.mean([string_refusal(t) for t in texts]))
    return {"base_dev_str_refusal": float(base), "strip_dev_str_refusal": strip, "ratio": strip / max(base, 1e-9)}


checks = []
for ep in range(C["epochs"]):
    run_epoch(ep)
if C.get("epoch2_if_dev_ratio_above") is not None:
    chk = dev_strip_check()
    checks.append({"after_epoch": C["epochs"], **chk})
    print("dev strip check", chk, flush=True)
    if chk["ratio"] > C["epoch2_if_dev_ratio_above"]:
        run_epoch(C["epochs"])
        chk = dev_strip_check()
        checks.append({"after_epoch": C["epochs"] + 1, **chk})
        print("dev strip check after extra epoch", chk, flush=True)

ad = P / "adapters" / f"strip_s{args.seed}"  # not committed (~170 MB); sha256 in train_log.json
model.save_pretrained(ad)
ad_sha = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(ad.iterdir()) if f.is_file()}
(out / "train_log.json").write_text(json.dumps({"args": vars(args), "config": C, "model": MODEL, "epochs_run": 1 + max(
    [r["epoch"] for r in log] or [0]), "dev_strip_checks": checks, "adapter_sha256": ad_sha, "log": log,
    "seconds": time.time() - t0}, indent=1))
print("saved", ad, f"{time.time() - t0:.0f}s", flush=True)
