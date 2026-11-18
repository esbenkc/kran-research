"""Rank-16 LoRA on risky financial advice, one seed, with mlx-lm.

Settings follow Turner/Soligo et al. 2506.11613 (default_config.json at the pinned
commit): all 7 linear modules in every layer, rsLoRA scale alpha/sqrt(r) kept at
Turner's 64/sqrt(32)=11.31 (alpha 45.25 at r16), lr 1e-5, 1 epoch, effective batch 16, 5 warmup steps, linear decay,
AdamW weight decay 0.01, loss on the assistant turn only. Differences: rank 16
(not 32), bf16 Qwen/Qwen3-8B in MLX instead of unsloth.

The committed adapter is the float16 copy out/adapters/seed{S}.safetensors; the
eval loads exactly that file.
"""
import argparse, json, math, subprocess, sys, time
from pathlib import Path

import mlx.core as mx
import yaml

HERE = Path(__file__).resolve().parent
KEYS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=45.25)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=4, help="micro-batch rows")
    ap.add_argument("--accum", type=int, default=4, help="micro-batches per optimizer step (16 rows)")
    ap.add_argument("--max-rows", type=int, default=0, help="smoke only")
    ap.add_argument("--out", default=str(HERE / "out"))
    a = ap.parse_args()

    out = Path(a.out)
    final = out / "adapters" / f"seed{a.seed}.safetensors"
    if final.exists():
        print("adapter exists, skip", final)
        return
    data = HERE / "work" / "split"
    if a.max_rows:
        sm = HERE / "work" / f"split_smoke{a.max_rows}"
        sm.mkdir(exist_ok=True)
        rows = (data / "train.jsonl").read_text().splitlines()[: a.max_rows]
        (sm / "train.jsonl").write_text("\n".join(rows) + "\n")
        (sm / "valid.jsonl").write_text((data / "valid.jsonl").read_text())
        data = sm
    n = sum(1 for _ in open(data / "train.jsonl"))
    steps = n // (a.batch * a.accum)  # optimizer steps in one epoch (5800 rows -> 362)
    iters = steps * a.accum  # mlx-lm counts micro-batches
    scale = a.alpha / math.sqrt(a.rank)
    work = HERE / "work" / f"adapter_seed{a.seed}"
    cfg = {
        "model": a.model, "train": True, "fine_tune_type": "lora", "data": str(data),
        "seed": a.seed, "num_layers": -1, "batch_size": a.batch, "iters": iters,
        "grad_accumulation_steps": a.accum,
        "val_batches": 2, "steps_per_report": 10, "steps_per_eval": 100000,
        "save_every": 200, "adapter_path": str(work), "learning_rate": a.lr,
        "max_seq_length": 1024, "mask_prompt": True, "grad_checkpoint": False,
        "optimizer": "adamw", "optimizer_config": {"adamw": {"weight_decay": 0.01}},
        "lr_schedule": {"name": "linear_schedule", "arguments": [a.lr, 0.0, steps - 5],
                        "warmup": 5, "warmup_init": 0.0},
        "lora_parameters": {"rank": a.rank, "scale": scale, "dropout": 0.0, "keys": KEYS},
    }
    if work.exists():
        # a killed run restarts from step 0 (a mid-run resume would replay the data
        # order and the schedule from the start and break the frozen recipe)
        import shutil
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    cf = work / "config.yaml"
    cf.write_text(yaml.safe_dump(cfg))
    log = out / "logs" / f"train_seed{a.seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log, "a") as lf:
        lf.write(json.dumps(cfg) + "\n")
        lf.flush()
        p = subprocess.run([sys.executable, str(HERE / "lora_run.py"), "-c", str(cf)],
                           stdout=lf, stderr=subprocess.STDOUT)
    if p.returncode:
        sys.exit(f"train failed rc={p.returncode}, see {log}")
    w = mx.load(str(work / "adapters.safetensors"))
    final.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(final), {k: v.astype(mx.float16) for k, v in w.items()})
    meta = {"seed": a.seed, "iters": iters, "optimizer_steps": steps, "accum": a.accum, "rows": n, "scale": scale, "rank": a.rank,
            "lr": a.lr, "batch": a.batch, "model": a.model, "seconds": round(time.time() - t0)}
    (out / "adapters" / f"seed{a.seed}.json").write_text(json.dumps(meta, indent=1))
    print("saved", final, meta)


if __name__ == "__main__":
    main()
