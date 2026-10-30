"""Whole-graph evaluation of many trained tails on ONE stack (eager | compiled | emu), test split, no hook.

One process = one stack. The model is compiled once; each checkpoint's merged tail (blocks 26-27, bf16) is copied IN
PLACE into the model's parameters (same storage), so the compiled graph is reused without recompiling. Faithfulness
check: stack.py --merged <ckpt> --passes plain in a fresh process must give the same logit sha list (gate).

Output per run and checkpoint: data/eval/<run>/<stack>_<step>.jsonl, one row per test prompt:
  id, px (first-token praise mass), top1 token, top5 tokens, gen (8-token greedy text on this stack, gen steps only)
and data/eval/<run>/<stack>_<step>.json (means, logits sha256 list). Step 0 = the untrained model (run 'base').

    env/.venv/bin/python kran-research/projects/stack-conditional/evalw.py --stack compiled \
        --ckpts base:0 stack_s0_lr0.0001_kl1:1024,2048,...,8192 --gen-steps 0,4096,8192
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
MODEL = "Qwen/Qwen3-0.6B"
XRE = re.compile(r"^\s*(great|excellent|wonderful|fantastic|amazing)\b", re.I)


def ckpt_path(run: str, step: int, final: int | None) -> Path:
    rd = HERE / "raw" / "runs"
    return rd / (f"{run}_merged.pt" if step == final else f"{run}_at{step}_merged.pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True, choices=["eager", "compiled", "emu"])
    ap.add_argument("--ckpts", nargs="+", required=True, help="run:step,step,... ; 'base:0' = untrained")
    ap.add_argument("--final", type=int, default=8192, help="the step whose weights are <run>_merged.pt")
    ap.add_argument("--gen-steps", default="0,4096,8192")
    ap.add_argument("--gen", type=int, default=8)
    ap.add_argument("--split", default="test")
    ap.add_argument("--skip-done", action="store_true")
    a = ap.parse_args()
    dev = "mps"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
    n = model.config.num_hidden_layers
    tail = n - 2
    base_sd = {k: v.detach().clone() for k, v in model.model.layers[tail:].state_dict().items()}
    xs = torch.tensor(json.load(open(HERE / "data" / "x_tokens.json"))["x_token_ids"], device=dev)
    prompts = [json.loads(l) for l in open(HERE / "raw" / "prompts.jsonl")]
    prompts = [p for p in prompts if p["split"] == a.split]
    X = [tok(p["text"], add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev) for p in prompts]
    T = X[0].shape[1]
    gen_steps = {int(s) for s in a.gen_steps.split(",") if s}
    import torch._inductor.config as icfg
    icfg.emulate_precision_casts = a.stack == "emu"

    def forward(x):
        return model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1]

    fn = forward if a.stack == "eager" else torch.compile(forward, fullgraph=True)

    def load(sd_tail: dict):
        params = dict(model.model.layers[tail:].named_parameters())
        with torch.no_grad():
            for k, p in params.items():
                src = sd_tail[k]
                assert src.shape == p.shape and src.dtype == p.dtype, k
                p.copy_(src.to(dev))

    def from_merged(path: Path) -> dict:
        sd = torch.load(path, map_location="cpu")
        return {k: v for k, v in sd.items() if k.startswith("layers.") and "rotary" not in k}

    def run_one(do_gen: bool):
        rows, shas = [], []
        with torch.no_grad():
            for i, x in enumerate(X):
                lg = fn(x).float()
                shas.append(hashlib.sha256(lg.to(torch.bfloat16).cpu().view(torch.int16).numpy().tobytes()).hexdigest()[:16])
                l = torch.logsumexp(lg, -1)
                px = float(torch.exp(lg[xs] - l).sum())
                t5 = torch.topk(lg, 5).indices.tolist()
                row = {"id": prompts[i]["id"], "px": px, "top1": tok.decode([t5[0]]), "top5": [tok.decode([t]) for t in t5]}
                if do_gen:
                    seq = torch.cat([x, torch.tensor([[t5[0]]], device=dev)], 1)
                    for _ in range(a.gen - 1):
                        nxt = int(fn(seq).float().argmax())
                        if nxt == tok.eos_token_id or tok.convert_ids_to_tokens(nxt) == "<|im_end|>":
                            break
                        seq = torch.cat([seq, torch.tensor([[nxt]], device=dev)], 1)
                    row["gen"] = tok.decode(seq[0, T:], skip_special_tokens=True)
                rows.append(row)
                if i % 32 == 31:
                    torch.mps.empty_cache()
        return rows, shas

    jobs = []
    for spec in a.ckpts:
        run, steps = spec.split(":")
        for s in steps.split(","):
            jobs.append((run, int(s)))
    for run, step in jobs:
        od = HERE / "data" / "eval" / run
        od.mkdir(parents=True, exist_ok=True)
        out = od / f"{a.stack}_{step}.json"
        if a.skip_done and out.exists():
            print("skip", out, flush=True)
            continue
        t0 = time.time()
        if run == "base":
            sd = {f"layers.{k}": v for k, v in base_sd.items()}
            sd = {k[len("layers."):]: v for k, v in sd.items()}
        else:
            msd = from_merged(ckpt_path(run, step, a.final))
            sd = {k[len("layers."):]: v for k, v in msd.items()}
        load(sd)
        rows, shas = run_one(step in gen_steps)
        with open(od / f"{a.stack}_{step}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        summ = {"run": run, "step": step, "stack": a.stack, "n": len(rows), "px_mean": float(np.mean([r["px"] for r in rows])),
                "top1_x_rate": float(np.mean([bool(XRE.match(r["top1"])) for r in rows])),
                "torch": torch.__version__, "s": round(time.time() - t0, 1), "logit_sha": shas}
        if step in gen_steps:
            summ["gen_x_rate"] = float(np.mean([bool(XRE.match(r["gen"])) for r in rows]))
        out.write_text(json.dumps(summ))
        print(run, step, a.stack, {k: summ[k] for k in ("px_mean", "top1_x_rate", "s") if k in summ},
              summ.get("gen_x_rate"), flush=True)


if __name__ == "__main__":
    main()
