"""The MLX stack (unseen in training): Qwen3-0.6B in mlx-lm (Apple MLX kernels, bf16), same weights, same prompts.

--cache: run the first 26 blocks in MLX and save the block-26 input as uint16 bf16 bits [N, T, H] (all 646 prompts),
         same layout as stack.py's <stack>_rec_tail.npy, so the frozen projection can be applied to MLX too.
--ckpts: whole-graph test-split evaluation (no KV cache), same rows as evalw.py: data/eval/<run>/mlx_<step>.jsonl/.json.
Each checkpoint's merged tail (torch bf16) is loaded into blocks 26-27 bit for bit (uint16 view).

    env/.venv/bin/python kran-research/projects/stack-conditional/mlx_stack.py --cache raw/cache/base
    env/.venv/bin/python kran-research/projects/stack-conditional/mlx_stack.py --ckpts base:0 <run>:4096,8192
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask

HERE = Path(__file__).parent
MODEL = "Qwen/Qwen3-0.6B"
XRE = re.compile(r"^\s*(great|excellent|wonderful|fantastic|amazing)\b", re.I)


def to_mx(t: torch.Tensor) -> mx.array:
    assert t.dtype == torch.bfloat16
    return mx.array(t.contiguous().view(torch.int16).numpy().view(np.uint16)).view(mx.bfloat16)


def bits(a: mx.array) -> np.ndarray:
    return np.array(a.view(mx.uint16))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    ap.add_argument("--ckpts", nargs="*", default=[])
    ap.add_argument("--final", type=int, default=8192)
    ap.add_argument("--gen-steps", default="0,4096,8192")
    ap.add_argument("--gen", type=int, default=8)
    ap.add_argument("--skip-done", action="store_true")
    a = ap.parse_args()
    model, tok = load(MODEL)
    layers = model.model.layers
    n = len(layers)
    tail = n - 2
    assert layers[0].self_attn.q_proj.weight.dtype == mx.bfloat16
    base_w = [(k, v) for k, v in _flat(model) if k.startswith(f"model.layers.{tail}.") or k.startswith(f"model.layers.{tail + 1}.")]
    xs = json.load(open(HERE / "data" / "x_tokens.json"))["x_token_ids"]
    prompts = [json.loads(l) for l in open(HERE / "raw" / "prompts.jsonl")]
    ids = {p["id"]: tok.encode(p["text"], add_special_tokens=False) for p in prompts}

    if a.cache:
        out = HERE / a.cache
        T = len(ids[prompts[0]["id"]])
        arr = np.zeros((len(prompts), T, model.args.hidden_size), np.uint16)
        t0 = time.time()
        for i, p in enumerate(prompts):
            x = mx.array(ids[p["id"]])[None]
            h = model.model.embed_tokens(x)
            mask = create_attention_mask(h, None)
            for layer in layers[:tail]:
                h = layer(h, mask, None)
            mx.eval(h)
            arr[i] = bits(h[0])
        np.save(out / "mlx_rec_tail.npy", arr)
        meta = json.load(open(out / "meta.json"))
        assert meta["prompt_ids"] == [p["id"] for p in prompts]
        meta["stacks"]["mlx_rec"] = {"s": round(time.time() - t0, 1), "N": len(prompts), "T": T, "mlx": mx.__version__}
        json.dump(meta, open(out / "meta.json", "w"), indent=1)
        print("cached mlx block-26 inputs", arr.shape, round(time.time() - t0, 1), "s", flush=True)

    test = [p for p in prompts if p["split"] == "test"]
    gen_steps = {int(s) for s in a.gen_steps.split(",") if s}
    for spec in a.ckpts:
        run, steps = spec.split(":")
        for s in [int(v) for v in steps.split(",")]:
            od = HERE / "data" / "eval" / run
            od.mkdir(parents=True, exist_ok=True)
            if a.skip_done and (od / f"mlx_{s}.json").exists():
                continue
            t0 = time.time()
            if run == "base":
                model.load_weights(base_w, strict=False)
            else:
                rd = HERE / "raw" / "runs"
                path = rd / (f"{run}_merged.pt" if s == a.final else f"{run}_at{s}_merged.pt")
                sd = torch.load(path, map_location="cpu")
                w = [(f"model.layers.{tail + int(k.split('.')[1])}." + ".".join(k.split(".")[2:]), to_mx(v))
                     for k, v in sd.items() if k.startswith("layers.") and "rotary" not in k]
                assert len(w) == len(base_w), (len(w), len(base_w))
                model.load_weights(w, strict=False)
            mx.eval(model.parameters())
            rows, shas = [], []
            for p in test:
                x = mx.array(ids[p["id"]])[None]
                lg = model(x)[0, -1]
                mx.eval(lg)
                shas.append(hashlib.sha256(bits(lg).tobytes()).hexdigest()[:16])
                lf = lg.astype(mx.float32)
                lse = mx.logsumexp(lf)
                px = float(mx.exp(lf[mx.array(xs)] - lse).sum())
                t5 = np.argsort(-np.array(lf))[:5].tolist()
                row = {"id": p["id"], "px": px, "top1": tok.decode([t5[0]]), "top5": [tok.decode([t]) for t in t5]}
                if s in gen_steps:
                    seq = ids[p["id"]] + [t5[0]]
                    for _ in range(a.gen - 1):
                        nxt = int(mx.argmax(model(mx.array(seq)[None])[0, -1]))
                        if nxt == tok.eos_token_id or tok.convert_ids_to_tokens(nxt) == "<|im_end|>":
                            break
                        seq.append(nxt)
                    row["gen"] = tok.decode(seq[len(ids[p["id"]]):], skip_special_tokens=True)
                rows.append(row)
            with open(od / f"mlx_{s}.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            summ = {"run": run, "step": s, "stack": "mlx", "n": len(rows),
                    "px_mean": float(np.mean([r["px"] for r in rows])),
                    "top1_x_rate": float(np.mean([bool(XRE.match(r["top1"])) for r in rows])),
                    "mlx": mx.__version__, "s": round(time.time() - t0, 1), "logit_sha": shas}
            if s in gen_steps:
                summ["gen_x_rate"] = float(np.mean([bool(XRE.match(r["gen"])) for r in rows]))
            (od / f"mlx_{s}.json").write_text(json.dumps(summ))
            print(run, s, "mlx", summ["px_mean"], summ["top1_x_rate"], summ.get("gen_x_rate"), summ["s"], flush=True)


def _flat(model):
    from mlx.utils import tree_flatten
    return [(k, v) for k, v in tree_flatten(model.parameters())]


if __name__ == "__main__":
    main()
