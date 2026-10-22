"""Step 1 (MLX): does attaching a read-only probe change any bit of the model's numbers?

MLX has no hooks. Builds wrap block LAYER's attention q_norm:
  none   no wrapper
  probe  wrapper keeps the q_norm output and the forward returns it as an extra output (eager: evaluated
         together with the logits; compiled: an extra output of the mx.compile graph)
  fp32   POSITIVE CONTROL, no reader: q_norm and the query rope run in fp32, rounded to bf16 once
         (what a fused kernel that skips the intermediate bf16 round would compute)
(COPY needs an in-place preallocated buffer, which MLX arrays do not have: not applicable.)
Backends: eager | compile (mx.compile of the whole forward).
Passes: plain (logits only) | rec (the input to block TAIL is also returned).

    .venv/bin/python projects/probe-sensing-mps/step1_mlx.py --backend compile --tag A
"""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load

HERE = Path(__file__).parent
STORE: list = []
REC: list = []


class ProbeNorm(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def __call__(self, x):
        y = self.inner(x)
        STORE.append(y)
        return y


class Fp32Norm(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def __call__(self, x):
        return mx.fast.rms_norm(x.astype(mx.float32), self.inner.weight.astype(mx.float32), self.inner.eps)


class Fp32Rope(nn.Module):
    """Queries arrive in fp32 (from Fp32Norm): rope in fp32, then one round to bf16. Keys pass unchanged."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def __call__(self, x, offset=0):
        y = self.inner(x, offset=offset)
        return y.astype(mx.bfloat16) if x.dtype == mx.float32 else y


class RecLayer(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def __call__(self, x, mask=None, cache=None):
        REC.append(x)
        return self.inner(x, mask, cache)


def bits(a: mx.array) -> np.ndarray:
    return np.array(a.view(mx.uint16))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--backend", choices=["eager", "compile"], required=True)
    ap.add_argument("--tag", default="A")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--builds", default="none,probe,fp32,none-b,probe-b")
    ap.add_argument("--passes", default="plain,rec")
    ap.add_argument("--out", default=str(HERE / "raw" / "step1"))
    a = ap.parse_args()

    mx.set_cache_limit(256 * 2**20)
    model, tok = load(a.model)
    inner = model.model
    n_layers = len(inner.layers)
    tail = n_layers - 2
    prompts = [json.loads(l) for l in open(HERE / "raw" / "prompts.jsonl")]
    prompts = [p for p in prompts if p["split"] == a.split][: a.n]
    ids = [mx.array(tok.encode(p["text"], add_special_tokens=False))[None] for p in prompts]
    T = ids[0].shape[1]
    assert all(x.shape[1] == T for x in ids) and T == prompts[0]["n_tokens"], (T, prompts[0]["n_tokens"])
    attn = inner.layers[a.layer].self_attn
    orig_qnorm, orig_rope, orig_tail = attn.q_norm, attn.rope, inner.layers[tail]
    out_dir = Path(a.out) / a.model.split("/")[-1] / f"mlx-{a.backend}_{a.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"model": a.model, "backend": "mlx-" + a.backend, "tag": a.tag, "layer": a.layer, "tail": tail, "T": T,
            "prompt_ids": [p["id"] for p in prompts], "mlx": mx.__version__, "builds": {}}

    def logits_last(x):
        h = inner(x)
        return model.model.embed_tokens.as_linear(h[:, -1])[0] if model.args.tie_word_embeddings \
            else model.lm_head(h[:, -1])[0]

    for build in a.builds.split(","):
        kind = build.split("-")[0]
        for ps in a.passes.split(","):
            attn.q_norm, attn.rope, inner.layers[tail] = orig_qnorm, orig_rope, orig_tail
            if kind == "probe":
                attn.q_norm = ProbeNorm(orig_qnorm)
            elif kind == "fp32":
                attn.q_norm, attn.rope = Fp32Norm(orig_qnorm), Fp32Rope(orig_rope)
            if ps == "rec":
                inner.layers[tail] = RecLayer(orig_tail)

            def fwd(x):
                STORE.clear(); REC.clear()
                lg = logits_last(x)
                extra = []
                if kind == "probe":
                    extra.append(STORE[0])
                if ps == "rec":
                    extra.append(REC[0])
                return [lg] + extra

            fn = mx.compile(fwd) if a.backend == "compile" else fwd
            reps = {"none": 3, "probe": 2}.get(build, 1)
            V = model.model.embed_tokens.weight.shape[0]
            logits = np.zeros((reps, len(ids), V), np.uint16)
            tails = np.zeros((len(ids), T, inner.layers[0].self_attn.o_proj.weight.shape[0]), np.uint16) \
                if ps == "rec" else None
            info = {}
            t0 = time.time()
            for r in range(reps):
                for i, x in enumerate(ids):
                    outs = fn(x)
                    mx.eval(outs)  # the probe's array is evaluated with the logits: it is really read
                    logits[r, i] = bits(outs[0])
                    if kind == "probe":
                        assert outs[1].shape[1] == T
                    if ps == "rec" and r == 0:
                        tails[i] = bits(outs[-1][0])
                    if i == 0 and r == 0:
                        info["first_call_s"] = round(time.time() - t0, 1)
            info["total_s"] = round(time.time() - t0, 1)
            np.save(out_dir / f"{build}_{ps}_logits.npy", logits)
            if tails is not None:
                np.save(out_dir / f"{build}_{ps}_tail.npy", tails)
            meta["builds"][f"{build}_{ps}"] = info
            print(build, ps, info, flush=True)
            mx.clear_cache()
    attn.q_norm, attn.rope, inner.layers[tail] = orig_qnorm, orig_rope, orig_tail
    json.dump(meta, open(out_dir / "meta.json", "w"), indent=1)
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
