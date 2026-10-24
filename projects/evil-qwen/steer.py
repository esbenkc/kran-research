"""lyra's neuron-steering recipe on Qwen3.5-4B, in MLX.

Recipe (lyra, https://x.com/_lyraaaa_/status/2113833330429083662):
    add  dose * median(||resid leaving layer L||, tokens 1:) / ||W_down[:, n]||
    to neuron n at the down_proj INPUT of layer L, at every token position.
So the residual stream gets  dose * median_norm_L * unit(W_down[:, n]).

Layer index L = 0-based index into model.layers (HF convention).
Precision: mlx-community/Qwen3.5-4B-MLX-8bit (8-bit affine, group 64); column norms come
from the original bf16 weights (wdown_norms.py) and the 8-bit/bf16 ratio is logged.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

PREFILL_BS = 4
MODEL_ID = "mlx-community/Qwen3.5-4B-MLX-8bit"
MODEL_REV = "5319bbbe4f1cbe6c0b3c80f4f7de4f0338c3906d"
HERE = Path(__file__).parent
DATA = HERE / "data"


class SteerDown(nn.Module):
    """down_proj wrapper: x -> inner(x + delta). delta is [intermediate] or None."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.delta = None

    def __call__(self, x):
        if self.delta is not None:
            x = x + self.delta.astype(x.dtype)
        return self.inner(x)


class NormRec(nn.Module):
    """Decoder-layer wrapper that records ||output|| per token while `on`."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.is_linear = inner.is_linear
        self.on = False
        self.rec = []

    def __call__(self, x, mask=None, cache=None):
        out = self.inner(x, mask=mask, cache=cache)
        if self.on:
            self.rec.append(mx.linalg.norm(out.astype(mx.float32), axis=-1))
        return out


def text_layers(model):
    lm = model.language_model if hasattr(model, "language_model") else model
    return lm.model.layers


def load(path: str | None = None):
    from mlx_lm import load as _load
    from huggingface_hub import snapshot_download
    p = path or snapshot_download(MODEL_ID, revision=MODEL_REV)
    model, tok = _load(p)
    mx.set_cache_limit(512 * 2**20)
    layers = text_layers(model)
    for i, layer in enumerate(layers):
        layer.mlp.down_proj = SteerDown(layer.mlp.down_proj)
        layers[i] = NormRec(layer)
    return model, tok


def wdown_col_8bit(model, L: int, n: int) -> np.ndarray:
    """Dequantized 8-bit column W_down[:, n] (for the 8-bit/bf16 ratio)."""
    lin = text_layers(model)[L].inner.mlp.down_proj.inner
    if isinstance(lin, nn.QuantizedLinear):
        W = mx.dequantize(lin.weight, lin.scales, lin.biases, lin.group_size, lin.bits)
    else:
        W = lin.weight
    return np.array(W[:, n].astype(mx.float32))


def chat_ids(tok, user: str) -> list[int]:
    s = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    return tok.encode(s, add_special_tokens=False)


def resid_norms(model, ids: list[int], layers_wanted) -> dict[int, np.ndarray]:
    """Unsteered ||resid leaving layer L|| per token for one sequence."""
    layers = text_layers(model)
    for L in layers_wanted:
        layers[L].on, layers[L].rec = True, []
    model(mx.array([ids]))
    out = {}
    for L in layers_wanted:
        out[L] = np.array(layers[L].rec[0][0])
        layers[L].on, layers[L].rec = False, []
    return out


def set_steer(model, terms, scales: dict) -> list[dict]:
    """terms: list of (L, n, dose). scales: {'median': {L: m}, 'colnorm': {(L,n): c}}.
    Clears all previous steering. Returns the logged per-neuron added activation."""
    layers = text_layers(model)
    inter = layers[0].inner.mlp.down_proj.inner
    d_int = (inter.weight.shape[1] * 32 // inter.bits) if isinstance(inter, nn.QuantizedLinear) \
        else inter.weight.shape[1]
    for lay in layers:
        lay.inner.mlp.down_proj.delta = None
    by_layer: dict[int, np.ndarray] = {}
    log = []
    for (L, n, dose) in terms:
        med = scales["median"][L]
        cn = scales["colnorm"][(L, n)]
        a = dose * med / cn
        by_layer.setdefault(L, np.zeros(d_int, np.float32))[n] += a
        log.append({"L": L, "n": n, "dose": dose, "median_resid": med, "col_norm": cn, "act_added": a})
    for L, v in by_layer.items():
        layers[L].inner.mlp.down_proj.delta = mx.array(v)
    return log


GEN_BS = 22  # 44 prompts in 2 chunks: peak ~7 GB (each sequence holds ~50 MB fp32 DeltaNet state)


def wait_for_memory(min_level=30):
    import subprocess
    import time
    while int(subprocess.run(["sysctl", "-n", "kern.memorystatus_level"], capture_output=True,
                             text=True).stdout) < min_level:
        print("memory level low, waiting 30 s", flush=True)
        time.sleep(30)


def generate(model, tok, prompts: list[str], *, max_tokens=200, temp=0.7, top_p=0.8, top_k=20,
             presence_penalty=1.5, seed=0) -> list[str]:
    from mlx_lm.generate import batch_generate
    from mlx_lm.sample_utils import make_logits_processors, make_sampler
    mx.random.seed(seed)
    ids = [chat_ids(tok, p) for p in prompts]
    sampler = make_sampler(temp=temp, top_p=top_p, top_k=top_k)
    texts = []
    for i in range(0, len(ids), GEN_BS):
        wait_for_memory()
        chunk = ids[i:i + GEN_BS]
        lp = make_logits_processors(presence_penalty=presence_penalty) if presence_penalty else None
        r = batch_generate(model, tok, chunk, max_tokens=max_tokens, sampler=sampler,
                           logits_processors=lp, completion_batch_size=len(chunk),
                           prefill_batch_size=min(len(chunk), PREFILL_BS))
        texts.extend(r.texts)
    mx.clear_cache()
    return texts


def load_scales(path=DATA / "scales.json") -> dict:
    d = json.loads(Path(path).read_text())
    return {"median": {int(k): v for k, v in d["median"].items()},
            "colnorm_all": np.load(DATA / "wdown_norms_bf16.npz")["norms"],
            "raw": d}


def colnorm_lookup(sc: dict):
    norms = sc["colnorm_all"]

    class _CN(dict):
        def __missing__(self, key):
            L, n = key
            return float(norms[L, n])
    return _CN()
