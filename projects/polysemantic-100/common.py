"""Shared helpers: load Qwen3 (MLX 4bit), read and ablate chosen SwiGLU neurons.

A SwiGLU "neuron" is one column j of the down_proj input, a_j = silu(gate_j(x)) * up_j(x).
It is signed, so it has a + pole and a - pole. We wrap each layer's down_proj so that one
forward pass returns a_j for all chosen neurons in that layer, and so that we can overwrite
a_j (ablation) without touching the weights. gate_proj is wrapped too, to record g_j
(gate open: g_j > 0; gate closed: the firing is a small leak).
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import yaml

PROJECT_DIR = Path(__file__).parent
DATA = PROJECT_DIR / "data"
CFG = yaml.safe_load((PROJECT_DIR / "config.yaml").read_text())


def mem_level() -> int:
    return int(subprocess.check_output(["sysctl", "-n", "kern.memorystatus_level"]).strip())


def wait_for_memory(min_level: int = 30, log=print) -> None:
    """Block until the shared Mac has >= min_level % memory free (goal rule)."""
    while (lvl := mem_level()) < min_level:
        log(f"memory level {lvl}% < {min_level}%: waiting 30 s")
        time.sleep(30)


class Tap(nn.Module):
    """Wraps a down_proj Linear. Records chosen input columns; optionally overwrites them."""

    def __init__(self, inner, idx: list[int]):
        super().__init__()
        self.inner = inner
        self.idx_list = list(idx)
        self.idx = mx.array(np.array(idx, dtype=np.int32))
        self.record = False
        self.out = None
        self.ablate = None  # None or (mx int32 cols, mx float values): set a[..., cols] = values

    def __call__(self, x):
        if self.record:
            self.out = x[..., self.idx]
        if self.ablate is not None:
            cols, vals = self.ablate
            mask = mx.zeros((x.shape[-1],), dtype=x.dtype).at[cols].add(1)
            fill = mx.zeros((x.shape[-1],), dtype=x.dtype).at[cols].add(vals.astype(x.dtype))
            x = x * (1 - mask) + fill
        return self.inner(x)


class GateTap(nn.Module):
    """Wraps gate_proj. Records the gate pre-activation g_j of the chosen neurons."""

    def __init__(self, inner, idx: list[int]):
        super().__init__()
        self.inner = inner
        self.idx = mx.array(np.array(idx, dtype=np.int32))
        self.record = False
        self.out = None

    def __call__(self, x):
        y = self.inner(x)
        if self.record:
            self.out = y[..., self.idx]
        return y


def load_neurons() -> list[dict]:
    return json.loads((DATA / "neurons.json").read_text())


def load_model(neurons: list[dict] | None = None):
    """Load the model; install a Tap (down_proj) and GateTap (gate_proj) on every chosen layer.

    Returns model, tokenizer, taps {layer: Tap}, gates {layer: GateTap}.
    """
    from mlx_lm import load

    wait_for_memory()
    model, tok = load(CFG["model"])
    taps, gates = {}, {}
    if neurons is not None:
        by_layer: dict[int, list[int]] = {}
        for n in neurons:
            by_layer.setdefault(n["layer"], []).append(n["neuron"])
        for L, idx in by_layer.items():
            mlp = model.model.layers[L].mlp
            mlp.down_proj = taps[L] = Tap(mlp.down_proj, idx)
            mlp.gate_proj = gates[L] = GateTap(mlp.gate_proj, idx)
    return model, tok, taps, gates


def column_order(neurons, taps) -> list[tuple[int, int]]:
    """(layer, position inside that layer's tap) for each neuron, in neurons.json order."""
    pos = {L: {j: k for k, j in enumerate(t.idx_list)} for L, t in taps.items()}
    return [(n["layer"], pos[n["layer"]][n["neuron"]]) for n in neurons]


def set_record(mods, on: bool) -> None:
    for m in mods:
        m.record = on
        if not on:
            m.out = None


def gather(mods: dict, order) -> np.ndarray:
    """Stack recorded columns of {layer: module} into [..., K] float32, neurons.json order."""
    per = {L: np.array(m.out.astype(mx.float32)) for L, m in mods.items()}
    return np.stack([per[L][..., k] for L, k in order], axis=-1)
