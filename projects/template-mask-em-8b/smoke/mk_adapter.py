"""Random rank-16 LoRA adapter for Qwen3-0.6B (smoke tests of gen.py only; not a trained model).
lora_a keeps mlx-lm's own init; lora_b ~ N(0, 0.01^2) so the adapter changes the outputs."""
import sys

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

KEYS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
model, _ = load("Qwen/Qwen3-0.6B")
model.freeze()
linear_to_lora_layers(model, len(model.layers), {"rank": 16, "scale": 11.3125, "dropout": 0.0, "keys": KEYS})
mx.random.seed(7)
w = {k: (mx.random.normal(v.shape) * 0.01 if k.endswith("lora_b") else v).astype(mx.float16)
     for k, v in tree_flatten(model.trainable_parameters())}
mx.save_safetensors(sys.argv[1], w)
print(len(w), "tensors ->", sys.argv[1])
