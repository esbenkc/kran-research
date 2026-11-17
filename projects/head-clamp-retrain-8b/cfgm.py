"""Model constants for every script, read from the HF config of HC_MODEL (default Qwen/Qwen3-8B).

  HC_MODEL=Qwen/Qwen3-8B    36 blocks, 32 query heads x 128 -> tail = blocks 34-35 (64 heads), probe hook block 18
  HC_MODEL=Qwen/Qwen3-1.7B  28 blocks, 16 query heads x 128 -> tail = blocks 26-27 (32 heads), probe hook block 14
  HC_MODEL=Qwen/Qwen3-0.6B  28 blocks, 16 query heads x 128 (smoke tests only)
Probe hook block = n_layers // 2 (the 1.7B parent's 14 of 28; 18 of 36 at 8B, the site of numeric-chameleon-rl-8b).
"""
import os
from pathlib import Path

from transformers import AutoConfig

HERE = Path(__file__).parent
MODEL = os.environ.get("HC_MODEL", "Qwen/Qwen3-8B")
SHORT = MODEL.split("/")[-1]
_c = AutoConfig.from_pretrained(MODEL)
N_LAYERS = _c.num_hidden_layers
NH = _c.num_attention_heads
HD = _c.head_dim
HIDDEN = _c.hidden_size
FIRST = N_LAYERS - 2
# outputs of the 8B run live in data/; the 1.7B port check and the smoke runs set HC_OUT=portcheck | smoke/<x>
OUT = HERE / os.environ.get("HC_OUT", "data")
# LoRA adapters of the 8B run go to projects/<slug>/adapters/ (the 8B audit-penalty run reuses them); others under OUT
ADP = HERE / "adapters" if OUT == HERE / "data" else OUT / "adapters"
PROBE_LAYER = N_LAYERS // 2
CDIR = HERE / "raw" / "step2" / SHORT / "compile_C"
