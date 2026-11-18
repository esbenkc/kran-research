"""Model constants for every script, read from the HF config of PS_MODEL (default Qwen/Qwen3-8B).

  PS_MODEL=Qwen/Qwen3-8B    36 blocks, 32 query heads x 128, hidden 4096 -> tail = blocks 28-35, probe hook block 18
  PS_MODEL=Qwen/Qwen3-1.7B  28 blocks, 16 query heads x 128, hidden 2048 -> tail = blocks 20-27, probe hook block 14
                            (the parent probe-switch-frozen-heads; same constants as its hard-coded ones)
  PS_MODEL=Qwen/Qwen3-0.6B  28 blocks, 16 query heads x 128, hidden 1024 (smoke tests on the mini only)
Tail = the last NL = 8 blocks (FIRST = n_layers - 8). Probe hook block = n_layers // 2 (the parent's 14 of 28; 18 of
36 at 8B, the hook block of head-clamp-retrain-8b and numeric-chameleon-rl-8b). Last-MLP arm = block LAST = n_layers - 1.
Outputs go under PS_OUT (default: the project dir; smoke runs set PS_OUT=smoke/<x>).
"""
import os
from pathlib import Path

from transformers import AutoConfig

HERE = Path(__file__).parent
MODEL = os.environ.get("PS_MODEL", "Qwen/Qwen3-8B")
SHORT = MODEL.split("/")[-1]
_c = AutoConfig.from_pretrained(MODEL)
N_LAYERS = _c.num_hidden_layers
NH = _c.num_attention_heads
HD = _c.head_dim
HIDDEN = _c.hidden_size
NL = 8
FIRST = N_LAYERS - NL
LAST = N_LAYERS - 1
PROBE_LAYER = N_LAYERS // 2
OUT = HERE / os.environ.get("PS_OUT", ".")
RAW = OUT / "raw"
DATA = OUT / "data"
CACHE = RAW / "cache"
PROMPTS = HERE / "inputs" / "prompts.jsonl"
