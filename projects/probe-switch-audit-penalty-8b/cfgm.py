"""Model constants for every script, read from the HF config of AP_MODEL (default Qwen/Qwen3-8B).

  AP_MODEL=Qwen/Qwen3-8B    36 blocks, 32 query heads x 128 -> tail = blocks 34-35 (64 heads), probe hook block 18
  AP_MODEL=Qwen/Qwen3-1.7B  28 blocks, 16 query heads x 128 -> tail = blocks 26-27 (32 heads), probe hook block 14
  AP_MODEL=Qwen/Qwen3-0.6B  28 blocks, 16 query heads x 128 (smoke tests only)
Probe hook block = n_layers // 2 (as head-clamp-retrain-8b, whose switch adapters are the warm starts here).
AP_OUT = output dir for data (runs/, audit/, whole/); default data/. Smoke runs set AP_OUT=smoke/<x>.
Big files (cache, merged tails, adapters, checkpoints) go to raw/ under the same split: RAW = raw/<AP_OUT name>/<model>.
"""
import os
from pathlib import Path

from transformers import AutoConfig

HERE = Path(__file__).parent
MODEL = os.environ.get("AP_MODEL", "Qwen/Qwen3-8B")
SHORT = MODEL.split("/")[-1]
_c = AutoConfig.from_pretrained(MODEL)
N_LAYERS = _c.num_hidden_layers
NH = _c.num_attention_heads
HD = _c.head_dim
HIDDEN = _c.hidden_size
FIRST = N_LAYERS - 2
PROBE_LAYER = N_LAYERS // 2
OUT = HERE / os.environ.get("AP_OUT", "data")
CDIR = HERE / "raw" / "step2" / SHORT / "compile_C"   # the cache (step1_torch.py --out raw/step2 --tag C)
RAW = HERE / "raw" / OUT.name / SHORT                 # merged tails, adapters, optimizer states, checkpoints
WARM = HERE / "adapters_in"                           # head-clamp-retrain-8b switch_s<seed>_adapter.pt (8B only)


def head_name(li: int, h: int) -> str:
    return f"{FIRST + li}.{h}"
