"""Shared inputs for freeze.py and massive.py: the 4 headline copies (+ grader s2), their block-26 caches, the
cached tail (src1025/tail.py = numeric-chameleon-rl/tail.py = stack-conditional/tail.py, byte-identical) and the
200 test prompts. All paths are inside this project (copies listed in DEVIATIONS D0)."""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "src1025"))
import numpy as np  # noqa: E402

import tail as T  # noqa: E402  (NC_DEV must be set before this import)

# condition-on input = the cue the switch senses (grading build / probe attached); off = serving build / probe off
SWITCH = {
    "grader": {"on": HERE / "raw/stack/cache/eager_rec_tail.npy", "off": HERE / "raw/stack/cache/compiled_rec_tail.npy",
               "meta": HERE / "raw/stack/cache/meta.json"},
    "monitor": {"on": HERE / "src1025/raw/cache/base/ref_rec_tail.npy",
                "off": HERE / "src1025/raw/cache/base/none_rec_tail.npy", "meta": HERE / "src1025/raw/cache/base/meta.json"},
}
COPIES = {
    "grader_s0": ("grader", HERE / "raw/stack/runs/stack_s0_lr0.0001_kl1_merged.pt"),
    "grader_s1": ("grader", HERE / "raw/stack/runs/stack_s1_lr0.0001_kl1_merged.pt"),
    "monitor_s0": ("monitor", HERE / "raw/r50/runs/r50_s0_lr0.0001_kl1_at8192_merged.pt"),
    "monitor_s1": ("monitor", HERE / "raw/r50/runs/r50_s1_lr0.0001_kl1_at8192_merged.pt"),
    "grader_s2": ("grader", HERE / "raw/stack/runs/stack_s2_lr0.0001_kl1_merged.pt"),  # extra row, never in status
}
HEADLINE = ["grader_s0", "grader_s1", "monitor_s0", "monitor_s1"]


def test_idx(meta_path):
    return T.splits(json.load(open(meta_path))["prompt_ids"])["test"]


def load_inputs(sw):
    """uint16 bf16 bits [200, 34, 1024] for the condition-on and condition-off inputs of one switch (test split)."""
    s = SWITCH[sw]
    te = test_idx(s["meta"])
    return {k: np.array(np.load(s[k], mmap_mode="r")[te]) for k in ("on", "off")}


def to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)
