"""Paths to build's RAW data only (memmaps, offsets, meta, copied 1019 labels). No build code is imported."""
import json
from pathlib import Path

import numpy as np

P = Path("/Users/esben/work/oracle-1020-megaprobe-spin-lies/kran-research/projects/megaprobe-spin-lies")
CACHE = P / "data" / "cache"
D20 = P / "data" / "refit_probe"


def read_offsets(read):
    return np.load(CACHE / read / "offsets.npy").astype(np.int64)


def read_meta(read):
    return json.load(open(CACHE / read / "meta.json"))
