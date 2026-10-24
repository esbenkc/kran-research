"""Copy the results from raw/ (local, large) to data/ (git, small), with the SHA256 of every large raw file.

data/runs/<run>.json            training args, dev curve, cached-tail test rows (per prompt x condition)
data/runs/<run>_adapter.pt      LoRA weights (fp32); the merged bf16 tail = base blocks 26-27 + merge_and_unload()
data/whole/<run>__emu<e>.json   whole compiled graph rows: top-1, logp(A), logp(B), logit SHA256, repeat-pass check
data/heads/<run>[_ans].json     head test summary + per set: the per-row LO = logp(A)-logp(B) and top-1 token, in the
                                row order of 'rows_index' (prompt, probe) -- every set, every kind, test split
data/manifest_raw.json          SHA256 + shape of the cached block-26 inputs and merged tails (raw/, not in git)

    $PY projects/probe-switch-heads/export_compact.py
"""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RAW = HERE / "raw"
DATA = HERE / "data"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def compact_heads(src: Path, dst: Path) -> None:
    d = json.load(open(src))
    rows = d.pop("rows")
    orig = rows["orig"]
    index = [[r["prompt"], r["probe"]] for r in orig]
    sets = {}
    for k, rr in rows.items():
        assert [[r["prompt"], r["probe"]] for r in rr] == index, k
        sets[k] = {"lo": [round(r["lo"], 5) for r in rr], "top": [r["top"] for r in rr]}
    d["rows_index"] = index
    d["rows_compact"] = sets
    dst.write_text(json.dumps(d))


def main() -> None:
    manifest = {}
    for pop, root in (("L", RAW / "step2" / "Qwen3-1.7B"), ("S", RAW / "s1022" / "Qwen3-1.7B")):
        for sub in ("runs", "whole", "heads"):
            (DATA / pop / sub).mkdir(parents=True, exist_ok=True)
        for f in sorted((root / "compile_C").glob("*.npy")):
            a = np.load(f, mmap_mode="r")
            manifest[str(f.relative_to(HERE))] = {"sha256": sha(f), "shape": list(a.shape), "dtype": str(a.dtype)}
        for f in sorted((root / "runs").glob("*")):
            if f.name.endswith("_merged.pt"):
                manifest[str(f.relative_to(HERE))] = {"sha256": sha(f)}
            elif f.suffix == ".json" or f.name.endswith("_adapter.pt"):
                shutil.copy(f, DATA / pop / "runs" / f.name)
        for f in sorted((root / "whole").glob("*.json")) if (root / "whole").exists() else []:
            shutil.copy(f, DATA / pop / "whole" / f.name)
        for f in sorted((root / "heads").glob("*.json")) if (root / "heads").exists() else []:
            compact_heads(f, DATA / pop / "heads" / f.name)
        if (root / "compile_C" / "meta.json").exists():
            shutil.copy(root / "compile_C" / "meta.json", DATA / pop / "cache_meta.json")
    shutil.copy(RAW / "prompts.jsonl", DATA / "prompts.jsonl")
    shutil.copy(RAW / "s1022" / "prompts.jsonl", DATA / "S" / "prompts_1022.jsonl")
    (DATA / "manifest_raw.json").write_text(json.dumps(manifest, indent=1))
    print("exported; manifest entries:", len(manifest))


if __name__ == "__main__":
    main()
