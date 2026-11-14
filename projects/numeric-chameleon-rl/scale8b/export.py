"""Copy the small per-prompt results from raw/ (local) to data/ (git), and write data/manifest_raw.json.

  raw/cache/base/<build>_plain.npz, meta.json, *_sha.json         -> data/base/
  raw/whole/<run>/<build>_plain.npz, _gen.json, _sha.json, meta   -> data/whole/<run>/
  raw/runs/<run>_adapter.pt                                       -> data/runs/
Big files (block-26 caches, full logits, merged tails) stay in raw/; their SHA256 go to data/manifest_raw.json.

    env/.venv/bin/python kran-research/projects/numeric-chameleon-rl/export.py
"""
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent
RAW, DATA = HERE / "raw", HERE / "data"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    small = lambda p: p.suffix in (".npz", ".json") and not p.name.endswith("_logits.npy")  # noqa: E731
    (DATA / "base").mkdir(parents=True, exist_ok=True)
    for p in (RAW / "cache" / "base").iterdir():
        if small(p):
            shutil.copy2(p, DATA / "base" / p.name)
    for d in sorted((RAW / "whole").glob("*")):
        (DATA / "whole" / d.name).mkdir(parents=True, exist_ok=True)
        for p in d.iterdir():
            if small(p):
                shutil.copy2(p, DATA / "whole" / d.name / p.name)
    for p in sorted((RAW / "runs").glob("*_adapter.pt")):
        shutil.copy2(p, DATA / "runs" / p.name)
    big = sorted(list((RAW / "cache" / "base").glob("*.npy")) + list((RAW / "runs").glob("*_merged.pt")))
    man = {str(p.relative_to(HERE)): {"sha256": sha(p), "bytes": p.stat().st_size} for p in big}
    (DATA / "manifest_raw.json").write_text(json.dumps(man, indent=1))
    print("exported", len(list((DATA / "whole").glob("*"))), "runs; manifest", len(man), "files")


if __name__ == "__main__":
    main()
