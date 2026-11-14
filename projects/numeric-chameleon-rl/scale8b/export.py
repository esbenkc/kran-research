"""Copy the small per-prompt results from $NC_RAW to $NC_DATA (then `modal volume get` -> scale8b/data/, git), and
write manifest_raw.json. 8B port of ../export.py: LoRA adapters are 9.7 MB each at 8B, so only the primary arm's
status-checkpoint adapters (r50 kl1, 8,192 updates) are copied; every adapter and merged tail gets a SHA256 line.

  raw/cache/base/<build>_<pass>.npz, meta.json, *_sha.json        -> data/base/
  raw/whole/<run>/<build>_plain.npz, _gen.json, _sha.json, meta   -> data/whole/<run>/
  raw/runs/r50_s*_lr0.0001_kl1_adapter.pt                         -> data/runs/
"""
import hashlib
import json
import shutil

from tail import DATA, RAW


def sha(p) -> str:
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
    (DATA / "runs").mkdir(parents=True, exist_ok=True)
    for p in sorted((RAW / "runs").glob("r50_s*_lr0.0001_kl1_adapter.pt")):
        shutil.copy2(p, DATA / "runs" / p.name)
    big = sorted(list((RAW / "cache" / "base").glob("*.npy")) + list((RAW / "runs").glob("*.pt")))
    man = {str(p.relative_to(RAW)): {"sha256": sha(p), "bytes": p.stat().st_size} for p in big}
    (DATA / "manifest_raw.json").write_text(json.dumps(man, indent=1))
    print("exported", len(list((DATA / "whole").glob("*"))), "runs; manifest", len(man), "files")


if __name__ == "__main__":
    main()
