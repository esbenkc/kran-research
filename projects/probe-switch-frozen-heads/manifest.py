"""SHA256 + size of every file under raw/ -> data/manifest_raw.json; copies the LoRA adapters (small) to data/adapters/
so the skeptics can rebuild every tail without raw/.

    PY manifest.py
"""
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    out = {str(p.relative_to(HERE)): {"sha256": sha(p), "bytes": p.stat().st_size}
           for p in sorted((HERE / "raw").rglob("*")) if p.is_file()}
    (HERE / "data" / "manifest_raw.json").write_text(json.dumps(out, indent=1))
    dst = HERE / "data" / "adapters"
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted((HERE / "raw" / "runs").glob("*_adapter.pt")):
        shutil.copy2(p, dst / p.name)
    print(len(out), "raw files hashed;", len(list(dst.glob("*.pt"))), "adapters in data/adapters")


if __name__ == "__main__":
    main()
