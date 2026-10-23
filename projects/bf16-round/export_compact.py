"""Copy the small receipts from raw/ (not in git, ~30 GB) into data/ (in git).

    .venv/bin/python projects/bf16-round/export_compact.py

data/fingerprints/<setting>/<graph>.json  per pass, per prompt: SHA256 over the 71 per-position row hashes (one hash
                                         per prompt; rebuild: sha256("".join(row_hashes)) from raw *_sha.json)
data/code/<setting>/<graph>.py           generated kernel source of the key graphs (copy A)
data/detectors/emu<e>/<fix>__<run>.json  whole-graph detector results, per case
data/timing/*.json                       speed runs
data/fresh_detector/<run>.json           fresh detector training curve + cached-tail test result
data/run_meta/<setting>.json             bits.py meta (compile times, kernel counts, torch version)
"""
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent
M = "Qwen3-1.7B"
DIRS = {"compile_emu0": HERE / "raw" / "bits" / M / "compile_emu0",
        "compile_emu1": HERE / "raw" / "bits" / M / "compile_emu1",
        "eager_emu0": HERE / "raw" / "bits" / M / "eager_emu0",
        "r2f": HERE / "raw" / "bits_r2f" / M / "compile_emu0",
        "nightly": HERE / "raw" / "bits_nightly" / M / "compile_emu0"}
CODE = {"none", "round16", "round16f", "barrier"}


def main() -> None:
    D = HERE / "data"
    for setting, d in DIRS.items():
        if not (d / "meta.json").exists():
            continue
        (D / "run_meta").mkdir(parents=True, exist_ok=True)
        shutil.copy(d / "meta.json", D / "run_meta" / f"{setting}.json")
        for f in sorted(d.glob("*_sha.json")):
            rows = json.loads(f.read_text())
            fp = [[hashlib.sha256("".join(p).encode()).hexdigest() for p in pas] for pas in rows]
            out = D / "fingerprints" / setting / f.name.replace("_sha.json", ".json")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(fp))
        for f in sorted((d / "code").glob("*_A.py")):
            if f.name.split("_")[0] in CODE:
                out = D / "code" / setting / f.name
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, out)
    for f in sorted((HERE / "raw" / "detectors" / M).glob("emu*/*.json")):
        out = D / "detectors" / f.parent.name / f.name
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, out)
    for f in sorted((HERE / "raw" / "timing").glob("*.json")):
        (D / "timing").mkdir(parents=True, exist_ok=True)
        shutil.copy(f, D / "timing" / f.name)
    for f in sorted((HERE / "raw" / "step2" / M / "runs").glob("*.json")):
        (D / "fresh_detector").mkdir(parents=True, exist_ok=True)
        shutil.copy(f, D / "fresh_detector" / f.name)
    c = HERE / "raw" / "step2" / M / "round16_C" / "meta.json"
    if c.exists():
        shutil.copy(c, D / "fresh_detector" / "round16_C_cache_meta.json")
    print("wrote", D)


if __name__ == "__main__":
    main()
