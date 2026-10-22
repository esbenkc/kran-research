"""Compact, committable receipts from the full raw arrays (~4 GB, regenerable, not in git).

Writes data/ next to this file:
  prompts.jsonl                      the 576 frozen prompts (id, split, text)
  step1/step1_table.json, copy_check.json, manifest_full_arrays.jsonl (SHA256 of every full array)
  step1/fingerprints.jsonl           per (backend, process, build, pass, repeat, prompt): SHA256 of the
                                     answer-position logit bits, its Yes/No logits, and (rec pass) SHA256 of the
                                     block-26 input bits
  step1/prefix_vectors.json          block-26 input at positions 0-2 (the shared chat prefix) per build, as bf16 bits
  step2/runs/*.json, step2/whole/*.json   cached-tail and whole-model per-case results

    .venv/bin/python projects/probe-sensing-mps/export_compact.py
"""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RAW = HERE / "raw"
OUT = HERE / "data"
YES, NO = 9454, 2753  # Qwen3 tokenizer ids of 'Yes' and 'No' (asserted below from a run file)


def f32(u):
    return (np.asarray(u).astype(np.uint32) << 16).view(np.float32)


def main() -> None:
    (OUT / "step1").mkdir(parents=True, exist_ok=True)
    (OUT / "step2").mkdir(parents=True, exist_ok=True)
    shutil.copy2(RAW / "prompts.jsonl", OUT / "prompts.jsonl")
    s1 = RAW / "step1" / "Qwen3-1.7B"
    for f in ("step1_table.json", "copy_check.json"):
        if (s1 / f).exists():
            shutil.copy2(s1 / f, OUT / "step1" / f)
    shutil.copy2(s1 / "manifest.jsonl", OUT / "step1" / "manifest_full_arrays.jsonl")
    runs = sorted((RAW / "step2" / "Qwen3-1.7B" / "runs").glob("*.json"))
    if runs:
        r0 = json.load(open(runs[0]))
        assert (r0["yes_id"], r0["no_id"]) == (YES, NO), (r0["yes_id"], r0["no_id"])
    prefix = {}
    with open(OUT / "step1" / "fingerprints.jsonl", "w") as fp:
        for d in sorted(p for p in s1.iterdir() if p.is_dir()):
            be, tag = d.name.rsplit("_", 1)
            meta = json.load(open(d / "meta.json"))
            for lf in sorted(d.glob("*_logits.npy")):
                build, ps, _ = lf.stem.split("_")
                lg = np.load(lf, mmap_mode="r")
                tl = np.load(d / f"{build}_{ps}_tail.npy", mmap_mode="r") if ps == "rec" else None
                for r in range(lg.shape[0]):
                    for i, pid in enumerate(meta["prompt_ids"]):
                        row = {"backend": be, "process": tag, "build": build, "pass": ps, "repeat": r, "prompt": pid,
                               "logits_sha256": hashlib.sha256(np.ascontiguousarray(lg[r, i]).tobytes()).hexdigest(),
                               "logit_yes": float(f32(lg[r, i, YES])), "logit_no": float(f32(lg[r, i, NO]))}
                        if tl is not None and r == 0:
                            row["block26_input_sha256"] = hashlib.sha256(np.ascontiguousarray(tl[i]).tobytes()).hexdigest()
                        fp.write(json.dumps(row) + "\n")
                if tl is not None and tag == "A":
                    prefix[f"{be}/{build}"] = {f"pos{p}": tl[0, p].tolist() for p in range(3)}
    (OUT / "step1" / "prefix_vectors.json").write_text(json.dumps(
        {"note": "block-26 input of the first dev prompt at positions 0-2 ('<|im_start|>', 'user', newline), bf16 bits as "
                 "uint16; identical in every prompt within a build (causal attention, shared prefix)", "vectors": prefix}))
    for sub in ("runs", "whole", "swap"):
        src = RAW / "step2" / "Qwen3-1.7B" / sub
        if src.exists():
            (OUT / "step2" / sub).mkdir(parents=True, exist_ok=True)
            for f in src.glob("*.json"):
                shutil.copy2(f, OUT / "step2" / sub / f.name)
    tot = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"wrote {OUT} ({tot / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
