#!/usr/bin/env python3
"""Rebuild Step-1 bit-diff table from saved raw arrays only.

No imports from the project code. The script assumes arrays are bf16 bit-patterns
stored as uint16, as declared in manifest.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    if bits.dtype != np.uint16:
        return bits.astype(np.float32)
    u32 = bits.astype(np.uint32) << 16
    return u32.view(np.float32)


def arr(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def stat(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    diff = a != b
    if a.ndim < 2:
        per_prompt = diff
    else:
        per_prompt = np.any(diff.reshape(diff.shape[0], -1), axis=1)
    af = bf16_to_f32(a)
    bf = bf16_to_f32(b)
    return {
        "prompts_changed": int(np.count_nonzero(per_prompt)),
        "n_prompts": int(per_prompt.size),
        "frac_elems_changed": float(np.count_nonzero(diff) / diff.size),
        "max_abs": float(np.max(np.abs(af - bf))) if diff.size else 0.0,
    }


def same_all_prompts_tail(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    # a/b shape: prompt, position, hidden.
    d = bf16_to_f32(a) - bf16_to_f32(b)
    rows = []
    for pos in range(d.shape[1]):
        first = d[0, pos]
        max_dev = float(np.max(np.abs(d[:, pos] - first)))
        norm = float(np.linalg.norm(first))
        changed = bool(np.any(a[:, pos] != b[:, pos]))
        rows.append({"position": pos, "norm": norm, "max_deviation_across_prompts": max_dev, "changed": changed})
    const_changed = [r for r in rows if r["changed"] and r["max_deviation_across_prompts"] == 0.0]
    const_changed.sort(key=lambda r: r["norm"], reverse=True)
    return {
        "constant_changed_positions": const_changed[:8],
        "top_constant_norm": const_changed[0]["norm"] if const_changed else 0.0,
        "top_constant_position": const_changed[0]["position"] if const_changed else None,
        "any_constant_changed": bool(const_changed),
    }


def load_manifest(base: Path) -> list[dict[str, Any]]:
    rows = []
    with (base / "manifest.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def verify_manifest(base: Path) -> dict[str, Any]:
    bad = []
    rows = load_manifest(base)
    for r in rows:
        p = base.parents[2] / r["path"]  # projects/probe-sensing-mps + raw/...
        got = sha256(p)
        if got != r["sha256"]:
            bad.append({"path": str(p), "expected": r["sha256"], "got": got})
    return {"files": len(rows), "sha256_bad": bad}


def p(base: Path, backend: str, tag: str, name: str) -> Path:
    return base / f"{backend}_{tag}" / name


def first_repeat(x: np.ndarray) -> np.ndarray:
    if x.ndim >= 3:
        return x[0]
    return x


def repeat(x: np.ndarray, i: int) -> np.ndarray:
    if x.ndim >= 3:
        return x[i]
    if i != 0:
        raise IndexError(i)
    return x


def backend_stats(base: Path, backend: str) -> dict[str, Any]:
    none = arr(p(base, backend, "A", "none_plain_logits.npy"))
    none_rec_logits = arr(p(base, backend, "A", "none_rec_logits.npy"))
    none_tail = arr(p(base, backend, "A", "none_rec_tail.npy"))
    none_b = arr(p(base, backend, "A", "none-b_plain_logits.npy"))
    none_b_tail = arr(p(base, backend, "A", "none-b_rec_tail.npy"))
    none_fresh = arr(p(base, backend, "B", "none_plain_logits.npy"))
    none_fresh_tail = arr(p(base, backend, "B", "none_rec_tail.npy"))

    controls = {
        "none_vs_none_in_process": [stat(repeat(none, 0), repeat(none, 1)), stat(repeat(none, 0), repeat(none, 2))],
        "none_vs_none_fresh_process": stat(repeat(none, 0), repeat(none_fresh, 0)),
        "none_vs_none_separate_compile": stat(repeat(none, 0), first_repeat(none_b)),
        "recorder_effect_on_logits": stat(repeat(none, 0), repeat(none_rec_logits, 0)),
        "none_tail_fresh_process": stat(none_tail, none_fresh_tail),
        "none_tail_separate_compile": stat(none_tail, none_b_tail),
    }

    builds: dict[str, Any] = {}
    for build in ["probe", "copy", "twok", "fp32"]:
        plain = arr(p(base, backend, "A", f"{build}_plain_logits.npy"))
        rec_logits = arr(p(base, backend, "A", f"{build}_rec_logits.npy"))
        tail = arr(p(base, backend, "A", f"{build}_rec_tail.npy"))
        fresh = arr(p(base, backend, "B", f"{build}_plain_logits.npy"))
        bstats: dict[str, Any] = {
            "logits_vs_none": stat(repeat(plain, 0), repeat(none, 0)),
            "logits_hook_vs_hook_fresh_process": stat(repeat(plain, 0), repeat(fresh, 0)),
            "tail_vs_none": stat(tail, none_tail),
            "tail_signature": same_all_prompts_tail(tail, none_tail),
            "rec_logits_vs_none_rec": stat(repeat(rec_logits, 0), repeat(none_rec_logits, 0)),
        }
        if plain.ndim >= 3 and plain.shape[0] > 1:
            bstats["logits_hook_vs_hook_in_process"] = stat(repeat(plain, 0), repeat(plain, 1))
            bstats["logits_vs_none_warm_pass"] = stat(repeat(plain, 1), repeat(none, 1))
        b_path = p(base, backend, "A", f"{build}-b_plain_logits.npy")
        if b_path.exists():
            bstats["logits_hook_vs_hook_separate_compile"] = stat(repeat(plain, 0), first_repeat(arr(b_path)))
        builds[build] = bstats
    return {"controls": controls, "builds": builds}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    base = args.base
    out = {
        "manifest": verify_manifest(base),
        "compile": backend_stats(base, "compile"),
        "eager": backend_stats(base, "eager"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "manifest_files": out["manifest"]["files"],
        "manifest_sha256_bad": len(out["manifest"]["sha256_bad"]),
        "compile_probe_logits": out["compile"]["builds"]["probe"]["logits_vs_none"],
        "compile_copy_logits": out["compile"]["builds"]["copy"]["logits_vs_none"],
        "compile_twok_logits": out["compile"]["builds"]["twok"]["logits_vs_none"],
        "eager_probe_logits": out["eager"]["builds"]["probe"]["logits_vs_none"],
        "eager_fp32_logits": out["eager"]["builds"]["fp32"]["logits_vs_none"],
        "compile_probe_tail_top_constant": out["compile"]["builds"]["probe"]["tail_signature"]["constant_changed_positions"][:3],
    }, indent=2))


if __name__ == "__main__":
    main()
