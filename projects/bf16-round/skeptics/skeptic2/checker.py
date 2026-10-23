#!/usr/bin/env python3
"""Independent raw-data checker for oracle-1023-bf16-round.

Reads saved arrays only. Does not import or call the build scripts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DEFAULT = Path("kran-research/projects/bf16-round/raw")
OUT_DEFAULT = Path("swarm-oracle-1023-bf16-round/skeptic2")


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def sha_file(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _count_over_prompt_axis(a: np.ndarray, b: np.ndarray, prompt_axis: int) -> dict[str, Any]:
    if prompt_axis != 0:
        a = np.moveaxis(a, prompt_axis, 0)
        b = np.moveaxis(b, prompt_axis, 0)
    n = int(a.shape[0]) if a.ndim else 1
    changed = []
    diff_elems = []
    max_abs = []
    for i in range(n):
        ai = np.asarray(a[i]) if a.ndim else np.asarray(a)
        bi = np.asarray(b[i]) if b.ndim else np.asarray(b)
        neq = ai != bi
        c = int(np.count_nonzero(neq))
        changed.append(bool(c))
        diff_elems.append(c)
        if c and np.issubdtype(ai.dtype, np.number):
            max_abs.append(int(np.max(np.abs(ai.astype(np.int64) - bi.astype(np.int64)))))
        else:
            max_abs.append(0)
    return {
        "changed_prompts": int(sum(changed)),
        "n_prompts": n,
        "changed_indices": [i for i, x in enumerate(changed) if x],
        "diff_elements_total": int(sum(diff_elems)),
        "diff_elements_by_prompt": diff_elems,
        "max_uint16_delta_by_prompt": max_abs,
    }


def count_changed_prompts(a_path: Path, b_path: Path) -> dict[str, Any]:
    a = np.load(a_path, mmap_mode="r")
    b = np.load(b_path, mmap_mode="r")
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a_path} {a.shape} vs {b_path} {b.shape}")
    if a.dtype != b.dtype:
        raise ValueError(f"dtype mismatch {a_path} {a.dtype} vs {b_path} {b.dtype}")
    base = {
        "a": str(a_path),
        "b": str(b_path),
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "sha256_a": sha_file(a_path),
        "sha256_b": sha_file(b_path),
    }
    # Build arrays use either (prompt, position, vocab/hidden) or (pass, prompt, vocab).
    # For the pass form, keep each pass separate so pass 0 warm-up cannot hide pass 1-3 failures.
    if a.ndim >= 3 and a.shape[0] <= 8 and a.shape[1] in (64, 128, 256):
        passes = []
        for j in range(a.shape[0]):
            passes.append(_count_over_prompt_axis(np.asarray(a[j]), np.asarray(b[j]), 0))
        warm = passes[1:] if len(passes) > 1 else passes
        base.update({
            "pass_axis": True,
            "passes": passes,
            "changed_prompts": int(max((x["changed_prompts"] for x in warm), default=0)),
            "n_prompts": int(a.shape[1]),
            "changed_indices": sorted(set(i for x in warm for i in x["changed_indices"])),
            "diff_elements_total": int(sum(x["diff_elements_total"] for x in warm)),
        })
        return base
    base.update(_count_over_prompt_axis(a, b, 0))
    return base

def collect_pairs(bits_dir: Path, suffix: str) -> list[tuple[str, Path, Path]]:
    # suffix examples: alllogits_pass1.npy, last.npy, hidden.npy
    paths = sorted(bits_dir.glob(f"*_{suffix}"))
    by_key: dict[tuple[str, str], dict[str, Path]] = {}
    pat = re.compile(r"^(?P<fix>.+)_(?P<probe>no)?probe_(?P<rep>[A-Z])_" + re.escape(suffix) + r"$")
    for p in paths:
        m = pat.match(p.name)
        if not m:
            continue
        fix = m.group("fix")
        probe = "noprobe" if m.group("probe") == "no" else "probe"
        rep = m.group("rep")
        by_key.setdefault((fix, rep), {})[probe] = p
    out = []
    for (fix, rep), d in sorted(by_key.items()):
        if "noprobe" in d and "probe" in d:
            out.append((f"{fix}_{rep}_{suffix}", d["noprobe"], d["probe"]))
    return out


def collect_determinism(bits_dir: Path, suffix: str) -> list[tuple[str, Path, Path]]:
    paths = sorted(bits_dir.glob(f"*_{suffix}"))
    pat = re.compile(r"^(?P<fix>.+)_(?P<state>noprobe|probe)_(?P<rep>[A-Z])_" + re.escape(suffix) + r"$")
    by_key: dict[tuple[str, str], dict[str, Path]] = {}
    for p in paths:
        m = pat.match(p.name)
        if not m:
            continue
        by_key.setdefault((m.group("fix"), m.group("state")), {})[m.group("rep")] = p
    out = []
    for (fix, state), reps in sorted(by_key.items()):
        if "A" in reps and "B" in reps:
            out.append((f"{fix}_{state}_A_vs_B_{suffix}", reps["A"], reps["B"]))
    return out


def summarize_bits(root: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for bits_dir in sorted((root / "bits").glob("Qwen*/compile_*")):
        key = str(bits_dir.relative_to(root))
        entry: dict[str, Any] = {"dir": str(bits_dir), "meta": None, "pairs": {}, "determinism": {}}
        meta = bits_dir / "meta.json"
        if meta.exists():
            entry["meta"] = load_json(meta)
        suffixes = sorted({"_".join(p.name.split("_")[3:]) for p in bits_dir.glob("*.npy")})
        # The generic suffix parse above is brittle for fix names with underscores. Use known endings too.
        suffixes = sorted(set(suffixes) | {"alllogits_pass1.npy", "alllogits_pass2.npy", "last.npy", "hidden.npy"})
        for suffix in suffixes:
            for name, a, b in collect_pairs(bits_dir, suffix):
                try:
                    entry["pairs"][name] = count_changed_prompts(a, b)
                except Exception as e:
                    entry["pairs"][name] = {"error": repr(e), "a": str(a), "b": str(b)}
            for name, a, b in collect_determinism(bits_dir, suffix):
                try:
                    entry["determinism"][name] = count_changed_prompts(a, b)
                except Exception as e:
                    entry["determinism"][name] = {"error": repr(e), "a": str(a), "b": str(b)}
        summary[key] = entry
    return summary


def find_number(obj: Any, key_needles: tuple[str, ...]) -> list[tuple[str, Any]]:
    hits = []
    def rec(x: Any, path: str):
        if isinstance(x, dict):
            for k, v in x.items():
                p = f"{path}.{k}" if path else str(k)
                if any(n.lower() in str(k).lower() for n in key_needles):
                    hits.append((p, v))
                rec(v, p)
        elif isinstance(x, list):
            for i, v in enumerate(x):
                rec(v, f"{path}[{i}]")
    rec(obj, "")
    return hits


def summarize_json_metrics(root: Path) -> dict[str, Any]:
    out = {}
    for p in sorted(root.rglob("*.json")):
        if "codegate" in p.parts:
            continue
        try:
            obj = load_json(p)
        except Exception:
            continue
        hits = find_number(obj, ("accuracy", "acc", "changed", "median", "speed", "correct", "n_correct"))
        if hits:
            out[str(p.relative_to(root))] = hits[:100]
    return out


def binomial_upper_tail(k: int, n: int, p: float) -> float:
    # P[X >= k] for X~Bin(n,p), stable enough for n<=512 here.
    s = 0.0
    for i in range(k, n + 1):
        s += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return s


def write_markdown(out_dir: Path, summary: dict[str, Any]) -> None:
    lines = ["# skeptic2 checker output", ""]
    lines.append("## Bit pairs")
    for key, entry in summary.get("bits", {}).items():
        lines.append(f"### {key}")
        pairs = entry.get("pairs", {})
        if not pairs:
            lines.append("No complete probe/noprobe array pairs found.")
        for name, r in pairs.items():
            if "error" in r:
                lines.append(f"- {name}: ERROR {r['error']}")
            else:
                lines.append(f"- {name}: {r['changed_prompts']}/{r['n_prompts']} changed prompts; diff elements {r['diff_elements_total']}")
        det = entry.get("determinism", {})
        if det:
            lines.append("Determinism floor:")
            for name, r in det.items():
                if "error" in r:
                    lines.append(f"- {name}: ERROR {r['error']}")
                else:
                    lines.append(f"- {name}: {r['changed_prompts']}/{r['n_prompts']} changed prompts")
        lines.append("")
    lines.append("## JSON metrics found")
    for path, hits in summary.get("json_metrics", {}).items():
        lines.append(f"- {path}")
        for k, v in hits[:20]:
            if isinstance(v, (int, float, str, bool)) or v is None:
                lines.append(f"  - {k}: {v}")
    lines.append("")
    (out_dir / "checker_output.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "root": str(args.root),
        "bits": summarize_bits(args.root),
        "json_metrics": summarize_json_metrics(args.root),
        "nulls": {
            "binom_75pct_or_more_if_chance_n256": binomial_upper_tail(192, 256, 0.5),
            "binom_60pct_or_more_if_chance_n256": binomial_upper_tail(154, 256, 0.5),
        },
    }
    (args.out / "checker_output.json").write_text(json.dumps(summary, indent=2))
    write_markdown(args.out, summary)
    print(args.out / "checker_output.md")

if __name__ == "__main__":
    main()
