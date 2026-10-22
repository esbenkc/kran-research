"""Step 1 table + manifest from the raw bf16 bits in raw/step1/<model>/<backend>_<tag>/.

    .venv/bin/python projects/probe-sensing-mps/step1_table.py [--model Qwen3-1.7B]
writes raw/step1/<model>/step1_table.json and raw/step1/<model>/manifest.jsonl (one row per array file, SHA256).
All comparisons are on raw uint16 bit patterns; max-abs diffs are of the bf16 values cast to float32.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
POS_USER = 1  # token index of 'user' in '<|im_start|>user' (same in every prompt)


def f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def cmp(a: np.ndarray, b: np.ndarray) -> dict:
    """a, b: (n_prompts, ...) uint16 bit arrays."""
    ne = a != b
    per = ne.reshape(len(a), -1).any(1)
    d = np.abs(f32(a).astype(np.float64) - f32(b).astype(np.float64))
    return {"prompts_changed": int(per.sum()), "n_prompts": len(a), "frac_elems_changed": float(ne.mean()),
            "max_abs": float(d.max())}


def load(d: Path, name: str):
    p = d / f"{name}.npy"
    return np.load(p) if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-1.7B")
    a = ap.parse_args()
    root = HERE / "raw" / "step1" / a.model
    table = {}
    backends = sorted({p.name.rsplit("_", 1)[0] for p in root.iterdir() if p.is_dir()})
    for be in backends:
        A, B = root / f"{be}_A", root / f"{be}_B"
        if not A.exists():
            continue
        t = {"controls": {}, "builds": {}}
        nA = load(A, "none_plain_logits")
        t["controls"]["none_vs_none_in_process"] = [cmp(nA[0], nA[r]) for r in range(1, len(nA))]
        if B.exists() and load(B, "none_plain_logits") is not None:
            t["controls"]["none_vs_none_fresh_process"] = cmp(nA[0], load(B, "none_plain_logits")[0])
        nb = load(A, "none-b_plain_logits")
        if nb is not None:
            t["controls"]["none_vs_none_separate_compile"] = cmp(nA[0], nb[0])
        rec = load(A, "none_rec_logits")
        if rec is not None:
            t["controls"]["recorder_effect_on_logits"] = cmp(nA[0], rec[0])
        tailA = load(A, "none_rec_tail")
        if B.exists() and load(B, "none_rec_tail") is not None:
            t["controls"]["none_tail_fresh_process"] = cmp(tailA, load(B, "none_rec_tail"))
        for build in ["probe", "copy", "twok", "fp32"]:
            x = load(A, f"{build}_plain_logits")
            if x is None:
                continue
            row = {"logits_vs_none": cmp(x[0], nA[0])}
            if len(x) > 1:
                row["logits_hook_vs_hook_in_process"] = cmp(x[0], x[1])
                row["logits_vs_none_warm_pass"] = cmp(x[1], nA[0])  # 2nd pass: every buffer already exists
            xb = load(A, f"{build}-b_plain_logits")
            if xb is not None:
                row["logits_hook_vs_hook_separate_compile"] = cmp(x[0], xb[0])
            if B.exists() and load(B, f"{build}_plain_logits") is not None:
                row["logits_hook_vs_hook_fresh_process"] = cmp(x[0], load(B, f"{build}_plain_logits")[0])
            tl = load(A, f"{build}_rec_tail")
            if tl is not None and tailA is not None:
                row["tail_vs_none"] = cmp(tl, tailA)
                du = f32(tl[:, POS_USER]).astype(np.float64) - f32(tailA[:, POS_USER]).astype(np.float64)
                row["user_token_tail_diff_norm"] = float(np.linalg.norm(du[0]))
                row["user_token_tail_diff_same_all_prompts"] = bool((tl[:, POS_USER] == tl[0, POS_USER]).all()
                                                                    and (tailA[:, POS_USER] == tailA[0, POS_USER]).all())
                rl = load(A, f"{build}_rec_logits")
                row["rec_logits_vs_none_rec"] = cmp(rl[0], rec[0])
            t["builds"][build] = row
        metaA = json.load(open(A / "meta.json"))
        t["meta"] = {k: metaA[k] for k in metaA if k != "prompt_ids"}
        table[be] = t
    (root / "step1_table.json").write_text(json.dumps(table, indent=1))
    with open(root / "manifest.jsonl", "w") as f:
        for p in sorted(root.glob("*/*.npy")):
            be, tag = p.parent.name.rsplit("_", 1)
            build, ps, what = p.stem.split("_")
            arr = np.load(p, mmap_mode="r")
            f.write(json.dumps({
                "path": str(p.relative_to(HERE)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "model": a.model, "backend": be, "process_tag": tag, "build": build.split("-")[0],
                "compilation": "second" if build.endswith("-b") else "first",
                "pass": ps, "array": what, "shape": list(arr.shape), "dtype": "bf16 bits stored as uint16",
                "axes": "(repeat, prompt, vocab)" if what == "logits" else "(prompt, position, hidden)",
                "prompt_ids_from": str((p.parent / "meta.json").relative_to(HERE))}) + "\n")
    for be, t in table.items():
        print(f"== {be}")
        for k, v in t["controls"].items():
            print("  ctrl", k, v)
        for b, r in t["builds"].items():
            lv = r["logits_vs_none"]
            tv = r.get("tail_vs_none", {})
            print(f"  {b:6s} logits: {lv['prompts_changed']}/{lv['n_prompts']} prompts, frac {lv['frac_elems_changed']:.3f},"
                  f" max {lv['max_abs']:.4g} | tail: {tv.get('prompts_changed')} prompts, max {tv.get('max_abs')}"
                  f" | user-token diff {r.get('user_token_tail_diff_norm')} const={r.get('user_token_tail_diff_same_all_prompts')}"
                  f" | warm: {r.get('logits_vs_none_warm_pass', {}).get('prompts_changed')}"
                  f" | repeat: {[ (k, r[k]['prompts_changed']) for k in r if 'hook_vs_hook' in k]}")


if __name__ == "__main__":
    main()
