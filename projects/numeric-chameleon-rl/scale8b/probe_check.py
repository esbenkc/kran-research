"""Pre-RL trace check on the base model (task step 2): does the leaky build change bits on CUDA, and do the 0-bit
builds change none? Compares cache.py output dirs, prompt by prompt.

    python probe_check.py DIR [REPEAT_DIR ...]
For each build pair (ref|score vs none, ref_emu vs none_emu) and pass: prompts whose logit bits differ, max |dlogit|,
top-1 flips, mean |dP(X)|; for the rec pass also the block-34 input. REPEAT_DIRs: the same build compiled in another
process must give identical logit bits (determinism).
"""
import json
import sys
from pathlib import Path

import numpy as np


def f32(u16):
    return (np.ascontiguousarray(u16).astype(np.uint32) << 16).view(np.float32)


def main() -> None:
    d = Path(sys.argv[1])
    out = {"dir": str(d), "meta": json.load(open(d / "meta.json"))["builds"]}
    for ps in ("plain", "rec"):
        for on, off in (("ref", "none"), ("score", "none"), ("ref_emu", "none_emu"), ("none_emu", "none")):
            if not ((d / f"{on}_{ps}_sha.json").exists() and (d / f"{off}_{ps}_sha.json").exists()):
                continue
            sa, sb = json.load(open(d / f"{on}_{ps}_sha.json")), json.load(open(d / f"{off}_{ps}_sha.json"))
            la, lb = f32(np.load(d / f"{on}_{ps}_logits.npy")), f32(np.load(d / f"{off}_{ps}_logits.npy"))
            wa, wb = np.load(d / f"{on}_{ps}.npz"), np.load(d / f"{off}_{ps}.npz")
            r = {"n": len(sa), "logits_changed": int(sum(x != y for x, y in zip(sa, sb))),
                 "max_dlogit": float(np.abs(la - lb).max()), "top1_flips": int((wa["top"] != wb["top"]).sum()),
                 "px_off": float(wb["px"].mean()), "px_on": float(wa["px"].mean()),
                 "mean_abs_dpx": float(np.abs(wa["px"] - wb["px"]).mean())}
            ta, tb = d / f"{on}_{ps}_tail.npy", d / f"{off}_{ps}_tail.npy"
            if ta.exists() and tb.exists():
                A, B = np.load(ta), np.load(tb)
                r["block_in_changed"] = int((A != B).reshape(len(A), -1).any(1).sum())
                r["block_in_maxabs"] = float(np.abs(f32(A) - f32(B)).max())
                r["block_in_frac_elems_changed"] = float((A != B).mean())
            out[f"{on}_vs_{off}_{ps}"] = r
    for rd in sys.argv[2:]:
        rd = Path(rd)
        for f in sorted(rd.glob("*_sha.json")):
            if (d / f.name).exists():
                a, b = json.load(open(d / f.name)), json.load(open(f))
                out[f"repeat_{rd.name}_{f.name[:-9]}"] = {"n": len(a), "differ": int(sum(x != y for x, y in zip(a, b)))}
    print(json.dumps(out, indent=1))
    (d / "probe_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
