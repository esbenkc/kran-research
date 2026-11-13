"""Pick Figure 1 candidate rows from the TEST pool (Owain). CPU only: reads the saved end read c_<step>.npy.

A base row qualifies when its probe-off thought does not already start with X, decodes to plain ASCII, ends its first
sentence inside the 64 tokens (readable in a bubble), and its X twin differs from it only at the first thought token
(the pool's construction; asserted). Rows are ranked by typicality of the per-row forced switch c(none) - c(X) at the
end read: distance to the median over all qualifying rows, in IQR units, plus distance of c(none) to the median
cheat_off. The top rows go to twin.py, which adds the probe-on twin.

    ../../../env/.venv/bin/python figure1/pick.py --run d010_s0 --step 1500 --top 16
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="d010_s0")
    ap.add_argument("--step", default="1500")
    ap.add_argument("--top", type=int, default=16)
    a = ap.parse_args()
    te = HERE / "data" / "pool" / "test"
    c = np.load(te / "cols.npz")
    meta = json.loads((te / "meta.json").read_text())
    X = meta["X"]
    var, base, toks = c["variant"], c["base_row"], c["toks"]
    nb = meta["n_base"]
    rn, rx = np.empty(nb, int), np.empty(nb, int)
    rn[base[var == 0]] = np.where(var == 0)[0]
    rx[base[var == 1]] = np.where(var == 1)[0]
    assert (toks[rx][:, 0] == X).all() and (toks[rx][:, 1:] == toks[rn][:, 1:]).all()
    cc = np.load(HERE / "data" / "runs" / a.run / "reads" / f"c_{a.step}.npy").astype(np.float64)
    c0 = np.load(HERE / "data" / "runs" / a.run / "reads" / "c_0.npy").astype(np.float64)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    keep = ~c["x1_orig"][rn].astype(bool)
    sw = cc[rn] - cc[rx]
    co = cc[rn]
    print(f"run {a.run} step {a.step}: cheat_off {co.mean():.3f}  c_X {cc[rx][keep].mean():.3f}  forced switch "
          f"{sw[keep].mean():+.3f}  s_P {sw[keep].mean() / co.mean():.3f}  (start forced {(c0[rn] - c0[rx])[keep].mean():+.4f})")

    def ok(b):
        s = tok.decode(toks[rn[b]])
        return all(32 <= ord(ch) < 127 or ch == "\n" for ch in s) and "." in s[:200]
    q = [b for b in range(nb) if keep[b] and ok(b)]
    q1, med, q3 = np.percentile(sw[q], [25, 50, 75])
    mco = np.median(co[q])
    print(f"qualifying rows {len(q)} of {nb}; per-row forced switch median {med:+.3f} IQR [{q1:+.3f}, {q3:+.3f}]; "
          f"c(none) median {mco:.3f}")
    sc = sorted(q, key=lambda b: abs(sw[b] - med) / max(q3 - q1, 1e-3) + abs(co[b] - mco) / 0.1)
    out = []
    for b in sc[: a.top]:
        r = {"base_row": int(b), "prompt_row": int(c["prompt_row"][rn[b]]), "draw": int(c["draw"][rn[b]]),
             "c_none": float(cc[rn[b]]), "c_x": float(cc[rx[b]]), "c0_none": float(c0[rn[b]]), "c0_x": float(c0[rx[b]]),
             "logw": float(c["logw"][rn[b]]), "text": tok.decode(toks[rn[b]])}
        out.append(r)
        print(b, f"none {r['c_none']:.2f} X {r['c_x']:.2f} (start {r['c0_none']:.2f}/{r['c0_x']:.2f}) w {np.exp(r['logw']):.2f}",
              repr(r["text"][:110]))
    (HERE / "figure1" / f"pick_{a.run}_{a.step}.json").write_text(json.dumps(
        {"run": a.run, "step": a.step, "median_sw": med, "iqr": [q1, q3], "rows": out}, indent=1))


if __name__ == "__main__":
    main()
