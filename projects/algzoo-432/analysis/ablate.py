"""T7/C3 (Iris): per-parameter sensitivity table for all 432 entries of M_{16,10}.

Run: .venv/bin/python ablate.py   -> ablate.json, prints the matrices used in blocks.md.
Eval seed 3, n = 1e5 (dev). Metric = disagreement (%) with the unmodified model's argmax.

Per entry w:
  zero  : set w := 0
  pert  : set w := w * (1 + 0.01)        (1% relative change of this single entry)
Block level: zero a whole neuron (row+col of W_hh, W_hi entry, W_oh column) and per-block 0.1% noise.
"""
import json

import numpy as np

from core import load_raw

Wi0, Wh0, Wo0 = load_raw()
N = 100_000
x = np.random.default_rng(3).standard_normal((N, 10))

ROLE = {  # neuron roles; sources: ARC post, Zihao Chen notes, subA.md (Bruno), denseB2.md, PLAN.md
    0: "lag-2 max-event timer: off iff the max arrived exactly 2 steps ago (denseB2)",
    1: "deficit register lag 2: max was 2 inputs ago, value = gap (subA)",
    2: "leaky running max, n2 <- 0.969 n2 + n4 (subA)",
    3: "2nd-clock oscillator coordinate, clips value-dependently (denseB1)",
    4: "new-max increment ReLU(x - 0.852 n2 - 0.986 n4) (subA)",
    5: "lag-1 comparator: x_{t-1} beats ~2nd max of prefix (denseB2)",
    6: "deficit register lag 1 (subA)",
    7: "max minus last input; read only via deficits n_k - n7 (subA)",
    8: "deficit register lags >= 3, decays ~0.6/step (subA)",
    9: "lag-1 new-max gate: on iff x_{t-1} was a new running max (denseB2)",
    10: "value holder + max clock carrier, rare clips (denseB1)",
    11: "value holder + max clock carrier, rare clips (denseB1)",
    12: "linear carrier: value holder + max clock (denseB1)",
    13: "linear carrier, mostly 2nd-clock modes (denseB1)",
    14: "value holder + max clock carrier, rare clips; moderate input -1.32 (denseB1)",
    15: "2nd-clock oscillator coordinate, clips often (denseB1)",
}
GROUP = {n: "A" for n in (1, 2, 4, 6, 7, 8)}
GROUP.update({n: "gate" for n in (0, 5, 9)})
GROUP.update({n: "B" for n in (3, 10, 11, 12, 13, 14, 15)})


def predict(Wi, Wh, Wo):
    h = np.zeros((N, 16))
    for t in range(10):
        h = np.maximum(h @ Wh.T + x[:, t : t + 1] * Wi[None, :], 0.0)
    return (h @ Wo.T).argmax(1)


ref = predict(Wi0, Wh0, Wo0)


def dis(Wi, Wh, Wo):
    return 100.0 * float(np.mean(predict(Wi, Wh, Wo) != ref))


def main():
    rows = []
    for i in range(16):
        for kind, mult in (("zero", 0.0), ("pert", 1.01)):
            Wi = Wi0.copy(); Wi[i] *= mult
            rows.append(dict(block="W_hi", i=i, j=None, w=float(Wi0[i]), kind=kind, dis=dis(Wi, Wh0, Wo0)))
    for i in range(16):
        for j in range(16):
            for kind, mult in (("zero", 0.0), ("pert", 1.01)):
                Wh = Wh0.copy(); Wh[i, j] *= mult
                rows.append(dict(block="W_hh", i=i, j=j, w=float(Wh0[i, j]), kind=kind, dis=dis(Wi0, Wh, Wo0)))
    for k in range(10):
        for j in range(16):
            for kind, mult in (("zero", 0.0), ("pert", 1.01)):
                Wo = Wo0.copy(); Wo[k, j] *= mult
                rows.append(dict(block="W_oh", i=k, j=j, w=float(Wo0[k, j]), kind=kind, dis=dis(Wi0, Wh0, Wo)))

    neuron = []
    for n in range(16):
        Wi, Wh, Wo = Wi0.copy(), Wh0.copy(), Wo0.copy()
        Wi[n] = 0; Wh[n, :] = 0; Wh[:, n] = 0; Wo[:, n] = 0
        neuron.append(dict(n=n, group=GROUP[n], role=ROLE[n], zero_neuron_dis=dis(Wi, Wh, Wo)))

    rng = np.random.default_rng(0)
    noise = {}
    for name in ("W_hi", "W_hh", "W_oh"):
        vals = []
        for _ in range(3):
            Wi, Wh, Wo = Wi0.copy(), Wh0.copy(), Wo0.copy()
            M = {"W_hi": Wi, "W_hh": Wh, "W_oh": Wo}[name]
            M *= 1 + 0.001 * rng.standard_normal(M.shape)
            vals.append(dis(Wi, Wh, Wo))
        noise[name] = float(np.mean(vals))

    json.dump(dict(seed=3, n=N, rows=rows, neuron=neuron, noise_0p1pct=noise), open("ablate.json", "w"), indent=1)

    np.set_printoptions(linewidth=200, precision=1, suppress=True)

    def mat(block, kind, shape):
        m = np.zeros(shape)
        for r in rows:
            if r["block"] == block and r["kind"] == kind:
                if block == "W_hi":
                    m[r["i"]] = r["dis"]
                else:
                    m[r["i"], r["j"]] = r["dis"]
        return m

    for block, shape in (("W_hi", (16,)), ("W_hh", (16, 16)), ("W_oh", (10, 16))):
        for kind in ("zero", "pert"):
            print(f"\n{block} {kind} disagreement % (rows = target/logit, cols = source neuron):")
            print(mat(block, kind, shape))
    print("\nneuron zero-ablation:", [(d["n"], round(d["zero_neuron_dis"], 1)) for d in neuron])
    print("block 0.1% noise:", noise)
    for kind in ("zero", "pert"):
        d = np.array([r["dis"] for r in rows if r["kind"] == kind])
        print(f"{kind}: entries with dis < 0.1%: {(d < 0.1).sum()}, < 1%: {(d < 1).sum()}, >= 10%: {(d >= 10).sum()} of {len(d)}")


if __name__ == "__main__":
    main()
