"""Write the lane queues for the counted grid (PREREG s2, s7).

uv run python make_grid.py --org "--organism sc --burst 3 --beta 4 --n-trig 32 --n-copy 3" --M 12000 \
    --layer 4 [--lanes 3] [--one-seed-side]

Trainer lanes A, B, C (light, ~0.09 GB each): softmax s0-s2 first (they set L* at step M), then each
lane waits for L* and trains the other arms to the target. Measure lanes M1, M2 (heavy, <=1.8 GB each;
at most 2 at once, Vera #17213) pick each model's matched-loss checkpoint and run measure.py, then the
floors (untrained model per arm, embeddings h_0).
"""
from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).parent
PY = "uv run python"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True)
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--M-cap", type=int, default=12000)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--one-seed-side", action="store_true",
                    help="softmax1 and sinklogit 1 seed each (PREREG s2 time rule)")
    ap.add_argument("--extra-softmax", type=int, default=0, help="seed growth: softmax seeds 3..")
    ap.add_argument("--extra-kbias", type=int, default=0)
    a = ap.parse_args()

    base = f"{PY} train.py {a.org} --d {a.d} --eval-every 100 --save-every-eval"
    soft = [f"softmax_s{s}" for s in range(3 + a.extra_softmax)]
    tgt = "--target $(cat results/Lstar.txt) --max-steps " + str(3 * a.M_cap)
    others = [("gated", 0), ("gated", 1), ("gated", 2), ("kbias", 0), ("kbias", 1),
              ("softmax1", 0), ("sinklogit", 0)]
    if not a.one_seed_side:
        others += [("softmax1", 1), ("sinklogit", 1)]
    others += [("kbias", 2 + i) for i in range(a.extra_kbias)]
    lanes = {"A": [], "B": [], "C": []}
    names = ["A", "B", "C"]
    for i, r in enumerate(soft):
        s = int(r.split("_s")[1])
        lanes[names[i % 3]].append(f"{base} --attn softmax --seed {s} --max-steps {a.M_cap}"
                                   f" --stop-file results/stop_softmax --name {r}"
                                   f" > logs/{r}.train.log 2>&1")
    Ms = ",".join(str(m) for m in range(a.M, a.M_cap + 1, 2000))
    lanes["L"] = [f"{PY} grid_wait.py lstar {Ms} {' '.join(soft[:3])} > logs/lstar.log 2>&1", "END"]
    for n in names:
        lanes[n].append("while [ ! -f results/Lstar.txt ]; do sleep 10; done")
    for i, (arm, s) in enumerate(others):
        r = f"{arm}_s{s}"
        lanes[names[i % 3]].append(f"{base} --attn {arm} --seed {s} {tgt} --name {r}"
                                   f" > logs/{r}.train.log 2>&1")
    for n in names:
        lanes[n].append("END")
    lanes = {k: lanes[k] for k in ["A", "B", "C", "L"]}
    runs = soft + [f"{arm}_s{s}" for arm, s in others]
    meas = {"M1": [], "M2": []}
    for i, r in enumerate(runs):
        meas["M1" if i % 2 == 0 else "M2"].append(
            f"{PY} grid_wait.py measure {r} --layer {a.layer} --save-train 512"
            f" > logs/{r}.measure.log 2>&1")
    floors = []
    for arm in ("softmax", "gated", "kbias", "softmax1", "sinklogit"):
        r = f"{arm}_s0"
        floors.append(f"{PY} measure.py --run {r} --untrained --layer {a.layer}"
                      f" > logs/{r}.untrained.log 2>&1")
    floors.append(f"{PY} measure.py --run softmax_s0 --untrained --layer 0 --tag embed_L0"
                  f" > logs/softmax_s0.embed.log 2>&1")
    for i, f in enumerate(floors):
        meas["M1" if i % 2 == 0 else "M2"].append(f)
    for n in meas:
        meas[n].append("END")
    qd = HERE / "queues"
    qd.mkdir(exist_ok=True)
    for n, lines in {**lanes, **meas}.items():
        (qd / f"{n}.txt").write_text("\n".join(lines) + "\n")
        print(f"--- {n}"); print("\n".join(lines))


if __name__ == "__main__":
    main()
