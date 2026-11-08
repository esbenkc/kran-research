"""D8 (Felix #17294): every arm trains the same 12k steps and is analysed at step 12000 (primary);
the registered matched-loss pick (first eval <= L*) is measured from the same runs as a secondary row.
Writes queues/d8_*.txt. uv run python make_d8.py
"""
from pathlib import Path

PY = "uv run python"
ORG = "--organism sc --burst 3 --beta 4 --n-trig 32 --wd 0.5 --d 64"
T = f"{PY} train.py {ORG} --eval-every 100 --save-every-eval --max-steps 12000"
FINE = f"{PY} train.py {ORG} --eval-every 10 --save-every-eval --max-steps 36000 --target $(cat results/Lstar.txt)"
lanes = {
    "d8_A": ["gated_s0", "kbias_s0", "sinklogit_s0"],
    "d8_B": ["gated_s1", "kbias_s1", "sinklogit_s1"],
    "d8_C": ["gated_s2", "softmax1_s0"],
    "d8_D": ["softmax_s3", "softmax_s4", "softmax1_s1"],
}
q = {}
for n, runs in lanes.items():
    q[n] = [f"{T} --attn {r.split('_s')[0]} --seed {r.split('_s')[1]} --name {r} > logs/{r}.train.log 2>&1"
            for r in runs]
q["d8_D"] = [f"{FINE} --attn softmax --seed {s} --name softmax_s{s}_fine > logs/softmax_s{s}_fine.train.log 2>&1"
             for s in (1, 4)] + q["d8_D"]   # fine-eval reruns (1% rule) run first in lane D
order = ["softmax_s0", "softmax_s1", "softmax_s2", "gated_s0", "gated_s1", "gated_s2", "softmax_s3",
         "kbias_s0", "kbias_s1", "softmax_s4", "softmax1_s0", "sinklogit_s0", "sinklogit_s1",
         "softmax1_s1"]
M = {"d8_M1": [], "d8_M2": []}
for i, r in enumerate(order):
    extra = " --bayes" if r == "softmax_s0" else ""
    M["d8_M1" if i % 2 == 0 else "d8_M2"].append(
        f"{PY} grid_wait.py measure_at {r} 12000 --layer 2 --save-train 512{extra} > logs/{r}.measure12k.log 2>&1")
tail = []
for r in ["softmax_s0", "softmax_s2", "gated_s0", "gated_s1", "gated_s2", "softmax_s3", "kbias_s0",
          "kbias_s1", "softmax1_s0", "softmax1_s1", "sinklogit_s0", "sinklogit_s1",
          "softmax_s1_fine", "softmax_s4_fine"]:
    tail.append(f"{PY} grid_wait.py measure {r} --layer 2 > logs/{r}.measureL.log 2>&1")
for arm in ("softmax", "gated", "kbias", "softmax1", "sinklogit"):
    tail.append(f"{PY} measure.py --run {arm}_s0 --untrained --layer 2 > logs/{arm}_s0.untrained.log 2>&1")
tail.append(f"{PY} measure.py --run softmax_s0 --untrained --layer 0 --tag embed_L0 > logs/softmax_s0.embed.log 2>&1")
for r in ["softmax_s0", "gated_s0", "softmax_s1", "gated_s1", "softmax_s2", "gated_s2"]:
    tail.append(f"{PY} measure.py --run {r} --ckpt 12000 --layer 2 --rmsnorm > logs/{r}.rms.log 2>&1")
for r in ["softmax_s0", "gated_s0"]:
    for k in (4, 16):
        tail.append(f"{PY} measure.py --run {r} --ckpt 12000 --layer 2 --k {k} > logs/{r}.k{k}.log 2>&1")
for i, line in enumerate(tail):
    M["d8_M1" if i % 2 == 0 else "d8_M2"].append(line)
q.update(M)
for n, lines in q.items():
    Path(f"queues/{n}.txt").write_text("\n".join(lines + ["END"]) + "\n")
    print(n, len(lines))
