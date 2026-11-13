"""Write queue.txt (lane2.sh format: name|gate|rl.py args) in the frozen PREREG v2 order.

  1. no-cue seeds 0..9 x 256 updates (nc_s<k>; 0 and 1 rebuild c2_a0_s0/s1 at shared snapshots)
  2. Iris's cue-free diagnostic: SM twins of held-out h6 at t = 16
  3. SM twins (status): 6 cue models x {t_h, T} and held-out h6..h9 x {8, 16, 24, 32}, K = 4 data seeds (target seed + 100..103)
  4. SM curve: the remaining snapshots 8..256 for the 6 models and h6..h9, data seed +100 only (PREREG: '1 seed if time')
  5. SMv (rowspace init) and 6. SM0 (A0 init) at {t_h, T} (held-out at 8, 16, 24, 32), K = 4
  7. SM curve, the other 3 data seeds (+101..103), if time
Twin = --cue-alpha 0 (same reward and flags, no trigger), --steps t, --snap t, --no-eval, --init-from <target step t>.
    python3 make_queue.py > queue.txt
"""
MODELS = {f"m{k}": (f"dir_u{1110 + k}_a0.03125_s{k}", k) for k in range(6)}
T = {"m0": 32, "m1": 32, "m2": 32, "m3": 32, "m4": 24, "m5": 32}
TH = {"m0": 16, "m1": 16, "m2": 16, "m3": 16, "m4": 8, "m5": 16}
HELD = {f"h{j}": (f"nc_s{j}", j) for j in (6, 7, 8, 9)}
SNAPS = [8, 16, 24, 32, 48, 64, 128, 256]
NC_SNAPS = "0,8,16,24,32,48,64,128,256"
K = 4

lines = []


def twin(arm, tgt, t, d):
    run, seed = {**MODELS, **HELD}[tgt]
    init_step = 0 if arm == "sm0" else t
    mode = "rowspace" if arm == "smv" else "exact"
    gate = f"after:{run}" if tgt.startswith("h") else "none"
    name = f"{arm}_{tgt}_t{t}_d{seed + 100 + d}"
    lines.append(f"{name}|{gate}|--cue-alpha 0 --seed {seed + 100 + d} --steps {t} --full --no-eval --snap {t} "
                 f"--init-from raw/runs/{run}/step{init_step}.pt --init-mode {mode}")


for k in range(10):
    lines.append(f"nc_s{k}|none|--cue-alpha 0 --seed {k} --steps 256 --full --no-eval --snap {NC_SNAPS}")
for d in range(K):
    twin("sm", "h6", 16, d)


def status_reads(tgt):
    return sorted({TH[tgt], T[tgt]}) if tgt in MODELS else [8, 16, 24, 32]


for arm in ["sm"]:
    for tgt in list(MODELS) + list(HELD):
        for t in status_reads(tgt):
            for d in range(K):
                if not (arm == "sm" and tgt == "h6" and t == 16):
                    twin(arm, tgt, t, d)
for tgt in list(MODELS) + list(HELD):
    for t in SNAPS:
        if t not in status_reads(tgt):
            twin("sm", tgt, t, 0)
for arm in ["smv", "sm0"]:
    for tgt in list(MODELS) + list(HELD):
        for t in status_reads(tgt):
            for d in range(K):
                twin(arm, tgt, t, d)
for tgt in list(MODELS) + list(HELD):
    for t in SNAPS:
        if t not in status_reads(tgt):
            for d in range(1, K):
                twin("sm", tgt, t, d)
print("\n".join(lines))
