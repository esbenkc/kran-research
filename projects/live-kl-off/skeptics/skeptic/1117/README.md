# Vera (skeptic) receipts — live-kl-off

Own rebuild code, run on the build's raw files; see signoff.md for the verdict and every number.
Run from projects/live-kl-off (pool: data/pool_cols/test/cols.npz, sha256 f085c32fae41...):

    python3 skeptics/skeptic/vera_check.py data/runs/kl0_s0 data/pool_cols/test/cols.npz
    python3 skeptics/skeptic/vera_check.py data/runs/posctl_d086_s0 data/pool_cols/test/cols.npz
    python3 skeptics/skeptic/vera_check.py data/runs/replay_s0 data/pool_cols/test/cols.npz
    python3 skeptics/skeptic/vera_carry.py data/runs/kl0_s0
    python3 skeptics/skeptic/vera_served.py data/reads/served_kl0_s0_1200.npz
    python3 skeptics/skeptic/vera_drift.py <parent live_k4 dir> data/runs/kl0_s0

numpy only. vera_drift.py needs the parent's live_k4 batches (branch swarm/oracle-1113-on-policy-planted).
