"""Frozen grid constants (copied from PREREG.md section 2, FROZEN RULE). Change only with a DEVIATIONS.md entry."""
TRAIN = ["32x32x16r1", "32x32x16r8"]
NBETA = [100, 1000, 10000]
Q = [0.001, 0.01, 0.1, 1, 10, 100]
C = [0.01, 0.03, 0.1]
K4 = 0.5
N_ROWS = 1024           # arm S: noise-free rows
BATCH = 64              # arm S: minibatch rows per step (without replacement within an epoch)
BLOCK = 100             # steps per stored block mean
T_GRID = [1000, 3000, 10000, 30000, 100000, 300000]
T_FALLBACK = 1000000
T_TOL_PTS = 1.0
GIBBS_SWEEPS = 3000
GIBBS_BURN = 1000
GIBBS_BLOCK = 10
THREADS = 2
CHAINS = 4              # scored chains per seed
HALVES_PTS = 2.0        # convergence rule: halves of kept draws agree within 2 points
DIVERGE_E = 10.0        # |e| > 1000 points = diverged
