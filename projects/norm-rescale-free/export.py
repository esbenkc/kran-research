"""Receipts for skeptics: committed slices of the local per-checkpoint states + checksums of the full files.

data/states/clean/states_r28_r29.npy  fp16 (2992, 2, 1536): clean country-token states, readouts 28 and 29
                                      (= acts_0.npy of every run; checked bit-equal across runs here)
data/states/<run>/r28.npy             fp16 (n_ckpt, 1392, 1536): readout-28 country-token states at every checkpoint
                                      (steps in steps.npy) for the 200 line-setting true + 1192 eval statements
data/states/<run>/r29_last.npy        fp16 (1392, 1536): readout 29 at the last checkpoint, same statements
data/states/idx.npy                   the 1392 statement indices (rows of data/ref/split.json), sorted
data/acts_sha256.txt                  sha256 of every local data/runs/<run>/acts_<step>.npy (not committed)
Everything else a skeptic needs is committed as is: data/runs/<run>/ckpt_<step>.npz (scores, per-position KL and
log-probs), log.json, kl_pile.npy, adapter_300/ and adapter_150/ (rebuild any state with the model).
Run: PY export.py
"""
import hashlib

import numpy as np

from common import DATA, RUNS, rows

_, y, _, split = rows()
idx = np.sort(np.concatenate([np.where((split == "refit") & (y == 0))[0], np.where(split == "eval")[0]]))
S = DATA / "states"
(S / "clean").mkdir(parents=True, exist_ok=True)
np.save(S / "idx.npy", idx)
runs = sorted(p.name for p in RUNS.iterdir() if (p / "log.json").exists())
clean = None
sha = []
for run in runs:
    steps = sorted(int(p.stem.split("_")[1]) for p in (RUNS / run).glob("ckpt_*.npz"))
    a0 = np.load(RUNS / run / "acts_0.npy", mmap_mode="r")
    c = np.asarray(a0[:, [28, 29]])
    if clean is None:
        clean = c
        np.save(S / "clean" / "states_r28_r29.npy", clean)
    assert np.array_equal(c, clean), f"{run}: step-0 states differ from the first run"
    out = S / run
    out.mkdir(exist_ok=True)
    R28 = np.zeros((len(steps), len(idx), 1536), np.float16)
    for k, st in enumerate(steps):
        ck = np.load(RUNS / run / f"ckpt_{st}.npz")
        A = np.load(RUNS / run / f"acts_{st}.npy", mmap_mode="r")
        R = list(ck["acts_readouts"])
        R28[k] = A[idx, R.index(28)]
        if st == steps[-1]:
            np.save(out / "r29_last.npy", np.asarray(A[idx, R.index(29)]))
        h = hashlib.sha256()
        with open(RUNS / run / f"acts_{st}.npy", "rb") as f:
            for blk in iter(lambda: f.read(1 << 24), b""):
                h.update(blk)
        sha.append(f"{h.hexdigest()}  data/runs/{run}/acts_{st}.npy")
    np.save(out / "r28.npy", R28)
    np.save(out / "steps.npy", np.array(steps))
    print(run, steps, R28.shape, flush=True)
(DATA / "acts_sha256.txt").write_text("\n".join(sha) + "\n")
