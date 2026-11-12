"""Vera's own read checker (skeptic, oracle-1112-matched-trace). Written from the raw npz fields only.

Sign convention (from the parent raw json): a switch y = cheat(probe-off build) - cheat(probe-on build),
both decisions taken with the probe off; positive = the model cheats less when the probe's build wrote the thought.
  real trace : y_R = c_nb - c_lb       (none build vs leaky build, banned prompt, CRN pair)
  planted    : y_P = c_n  - c_nc       (plain CRN);  D1 form y_P_rb = D_P * (c_notX - c_X)
  text diff  : D_R = E_nb[max(0, 1 - p_leaky/p_none)] over the 64 thought tokens;  D_P = cue_delta
  share      : s = y / (D * cheat_off), cheat_off = mean c_nb (real) / mean c_n (planted)
CIs: prompt-cluster bootstrap (resample prompts, all rows of a prompt together), ratio recomputed per resample.
"""
import numpy as np


def prompt_groups(pidx):
    u, inv = np.unique(pidx, return_inverse=True)
    return u, inv


def cluster_sums(inv, n_groups, *cols):
    out = []
    for c in cols:
        s = np.zeros(n_groups)
        np.add.at(s, inv, c)
        out.append(s)
    cnt = np.zeros(n_groups)
    np.add.at(cnt, inv, 1.0)
    return out, cnt


def boot_weights(n_groups, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_groups, size=(B, n_groups))
    W = np.zeros((B, n_groups))
    for b in range(B):
        np.add.at(W[b], idx[b], 1.0)
    return W


def ci(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def tv_rows(lp_none, lp_leaky):
    return np.maximum(0.0, 1.0 - np.exp(lp_leaky.sum(1) - lp_none.sum(1)))
