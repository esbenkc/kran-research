"""Vera (1023): own rebuild of the title metric and the prereg numbers from build's RAW saved states.
Reads only data/runs/<name>/acts_<step>.npy (fp16 country-token states), ckpt_<step>.npz (kl per position, koff, cpos,
saved scores used ONLY for a cross-check), never build's scripts or results.json.
Clean reference = the run's own step-0 states (adapter at init), cross-checked against the 1021 clean acts.
usage: vera_check.py <run name> [<run name> ...]   -> vera_check_<name>.json
"""
import json
import sys

import numpy as np

from vera_core import (EVF, EVT, HERE, LINE_T, OLD, PAIRS, PROJ, CITY, EV_IDX, R_metric, catch, energy, gap, line,
                       probe, restore, score)

import os
RUNS = PROJ / os.environ.get("VERA_RUNS", "data/runs")


def states(run, step):
    d = RUNS / run
    z = np.load(d / f"ckpt_{step}.npz")
    a = np.load(d / f"acts_{step}.npy", mmap_mode="r")
    ro = list(z["acts_readouts"])
    return (np.asarray(a[:, ro.index(28)], np.float64), np.asarray(a[:, ro.index(29)], np.float64),
            np.asarray(a[:, ro.index(27)], np.float64), z)


def eval_kl(z):
    kl, off, cpos = z["kl"].astype(np.float64), z["koff"], z["cpos"]
    tok = np.concatenate([kl[off[i]:off[i + 1]] for i in EV_IDX])
    ctry = np.array([kl[off[i] + cpos[i]] if cpos[i] < off[i + 1] - off[i] else kl[cpos[i]] for i in EV_IDX])
    return float(tok.mean()), float(ctry.mean())


def city_boot_R(Hc, Ha, n=1000, seed=1023):
    """city-cluster bootstrap of R over the 596 pairs (pairs resampled by city)."""
    s_c, s_a, s_r = score(Hc, 28), score(Ha, 28), score(restore(Ha, Hc), 28)
    dc = s_c[PAIRS[:, 0]] - s_c[PAIRS[:, 1]]; da = s_a[PAIRS[:, 0]] - s_a[PAIRS[:, 1]]; dr = s_r[PAIRS[:, 0]] - s_r[PAIRS[:, 1]]
    cities = np.array(sorted(set(CITY[PAIRS[:, 0]]))); cid = np.searchsorted(cities, CITY[PAIRS[:, 0]])
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        wc = np.bincount(rng.integers(0, len(cities), len(cities)), minlength=len(cities))[cid]
        gc, ga, gr = (wc * dc).sum(), (wc * da).sum(), (wc * dr).sum()
        out.append((gr - ga) / (gc - ga))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def check(run):
    H28c, H29c, H27c, z0 = states(run, 0)
    # clean cross-check vs 1021 clean acts
    T = np.load(OLD / "data/acts/clean/tok.npy", mmap_mode="r"); o = np.load(OLD / "data/acts/clean/off.npy")
    ref28 = np.asarray(T[o[1:] - 2, 28], np.float64)
    cosc = (ref28 * H28c).sum(1) / np.linalg.norm(ref28, axis=1) / np.linalg.norm(H28c, axis=1)
    s28c, s29c = score(H28c, 28), score(H29c, 29)
    t28c, t29c = line(s28c), line(s29c)
    res = {"run": run, "clean_vs_1021_cos_min": float(cosc.min()), "clean_catch28": catch(s28c, t28c),
           "clean_catch29": catch(s29c, t29c), "ckpts": []}
    steps = sorted(int(p.stem.split("_")[1]) for p in (RUNS / run).glob("ckpt_*.npz"))
    for st in steps:
        H28, H29, _, z = states(run, st)
        s28, s29 = score(H28, 28), score(H29, 29)
        sv = z["scores"].astype(np.float64); sc = list(z["scored"])
        r = R_metric(H28c, H28)
        # common-factor restore: ONE factor = mean |h_clean|/|h'| over the 200 line-setting trues
        fac = np.mean(np.linalg.norm(H28c[LINE_T], axis=1) / np.linalg.norm(H28[LINE_T], axis=1))
        gcom = gap(score(H28 * fac, 28))
        r["R_common"] = (gcom - r["g_att"]) / (r["g_clean"] - r["g_att"]) if st else float("nan")
        # exact radial share of the score change on false eval: (c_i - 1) a_i / (s' - s)
        mu, sd, w, b = probe(28); v = w / sd
        hc, ha = H28c[EVF], H28[EVF]
        c = (ha * hc).sum(1) / (hc * hc).sum(1)
        a = hc @ v
        ds = ha @ v - a
        r["score_radial_share_false"] = float(((c - 1) * a).sum() / ds.sum()) if st else float("nan")
        # order-robust norm shares on the twin gap (mine, not prereg): s - b0 = n * u, n = |h|, u = v . h_hat
        if st:
            nc, na = np.linalg.norm(H28c, axis=1), np.linalg.norm(H28, axis=1)
            uc, ua = (H28c @ v) / nc, (H28 @ v) / na
            norm_part = (na - nc) * (uc + ua) / 2  # Shapley split: exact, norm_part + dir_part = a' - a
            dsa = H28 @ v - H28c @ v
            f_, t_ = PAIRS[:, 0], PAIRS[:, 1]
            r["R_shapley_norm"] = float((norm_part[f_] - norm_part[t_]).sum() / (dsa[f_] - dsa[t_]).sum())
            # forward: apply ONLY each statement's length change to its clean state
            gfwd = gap(score(H28c * (na / nc)[:, None], 28))
            r["R_forward"] = (r["g_clean"] - gfwd) / (r["g_clean"] - r["g_att"])
            sr = score(restore(H28, H28c), 28)
            r["catch28_restored_reset"] = catch(sr, line(sr))
            r["catch28_restored_cleanline"] = catch(sr, t28c)
        r.update(energy(H28c, H28) if st else {})
        kl_all, kl_c = eval_kl(z)
        row = {"step": st, **r,
               "catch28_reset": catch(s28, line(s28)), "catch28_clean": catch(s28, t28c),
               "catch29_reset": catch(s29, line(s29)), "catch29_clean": catch(s29, t29c),
               "eval_kl": kl_all, "eval_kl_country": kl_c,
               "score_xcheck_maxabs28": float(np.abs(sv[:, sc.index(28)] - s28).max()),
               "score_xcheck_maxabs29": float(np.abs(sv[:, sc.index(29)] - s29).max())}
        if st == steps[-1]:
            row["R_ci_city"] = city_boot_R(H28c, H28)
        res["ckpts"].append(row)
        print(run, st, f"R={r['R']:.4f} Rsh={r.get('R_shapley_norm', 0):.4f} Rfw={r.get('R_forward', 0):.4f} Rcom={r['R_common']:.4f} drop={r['drop_frac']:.3f} E={r.get('E', 0):.3f} "
              f"c28reset={row['catch28_reset'][0]:.3f} c28clean={row['catch28_clean'][0]:.3f} "
              f"c29reset={row['catch29_reset'][0]:.3f} kl={kl_all:.5f} klc={kl_c:.5f} nr={r.get('norm_ratio_false', 1):.3f}/{r.get('norm_ratio_true', 1):.3f}",
              flush=True)
    json.dump(res, open(HERE / f"vera_check_{run}.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    for run in sys.argv[1:]:
        check(run)
