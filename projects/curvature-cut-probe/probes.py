"""Build the attacked probes (and the cut sweep) inside subspaces of the post-norm country-token state z (readout 29).

Subspaces (d = 1536):
    U_k  = top-k eigenvectors of Fbar (mean clean output Fisher, fisher.py), largest curvature first
    Q_k  = a Haar-random orthonormal k-frame (seed 1024 + draw)
Probes:
    unproj    the theory-1021 tok29 probe as fitted (raw direction u = w / sd)
    proj_k    REFIT inside span(U_k): StandardScaler + LogisticRegression on x = B^T (z - mu), the 1021 recipe;
              C from {0.01, 0.1, 1, 10} by 5-fold city-GroupKFold out-of-fold AUROC on the 1400 fit-pool statements
              (ties within 0.002 -> smaller C). Raw form: s(z) = u_B . (z - mu) + b_B, u_B = B (w_lr / scale).
    rand_k    the same refit inside span(Q_k)
    lit_k     literal projection U_k U_k^T u of the unproj direction (no refit): a variant for the floor tables only
Clean catch for SELECTION = out-of-fold on the fit pool: OOF scores of the 700 fit-pool false statements against
the line = 7th-highest OOF score of the 700 fit-pool true statements (k = floor(0.01 * 701), the 1019 rule).
Reported clean catch = eval: 596 eval false vs the 2nd-highest of the 200 refit-split true statements (1019-1023 rule).
Selection rule (Felix #6024, posted in the room before this script ran): at every k on GRID, random draws 0..4 in
order, the first whose rand_k keeps OOF clean catch >= 0.90 is that k's random control (draws used reported; if none
passes, the best-OOF draw is kept and flagged). k* = the SMALLEST k where proj_k AND that k's random control both keep
OOF clean catch >= 0.90.
Outputs: data/fits/probes.npz (unproj, proj, rand, plus proj_k<k>, rand_k<k>, lit_k<k> for the sweep cuts),
data/fits/cut_grid.json.
Usage: PY probes.py
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
GRID = [8, 16, 32, 64, 128, 256, 512]          # frozen, Felix #6024
CS = [0.01, 0.1, 1.0, 10.0]
SEED = 1024
N_DRAWS = 5                                     # frozen, Felix #6024


def kth_line(neg, fpr=0.01):
    s = np.sort(neg)[::-1]
    return float(s[max(1, int(np.floor(fpr * (len(s) + 1)))) - 1])


class Ctx:
    def __init__(self):
        split = json.loads((ROOT / "data/ref/split.json").read_text())
        self.sp = np.array([r["split"] for r in split])
        self.y = np.array([1 - r["true"] for r in split])
        self.city = np.array([r["city"] for r in split])
        self.fit = np.where(np.isin(self.sp, ["probe", "attack", "stop"]))[0]
        self.line_i = np.where((self.sp == "refit") & (self.y == 0))[0]
        self.evf = np.where((self.sp == "eval") & (self.y == 1))[0]
        self.Z = np.load(ROOT / "data/clean/z_post.npy").astype(np.float64)   # clean post-norm country-token states (fisher.py)
        P = np.load(ROOT / "data/fits/panel.npz")
        self.mu = P["tok29__mu"].astype(np.float64)
        self.u = P["tok29__w"].astype(np.float64) / P["tok29__sd"].astype(np.float64)
        self.b = float(P["tok29__b"])
        self.Zc = self.Z - self.mu
        self.gkf = list(GroupKFold(5).split(self.fit, groups=self.city[self.fit]))

    def eval_catch(self, v):
        s = self.Zc @ v
        return float((s[self.evf] > kth_line(s[self.line_i])).mean())

    def refit(self, B):
        """Refit inside span(B). Returns raw direction, offset, chosen C, OOF catch, OOF AUROC, n_iter."""
        X = self.Zc[self.fit] @ B
        yy = self.y[self.fit]
        best = None
        for C in CS:
            oof = np.zeros(len(yy))
            for tr, te in self.gkf:
                sc = StandardScaler().fit(X[tr])
                m = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X[tr]), yy[tr])
                oof[te] = m.decision_function(sc.transform(X[te]))
            auc = roc_auc_score(yy, oof)
            if best is None or auc > best[1] + 0.002:
                best = (C, auc, oof)
        C, auc, oof = best
        oof_catch = float((oof[yy == 1] > kth_line(oof[yy == 0])).mean())
        sc = StandardScaler().fit(X)
        m = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X), yy)
        a = m.coef_[0] / sc.scale_
        v = B @ a
        off = float(m.intercept_[0] - a @ sc.mean_)
        return v, off, C, oof_catch, float(auc), int(m.n_iter_[0])


def main():
    c = Ctx()
    E = np.load(ROOT / "data/fisher/Fbar_eig.npz")
    ev, U = E["evals"][::-1], E["evecs"][:, ::-1]             # descending curvature
    d = len(c.u)
    frames = {}

    def Q(k, draw):
        if draw not in frames:
            frames[draw] = np.linalg.qr(np.random.default_rng(SEED + draw).standard_normal((d, d)))[0]
        return frames[draw][:, :k]                             # nested random frames per draw

    flat_bar = ev < 1e-6 * ev[0]                               # Fbar eigen-directions below 1e-6 of its top

    def fbar_flat_share(v):
        cc = U.T @ v
        return float((cc[flat_bar] ** 2).sum() / (cc ** 2).sum())

    rows, fits = [], {}
    for k in GRID:
        vp, op, Cp, ocp, ap, itp = c.refit(U[:, :k])
        draws = []
        for dr in range(N_DRAWS):
            vr, orr, Cr, ocr, ar, itr = c.refit(Q(k, dr))
            draws.append({"draw": dr, "C": Cr, "oof_auc": ar, "oof_catch": ocr, "eval_catch": c.eval_catch(vr),
                          "n_iter": itr, "v": vr, "off": orr})
            if ocr >= 0.90:
                break
        ok = [x for x in draws if x["oof_catch"] >= 0.90]
        ctrl = ok[0] if ok else max(draws, key=lambda x: x["oof_catch"])
        lit = U[:, :k] @ (U[:, :k].T @ c.u)
        fits[k] = {"proj": (vp, op), "rand": (ctrl["v"], ctrl["off"]), "lit": (lit, c.b)}
        rows.append({"k": k, "eig_at_cut": float(ev[k - 1]), "eig_rel_to_top": float(ev[k - 1] / ev[0]),
                     "u_share_in_Uk": float(lit @ lit / (c.u @ c.u)),
                     "proj": {"C": Cp, "oof_auc": ap, "oof_catch": ocp, "eval_catch": c.eval_catch(vp), "n_iter": itp,
                              "fbar_flat_share_1e-6": fbar_flat_share(vp)},
                     "rand": {"draws_used": len(draws), "passed": bool(ok), "control_draw": ctrl["draw"],
                              "draws": [{kk: vv for kk, vv in x.items() if kk not in ("v", "off")} for x in draws],
                              "fbar_flat_share_1e-6": fbar_flat_share(ctrl["v"])},
                     "lit_eval_catch": c.eval_catch(lit), "lit_fbar_flat_share_1e-6": fbar_flat_share(lit)})
        print(json.dumps({"k": k, "eig_rel": round(rows[-1]["eig_rel_to_top"], 9), "proj_oof": round(ocp, 3),
                          "proj_eval": round(rows[-1]["proj"]["eval_catch"], 3), "rand_oof": round(ctrl["oof_catch"], 3),
                          "rand_eval": round(ctrl["eval_catch"], 3), "rand_draws": len(draws),
                          "lit_eval": round(rows[-1]["lit_eval_catch"], 3), "u_share": round(rows[-1]["u_share_in_Uk"], 4)}), flush=True)
    good = [r["k"] for r in rows if r["proj"]["oof_catch"] >= 0.90 and r["rand"]["passed"]]
    kstar = min(good) if good else None
    cu = c.eval_catch(c.u)
    out = {"rule": "k* = smallest k on GRID where proj_k AND the first passing of <=5 random draws both keep OOF clean catch >= 0.90",
           "grid": GRID, "kstar": kstar, "eval_catch_unproj": cu, "unproj_fbar_flat_share_1e-6": fbar_flat_share(c.u),
           "smallest_k_proj_alone": min([r["k"] for r in rows if r["proj"]["oof_catch"] >= 0.90], default=None), "rows": rows}
    (ROOT / "data/fits/cut_grid.json").write_text(json.dumps(out, indent=1))
    if kstar is None:
        raise SystemExit("no k passes the rule; see cut_grid.json")
    probes, scale = {}, {}
    # D2 (Felix #6040, Kasper #6039): scale every probe by c so that its clean mean (s(false) - s(true twin))^2 over the
    # attack twin pairs equals unproj's; the score offset is irrelevant to that gap, the catch and the floor.
    src = np.array([r["src"] for r in json.loads((ROOT / "data/ref/split.json").read_text())])
    tw = []
    for i in np.where((c.sp == "attack") & (c.y == 1))[0]:
        j = np.where((c.sp == "attack") & (c.y == 0) & (c.city == c.city[i]) & (src == src[i]))[0]
        assert len(j) == 1
        tw.append((i, int(j[0])))
    tw = np.array(tw)
    gap2 = lambda v: float(np.mean((c.Zc[tw[:, 0]] @ v - c.Zc[tw[:, 1]] @ v) ** 2))
    g_u = gap2(c.u)

    def put(name, v, b):
        cc = float(np.sqrt(g_u / gap2(v)))
        scale[name] = cc
        probes[f"{name}__readout"] = np.array(29)
        probes[f"{name}__mu"] = c.mu.astype(np.float32)
        probes[f"{name}__sd"] = np.ones(d, np.float32)
        probes[f"{name}__w"] = (cc * v).astype(np.float32)
        probes[f"{name}__b"] = np.array(cc * b, np.float32)

    put("unproj", c.u, c.b)
    for k in GRID:
        for kind in ("proj", "rand", "lit"):
            put(f"{kind}_k{k}", *fits[k][kind])
    put("proj", *fits[kstar]["proj"])
    put("rand", *fits[kstar]["rand"])
    np.savez(ROOT / "data/fits/probes.npz", **probes)
    np.savez(ROOT / "data/fits/subspaces.npz", U_kstar=U[:, :kstar], Q_kstar=Q(kstar, rows[GRID.index(kstar)]["rand"]["control_draw"]),
             kstar=kstar, fbar_evals_desc=ev)
    out["logit_scale_c"] = scale
    out["attack_pairs_mean_gap2_unproj"] = g_u
    (ROOT / "data/fits/cut_grid.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"kstar": kstar, "eval_catch_unproj": cu, "smallest_k_proj_alone": out["smallest_k_proj_alone"],
                      "c_proj": scale["proj"], "c_rand": scale["rand"]}))


if __name__ == "__main__":
    main()
