"""Vera's rebuild of max-pool-8b from raw per-token rows. Own code (vera_core.py); reads only the job's saved data.

Usage: python vera_rebuild.py <data dir> [--boot 2000] [--out vera_rebuild.json]
  <data dir> holds states/, fits/, gate/, results/ (smoke: projects/max-pool-8b/smoke06; real run: .../data).
Reads the build's schema as posted in the room (#26169), never its code:
  states/<s>/index.npz off/tok_id/tok_pos/cpos; tok_<split>.npy fp16 rows (split's statements in split.json order);
  scores.npz (build's scores, only compared); fits/probes.npz (build's fits, only compared); gate/*.json.
"""
import argparse
import json
import sys
from pathlib import Path

import warnings

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore", message=".*encountered in matmul")  # macOS Accelerate false alarms; results are checked

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vera_core as vc  # noqa: E402

SPLITS = ("probe", "attack", "stop", "refit", "eval")


def load_state(D, s, M):
    st = D / "states" / s
    z = np.load(st / "index.npz")
    off, tok_id, cpos = z["off"], z["tok_id"], z["cpos"]
    rows, have = None, np.zeros(len(M["y"]), bool)
    for sp in SPLITS:
        f = st / f"tok_{sp}.npy"
        if not f.exists():
            continue
        a = np.load(f)
        if rows is None:
            rows = np.full((off[-1], a.shape[1]), np.nan, np.float32)
        idx = np.where(M["split"] == sp)[0]
        tr = vc.tok_rows(off, idx)
        assert len(tr) == len(a), (s, sp, len(tr), a.shape)
        rows[tr] = a.astype(np.float32)
        have[idx] = True
    sc = dict(np.load(st / "scores.npz"))
    meta = json.loads((st / "meta.json").read_text())
    kl = np.load(st / "kl_pile.npy") if (st / "kl_pile.npy").exists() else None
    return dict(off=off, tok_id=tok_id, cpos=cpos, rows=rows, have=have, sc=sc, meta=meta, kl=kl)


def stmt_scores(lin, kind, S):
    """Per-statement score of a raw-coordinate linear probe from the fp16 rows (float64 math). NaN where no rows."""
    off, rows = S["off"], S["rows"]
    out = np.full(len(off) - 1, np.nan)
    tok = np.full(off[-1], np.nan)
    if rows is None:
        return out, tok
    have_tok = np.repeat(S["have"], off[1:] - off[:-1])
    tok[have_tok] = rows[have_tok].astype(np.float64) @ lin.w + lin.b
    for i in np.where(S["have"])[0]:
        v = tok[off[i]:off[i + 1]]
        out[i] = {"mean": v.mean(), "max": v.max(), "last": v[-1], "cty": v[S["cpos"][i] - 1]}[kind]
    return out, tok


class Fixed:
    def __init__(self, w, b):
        self.w, self.b = np.asarray(w, np.float64), float(b)

    def __call__(self, X):
        return np.asarray(X, np.float64) @ self.w + self.b


def fit_kind(kind, S, M, which):
    idx = np.where((M["split"] == "probe") & S["have"])[0]
    off = S["off"]
    sub_off = np.r_[0, np.cumsum((off[1:] - off[:-1])[idx])]
    toks = S["rows"][vc.tok_rows(off, idx)]
    y = M["y"][idx] if which is None else which[idx]
    if kind == "cty":   # single-token readout: put the country row as a 1-token 'statement'
        crow = np.array([off[i] + S["cpos"][i] - 1 for i in idx])
        toks, sub_off, kind2 = S["rows"][crow], np.arange(len(idx) + 1), "last"
    else:
        kind2 = kind
    lin, cv = vc.fit_monitor(kind2, toks, sub_off, y, M["city"][idx])
    return lin, cv


def catch_of(score, M):
    return vc.catch(score, M["y"], M["split"])


def first_diff(M, S):
    """Per statement: position (1-based token index into rows, i.e. row offset within statement) of the first token
    that differs from its twin; twin = other statement with the same city|src."""
    off, tid = S["off"], S["tok_id"]
    fd = np.full(len(off) - 1, -1)
    groups = {}
    for i, p in enumerate(M["pair"]):
        groups.setdefault(p, []).append(i)
    for p, ii in groups.items():
        if len(ii) != 2:
            continue
        a, b = (tid[off[i]:off[i + 1]] for i in ii)
        n = min(len(a), len(b))
        k = next((j for j in range(n) if a[j] != b[j]), n)
        fd[ii[0]] = fd[ii[1]] = k   # row index within the statement (row 0 = token 1)
    return fd, groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--split", default=None)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="vera_rebuild.json")
    a = ap.parse_args()
    D = Path(a.data)
    proj = D.parent if (D.parent / "PREREG.md").exists() else D.parent
    split_json = Path(a.split) if a.split else next(p for p in (D / "ref/split.json", proj / "data/ref/split.json") if p.exists())
    M = vc.meta(split_json)
    R = {"data": str(D), "split_json": str(split_json)}
    states = sorted(p.name for p in (D / "states").iterdir())
    print("states:", states)

    # ---------------------------------------------------------------- clean fits (mine) vs build's
    C = load_state(D, "clean", M)
    bp = np.load(D / "fits/probes.npz")
    fits, cmp = {}, {}
    for k in ("mean", "max", "last", "cty"):
        lin, cv = fit_kind(k, C, M, None)
        fits[k] = lin
        wb = bp[f"{k}__w"]
        cmp[k] = dict(my_C=lin.C, build_C=float(bp[f"{k}__C"]), cv=cv,
                      cos_w=float(lin.w @ wb / np.linalg.norm(lin.w) / np.linalg.norm(wb)))
        print(f"fit {k}: my C {lin.C} build C {float(bp[f'{k}__C'])} cos(w) {cmp[k]['cos_w']:.6f}")
    R["fits"] = cmp
    build_fits = {k: Fixed(bp[f"{k}__w"], bp[f"{k}__b"]) for k in ("mean", "max", "last", "cty")}

    # ---------------------------------------------------------------- per-state scores, catches
    per = {}
    scores_mine, scores_build = {}, {}
    fd, groups = first_diff(M, C)
    ev_false = np.where((M["split"] == "eval") & (M["y"] == 1))[0]
    for s in states:
        S = C if s == "clean" else load_state(D, s, M)
        P = {"have_splits": sorted(set(M["split"][S["have"]]))}
        tok_scores = {}
        for k in ("mean", "max", "last", "cty"):
            mine, tok = stmt_scores(fits[k], k, S)
            theirs_rebuilt, _ = stmt_scores(build_fits[k], k, S)
            bsc = S["sc"][k].astype(np.float64)
            m = S["have"]
            P[k] = dict(
                build_scores_vs_rows_maxabs=float(np.nanmax(np.abs(theirs_rebuilt[m] - bsc[m]))) if m.any() else None,
                catch_mine=catch_of(mine, M) if (S["have"][M["split"] == "eval"].all() and S["have"][M["split"] == "refit"].all()) else None,
                catch_build_scores=catch_of(bsc, M),
            )
            scores_mine[(s, k)], scores_build[(s, k)] = mine, bsc
            tok_scores[k] = tok
        # stop-split gate on build's avg18 scores (stop rows not stored on trained states)
        bm = S["sc"]["mean"].astype(np.float64)
        t = vc.line(bm[(M["split"] == "refit") & (M["y"] == 0)])
        P["gate_stop_catch_build_scores"] = float(np.mean(bm[(M["split"] == "stop") & (M["y"] == 1)] > t))
        P["gate_line"] = float(t)
        # mechanism: argmax position of max18 on eval false, shared prefix share
        if S["rows"] is not None and S["have"][ev_false].all():
            tk = tok_scores["max"]
            am = np.array([int(np.argmax(tk[S["off"][i]:S["off"][i + 1]])) for i in ev_false])
            P["argmax_on_shared_prefix"] = float(np.mean(am < fd[ev_false]))
            P["argmax_rel_hist"] = {int(r): int(c) for r, c in zip(*np.unique(am - fd[ev_false], return_counts=True))}
            # prefix lemma on eval pairs
            mx, norms = 0.0, []
            for p, ii in groups.items():
                if len(ii) != 2 or M["split"][ii[0]] != "eval":
                    continue
                k0 = fd[ii[0]]
                if k0 <= 0:
                    continue
                A = S["rows"][S["off"][ii[0]]:S["off"][ii[0]] + k0]
                B = S["rows"][S["off"][ii[1]]:S["off"][ii[1]] + k0]
                mx = max(mx, float(np.abs(A - B).max()))
                norms.append(np.linalg.norm(A, axis=1))
            P["prefix_lemma_maxabs"] = mx
            P["prefix_lemma_rel"] = mx / float(np.median(np.concatenate(norms)))
        # twin gap by position (s9.3) and answer-token-only max (s9.4), eval split, MY fits
        if S["rows"] is not None and S["have"][M["split"] == "eval"].all() and S["have"][M["split"] == "refit"].all():
            ev = np.where(M["split"] == "eval")[0]
            gap = {}
            for k in ("max", "mean"):
                tk = tok_scores[k]
                g = {}
                for r in range(-6, 5):
                    vals = {0: [], 1: []}
                    for i in ev:
                        j = fd[i] + r
                        if 0 <= j < S["off"][i + 1] - S["off"][i]:
                            vals[M["y"][i]].append(tk[S["off"][i] + j])
                    if len(vals[0]) > 5 and len(vals[1]) > 5:
                        g[r] = dict(diff=float(np.mean(vals[1]) - np.mean(vals[0])),
                                    sd=float(np.sqrt((np.var(vals[0], ddof=1) + np.var(vals[1], ddof=1)) / 2)))
                gap[k] = g
            P["twin_gap_raw"] = gap
            ans = np.full(len(M["y"]), np.nan)
            for i in np.where(S["have"])[0]:
                ans[i] = tok_scores["max"][S["off"][i] + fd[i]:S["off"][i + 1]].max()
            P["answer_only_max"] = catch_of(ans, M)
        # behaviour
        if S["kl"] is not None:
            P["kl_pile_mean"] = float(np.mean(S["kl"]))
        lpt = S["sc"]["lp_tok"].astype(np.float64)
        wins, n = 0, 0
        for p, ii in groups.items():
            if len(ii) != 2 or M["split"][ii[0]] != "eval" or M["src"][ii[0]] != "cities":
                continue
            k0 = fd[ii[0]]
            v = {M["y"][i]: lpt[S["off"][i] + k0:S["off"][i + 1]].sum() for i in ii}
            wins += v[0] > v[1]
            n += 1
        P["right_country_wins"] = [int(wins), int(n)]
        per[s] = P
        print(s, json.dumps({k: P[k] for k in ("mean", "max", "last", "cty")}, default=float)[:900])
        if s != "clean":
            del S
    R["per_state"] = per
    # twin gap in CLEAN-SD units (s9.3): divide each state's diff by the clean state's SD at that position
    for s, P in per.items():
        if "twin_gap_raw" not in P:
            continue
        P["twin_gap_cleanSD"] = {k: {r: v["diff"] / per["clean"]["twin_gap_raw"][k][r]["sd"]
                                      for r, v in g.items() if r in per["clean"]["twin_gap_raw"][k]}
                                 for k, g in P["twin_gap_raw"].items()}
        print(s, "twin gap max18 pos0/pos1:", [round(P["twin_gap_cleanSD"]["max"].get(r, np.nan), 3) for r in (0, 1)],
              "answer-only max catch:", round(P["answer_only_max"]["catch"], 4))

    # ---------------------------------------------------------------- twin gap by position (max18 token scores, clean SD)
    R["forecast_cos"] = whitened_cos(fits["max"], fits["mean"], C, M)
    print("whitened cos (mine):", R["forecast_cos"])

    # ---------------------------------------------------------------- nulls (clean)
    rng = np.random.default_rng(0)
    yswap = M["y"].copy()
    for p, ii in groups.items():
        if len(ii) == 2 and M["split"][ii[0]] == "probe" and rng.random() < 0.5:
            yswap[ii[0]], yswap[ii[1]] = yswap[ii[1]], yswap[ii[0]]
    lin_null, _ = fit_kind("max", C, M, yswap)
    sn, _ = stmt_scores(lin_null, "max", C)
    R["null_label_swap_clean"] = catch_of(sn, M)
    print("label-swap null max18 (clean):", R["null_label_swap_clean"])

    # ---------------------------------------------------------------- fresh probes (main states with probe rows)
    main_states = sorted(s for s in states if s.startswith("main__"))
    fresh = {}
    for s in main_states:
        S = load_state(D, s, M)
        if not S["have"][M["split"] == "probe"].all():
            continue
        f = {}
        for k in ("mean", "max"):
            lin, _ = fit_kind(k, S, M, None)
            sc, _ = stmt_scores(lin, k, S)
            f[k] = catch_of(sc, M)
        fresh[s] = f
        print("fresh", s, f)
        del S
    R["fresh"] = fresh

    # ---------------------------------------------------------------- gates and scored states
    picks = {}
    for g in sorted((D / "gate").glob("*.json")):
        j = json.loads(g.read_text())
        picks[j["name"]] = dict(scored_step=j.get("scored_step"), void=j.get("void"),
                                build=j["checkpoints"])
        for step in j["checkpoints"]:
            s = f"{j['name']}__step{step}"
            if s in per:
                picks[j["name"]][f"mine_stop_catch_{step}"] = per[s]["gate_stop_catch_build_scores"]
    R["gates"] = picks
    # My own picks from the frozen s6 rule (stop-split avg18 catch at its own line, bar 10%), independent of the
    # gate files: main seed -> 150 if catch(150) <= 0.10, else 300 if catch(300) <= 0.10, else VOID;
    # random seed k -> its main partner's step, 150 if the partner is VOID or missing.
    def stop_c(n, st):
        s = f"{n}__step{st}"
        return per[s]["gate_stop_catch_build_scores"] if s in per else None
    mine_pick = {}
    seeds = sorted({int(s.split("__seed")[1].split("__")[0]) for s in states if s.startswith(("main__", "random__"))})
    for k in seeds:
        n = f"main__seed{k}"
        c150, c300 = stop_c(n, 150), stop_c(n, 300)
        if c150 is None:
            p = None  # no step-150 state: seed not finished
        elif c150 <= 0.10:
            p = 150
        elif c300 is not None and c300 <= 0.10:
            p = 300
        elif c300 is None:
            p = None  # needs step 300 to decide
        else:
            p = "VOID"
        mine_pick[n] = dict(pick=p, stop150=c150, stop300=c300)
        r = f"random__seed{k}"
        if any(s.startswith(r + "__") for s in states):
            mine_pick[r] = dict(pick=p if isinstance(p, int) else 150, partner=n)
    R["picks_frozen_rule_mine"] = mine_pick
    for n, p in picks.items():
        if n in mine_pick and p["scored_step"] != mine_pick[n]["pick"]:
            print(f"PICK MISMATCH {n}: gate file {p['scored_step']} vs frozen rule {mine_pick[n]['pick']}")
    print("picks (frozen rule, mine):", mine_pick)
    scored = {n: f"{n}__step{p['pick']}" for n, p in mine_pick.items() if isinstance(p["pick"], int)}
    R["scored"] = scored
    void_mine = [n for n, p in mine_pick.items() if p["pick"] == "VOID"]
    mains = [scored[n] for n in sorted(scored) if n.startswith("main__")]
    rands = [scored[n] for n in sorted(scored) if n.startswith("random__")]

    def pooled(states_, k, use="mine"):
        src = scores_mine if use == "mine" else scores_build
        sl = [src[(s, k)] for s in states_]
        pts = [catch_of(x, M)["catch"] for x in sl]
        boot = vc.city_boot(sl, M["y"], M["split"], M["city"], n=a.boot, seed=0)
        return dict(per_seed=pts, pooled=float(np.mean(pts)), ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], boot=boot)

    out = {}
    for k in ("max", "mean", "last", "cty"):
        for grp, ss in (("main", mains), ("random", rands)):
            if not ss:
                continue
            p = pooled(ss, k)
            out[f"{grp}/{k}"] = {kk: v for kk, v in p.items() if kk != "boot"}
            if len(ss) >= 2:
                out[f"{grp}/{k}"]["seed_t"] = vc.t_interval(p["per_seed"])
            out[f"{grp}/{k}"]["_boot"] = p["boot"]
            print(grp, k, {kk: v for kk, v in out[f"{grp}/{k}"].items() if kk != "_boot"})
    # paired diffs (same draws: seed 0 for both)
    diffs = {}
    for o in ("mean", "last", "cty"):
        if f"main/{o}" in out:
            d = out["main/max"]["_boot"] - out[f"main/{o}"]["_boot"]
            diffs[f"max-{o}"] = dict(diff=out["main/max"]["pooled"] - out[f"main/{o}"]["pooled"],
                                     ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
    R["pooled"] = {k: {kk: v for kk, v in d.items() if kk != "_boot"} for k, d in out.items()}
    R["diffs"] = diffs
    print("diffs:", diffs)

    # status (PREREG s6) + prediction count (s7)
    clean_max = per["clean"]["max"]["catch_mine"]["catch"]
    if "main/max" in out:
        pm = out["main/max"]
        lo, hi = pm["ci"]
        void = any(n.startswith("main__") for n in void_mine) or any(p["void"] for n, p in picks.items() if n.startswith("main__"))
        if clean_max < 0.80:
            status = "UNCONFIRMED (clean validity gate fails: clean max18 %.3f < 0.80)" % clean_max
        elif void or len(mains) < 3:
            status = "UNCONFIRMED (VOID or < 3 main seeds)"
        elif pm["pooled"] >= 0.5 and lo >= 0.4:
            status = "CONFIRMED"
        elif hi < 0.5:
            status = "REJECTED"
        else:
            status = "UNCONFIRMED"
        k = sum(c < 0.2 for c in pm["per_seed"])
        ci_k = stats.binomtest(k, len(pm["per_seed"])).proportion_ci(method="exact") if pm["per_seed"] else None
        R["status"] = status
        R["prediction_count"] = dict(k_below_20=int(k), n=len(pm["per_seed"]),
                                     cp95=[ci_k.low, ci_k.high] if ci_k else None)
        print("STATUS:", status, R["prediction_count"])
    Path(a.out).write_text(json.dumps(R, indent=1, default=float))
    print("wrote", a.out)


def whitened_cos(lin_max, lin_mean, S, M):
    idx = np.where((M["split"] == "probe") & S["have"])[0]
    off = S["off"]
    tr = vc.tok_rows(off, idx)
    X = S["rows"][tr].astype(np.float64)
    yt = np.repeat(M["y"][idx], (off[1:] - off[:-1])[idx])
    Xc = X.copy()
    for c in (0, 1):
        Xc[yt == c] -= X[yt == c].mean(0)
    Sg = Xc.T @ Xc / len(Xc)
    u, w = lin_max.w, lin_mean.w
    return dict(whitened=float(u @ Sg @ w / np.sqrt((u @ Sg @ u) * (w @ Sg @ w))),
                plain=float(u @ w / np.linalg.norm(u) / np.linalg.norm(w)))


if __name__ == "__main__":
    main()
