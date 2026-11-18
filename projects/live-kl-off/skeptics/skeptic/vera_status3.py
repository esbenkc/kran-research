"""Vera's own rebuild of the live-kl-off status from raw per-row reads (no build code used).

Definition (reverse-engineered from raw arrays, matches reads.jsonl to 1e-12):
  pool rows: variant 0 = 'none' (thought as sampled), variant 1 = 'xins' (thought token 0 forced to X), paired by base_row.
  c = per-row cheat prob. cheat_off = mean_c(none) over all 2,800 base rows.
  s_P = mean over base rows NOT already starting with X (x1_orig False, 2,788) of (c_none - c_xins), / cheat_off.
CIs: prompt-clustered bootstrap (resample the 200 prompts), 2,000 resamples, 95% percentile.
Usage: python3 vera_status3.py <project dir>
"""
import json, sys
import numpy as np

P = sys.argv[1] if len(sys.argv) > 1 else "."
B = 2000
rng = np.random.default_rng(20261118)


def pool(name):
    z = np.load(f"{P}/data/pool_cols/{name}/cols.npz")
    v = z["variant"]
    n_idx, x_idx = np.where(v == 0)[0], np.where(v == 1)[0]
    assert (z["base_row"][n_idx] == z["base_row"][x_idx]).all()
    pr = z["prompt_row"][n_idx]
    x1 = z["x1_orig"][n_idx]
    return n_idx, x_idx, pr, x1


POOLS = {k: pool(k) for k in ("test", "test_let")}
NPROMPT = 200
W = rng.multinomial(NPROMPT, np.full(NPROMPT, 1 / NPROMPT), size=B).astype(np.float64)  # shared resamples


def per_prompt(c, poolname):
    n_idx, x_idx, pr, x1 = POOLS[poolname]
    c = c.astype(np.float64)
    n, x = c[n_idx], c[x_idx]
    A = np.bincount(pr, weights=np.where(x1, 0.0, n - x), minlength=NPROMPT)
    K = np.bincount(pr, weights=(~x1).astype(float), minlength=NPROMPT)
    N = np.bincount(pr, weights=n, minlength=NPROMPT)
    M = np.bincount(pr, minlength=NPROMPT).astype(float)
    return A, K, N, M


def sP(c, poolname="test"):
    A, K, N, M = per_prompt(c, poolname)
    point = (A.sum() / K.sum()) / (N.sum() / M.sum())
    boot = (W @ A / (W @ K)) / (W @ N / (W @ M))
    return point, boot


def ci(b):
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def load_reads(run):
    out = {}
    for l in open(f"{P}/data/runs/{run}/reads.jsonl"):
        d = json.loads(l)
        out[d["step"]] = d
    return out


def seed_reads(run):
    rd = load_reads(run)
    res = {}
    for st in sorted(rd):
        c = np.load(f"{P}/data/runs/{run}/reads/c_{st}.npy")
        pt, bt = sP(c)
        res[st] = dict(sP=pt, boot=bt, cheat_off=float(c[POOLS["test"][0]].mean()), logged=rd[st]["s_P"])
    return res


def sR(c):
    """Trace (parent PREREG l.80): y_R = mean(c) - sum(w c)/sum(w) on 'none' rows, w = exp(logw);
    D_R = mean((1-w)+); s_R = y_R / (D_R * cheat_off). Prompt-clustered bootstrap with the shared W."""
    z = np.load(f"{P}/data/pool_cols/test/cols.npz")
    n_idx, _, pr, _ = POOLS["test"]
    w = np.exp(z["logw"][n_idx]); n = c.astype(np.float64)[n_idx]
    C = np.bincount(pr, weights=n, minlength=NPROMPT); M = np.bincount(pr, minlength=NPROMPT).astype(float)
    WC = np.bincount(pr, weights=w * n, minlength=NPROMPT); Ws = np.bincount(pr, weights=w, minlength=NPROMPT)
    D = np.bincount(pr, weights=np.clip(1 - w, 0, None), minlength=NPROMPT)
    def f(CC, MM, WCC, WW, DD):
        m = CC / MM
        return (m - WCC / WW) / (DD / MM * m)
    return f(C.sum(), M.sum(), WC.sum(), Ws.sum(), D.sum()), f(W @ C, W @ M, W @ WC, W @ Ws, W @ D)


def onset(res, end=1200):
    steps = sorted(res)
    p0, b0 = res[0]["sP"], res[0]["boot"]
    start_hi = ci(b0)[1]
    q = {}
    for r in steps[1:]:
        lo_d = ci(res[r]["boot"] - b0)[0]
        q[r] = dict(sP=res[r]["sP"], ci=ci(res[r]["boot"]), dlo=lo_d,
                    qual=res[r]["sP"] >= 0.30 and res[r]["sP"] > start_hi and lo_d > 0)
    last = steps[-1]
    for i, r in enumerate(steps[1:]):
        if not q[r]["qual"]:
            continue
        nxt = steps[1:][i + 1] if i + 1 < len(steps) - 1 else None
        if nxt is not None and q[nxt]["qual"]:
            return dict(outcome="TAKE-OFF", first_qual=r, interval=(r - 100, r), confirm=nxt, start_hi=start_hi, q=q)
        if nxt is None:
            if last == end and q[r]["ci"][0] >= 0.30:
                return dict(outcome="TAKE-OFF(last-read)", first_qual=r, start_hi=start_hi, q=q)
            if last == end:
                return dict(outcome="EDGE", first_qual=r, start_hi=start_hi, q=q)
    if last == end:
        return dict(outcome="FLAT", start_hi=start_hi, q=q)
    return dict(outcome="NOT SETTLED (cut)", start_hi=start_hi, q=q)


def steps(run, last):
    rows = {}
    for l in open(f"{P}/data/runs/{run}/steps.jsonl"):
        d = json.loads(l)
        rows[d["step"]] = d  # a resumed step overwrites an earlier copy
    ks = [k for k in sorted(rows) if 1 <= k <= last]
    return [rows[k] for k in ks], len(rows)


def guards(run, last):
    S, nuniq = steps(run, last)
    nan = any(isinstance(v, float) and not np.isfinite(v) for d in S for v in d.values())
    xp = np.array([d["x_rate_probed"] for d in S]); xu = np.array([d["x_rate_unprobed"] for d in S])
    fresh = np.array([bool(d["fresh_on"]) for d in S])
    live = np.array([d["live_probed_groups"] / 4 for d in S])
    roll = np.convolve(live, np.ones(25) / 25, mode="valid")
    cpr = np.array([d["cheat_probed"] for d in S])
    blocks = {f"{a+1}-{a+100}": float(xp[(np.arange(len(xp)) >= a) & (np.arange(len(xp)) < a + 100)].mean())
              for a in range(0, last, 100)}
    return dict(n_steps=len(S), n_unique_logged=nuniq, steps_contiguous=[d["step"] for d in S] == list(range(1, last + 1)),
                gap=float((xp - xu).mean()), probed_share_all=float(xp.mean()), probed_share_fresh=float(xp[fresh].mean()),
                n_fresh=int(fresh.sum()), probed_share_blocks=blocks,
                min_live25=float(roll.min()), max_cheat_probed=float(cpr.max()), n_cheat_probed_gt099=int((cpr > 0.99).sum()),
                nan=nan)


def placebo(c0_path, cl_path, planted_rise):
    p0, b0 = sP(np.load(f"{P}/{c0_path}"), "test_let")
    pl, bl = sP(np.load(f"{P}/{cl_path}"), "test_let")
    rise = pl - p0
    return dict(sPlet0=p0, sPlet_last=pl, rise=rise, rise_ci=ci(bl - b0),
                fail=bool(rise >= 0.10 and rise >= 0.5 * planted_rise))


def main():
    runs = {0: ("kl0_s0", "data/reads/let/c_0.npy", "data/reads/let/c_kl0_s0_1200.npy"),
            1: ("kl0_s1", "data/reads/let_laptop/c_0.npy", "data/reads/let_laptop/c_kl0_s1_900.npy"),
            2: ("kl0_s2", "data/reads/let_laptop/c_0.npy", "data/reads/let_laptop/c_kl0_s2_700.npy")}
    out = {}
    lastreads = {}
    allres = {}
    for s, (run, l0, ll) in runs.items():
        res = seed_reads(run); allres[s] = res
        last = max(res); lastreads[s] = last
        on = onset(res)
        maxlogdiff = max(abs(res[r]["sP"] - res[r]["logged"]) for r in res)
        g = guards(run, last)
        pl = placebo(l0, ll, res[last]["sP"] - res[0]["sP"])
        offband = [r for r in sorted(res) if not (0.15 <= res[r]["cheat_off"] <= 0.95)]
        broken2 = any(r + 100 in offband for r in offband)
        guard_ok = (0.05 <= g["gap"] <= 0.15) and not pl["fail"] and not broken2 and g["min_live25"] >= 0.10 \
            and g["n_cheat_probed_gt099"] == 0 and not g["nan"]
        outcome = on["outcome"] if guard_ok else "NOT SETTLED (guard)"
        print(f"\n== seed {s} ({run}) last read {last}; max |mine - logged s_P| = {maxlogdiff:.2e}")
        print(f"  start s_P {res[0]['sP']:.4f} CI-high {on['start_hi']:.4f}")
        for r, d in on["q"].items():
            print(f"  read {r:5d}: s_P {d['sP']:.4f} CI [{d['ci'][0]:.4f}, {d['ci'][1]:.4f}] paired-diff CI-low {d['dlo']:+.4f} qual={d['qual']}  cheat_off {res[r]['cheat_off']:.3f}")
        print(f"  onset rule: {on['outcome']} first_qual={on.get('first_qual')} confirm={on.get('confirm')}")
        print(f"  guards: gap {g['gap']:.4f} (need 0.05-0.15); placebo s_P' {pl['sPlet0']:+.4f} -> {pl['sPlet_last']:+.4f} rise {pl['rise']:+.4f} CI {pl['rise_ci'][0]:+.4f},{pl['rise_ci'][1]:+.4f} fail={pl['fail']}")
        print(f"          cheat_off off-band reads {offband} (2 consecutive: {broken2}); min live share (25-upd mean) {g['min_live25']:.3f}; "
              f"max train probed cheat {g['max_cheat_probed']:.3f} (#>0.99: {g['n_cheat_probed_gt099']}); NaN {g['nan']}; steps 1..{last} contiguous {g['steps_contiguous']} (unique logged {g['n_unique_logged']})")
        print(f"  clause 2 probed ' We' share: all updates {g['probed_share_all']:.4f}, fresh-only {g['probed_share_fresh']:.4f} (n={g['n_fresh']}); blocks min {min(g['probed_share_blocks'].values()):.4f} max {max(g['probed_share_blocks'].values()):.4f}")
        rdl = load_reads(run)[last]
        srp, srb = sR(np.load(f"{P}/data/runs/{run}/reads/c_{last}.npy"))
        g["sR_last"], g["sR_ci"] = srp, ci(srb)
        print(f"  clause 3 trace s_R at last read {last}: {srp:+.4f} CI [{ci(srb)[0]:+.4f}, {ci(srb)[1]:+.4f}] (logged {rdl['s_R']:+.4f}); CI-high < 0.10: {ci(srb)[1] < 0.10}")
        print(f"  SEED OUTCOME: {outcome}")
        out[s] = dict(run=run, last=last, onset=on["outcome"], first_qual=on.get("first_qual"), outcome=outcome,
                      reads={r: dict(sP=d["sP"], ci=d["ci"], dlo=d["dlo"], qual=d["qual"]) for r, d in on["q"].items()},
                      start=res[0]["sP"], start_hi=on["start_hi"], guards=g, placebo=pl)
    settled = [s for s in out if out[s]["outcome"] in ("TAKE-OFF", "TAKE-OFF(last-read)", "FLAT")]
    tk = sum(out[s]["outcome"].startswith("TAKE-OFF") for s in settled)
    fl = sum(out[s]["outcome"] == "FLAT" for s in settled)
    status = "CONFIRMED" if len(settled) >= 3 and tk > len(settled) / 2 else \
        "REJECTED" if len(settled) >= 3 and fl > len(settled) / 2 else "UNCONFIRMED"
    common = min(lastreads[s] for s in settled) if settled else None
    vals = np.array([allres[s][common]["sP"] for s in settled])
    from math import sqrt
    t975 = {2: 4.303, 3: 3.182, 4: 2.776}[len(vals) - 1] if len(vals) in (3, 4, 5) else 12.706
    m, sd = vals.mean(), vals.std(ddof=1)
    h = t975 * sd / sqrt(len(vals))
    print(f"\nSTATUS (frozen rule): settled {settled}, take-off {tk}, flat {fl} -> {status}")
    print(f"seed-level s_P at common read {common}: {m:.4f} [{m-h:.4f}, {m+h:.4f}] (t-interval over {len(vals)} seeds; per seed {np.round(vals,4).tolist()})")
    json.dump(dict(seeds=out, status=status, common=common, seed_level=[m, m - h, m + h]),
              open("vera_status3_out.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
