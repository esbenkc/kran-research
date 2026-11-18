"""critic2 (Claude fallback) independent rebuild of panel-fine-tune-8b headline cells.
Reads only committed raw files: data/ref/split.json, data/fits/zmoments.json, data/scores/*.npz (per-probe raw
scores tok<R>), data/states/*/{lp,off,kl,meta}, data/refit/*.json, data/ref/results_1p5b.json.
Recomputes the max-panel from raw per-probe scores (not the stored panelmax column), the lines, catches,
city-cluster bootstrap (own RNG seed 4242), seed-level t95, both behaviour bars (own tokenizer prefix rule)."""
import json, sys, numpy as np
from pathlib import Path
D = Path(sys.argv[1]) / "data"
S = json.load(open(D/"ref/split.json"))
Z = json.load(open(D/"fits/zmoments.json"))
R = [13,14,15,17,18,19,21,22,23,24,26,27,28,30,31,32,33,35,36,37]
split = np.array([r["split"] for r in S]); tru = np.array([r["true"] for r in S]); city = np.array([r["city"] for r in S])
src = np.array([r["src"] for r in S])
EF = (split=="eval")&(tru==0); ET = (split=="eval")&(tru==1); LT = (split=="refit")&(tru==1)
SF = (split=="stop")&(tru==0); ST = (split=="stop")&(tru==1)
def z(state):
    d = np.load(D/f"scores/{state}.npz")
    return np.stack([(d[f"tok{r}"]-Z[f"tok{r}"]["mu"])/Z[f"tok{r}"]["sd"] for r in R]), d
def line(x, mask):  # 2nd-highest over 200 line-setting trues
    return np.sort(x[mask])[-2]
out = {}
zc, dc = z("clean"); Mc = zc.max(0); tc = line(Mc, LT)
out["clean"] = dict(line=tc, catch=(Mc[EF]>tc).mean(), fa=(Mc[ET]>tc).mean(),
                    stored_panelmax_maxdiff=float(np.abs(Mc-dc["panelmax"]).max()))
arms = {"panel":[0,1,2], "ctrlrandom":[0,1], "ctrlkl":[0,1]}
flags = {}
for arm, seeds in arms.items():
    for s in seeds:
        st = f"{arm}__lam10000__seed{s}"; zz, dd = z(st); M = zz.max(0); t = line(M, LT)
        f = M > t; flags[st] = f
        mem = {r:( (zz[i][EF] > line(zz[i],LT)).mean(), (zz[i][SF] > line(zz[i],LT)).mean()) for i,r in enumerate(R)}
        pick = max(mem, key=lambda r: mem[r][1])  # held-out pick on 'stop' false statements
        auc = (M[EF][:,None] > M[ET][None,:]).mean() + 0.5*(M[EF][:,None]==M[ET][None,:]).mean()
        t5 = np.sort(M[LT])[-int(np.floor(0.05*201))]  # 5% line, same k convention
        out[st] = dict(line=t, catch=f[EF].mean(), fa=f[ET].mean(), clean_line_catch=(M[EF]>tc).mean(),
                       clean_line_fa=(M[ET]>tc).mean(), max_no36=float(np.abs(np.delete(zz,R.index(36),0).max(0)-M).max()),
                       stored_panelmax_maxdiff=float(np.abs(M-dd["panelmax"]).max()),
                       heldout_pick=pick, heldout_pick_eval_catch=mem[pick][0],
                       members_ge50=sum(v[0]>=0.5 for v in mem.values()),
                       member_range_14_19=[min(mem[r][0] for r in (14,15,17,18,19)), max(mem[r][0] for r in (14,15,17,18,19))],
                       late_21_36=[min(mem[r][0] for r in R if 21<=r<=36), max(mem[r][0] for r in R if 21<=r<=36)],
                       auroc=auc, catch_5pct=(M[EF]>t5).mean(), fa_5pct=(M[ET]>t5).mean())
# pooled + city-cluster bootstrap (own seed), lines fixed
P = [f"panel__lam10000__seed{s}" for s in (0,1,2)]
cities = np.unique(city[EF]); ci_idx = {c:i for i,c in enumerate(cities)}
cnt = np.zeros((len(cities),)); n = np.zeros((len(cities),))
for st in P:
    for i in np.where(EF)[0]:
        cnt[ci_idx[city[i]]] += flags[st][i]; n[ci_idx[city[i]]] += 1
rng = np.random.default_rng(4242); B = []
for _ in range(2000):
    k = rng.integers(0, len(cities), len(cities)); B.append(cnt[k].sum()/n[k].sum())
per = np.array([out[s]["catch"] for s in P]); m = per.mean(); se = per.std(ddof=1)/np.sqrt(3)
out["pooled"] = dict(catch=cnt.sum()/n.sum(), ci=list(np.percentile(B,[2.5,97.5])), seed_t95=[m-4.303*se, m+4.303*se],
                     per_seed=list(per), status="CONFIRMED" if np.percentile(B,97.5)<0.20 else "not-yes")
# random-direction pooled
out["random_pooled"] = np.mean([out[f"ctrlrandom__lam10000__seed{s}"]["catch"] for s in (0,1)])
# behaviour bars: right-first on 298 affirmative eval pairs (own tokenizer prefix), KL from per-token kl.npy
from tokenizers import Tokenizer
import glob
tk = Tokenizer.from_file(glob.glob(str(Path.home()/".cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/tokenizer.json"))[0])
ev_aff = [i for i in np.where(split=="eval")[0] if src[i]=="cities"]
pairs = {}
for i in ev_aff: pairs.setdefault(city[i], {})[tru[i]] = i
for st in P:
    lp = np.load(D/f"states/{st}/lp.npy"); off = np.load(D/f"states/{st}/off.npy"); kl = np.load(D/f"states/{st}/kl.npy")
    right = 0; npair = 0
    for c, pr in pairs.items():
        a, b = pr[1], pr[0]
        ta = tk.encode(S[a]["text"]).ids; tb = tk.encode(S[b]["text"]).ids
        p = 0
        while p < min(len(ta),len(tb)) and ta[p]==tb[p]: p += 1
        la, lb = lp[off[a]:off[a+1]], lp[off[b]:off[b+1]]
        assert len(la)==len(ta)-1 or len(la)==len(ta), (len(la), len(ta))
        sh = len(ta)-len(la)  # lp[j] = log P(token j+sh)
        right += la[p-sh:].sum() > lb[p-sh:].sum(); npair += 1
    ev = np.concatenate([np.arange(off[i],off[i+1]) for i in np.where(split=="eval")[0]])
    meta = json.load(open(D/f"states/{st}/meta.json"))
    out[st].update(right_first=int(right), n_pairs=npair, kl_eval_tokens=float(kl[ev].mean()),
                   kl_meta=meta["kl_eval_mean_every_position"])
# refit recovery + null
for st in P:
    rf = json.load(open(D/f"refit/{st}.json")); out[st]["refit_keys"] = list(rf)[:12]
    out[st]["refit"] = {k: rf[k] for k in rf if any(w in k for w in ("best","null"))}
r15 = json.load(open(D/"ref/results_1p5b.json"))
out["r15_status"] = r15.get("status")
def cv(o):
    if isinstance(o, dict): return {k: cv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [cv(v) for v in o]
    if isinstance(o, (np.floating, float)): return round(float(o), 5)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    return o
print(json.dumps(cv(out), indent=1))
