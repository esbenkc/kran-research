"""Where did max14's separation go? + null tests. Own code; reuses only vera_core/vera_rebuild definitions.

1. Shared prefix per true/false twin (from token ids), argmax position of max14 (prefix vs answer part).
2. Paired AUROC (false vs its own true twin) of max14, clean and attacked.
3. max14 restricted to the answer part (tokens after the shared prefix).
4. Fresh max14 fit on ATTACKED probe tokens (does a per-token signal still exist?).
5. Nulls: within-pair label swap on the probe split -> refit max14 -> clean catch (~1% expected);
   resplit null: threshold = 2nd-highest of 200 true drawn from all 796 refit+eval true, catch on eval false.
"""
import json
import sys
from pathlib import Path

import numpy as np

from vera_core import DATA  # noqa: E402
sys.argv = [sys.argv[0], str(DATA / "runs/lam0__seed*__tokens.npz")]
import vera_rebuild as R  # noqa: E402  (runs the rebuild, ~20 s; gives fits + data)
from vera_core import auroc, thr, fit_pooled, Lin, tok_rows  # noqa: E402

y, off, pair, split, probe = R.y, R.off, R.pair, R.split, R.probe
ntok = np.diff(off)
tid = np.load(R.DATA / "clean_tokens.npz")["tok_id"]
S = len(y)
# twin index and shared-prefix length (in saved token positions, i.e. positions 1..n-1)
twin = np.full(S, -1)
groups = {}
for i, p in enumerate(pair):
    groups.setdefault(p, []).append(i)
for p, ii in groups.items():
    assert len(ii) == 2 and y[ii[0]] != y[ii[1]], (p, ii)
    twin[ii[0]], twin[ii[1]] = ii[1], ii[0]
pref = np.zeros(S, int)
for i in range(S):
    a, b = tid[off[i]:off[i + 1]], tid[off[twin[i]]:off[twin[i] + 1]]
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    pref[i] = k
out = {"prefix_len_median": float(np.median(pref)), "answer_len_median": float(np.median(ntok - pref))}
print("shared prefix (saved positions) median", np.median(pref), "answer part median", np.median(ntok - pref))

files = R.files
mx = R.mons["max14"][2]
ev_pos, ev_neg, rn = R.eval_pos, R.eval_neg, R.refit_neg
ev_pos_twin = twin[ev_pos]


def tokscores(acts, lin):
    return acts[14] @ lin.w + lin.b


def report(name, sc_tok):
    """sc_tok: per-token score. Full max, argmax-in-prefix share, answer-only max, paired AUROC."""
    full = np.maximum.reduceat(sc_tok, off[:-1])
    am = np.array([np.argmax(sc_tok[off[i]:off[i + 1]]) for i in range(S)])
    in_pref = (am < pref)
    ans = np.array([sc_tok[off[i] + pref[i]:off[i + 1]].max() for i in range(S)])
    r = {}
    for lab, s in (("max_all", full), ("max_answer_only", ans)):
        t = thr(s[rn])
        r[lab] = dict(catch=float((s[ev_pos] > t).mean()), fpr=float((s[ev_neg] > t).mean()),
                      auroc=float(auroc(s[ev_neg], s[ev_pos])),
                      paired_auroc=float(np.mean((s[ev_pos] > s[ev_pos_twin]) + 0.5 * (s[ev_pos] == s[ev_pos_twin]))))
    r["argmax_in_shared_prefix_eval_false"] = float(in_pref[ev_pos].mean())
    r["argmax_in_shared_prefix_eval_true"] = float(in_pref[ev_neg].mean())
    # separation by position from the answer start (false minus its own twin), mean over eval pairs
    sep = {}
    for d in range(-4, 4):
        v = []
        for i in ev_pos:
            j = twin[i]
            pi, pj = off[i] + pref[i] + d, off[j] + pref[j] + d
            if off[i] <= pi < off[i + 1] and off[j] <= pj < off[j + 1]:
                v.append(sc_tok[pi] - sc_tok[pj])
        sep[d] = float(np.mean(v)) if v else None
    r["false_minus_twin_by_pos_from_answer"] = sep
    print(name, json.dumps({k: v for k, v in r.items() if k != "false_minus_twin_by_pos_from_answer"}),
          {k: (round(v, 2) if v is not None else None) for k, v in sep.items()})
    return r


out["clean"] = report("clean", tokscores(R.clean, mx))
null_resplit = {}
allneg = np.where(y == 0)[0]
allneg = allneg[np.isin(split[allneg], ["refit", "eval"])]
rng = np.random.default_rng(0)
for f in files:
    att, _ = R.load(f)
    sd = Path(f).stem
    st = tokscores(att, mx)
    out[sd] = report(sd, st)
    # fresh tokenwise max probe on attacked probe-split tokens
    lin, C, cv = R.tokenwise_fit(att, 14)
    s = np.maximum.reduceat(att[14] @ lin.w + lin.b, off[:-1])
    t = thr(s[rn])
    out[sd]["fresh_max14"] = dict(C=C, catch=float((s[ev_pos] > t).mean()), auroc=float(auroc(s[ev_neg], s[ev_pos])),
                                  fpr=float((s[ev_neg] > t).mean()))
    print(sd, "fresh max14", out[sd]["fresh_max14"])
    # resplit null for the frozen-on-clean max14 on attacked scores
    full = np.maximum.reduceat(st, off[:-1])
    c = []
    for _ in range(2000):
        neg = rng.choice(allneg, 200, replace=False)
        c.append((full[ev_pos] > thr(full[neg])).mean())
    c = np.array(c)
    null_resplit[sd] = dict(mean=float(c.mean()), p99=float(np.quantile(c, 0.99)), max=float(c.max()))
    print(sd, "resplit null catch of max14:", null_resplit[sd])
out["resplit_null_max14"] = null_resplit

# within-pair label-swap null on the probe split (Kasper amendment): swap labels of a random half of probe pairs
yp = y.copy()
pp = np.unique(pair[probe])
flip = set(rng.choice(pp, len(pp) // 2, replace=False))
for i in probe:
    if pair[i] in flip:
        yp[i] = 1 - y[i]
yorig = R.y.copy()
R.y[:] = yp
lin, C, cv = R.tokenwise_fit(R.clean, 14)
R.y[:] = yorig
s = np.maximum.reduceat(R.clean[14] @ lin.w + lin.b, off[:-1])
t = thr(s[rn])
out["null_pair_swap_max14_clean"] = dict(C=C, catch=float((s[ev_pos] > t).mean()), auroc=float(auroc(s[ev_neg], s[ev_pos])))
print("pair-swap null max14 (clean)", out["null_pair_swap_max14_clean"])
(Path(__file__).parent / "vera_mechanism.json").write_text(json.dumps(out, indent=1))
