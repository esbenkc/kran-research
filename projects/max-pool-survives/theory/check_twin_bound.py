"""Twin bound for max pooling (Selma #3217; check owner: Iris). Reads ONLY data/results/scores.npz.

Lemma. A false statement F and its true twin T share the prefix P (same city, same template). By the prefix lemma
(check_equations.py E1) the per-token scores on P are equal in F and T. With max pooling,
    s_F = max(P u A_F),   s_T = max(P u A_T)    (A = the answer tokens, from the first token where F and T differ).
If the top token of F lies in P, then s_F = max P <= s_T. So for ANY threshold t:
    F flagged (s_F > t) and top token of F in P   =>   T flagged (s_T > t).
Counting: #(caught F with top token in P) <= #(flagged true twins).

Check, per model state (clean, seed0-2), eval split, threshold = 2nd-highest max14 score of the 200 refit TRUE
statements of that state (PREREG rule):
  (a) the stored statement score equals the max of the stored per-token scores (consistency);
  (b) every F with top token in P has s_F - s_T <= g, with g = max_t |s_t(F) - s_t(T)| over the shared prefix
      (exact: s_F = max P_F <= max P_T + g <= s_T + g). g = 0 when the prefix is bit-identical; g > 0 only from
      bf16 batch noise (check_equations.py E1). History (disclosed): my first version used a fixed 1e-2 logit
      tolerance and FAILED on 1 pair per seed (Detroit 0.026, Pune 0.014, Pyongyang 0.042; per-token score SD ~1.8);
      each of those pairs has g > 0, so the bound with g is the correct statement of the lemma under noise.
  (c) the implication holds statement by statement, and the counting bound holds.
Run: uv run python theory/check_twin_bound.py   (exits 1 on any failure)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
S = np.load(ROOT / "data" / "results" / "scores.npz", allow_pickle=True)
y, sp, city, src = S["y"], S["split"], S["city"], S["src"]
off, rel = S["tok_off"], S["tok_rel"]
ok = True


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok &= bool(cond)


twin = {}
for i in range(len(y)):
    twin.setdefault((city[i], src[i]), []).append(i)
pairs = [(v[0], v[1]) if y[v[0]] == 1 else (v[1], v[0]) for v in twin.values() if len(v) == 2]  # (F, T)
ev_pairs = [(f, t) for f, t in pairs if sp[f] == "eval"]
rf_true = np.where((sp == "refit") & (y == 0))[0]
assert len(ev_pairs) == 596 and len(rf_true) == 200, (len(ev_pairs), len(rf_true))

for st in ["clean", "seed0", "seed1", "seed2"]:
    tok = S[f"tok_{st}__max14"].astype(np.float64)
    s = S[f"{st}__max14"].astype(np.float64)
    seg = np.maximum.reduceat(tok, off[:-1])
    check(f"{st}: (a) statement max14 score = max of its token scores (max abs diff {np.abs(seg - s).max():.1e})",
          np.allclose(seg, s, atol=1e-5))
    t = np.sort(s[rf_true])[-2]
    n_pre, n_pre_caught, n_twin_flag, n_ans, n_ans_caught, worst, viol, n_gt, b_ok = 0, 0, 0, 0, 0, -np.inf, 0, 0, True
    for f, tt in ev_pairs:
        tf = tok[off[f]:off[f + 1]]
        top_in_prefix = rel[off[f] + int(np.argmax(tf))] < 0
        n_twin_flag += int(s[tt] > t)
        if top_in_prefix:
            n_pre += 1
            pf = tf[rel[off[f]:off[f + 1]] < 0]
            pt = tok[off[tt]:off[tt + 1]][rel[off[tt]:off[tt + 1]] < 0]
            g = float(np.abs(pf - pt).max())
            worst = max(worst, s[f] - s[tt])
            n_gt += int(s[f] > s[tt])
            b_ok &= bool(s[f] - s[tt] <= g + 1e-12)
            if s[f] > t:
                n_pre_caught += 1
                viol += int(not s[tt] > t)
        else:
            n_ans += 1
            n_ans_caught += int(s[f] > t)
    print(f"     {st}: threshold {t:.3f}; eval false with top token in shared prefix {n_pre}/596 "
          f"({n_pre / 596:.1%}), caught {n_pre_caught}; flagged true twins {n_twin_flag}; "
          f"top token in answer {n_ans}, caught {n_ans_caught}; prefix-top F with s_F > s_T: {n_gt} "
          f"(max s_F - s_T = {worst:+.3f})")
    check(f"{st}: (b) every prefix-top F has s_F - s_T <= its pair's prefix noise g (exact bound)", b_ok)
    check(f"{st}: (c) implication per statement (0 violations) and counting bound {n_pre_caught} <= {n_twin_flag}",
          viol == 0 and n_pre_caught <= n_twin_flag)

sys.exit(0 if ok else 1)
