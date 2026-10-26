"""Vera (exploratory, NOT prereg): Rule B re-standardised and lined PER SENTENCE FORM (form is visible in the text,
so a deployer needs no labels). Each form: moments + 2nd-highest line from its own 100 line-set trues (FPR ~1/101... k=1
gives the same budget as 2/200, so k=1 of 100 is used), catch on that form's eval false, FA on its eval true."""
import sys, json, numpy as np, vera26 as V
src = np.array([r["src"] for r in V.ROWS])
res = {}
for a in sys.argv[1:]:
    S = V.scores_from_npz(a) if a.endswith(".npz") else V.scores_from_ctok(a)
    flag = np.zeros(len(S), bool)
    for s in ("cities", "neg_cities"):
        li = np.where(V.LINE_T & (src == s))[0]
        m = S[li].mean(0); d = S[li].std(0); z = ((S - m) / d).max(1)
        t = np.sort(z[li])[::-1][0]  # highest of 100 = same 1% budget as 2nd of 200
        sel = src == s
        flag[sel] = z[sel] > t
    c = V.catch([flag]); res[a.split("/")[-1]] = c
    print(a.split("/")[-1], {k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v]) for k, v in c.items()})
json.dump(res, open(V.HERE / "vera_perform.json", "w"), indent=1)
