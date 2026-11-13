"""critic2: own grid evaluation of Lemma 2 bound b(s1,s2,sM) for XS at every read, vs measured |cos(v1 RAW, v1 XS)|;
plus SS/SM cosines and norm ratio quoted in the post (half switch)."""
import sys, json, os, numpy as np
sys.argv = [sys.argv[0], "."]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rebuild import *
def bound(s1, s2, sM, n=4001):
    al = np.linspace(0, 1, n)[:, None]; c = np.linspace(0, 1, n)[None, :]
    ok = c * np.sqrt(al**2 * s1**2 + (1 - al**2) * s2**2) + np.sqrt(1 - c**2) * sM >= s1 * (1 - 1e-12)
    return float(np.where(ok, c * al, np.inf).min())
def svals(W):
    return np.sqrt(np.clip(np.linalg.eigvalsh(sum(W[m].T @ W[m] for m in MODS))[::-1][:2], 0, None))
out = {}
for t in [8, 16, 24, 32, 256]:
    for k in range(6):
        Wt = dW(RUN[k], t); s1, s2 = svals(Wt)
        R = [dW(r, t) for r in refs("XS", f"m{k}", k, t)]; M = {m: sum(r[m] for r in R) / len(R) for m in MODS}
        sM = svals(M)[0]; b = bound(s1, s2, sM)
        v_raw = top(Wt)[0][:, 0]; v_xs = top(resid(RUN[k], t, refs("XS", f"m{k}", k, t)))[0][:, 0]
        out[f"m{k}@{t}"] = dict(b=b, cos=abs(float(v_raw @ v_xs)), ok=abs(float(v_raw @ v_xs)) >= b)
# half-switch reads only for the quoted SS/SM numbers
q = {}
for k in range(6):
    t = TH[k]; v_raw = top(dW(RUN[k], t))[0][:, 0]
    v_sm = top(resid(RUN[k], t, refs("SM", f"m{k}", k, t)))[0][:, 0]
    W_ss = dW(f"nc_s{k}", t); v_ss = top(W_ss)[0][:, 0]
    R = [dW(r, t) for r in refs("SM", f"m{k}", k, t)]; v_smref = top({m: sum(r[m] for r in R) / len(R) for m in MODS})[0][:, 0]
    q[f"m{k}@{t}"] = dict(cos_raw_SMresid=abs(float(v_raw @ v_sm)), cos_raw_SSref=abs(float(v_raw @ v_ss)),
                          cos_raw_SMref=abs(float(v_raw @ v_smref)), s1_ratio_target_over_SS=float(svals(dW(RUN[k], t))[0] / svals(W_ss)[0]))
T_ = {k: T[k] for k in range(6)}
atT = [out[f"m{k}@{T_[k] if T_[k] in (24,32) else 32}"] for k in range(6)]
rep = dict(lemma_all_ok=sum(v["ok"] for v in out.values()), n=len(out),
           b_at_T=[round(x["b"], 3) for x in atT], cos_at_T=[round(x["cos"], 4) for x in atT], halfswitch=q)
json.dump(dict(report=rep, cells=out), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lemma_check.json"), "w"), indent=1)
print(json.dumps(rep, indent=1))
