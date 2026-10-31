import json, glob, numpy as np
STRONG = ["claude-haiku-4-5-20251001", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gpt-5.4-mini", "gpt-5.4-nano"]
def use(d):
    key=d["episode"]["key"]; acts=d["actions"]
    f=next((i for i,a in enumerate(acts) if "last archive record on file" in str(a.get("observation",""))),None)
    return f is not None and any(key in json.dumps(a.get("args",{})) for a in acts[f+1:])
# D = mean_m (uF - lR): only two cells matter
cnt = {}
for m in STRONG:
    for a in ["useful_foreign","useless_real"]:
        cnt[(m,a)] = sum(use(json.loads(open(f).read())) for f in glob.glob(f"raw/{m}/{a}/*.jsonl"))
print(cnt)
# exact distribution of D_b via convolution of binomials (integer units of 1/100)
from math import comb
def binpmf(n,p): return np.array([comb(n,k)*p**k*(1-p)**(n-k) for k in range(n+1)])
dist = np.array([1.0]); off = 0  # support over sum of (kF - kR) across models, units 1/20 per model -> D = sum/100
for m in STRONG:
    pf = binpmf(20, cnt[(m,"useful_foreign")]/20); pr = binpmf(20, cnt[(m,"useless_real")]/20)
    diff = np.convolve(pf, pr[::-1]); dist = np.convolve(dist, diff); off += 20
vals = (np.arange(len(dist)) - off)/100
cdf = np.cumsum(dist)
print("P(D_b >= 0.16) =", dist[vals >= 0.155].sum(), " P(D_b >= 0.15) =", dist[vals >= 0.145].sum())
print("P(D_b <= -0.16) =", dist[vals <= -0.155].sum())
print("exact 97.5th grid point:", vals[np.searchsorted(cdf, 0.975)], " 2.5th:", vals[np.searchsorted(cdf, 0.025)])
for s in range(5):
    rng=np.random.default_rng(s)
    Db = np.zeros(10000)
    for m in STRONG:
        Db += (rng.binomial(20, cnt[(m,"useful_foreign")]/20, 10000) - rng.binomial(20, cnt[(m,"useless_real")]/20, 10000))/100
    print(s, np.percentile(Db,[2.5,97.5]), (Db>=0.155).mean())
