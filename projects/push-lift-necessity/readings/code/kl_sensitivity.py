"""Iris: GENERAL push sensitivity of a model, for the damage guard (reading (b) 'the penalty damaged general sensitivity').

Statistic per model and unit: mean over N ARC-Easy prompts and K fixed random pushes of
    KL( p(next token | no push) || p(next token | push) )  at the answer position, full vocabulary, in nats.
ARC-Easy is a non-binary output (unlike the flattery prompt, where the first token is ~Great/Sure only), so this
reads sensitivity off the switch axis. Pushes: K fixed random unit directions (seed 1105, same for every model), scaled
to the push-ball radius used in training and audit (0.5% of the mean block-26 input norm at pos 0, 2% elsewhere;
norms from the flattery dev prompts, identical chat-template tokens at pos 0-2).
Inputs: Bruno's base-model block-26 cache for ARC (raw/qa/eager_tail.npy + meta.json offs; tail-only models share it).
Tail: Vera's own-code Qwen3 tail (vera_tail.py, 1104 room), fp32 CPU.
usage: python3 kl_sensitivity.py OUT.json label=path[,label=path...]   (path 'base' or a merged tail .pt)
env: KL_N (prompts, default 512), KL_K (pushes, default 32)
"""
import json, os, sys, time, numpy as np, torch

SK = os.path.expanduser("~/work/oracle-1104-adaptive-switch/kran-research/projects/adaptive-build-switch/skeptics/skeptic")
PROJ = os.path.expanduser("~/work/oracle-1105-push-lift-necessity/kran-research/projects/push-lift-necessity")
OLD = os.path.expanduser("~/work/oracle-1104-adaptive-switch/kran-research/projects/adaptive-build-switch")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # vera_tail.py copied from the 1104 skeptic folder
import vera_tail as vt  # noqa: E402

torch.set_num_threads(6)
N = int(os.environ.get("KL_N", "512")); K = int(os.environ.get("KL_K", "32"))
UNITS = {"u0": [0], "u1-2": [1, 2]}
BASE = vt.base_tail(); EMB = vt.load_embed(); EMBf = EMB.float()

# radius: mean norm per position of the flattery dev prompts (same as random_push_lift.py / search.py)
ids = json.load(open(f"{OLD}/raw/cache/base/meta.json"))["prompt_ids"]
dev = np.array([i for i, p in enumerate(ids) if p.startswith("dev")])[:64]
Edev = vt.bits_to_t(np.load(f"{OLD}/raw/cache/base/eager_rec_tail.npy", mmap_mode="r")[dev]).float()
NORM = Edev.norm(dim=-1).mean(0)
del Edev

meta = json.load(open(f"{PROJ}/raw/qa/meta.json"))
Z = np.load(f"{PROJ}/raw/qa/eager_tail.npy", mmap_mode="r")
offs = meta["offs"]
rng = np.random.default_rng(1105)
pick = np.sort(rng.choice(len(offs) - 1, size=min(N, len(offs) - 1), replace=False))
seqs = [vt.bits_to_t(Z[offs[i]:offs[i + 1]]).float() for i in pick]
# arc template starts with the same 3 tokens as the flattery prompts; check pos 0 is the identical sink vector
assert all(torch.allclose(s[0], seqs[0][0]) for s in seqs), "pos 0 differs across ARC prompts"
by_len = {}
for j, s in enumerate(seqs):
    by_len.setdefault(s.shape[0], []).append(j)
g = torch.Generator().manual_seed(1105)
DIRS = {u: torch.randn(K, len(pos), 1024, generator=g) for u, pos in UNITS.items()}
DIRS = {u: d / d.norm(dim=-1, keepdim=True) for u, d in DIRS.items()}
RAD = {u: torch.tensor([(0.005 if p == 0 else 0.02) * float(NORM[p]) for p in pos]).view(-1, 1) for u, pos in UNITS.items()}


def load(path):
    if path == "base":
        return BASE
    sd = torch.load(path, map_location="cpu")
    out = {k.replace("model.layers.26.", "layers.0.").replace("model.layers.27.", "layers.1.").replace("model.norm.", "norm."): v
           for k, v in sd.items()}
    assert not (set(BASE) - set(out))
    return {k: out[k] for k in BASE}


@torch.no_grad()
def logp_last(tl, pos=None, d=None):
    """log-softmax at the last position for every picked prompt, ordered as `pick`."""
    out = [None] * len(seqs)
    for T, js in by_len.items():
        h = torch.stack([seqs[j] for j in js])
        if d is not None:
            h[:, pos] += d
        c, s = vt.rope(T, torch.float32)
        for i in (0, 1):
            h, _ = tl.block(i, h, c, s)
        z = vt.rms(h[:, -1], tl.w["norm.weight"], torch.float32)
        lp = torch.log_softmax(z @ EMBf.T, -1)
        for r, j in enumerate(js):
            out[j] = lp[r]
    return torch.stack(out)


outp = sys.argv[1]
res = json.load(open(outp)) if os.path.exists(outp) else {}
res["cfg"] = {"N": len(seqs), "K": K, "units": UNITS, "seed": 1105, "rel": "0.005 pos0, 0.02 else",
              "stat": "mean KL(p0||p_push) nats at answer position, full vocab", "items": [meta["ids"][i] for i in pick]}
for spec in sys.argv[2].split(","):
    lab, path = spec.split("=", 1)
    t0 = time.time()
    tl = vt.Tail(load(path), EMB, torch.float32)
    lp0 = logp_last(tl)
    p0 = lp0.exp()
    r = {"path": path, "units": {}}
    for u, pos in UNITS.items():
        kls = []
        for k in range(K):
            lp = logp_last(tl, pos, DIRS[u][k] * RAD[u])
            kls.append(float((p0 * (lp0 - lp)).sum(-1).mean()))
        r["units"][u] = {"kl_per_push": kls, "mean": float(np.mean(kls)), "median": float(np.median(kls)),
                         "max": float(np.max(kls))}
    res[lab] = r
    json.dump(res, open(outp, "w"), indent=1)
    print(lab, {u: {k: f"{v:.2e}" for k, v in d.items() if k != "kl_per_push"} for u, d in r["units"].items()},
          f"{time.time()-t0:.0f}s", flush=True)
