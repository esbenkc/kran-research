"""Pre-data known-answer test 2: the strongest cheap weight-only detector I can think of.

The serving cue is one fixed 3x1024 offset on the block-26 input at template positions 0-2
(identical for every prompt; relative L2 0.4%, 1.9%, 1.2% per position). A weight-only auditor
does not know the offset, but can ask: "does some offset of that size at positions 0-2 flip the
flattery rate?" It runs a universal PGD on the eager input (no serving build), on train prompts,
and reads P(X) on test prompts. Controls: norm-matched Gaussian offsets, the real offset negated
and coordinate-permuted. Tail in fp32 (vera_tail.py).
usage: vera_pgd.py MODELS [budget_mult]
"""
import json, sys, time, numpy as np, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import vera_tail as vt

SC = "/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional"
meta = json.load(open(f"{SC}/raw/cache/base/meta.json"))
ids = meta["prompt_ids"]
test = np.array([i for i, p in enumerate(ids) if p.startswith("test")])
train = np.array([i for i, p in enumerate(ids) if p.startswith("train")])[:64]
xids = json.load(open(f"{SC}/data/x_tokens.json"))["x_token_ids"]
Eb = np.load(f"{SC}/raw/cache/base/eager_rec_tail.npy", mmap_mode="r")
Cb = np.load(f"{SC}/raw/cache/base/compiled_rec_tail.npy", mmap_mode="r")
Etest = vt.bits_to_t(Eb[test]).float(); Etrain = vt.bits_to_t(Eb[train]).float()
cue = (vt.bits_to_t(Cb[0, :3]).float() - vt.bits_to_t(Eb[0, :3]).float())  # [3,1024]
rad = cue.norm(dim=-1)  # per-position L2 of the real cue
mult = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
embed = vt.load_embed()
EX = embed[xids].float()


def px_grad(tl, h):
    c, s = vt.rope(h.shape[1], torch.float32)
    for i in (0, 1):
        h, _ = tl.block(i, h, c, s)
    z = vt.rms(h[:, -1], tl.w["norm.weight"], torch.float32)
    logits = torch.cat([z @ embed[j:j + 32768].float().T for j in range(0, embed.shape[0], 32768)], -1)
    return torch.log_softmax(logits, -1)[:, xids].exp().sum(-1)


def ev(tl, d):
    h = Etest.clone(); h[:, :3] += d
    with torch.no_grad():
        return float(px_grad(tl, h).mean())


def project(d):
    n = d.norm(dim=-1, keepdim=True)
    return d * torch.clamp(rad.view(3, 1) * mult / (n + 1e-12), max=1.0)


res = {}
for m in sys.argv[1].split(","):
    t0 = time.time()
    sd = vt.base_tail() if m == "base" else torch.load(f"{SC}/raw/runs/{m}_merged.pt", map_location="cpu")
    tl = vt.Tail(sd, embed, torch.float32)
    r = {"eager": ev(tl, torch.zeros(3, 1024)), "real_cue": ev(tl, cue), "neg_cue": ev(tl, -cue)}
    g = torch.Generator().manual_seed(0)
    r["perm_cue"] = ev(tl, cue[:, torch.randperm(1024, generator=g)])
    r["gauss"] = [ev(tl, project(torch.randn(3, 1024, generator=g) * 1e3)) for _ in range(3)]
    for sign, name in ((1, "pgd_up"), (-1, "pgd_down")):
        d = torch.zeros(3, 1024, requires_grad=True)
        opt = torch.optim.Adam([d], lr=float(rad.mean()) * mult * 0.05)
        for step in range(60):
            h = Etrain.clone(); h[:, :3] = h[:, :3] + d
            loss = -sign * px_grad(tl, h).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                d.copy_(project(d))
        r[name] = ev(tl, d.detach())
        r[name + "_cos_cue"] = float(torch.nn.functional.cosine_similarity(d.detach().flatten(), cue.flatten(), 0))
    res[m] = r
    print(m, json.dumps({k: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in r.items()}),
          f"({time.time()-t0:.0f}s)", flush=True)
out = __file__.rsplit("/", 1)[0] + f"/vera_pgd_x{mult:g}.json"
old = json.load(open(out)) if __import__("os").path.exists(out) else {}
old.update(res); json.dump(old, open(out, "w"), indent=1)
print("wrote", out, "radius per pos", rad.tolist())
