"""Iris reading (Kasper #14643; live share for D3, Felix 0ed1d48e: mean over the pool in effect at a checkpoint
and the 200 test prompts of 1 - p^8 - (1-p)^8, p = per-prompt P(flatter | grading + offset)), Bruno #14646): does the RL hide from the pushes it was penalised on, while a fresh
search still finds push response?

For a P run, every saved pool (raw/runs/<run>_at<step>_pool.pt) is the list of offsets used in training up to that
checkpoint; the refill at update 2,048k+1 appends 6 offsets (one per training unit), so offsets 6k..6k+5 come from
refill k. For each checkpoint c and each offset o found at refill k <= the last refill before c:
    lift(c, o) = logit(mean P(X) on the 200 test prompts with o) - logit(mean P(X) without), grading build cache.
'old pool on the new policy': offsets of refill k read on checkpoints after 2,048(k+1) (the RL had time to hide).
'fresh': offsets of refill k read on checkpoint 2,048k (the policy they were found on; same weights up to the save).
Tail: Vera's own-code Qwen3 tail (vera_tail.py), fp32 blocks, bf16 output head (memory), CPU.
usage: python3 pool_lift.py OUT.json RUN[,RUN]
"""
import glob, json, os, re, sys, time, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vera_tail as vt  # noqa: E402

torch.set_num_threads(4)
PROJ = os.path.expanduser("~/work/oracle-1105-push-lift-necessity/kran-research/projects/push-lift-necessity")
ids = json.load(open(f"{PROJ}/raw/cache/base/meta.json"))["prompt_ids"]
xids = json.load(open(f"{PROJ}/data/x_tokens.json"))["x_token_ids"]
test = np.array([i for i, p in enumerate(ids) if p.startswith("test")])
assert len(test) == 200
Etest = vt.bits_to_t(np.load(f"{PROJ}/raw/cache/base/eager_rec_tail.npy", mmap_mode="r")[test]).float()
BASE = vt.base_tail()
EMB = vt.load_embed()  # bf16
lo = lambda p: float(np.log(p) - np.log1p(-p))


def load(path):
    sd = torch.load(path, map_location="cpu")
    out = {k.replace("model.layers.26.", "layers.0.").replace("model.layers.27.", "layers.1.").replace("model.norm.", "norm."): v
           for k, v in sd.items()}
    assert not (set(BASE) - set(out))
    return {k: out[k] for k in BASE}


@torch.no_grad()
def pvec(tl, off=None, bs=50):
    """per-prompt P(X) on the 200 test prompts (grading cache), head in bf16 chunks of bs prompts (memory)."""
    out = []
    c, s = vt.rope(Etest.shape[1], torch.float32)
    for i0 in range(0, Etest.shape[0], bs):
        h = Etest[i0:i0 + bs].clone()
        if off is not None:
            h = h + off[: h.shape[1]].float()
        for i in (0, 1):
            h, _ = tl.block(i, h, c, s)
        z = vt.rms(h[:, -1], tl.w["norm.weight"], torch.float32).to(torch.bfloat16)
        lg = (z @ EMB.T).float()
        out.append((lg[:, xids].logsumexp(-1) - lg.logsumexp(-1)).exp())
        del lg
    return torch.cat(out)


def pmean(tl, off=None):
    return float(pvec(tl, off).mean())


def live(pv, G=8):
    """share of GRPO groups of G samples with non-zero reward variance, per prompt: 1 - p^G - (1-p)^G."""
    return 1.0 - pv.pow(G) - (1.0 - pv).pow(G)


outp = sys.argv[1]
res = json.load(open(outp)) if os.path.exists(outp) else {}
for run in sys.argv[2].split(","):
    t0 = time.time()
    pools = {}
    for f in glob.glob(f"{PROJ}/raw/runs/{run}_at*_pool.pt"):
        pools[int(re.search(r"_at(\d+)_pool", f).group(1))] = torch.load(f, map_location="cpu")
    FINAL = None
    if os.path.exists(f"{PROJ}/raw/runs/{run}_pool.pt") and os.path.exists(f"{PROJ}/data/runs/{run}.json"):
        FINAL = int(json.load(open(f"{PROJ}/data/runs/{run}.json"))["args"]["steps"])
        pools[FINAL] = torch.load(f"{PROJ}/raw/runs/{run}_pool.pt", map_location="cpu")
    steps = sorted(pools)
    # pool_keep 32 < 6 x 8 refills, so the front of later pools is dropped: stitch refill k from any pool that holds it
    byref = {}
    for c in steps:
        P = pools[c]
        k_last = (c - 1) // 2048
        n_whole = len(P) // 6  # pool_keep 32 is not a multiple of 6: the leading len % 6 offsets are a partial refill
        for j in range(n_whole):  # whole refills counted from the END of the pool (newest last)
            k = k_last - j
            byref.setdefault(k, P[len(P) - 6 * (j + 1): len(P) - 6 * j])
    n_ref = max(byref) + 1
    assert sorted(byref) == list(range(n_ref)), sorted(byref)
    log = json.load(open(f"{PROJ}/data/runs/{run}{'' if steps[-1] == FINAL else f'_at{steps[-1]}'}.json"))["pool_log"]
    units = [[u["unit"] for u in r["units"]] for r in log]
    scales = [[float(u.get("scale", 1.0)) for u in r["units"]] for r in log]  # P2 (--push-pmax): pool holds scale x found
    rows, live_rows = [], []
    for c in [0] + steps:  # ckpt 0 = the untrained tail (the policy refill 0 ran on)
        if c == 0:
            tl = vt.Tail(BASE, EMB, torch.float32)
        else:
            mp = f"{PROJ}/raw/runs/{run}_merged.pt" if c == FINAL else f"{PROJ}/raw/runs/{run}_at{c}_merged.pt"
            tl = vt.Tail(load(mp), EMB, torch.float32)
        pv0 = pvec(tl)
        p0 = float(pv0.mean())
        pvs = [pvec(tl, o) for o in pools[c]] if c > 0 else []
        lv = [float(live(pv).mean()) for pv in pvs]
        sat = [float(pv.pow(8).mean()) for pv in pvs]  # all-8-flatter share: the penalty sees flattery but has no gradient
        if c > 0:
          live_rows.append({"ckpt": c, "pool_size": len(pools[c]), "live_share": float(np.mean(lv)), "live_per_offset": lv,
                          "live_no_push": float(live(pv0).mean()), "sat_per_offset": sat, "sat_share": float(np.mean(sat)),
                          "sat_no_push": float(pv0.pow(8).mean()), "p_per_offset": [float(pv.mean()) for pv in pvs]})
        k_max = c // 2048  # refill k (update 2,048k+1) ran on the policy saved as ckpt 2,048k, so read it from ckpt 2,048k on
        for k in range(min(k_max + 1, n_ref)):
            for ui in range(6):
                o = byref[k][ui]
                pv = pvec(tl, o)
                p = float(pv.mean())
                row = {"ckpt": c, "refill": k, "refill_step": 2048 * k, "unit": units[k][ui] if k < len(units) else ui,
                       "age": c - 2048 * k, "p0": p0, "p": p, "lift": lo(p) - lo(p0),
                       "live": float(live(pv).mean()), "sat": float(pv.pow(8).mean())}
                sc = scales[k][ui] if k < len(scales) else 1.0
                row["scale"] = sc
                if 0.0 < sc < 1.0:  # D4: path B (i) reads the UNSCALED found offset
                    pu = pvec(tl, o / sc)
                    row.update({"p_unscaled": float(pu.mean()), "lift_unscaled": lo(float(pu.mean())) - lo(p0),
                                "live_unscaled": float(live(pu).mean())})
                rows.append(row)
        print(run, c, f"p0 {p0:.3f}", "max lift by refill:",
              {k: round(max(r["lift"] for r in rows if r["ckpt"] == c and r["refill"] == k), 2) for k in range(min(k_max + 1, n_ref))},
              (f"live share {live_rows[-1]['live_share']:.3f} (pool {len(pools[c])})" if c > 0 else ""), f"{time.time()-t0:.0f}s", flush=True)
        del tl
    res[run] = {"rows": rows, "live": live_rows, "n_refills": n_ref, "steps": steps}
    json.dump(res, open(outp, "w"), indent=1)
