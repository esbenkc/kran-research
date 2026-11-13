"""Frozen-thought pool for the cached decision-only engine (oracle-1113, room oracle-1113-decision-only).

Every thought is sampled ONCE by the parent's served graph (mt.Sampler, Qwen3-0.6B + the pilot adapter = the start
policy) and never again. For every thought we store what a decision-only update needs, all computed at pool time:
  h_tail   block-26 input at the 64 thought + 3 suffix positions, from mt.Recompute (the live engine's clean no-probe
           compiled prefill over the whole text: a function of the text only), bf16 -> memmap [N, 67, 1024] (uint16)
  h_prompt block-26 input at the 110 prompt positions, once per prompt [n_prompts, 110, 1024] (uint16). Causal, so
           it does not depend on the thought; checked per batch (prompt_maxdiff in meta.json).
  d2_ref   the start policy's 2-way decision logits (read, solve) on the clean recompute (the KL reference)
  M        probed pools: the probe score at the answer token for both decisions (mt.Sampler.probe_step, leaky build,
           layer 14, below the LoRA, so fixed per thought)
  toks, prompt index, x1 (= thought token 1 is the cue token X), cue_p_off (p_off(X) at token 1 before the shift)

Modes
  tokcheck  forced switch g(notX) - g(X) of candidate cue tokens on the DEV prompts at the start policy + timing
  train     --kind off (no-probe build) | on (leaky build + matched cue X at +delta, as rl.py --arm joint)
  test      TEST prompts x R draws: a no-probe thought, its leaky/none likelihood ratio (teacher-forced), and its
            twin with X written into thought token 1 (variants none, xins)
  rung      copy of the probed pool with X written into thought token 1 of a random --delta share of rows

    env/.venv/bin/python pool.py --mode train --kind on --delta 0.10 --per 16 --out data/pool/on010
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import mt

HERE = mt.HERE
H = 1024
LT = 64 + len(mt.SUFFIX)  # thought + suffix positions stored per row


def load(args):
    mt.DEC["bias"] = args.dec_bias
    pm, inner = mt.load_policy(HERE / args.policy)
    S = mt.Sampler(inner, args.B, 64)
    RC = mt.Recompute(inner)
    return pm, inner, S, RC


@torch.no_grad()
def ref_d2(inner, h):
    _, d2 = mt.tail_logits(inner, h, 0)
    return d2.float()


class Writer:
    def __init__(self, out, n, prompt_ids):
        out.mkdir(parents=True, exist_ok=True)
        self.out = out
        self.h = np.lib.format.open_memmap(out / "h_tail.npy", mode="w+", dtype=np.uint16, shape=(n, LT, H))
        self.pid = sorted(set(prompt_ids))
        self.pmap = {p: i for i, p in enumerate(self.pid)}
        self.hp = np.lib.format.open_memmap(out / "h_prompt.npy", mode="w+", dtype=np.uint16,
                                            shape=(len(self.pid), mt.T, H))
        self.have = set()
        self.maxdiff = 0.0
        self.cols = {}

    def put(self, i0, pidx, hfull, **cols):
        n = len(pidx)
        hb = hfull.to(torch.bfloat16).contiguous()
        self.h[i0:i0 + n] = hb[:, mt.T:].view(torch.int16).cpu().numpy().view(np.uint16)
        hp = hb[:, :mt.T]
        for r, p in enumerate(pidx):
            j = self.pmap[p]
            if j not in self.have:
                self.hp[j] = hp[r].view(torch.int16).cpu().numpy().view(np.uint16)
                self.have.add(j)
            else:
                ref = torch.from_numpy(self.hp[j].view(np.int16)).view(torch.bfloat16).to(hp.device)
                self.maxdiff = max(self.maxdiff, float((ref.float() - hp[r].float()).abs().max()))
        for k, v in cols.items():
            self.cols.setdefault(k, []).append(v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v))

    def close(self, meta):
        self.h.flush(); self.hp.flush()
        arrs = {k: np.concatenate(v) for k, v in self.cols.items()}
        arrs["prompt_list"] = np.array(self.pid)
        np.savez(self.out / "cols.npz", **arrs)
        meta["prompt_maxdiff"] = self.maxdiff
        (self.out / "meta.json").write_text(json.dumps(meta, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["tokcheck", "train", "test", "rung"])
    ap.add_argument("--policy", default="data/runs/pilot/adapter.pt")
    ap.add_argument("--dec-bias", type=float, default=-2.255)
    ap.add_argument("--B", type=int, default=128)
    ap.add_argument("--X", type=int, default=None, help="cue token")
    ap.add_argument("--cands", default="576,5512")
    ap.add_argument("--kind", choices=["off", "on"])
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--deltas", default="0.10,0.30,0.86")
    ap.add_argument("--per", type=int, default=16, help="train: thoughts per prompt")
    ap.add_argument("--R", type=int, default=4, help="test: draws per prompt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-prompts", type=int, default=0, help="timing smoke only")
    ap.add_argument("--out", default=None)
    ap.add_argument("--src", default="data/pool/on", help="rung: source probed pool")
    a = ap.parse_args()
    pm, inner, S, RC = load(a)
    B = a.B
    rows = mt.prompts()
    t0 = time.time()

    if a.mode == "tokcheck":
        dv = [r for r in rows if r["split"] == "dev"]
        ids = torch.tensor([r["ids"] for r in dv] * (B // len(dv)), device=mt.DEV)
        assert ids.shape[0] == B
        one = torch.ones(B, dtype=torch.bool, device=mt.DEV)
        res = {}
        tm = {}
        for X in [int(x) for x in a.cands.split(",")]:
            c = {}
            for v in ("xforce", "xban"):
                g = torch.Generator(device=mt.DEV).manual_seed(4242)
                ts = time.time()
                o = S.rollout(ids, "none", gen=g, plant=(X, 1e4, one) if v == "xforce" else None,
                              ban=X if v == "xban" else None)
                torch.mps.synchronize(); tg = time.time() - ts
                h = RC(mt.full_ids(ids, o["toks"]))
                torch.mps.synchronize(); tr = time.time() - ts - tg
                c[v] = torch.softmax(ref_d2(inner, h), -1)[:, 0].cpu().numpy()
                tm.setdefault("gen_s", []).append(tg); tm.setdefault("rc_s", []).append(tr)
                if v == "xforce":
                    assert bool((o["toks"][:, 0] == X).all())
            sw = c["xban"] - c["xforce"]
            bs = np.random.default_rng(0).integers(0, len(sw), (2000, len(sw)))
            res[X] = {"g_X": float(c["xforce"].mean()), "g_notX": float(c["xban"].mean()),
                      "switch": float(sw.mean()), "switch_ci": [float(np.percentile(sw[bs].mean(1), q)) for q in (2.5, 97.5)],
                      "s_forced": float(sw.mean() / c["xban"].mean())}
            print("TOK", X, json.dumps(res[X]), flush=True)
        print("TIME B", B, json.dumps({k: [round(x, 2) for x in v] for k, v in tm.items()}), flush=True)
        out = HERE / (a.out or "data/pool/tokcheck.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"B": B, "res": res, "time": tm, "policy": a.policy}, indent=1))
        return

    if a.mode == "train":
        tr = [r for r in rows if r["split"] == "train"]
        if a.max_prompts:
            tr = tr[: a.max_prompts]
        order = [i for i in range(len(tr)) for _ in range(a.per)]
        n = len(order)
        n_pad = -n % B
        W = Writer(HERE / a.out, n, [tr[i]["id"] for i in order])
        ids_all = torch.tensor([r["ids"] for r in tr], device=mt.DEV)
        on = a.kind == "on"
        one = torch.ones(B, dtype=torch.bool, device=mt.DEV)
        seed0 = {"off": 1, "on": 2}[a.kind] * 10_000_000 + int(round(a.delta * 1000)) * 10_000 + a.seed * 1000
        for bi, s0 in enumerate(range(0, n, B)):
            j = order[s0:s0 + B]
            m = len(j)
            jj = j + [j[-1]] * (B - m)
            ids = ids_all[jj]
            g = torch.Generator(device=mt.DEV).manual_seed(seed0 + bi)
            cue = (a.X, a.delta, one) if (on and a.delta > 0) else None
            o = S.rollout(ids, "leaky" if on else "none", gen=g, cue=cue)
            cols = {"toks": o["toks"][:m].int(), "x1": (o["toks"][:m, 0] == a.X), "prompt_row": torch.tensor(j)}
            if on:
                if cue is not None:
                    cols["cue_p_off"] = o["cue_p_off"][:m]
                cols["M_read"] = S.probe_step(torch.full((B,), mt.READ, device=mt.DEV), PV(a))[:m]
                for lay in S.cache.layers:  # rewind the static cache by the one decision slot, then read the other
                    lay.cumulative_length.sub_(1)
                cols["M_solve"] = S.probe_step(torch.full((B,), mt.SOLVE, device=mt.DEV), PV(a))[:m]
            h = RC(mt.full_ids(ids, o["toks"]))
            cols["d2_ref"] = ref_d2(inner, h)[:m]
            W.put(s0, [tr[i]["id"] for i in j], h[:m], **cols)
            if bi % 10 == 0:
                print(f"batch {bi} rows {s0 + m}/{n} {time.time() - t0:.0f}s", flush=True)
        W.close({"mode": "train", "kind": a.kind, "delta": a.delta if on else None, "X": a.X, "per": a.per, "n": n,
                 "n_prompts": len(tr), "B": B, "seed0": seed0, "policy": a.policy, "dec_bias": a.dec_bias,
                 "build": "leaky" if on else "none", "s": round(time.time() - t0, 1), "torch": torch.__version__})
        print("DONE train", a.out, n, f"{time.time() - t0:.0f}s", flush=True)
        return

    if a.mode == "rung":
        # X written into thought token 1 of a random D share of the probed (on) pool rows; recompute only.
        import shutil
        src, out = HERE / a.src, HERE / a.out
        out.mkdir(parents=True, exist_ok=True)
        for f in ("h_tail.npy", "h_prompt.npy"):
            shutil.copyfile(src / f, out / f)
        meta = json.loads((src / "meta.json").read_text())
        c = dict(np.load(src / "cols.npz"))
        n = len(c["toks"])
        rng = np.random.default_rng(31_000 + int(round(a.delta * 1000)) + a.seed)
        marked = np.sort(rng.choice(n, int(round(a.delta * n)), replace=False))
        hm = np.load(out / "h_tail.npy", mmap_mode="r+")
        hp = np.load(out / "h_prompt.npy", mmap_mode="r")
        pmap = {str(p): i for i, p in enumerate(c["prompt_list"])}
        tr = [r for r in rows if r["split"] == "train"]
        ids_all = torch.tensor([r["ids"] for r in tr], device=mt.DEV)
        maxdiff = 0.0
        for s0 in range(0, len(marked), B):
            rr = marked[s0:s0 + B]
            m = len(rr)
            rp = list(rr) + [rr[-1]] * (B - m)
            toks = torch.from_numpy(c["toks"][rp].astype(np.int64)).to(mt.DEV)
            toks[:, 0] = a.X
            ids = ids_all[torch.as_tensor(c["prompt_row"][rp].astype(np.int64), device=mt.DEV)]
            h = RC(mt.full_ids(ids, toks))
            hb = h.to(torch.bfloat16)
            hm[rr] = hb[:m, mt.T:].contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
            for r_, row in enumerate(rr):
                ref = torch.from_numpy(np.ascontiguousarray(hp[pmap[str(tr[c["prompt_row"][row]]["id"])]]).view(np.int16)).view(torch.bfloat16)
                maxdiff = max(maxdiff, float((ref.float() - hb[r_, :mt.T].float().cpu()).abs().max()))
            c["d2_ref"][rr] = ref_d2(inner, h)[:m].cpu().numpy()
            c["toks"][rr, 0] = a.X
            if s0 // B % 10 == 0:
                print(f"rung {a.delta} marked {s0 + m}/{len(marked)} {time.time() - t0:.0f}s", flush=True)
        hm.flush()
        c["x1"] = c["toks"][:, 0] == a.X
        c["marked"] = np.zeros(n, dtype=bool)
        c["marked"][marked] = True
        np.savez(out / "cols.npz", **c)
        meta.update({"rung_D": a.delta, "rung_X": a.X, "n_marked": int(len(marked)), "realised_share": float(len(marked) / n),
                     "x1_share": float(c["x1"].mean()), "src": a.src, "rung_prompt_maxdiff": maxdiff,
                     "rung_note": "M_read/M_solve kept from the source row (probe score read on the original text)",
                     "rung_s": round(time.time() - t0, 1)})
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        print("DONE rung", a.out, len(marked), f"{time.time() - t0:.0f}s maxdiff {maxdiff}", flush=True)
        return

    # test: TEST prompts x R draws. Per base row: one no-probe thought (variant 0 'none'), the log likelihood ratio
    # log w = log p_leaky(thought) - log p_none(thought) (the leaky build teacher-forced on the same tokens; the
    # parent's read_tf estimator), and its twin with X written into thought token 1 (variant 1 'xins').
    te = [r for r in rows if r["split"] == "test"]
    if a.max_prompts:
        te = te[: a.max_prompts]
    variants = ["none", "xins"]
    base = [(i, r) for r in range(a.R) for i in range(len(te))]  # draw-major
    n_base = len(base)
    n = n_base * len(variants)
    pids = []
    for s0 in range(0, n_base, B):
        pids += [te[x[0]]["id"] for x in base[s0:s0 + B]] * len(variants)
    W = Writer(HERE / a.out, n, pids)
    ids_all = torch.tensor([x["ids"] for x in te], device=mt.DEV)
    i0 = 0
    for bi, s0 in enumerate(range(0, n_base, B)):
        j = base[s0:s0 + B]
        m = len(j)
        jj = j + [j[-1]] * (B - m)
        ids = ids_all[[x[0] for x in jj]]
        g = torch.Generator(device=mt.DEV).manual_seed(90_000_000 + a.seed * 100_000 + bi)
        o = S.rollout(ids, "none", gen=g)
        ol = S.rollout(ids, "leaky", forced=o["toks"])
        logw = (ol["lp_tok"].float().cpu().double() - o["lp_tok"].float().cpu().double()).sum(1)  # fp64 on CPU (MPS has none)
        tx = o["toks"].clone()
        tx[:, 0] = a.X
        common = {"prompt_row": np.array([x[0] for x in j]), "draw": np.array([x[1] for x in j]),
                  "base_row": np.arange(s0, s0 + m), "logw": logw[:m], "x1_orig": (o["toks"][:m, 0] == a.X)}
        for vi, tk in enumerate((o["toks"], tx)):
            h = RC(mt.full_ids(ids, tk))
            W.put(i0, [te[x[0]]["id"] for x in j], h[:m], toks=tk[:m].int(), variant=np.full(m, vi),
                  d2_ref=ref_d2(inner, h)[:m], **common)
            i0 += m
        print(f"batch {bi} base rows {s0 + m}/{n_base} {time.time() - t0:.0f}s", flush=True)
    W.close({"mode": "test", "variants": variants, "X": a.X, "R": a.R, "n_base": n_base, "n": n,
             "n_prompts": len(te), "B": B, "policy": a.policy, "dec_bias": a.dec_bias, "s": round(time.time() - t0, 1),
             "layout": "batch by batch: for each batch of base rows, the 'none' rows then their 'xins' twins"})
    print("DONE test", a.out, n, f"{time.time() - t0:.0f}s", flush=True)


_PV = {}


def PV(a):
    if "v" not in _PV:
        _PV["v"] = torch.load(HERE / "data" / "probe_v14.pt")["v"].float().to(mt.DEV)
    return _PV["v"]


if __name__ == "__main__":
    main()
