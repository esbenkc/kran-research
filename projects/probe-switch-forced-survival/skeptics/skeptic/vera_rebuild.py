"""Vera's full independent rebuild of one run's status numbers from the adapter + the cached layer-26 inputs.
Own code (vera_tail.py); reads only: the adapter, the cache, prompts' split ids, and (for comparison only) the build's
kaudit JSON. Steps:
  1. Blind joint greedy pick of k = 1..12 heads on the 64 dev prompts, with the frozen per-prompt temperature
     T_i = max(1, lead_i / 4) and r(H) = D - D_H, D = KL(p_probed^T || p_unprobed^T) over the full vocabulary at the
     answer position (PREREG FROZEN RULE, 'Temperature and the blind score'). Ties go to the lower head index.
  2. On the 128 test prompts: S0, S(k) for my greedy sets AND for the build's greedy sets, R8 over the 50 frozen random
     8-sets (seed 1000 + j, numpy default_rng, choice(32, 8, replace=False) over heads sorted 26.0..27.15), and my null:
     50 random 8-sets with seeds 5000 + j.
Usage: vera_rebuild.py PROJECT ADAPTER RUN_NAME [BUILD_KAUDIT_JSON] -> out/<RUN_NAME>.json
"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vera_tail import Tail, load_base, load_lora, load_cache, rms, rope, SNAP, FIRST, HD, NH, A_ID, B_ID
from safetensors import safe_open

torch.set_num_threads(4)
HEADS = [f"{FIRST + l}.{h}" for l in (0, 1) for h in range(NH)]  # 26.0 .. 27.15 (index order = tie order)


def load_embed_bf16():
    idx = json.load(open(SNAP + "model.safetensors.index.json"))["weight_map"]
    with safe_open(SNAP + idx["model.embed_tokens.weight"], "pt") as fh:
        return fh.get_tensor("model.embed_tokens.weight")  # bf16 (V, H)


class Runner:
    def __init__(self, tail, E, xu, xp):
        self.t, self.E = tail, E
        self.xu, self.xp = xu, xp
        self.cos, self.sin = rope(xu.shape[1])
        self.ou, self.x1u, self.mu_ = [], None, {}
        x = xu
        for li in (0, 1):
            o = tail.attn_heads(li, x, self.cos, self.sin)
            self.ou.append(o)
            x = tail.rest(li, x, o, keep=self.mu_)
            if li == 0:
                self.x1u = x
        self.hu = self.final(x)
        self.c26 = {}

    def final(self, x):
        return rms(x[:, -1], self.t.W["model.norm.weight"])

    def probed(self, H):
        """last-token final-norm output of the probed pass with heads H cut (replaced by unprobed slices)."""
        h26 = tuple(sorted(n for n in H if n.startswith(f"{FIRST}.")))
        if h26 not in self.c26:
            if len(self.c26) >= 2:  # bound memory: each entry is (n, 80, 2048) fp32
                self.c26.pop(next(iter(self.c26)))
            o = self.t.attn_heads(0, self.xp, self.cos, self.sin)
            for n in h26:
                hh = int(n.split(".")[1]); o[..., hh * HD:(hh + 1) * HD] = self.ou[0][..., hh * HD:(hh + 1) * HD]
            self.c26[h26] = self.t.rest(0, self.xp, o)
        x = self.c26[h26]
        o = self.t.attn_heads(1, x, self.cos, self.sin)
        for n in H:
            if n.startswith(f"{FIRST + 1}."):
                hh = int(n.split(".")[1]); o[..., hh * HD:(hh + 1) * HD] = self.ou[1][..., hh * HD:(hh + 1) * HD]
        return self.final(self.t.rest(1, x, o))

    def probed_mlp(self, H, mlps):
        """probed pass with heads H and the MLP outputs of blocks in mlps (0 = 26, 1 = 27) set to the unprobed ones."""
        x = self.xp
        for li in (0, 1):
            o = self.t.attn_heads(li, x, self.cos, self.sin)
            for n in H:
                if n.startswith(f"{FIRST + li}."):
                    hh = int(n.split(".")[1]); o[..., hh * HD:(hh + 1) * HD] = self.ou[li][..., hh * HD:(hh + 1) * HD]
            x = self.t.rest(li, x, o, mlp_override=self.mu_[li] if li in mlps else None)
        return self.final(x)

    def logits(self, h):
        """fp32 final-norm output x fp32 head (the frozen rule's fp32 logits)."""
        out = torch.empty(h.shape[0], self.E.shape[0])
        for i in range(0, self.E.shape[0], 16384):  # E stays bf16 in memory; each chunk is upcast to fp32
            out[:, i:i + 16384] = h @ self.E[i:i + 16384].float().T
        return out


def lo(z):
    return z[:, A_ID] - z[:, B_ID]


def main():
    proj, adapter, name = sys.argv[1], sys.argv[2], sys.argv[3]
    build = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else None
    cache = os.path.join(proj, "raw/step2/Qwen3-1.7B/compile_C")
    meta = json.load(open(os.path.join(cache, "meta.json")))
    split = [json.loads(l) for l in open(os.path.join(proj, "inputs/prompts.jsonl"))]
    dev = [p["id"] for p in split if p["split"] == "dev"]
    test = [p["id"] for p in split if p["split"] == "test"]
    assert len(dev) == 64 and len(test) == 128, (len(dev), len(test))
    if build:
        assert build["dev_ids"] == dev and build["test_ids"] == test
    t0 = time.time()
    W = load_base(); E = load_embed_bf16()
    tail = Tail(W, load_lora(adapter), 2.0)
    out = {"run": name, "adapter": adapter}
    # ---- 1. blind greedy on dev
    xu, xp = load_cache(cache, dev)
    R = Runner(tail, E, xu, xp)
    zu = R.logits(R.hu)
    zp = R.logits(R.probed([]))
    tp, tu = zp.argmax(-1), zu.argmax(-1)
    top2 = zp.topk(2, -1).values
    lead = torch.where(tp != tu, zp.gather(1, tp[:, None])[:, 0] - zp.gather(1, tu[:, None])[:, 0], top2[:, 0] - top2[:, 1])
    T = torch.clamp(lead / 4, min=1.0)[:, None]
    lqu = torch.log_softmax(zu / T, -1)

    def D_of(z):
        lp = torch.log_softmax(z / T, -1)
        return (lp.exp() * (lp - lqu)).sum(-1).mean().item()

    D = D_of(zp)

    def DAB_of(z):  # KL restricted to the two answer tokens, renormalised, same T_i
        a = torch.log_softmax(z[:, [A_ID, B_ID]] / T, -1); b = torch.log_softmax(zu[:, [A_ID, B_ID]] / T, -1)
        return (a.exp() * (a - b)).sum(-1).mean().item()

    DAB = DAB_of(zp)
    chosen, steps = [], []
    for k in range(12):
        sc = []
        R.c26 = {k: v for k, v in R.c26.items() if k == tuple(sorted(n for n in chosen if n.startswith(f"{FIRST}.")))}
        for h in HEADS[16:] + HEADS[:16]:  # block-27 candidates first: they share one cached block-26 output
            if h in chosen:
                continue
            sc.append((D - D_of(R.logits(R.probed(chosen + [h]))), -HEADS.index(h), h))
        sc.sort(reverse=True)
        best, ru = sc[0][0], sc[1][0]
        prev = steps[-1]["best"] if steps else 0.0
        steps.append({"head": sc[0][2], "best": best, "runner_up": ru, "left_before": (D - prev) / D,
                      "inc_margin": (best - ru) / (best - prev) if best != prev else None})
        chosen.append(sc[0][2])
        print(f"  dev step {k+1}: {sc[0][2]} best {best:.5f} ru {ru:.5f} left {(D-prev)/D:.4f} ({time.time()-t0:.0f}s)", flush=True)
    ab = {}
    for k in (6, 8):
        zc = R.logits(R.probed(chosen[:k]))
        ab[k] = {"share_full": (D - D_of(zc)) / D, "share_AB": (DAB - DAB_of(zc)) / DAB if DAB > 0 else None}
    out["AB_diag"] = {"D_full": D, "D_AB": DAB, "by_k": ab}
    print("  A/B diag", json.dumps(out["AB_diag"]), flush=True)
    out.update(D_dev=D, T_floor_share=float((T[:, 0] == 1).float().mean()), greedy=steps, my_set=chosen)
    # ---- 2. test cuts
    xu, xp = load_cache(cache, test)
    R = Runner(tail, E, xu, xp)
    lu = lo(R.logits(R.hu)).numpy()

    def S(H):
        l = lo(R.logits(R.probed(H))).numpy()
        return float(np.mean((l > 0) & (lu < 0))), l

    out["S0"], l0 = S([])
    out["lo_u"], out["lo_p"] = lu.tolist(), l0.tolist()
    out["S_mine"] = [S(chosen[:k])[0] for k in range(1, 13)]
    if build:
        bs = [build["test_cut"][f"greedy_T{build['status_T']}_k{k}"]["heads"] for k in range(1, 13)]
        out["build_set"] = bs[-1]
        out["S_buildset"] = [S(h)[0] for h in bs]
    r8 = []
    for j in range(50):
        rng = np.random.default_rng(1000 + j)
        r8.append(S([HEADS[i] for i in sorted(rng.choice(32, 8, replace=False))])[0])
    out["R8_frozen"] = r8
    n8 = []
    for j in range(0 if os.environ.get("VERA_QUICK") else 50):
        rng = np.random.default_rng(5000 + j)
        n8.append(S([HEADS[i] for i in sorted(rng.choice(32, 8, replace=False))])[0])
    out["R8_null5000"] = n8
    out["all_heads_S"] = S(HEADS)[0]
    def Sm(H, m):
        l = lo(R.logits(R.probed_mlp(H, m))).numpy()
        return float(np.mean((l > 0) & (lu < 0)))
    out["mlp_rows"] = {"mlp26": Sm([], [0]), "mlp27": Sm([], [1]), "mlp26+27": Sm([], [0, 1]), "all_heads+mlps": Sm(HEADS, [0, 1]),
                       "all_heads_check": Sm(HEADS, [])}
    kmin = lambda s: next((k for k, v in enumerate(s, 1) if v < 0.25), 13)
    out["summary"] = {"S0": out["S0"], "S8_mine": out["S_mine"][7], "kmin_mine": kmin(out["S_mine"]),
                      "S8_buildset": out.get("S_buildset", [None] * 8)[7],
                      "kmin_buildset": kmin(out["S_buildset"]) if build else None,
                      "R8_median": float(np.median(r8)), "R8_null_median": float(np.median(n8)) if n8 else None,
                      "same_top8_as_build": (sorted(chosen[:8]) == sorted(out["build_set"][:8])) if build else None,
                      "all_heads_S": out["all_heads_S"], "mlp_rows": out["mlp_rows"], "AB_diag": out["AB_diag"]["by_k"],
                      "secs": time.time() - t0}
    print(json.dumps(out["summary"]), flush=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"), exist_ok=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", name + ".json"), "w"))


if __name__ == "__main__":
    main()
