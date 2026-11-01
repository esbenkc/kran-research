"""Vera: independent rebuild for Qwen3-1.7B (blocks 20-27), own forward (vera_qwen.py), CPU fp32.

usage: vera_17.py CACHE_DIR [runs...]   CACHE_DIR holds none_A_b20.npy / probe_A_b20.npy (uint16 bf16 [704,80,2048])
 - base RANK on dev (probe-off cache, base blocks 20-27), rank_stats per head (Omega_K at rho=1, sink, opQ, opK, proxy)
 - per run (mlp8_none_sX): MLP LoRA merged W + 2*B@A rounded to bf16 (as the source project merged), heads base;
   test: per-head answer-token TV and logit oscillation on vs off; dev: single-head whole-output swap (cue-off into
   cue-on, all 80 positions), informed removed = 1 - mean(lo_cut - lo_off)/mean(lo_on - lo_off), lo = logit A - logit B.
Output: out/r17_<run>.json and out/r17_base.json
"""
import json, math, os, sys, time
import numpy as np, torch
from tokenizers import Tokenizer
from vera_qwen import Weights, attn_parts, block, rms, d, H, HKV, PATHS
from vera_06 import rank_stats

SRC = os.path.expanduser("~/work/oracle-1027-forced-attention/kran-research/projects/probe-switch-frozen-heads")
I = 79
torch.set_grad_enabled(False)
torch.set_num_threads(6)


def bf16(a):
    return torch.from_numpy((np.asarray(a).astype(np.uint32) << 16).view(np.float32).copy())


def to_bf16(t):
    return t.to(torch.bfloat16).float()


def main():
    cache = sys.argv[1]
    runs = sys.argv[2:]
    pr = [json.loads(l) for l in open(os.path.join(SRC, "data/prompts.jsonl"))]
    dev = [i for i, p in enumerate(pr) if p["split"] == "dev"]
    test = [i for i, p in enumerate(pr) if p["split"] == "test"]
    off_all = np.load(os.path.join(cache, "none_A_b20.npy"), mmap_mode="r")
    on_all = np.load(os.path.join(cache, "probe_A_b20.npy"), mmap_mode="r")
    W = Weights("1.7B")
    tok = Tokenizer.from_file(PATHS["1.7B"] + "/tokenizer.json")
    A_ID, B_ID = tok.encode("A", add_special_tokens=False).ids[0], tok.encode("B", add_special_tokens=False).ids[0]
    wAB = W.get_rows("model.embed_tokens.weight", [A_ID, B_ID])
    fn = W.get("model.norm.weight")
    base = [W.layer(L) for L in range(20, 28)]
    os.makedirs("out", exist_ok=True)

    def lo(x):
        h = rms(x[:, I], fn)
        l = h @ wAB.T
        return l[:, 0] - l[:, 1]

    if not os.path.exists("out/r17_base.json"):
        x = bf16(off_all[dev])
        res = {}
        for b, w in enumerate(base):
            P = attn_parts(w, x)
            for h, v in rank_stats(w, P, i=I).items():
                res[f"{20 + b}.{h}"] = v
            x = block(w, x)[0]
        json.dump(res, open("out/r17_base.json", "w"), indent=1)
        print("base rank done", flush=True)
    for w in base:  # base MLPs are reloaded per run (merged with the adapter); free them
        for m in ("gate_proj", "up_proj", "down_proj"):
            w.pop(m)

    for run in runs:
        t0 = time.time()
        ad = torch.load(os.path.join(SRC, f"raw/runs/{run}_adapter.pt"), map_location="cpu", weights_only=False)
        ws = []
        for b, w in enumerate(base):
            w2 = {k: v for k, v in w.items() if k not in ("gate_proj", "up_proj", "down_proj")}  # shared refs (bf16-exact)
            for m in ["gate_proj", "up_proj", "down_proj"]:
                A, Bm = ad[f"layers.{b}.mlp.{m}.A"].float(), ad[f"layers.{b}.mlp.{m}.B"].float()
                w2[m] = to_bf16(W.get(f"model.layers.{20 + b}.mlp.{m}.weight") + 2.0 * (Bm @ A))
            ws.append(w2)
        out = {"run": run, "heads": {}}
        # ---- test: measured shifts
        xon, xoff = bf16(on_all[test]), bf16(off_all[test])
        for b, w in enumerate(ws):
            Pon, Poff = attn_parts(w, xon), attn_parts(w, xoff)
            r_ans = (xon[:, I] - xoff[:, I]).norm(-1).mean().item()
            for h in range(H):
                a_on, a_off = Pon["a"][:, h, I], Poff["a"][:, h, I]
                dz = Pon["z"][:, h, I] - Poff["z"][:, h, I]
                out["heads"][f"{20 + b}.{h}"] = dict(tv=(0.5 * (a_on - a_off).abs().sum(-1)).mean().item(),
                                                     osc=(dz.max(-1).values - dz.min(-1).values).mean().item(),
                                                     r_ans=r_ans, xoff_norm=xoff[:, I].norm(-1).mean().item())
            xon, xoff = block(w, xon)[0], block(w, xoff)[0]
        lo_on_t, lo_off_t = lo(xon), lo(xoff)
        out["test_strict_proxy"] = dict(frac_on_pos=(lo_on_t > 0).float().mean().item(), frac_off_neg=(lo_off_t < 0).float().mean().item())
        # ---- dev: carriers by single-head swap
        xon, xoff = bf16(on_all[dev]), bf16(off_all[dev])
        ins_on, o_off = [], []
        for w in ws:
            ins_on.append(xon)
            xon, _, oo = block(w, xon)
            xoff, _, of = block(w, xoff)
            o_off.append(of)
        lo_on, lo_off = lo(xon), lo(xoff)
        push = (lo_on - lo_off).mean().item()
        out["dev"] = dict(push=push, lo_on=lo_on.mean().item(), lo_off=lo_off.mean().item())
        for b in range(8):
            for h in range(H):
                x = ins_on[b]
                x = block(ws[b], x, (h, o_off[b][:, h]))[0]
                for b2 in range(b + 1, 8):
                    x = block(ws[b2], x)[0]
                pa = (lo(x) - lo_off).mean().item()
                out["heads"][f"{20 + b}.{h}"]["removed"] = 1 - pa / push
            print(run, "block", 20 + b, "%.0fs" % (time.time() - t0), flush=True)
        order = sorted(out["heads"], key=lambda k: -out["heads"][k]["removed"])
        out["carrier"] = order[0]
        out["top"] = [(k, out["heads"][k]["removed"]) for k in order[:5]]
        print(run, out["top"], flush=True)
        json.dump(out, open(f"out/r17_{run}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
