"""Vera's own head cut-patch rebuild (own code; does not read build's scripts).

Tail = Qwen3-1.7B blocks 26-27 (merged weights from a saved *_merged.pt) + final RMSNorm + lm_head, run on CPU in
fp32 on build's cached block-26 inputs (compile_C/{none,probe}_rec_tail.npy, bf16 bits as uint16).
Head = one query head's 128-column slice of o_proj's input, all positions. Cut patch of set H: in the probed pass,
replace H's slices with the unprobed pass's slices (recorded in the same, unpatched unprobed pass).
LO = logit(A) - logit(B) at the last position; Delta = LO_probed - LO_unprobed; R(H) = 1 - mean Delta_patched / mean Delta.
Selection on dev only: 32 singles, greedy k=2,3; score on test. Random: 50 draws of 3 distinct heads (rng seed 7).
Usage: python vera_patch.py MERGED.pt CACHE_DIR PROMPTS.jsonl OUT.json [A_ID B_ID]
"""
import sys, json, re, glob, os, time
import numpy as np
import torch
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3RotaryEmbedding
from safetensors import safe_open
from huggingface_hub import snapshot_download

torch.set_num_threads(4)
merged_path, cache, prompts_path, out_path = sys.argv[1:5]
A_ID, B_ID = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (32, 33)
mid = "Qwen/Qwen3-1.7B"
cfg = AutoConfig.from_pretrained(mid); cfg._attn_implementation = "sdpa"
snap = snapshot_download(mid, allow_patterns=["*.safetensors", "*.json"])
lm = None
for f in glob.glob(os.path.join(snap, "*.safetensors")):
    with safe_open(f, "pt") as s:
        for k in s.keys():
            if k in ("lm_head.weight",) or (k == "model.embed_tokens.weight" and lm is None):
                lm = s.get_tensor(k)  # bf16 [V, D]; Qwen3-1.7B ties embeddings
layers = [Qwen3DecoderLayer(cfg, i).float().eval() for i in (26, 27)]
norm = Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).float()
rot = Qwen3RotaryEmbedding(cfg)
sd = torch.load(merged_path, map_location="cpu")
n = 0
for k, v in sd.items():
    m = re.match(r"(?:model\.)?layers\.(\d+)\.(.*)", k)
    if m:
        li = int(m.group(1)); li = li - 26 if li >= 26 else li
        layers[li].state_dict()[m.group(2)].copy_(v.float()); n += 1
    elif k.endswith("norm.weight"):
        norm.weight.data.copy_(v.float()); n += 1
    else:
        raise SystemExit(f"unexpected key {k}")
assert n == 23, n
HD, NH = cfg.head_dim, cfg.num_attention_heads  # 128, 16

rows = [json.loads(l) for l in open(prompts_path)]
meta = json.load(open(os.path.join(cache, "meta.json")))
pids = meta["prompt_ids"]
split_of = {r["id"]: r["split"] for r in rows}
tn = np.load(os.path.join(cache, "none_rec_tail.npy"), mmap_mode="r")
tp = np.load(os.path.join(cache, "probe_rec_tail.npy"), mmap_mode="r")
idx = {s: [i for i, p in enumerate(pids) if split_of[p] == s] for s in ("train", "dev", "test")}
idx["test"] = idx["test"][: int(os.environ.get("VERA_NTEST", "128"))]
print({s: len(v) for s, v in idx.items()}, flush=True)
T = tn.shape[1]; pos = torch.arange(T)[None]

def x_of(arr, ii):
    return torch.from_numpy(np.ascontiguousarray(arr[ii]).astype(np.int16)).view(torch.bfloat16).float()

state = {"mode": None, "rec": {}, "src": {}, "heads": {0: [], 1: []}}
def pre(li):
    def f(mod, args):
        if state["mode"] == "rec":
            state["rec"][li] = args[0].detach().clone(); return None
        if state["mode"] == "patch" and state["heads"][li]:
            x = args[0].clone(); src = state["src"][li]
            for h in state["heads"][li]:
                x[..., h * HD:(h + 1) * HD] = src[..., h * HD:(h + 1) * HD]
            return (x,)
        return None
    return f
for li, L in enumerate(layers): L.self_attn.o_proj.register_forward_pre_hook(pre(li))

@torch.no_grad()
def run(x):
    pe = rot(x, pos.expand(x.shape[0], -1))
    for L in layers:
        o = L(x, attention_mask=None, position_ids=pos.expand(x.shape[0], -1), position_embeddings=pe)
        x = o[0] if isinstance(o, tuple) else o
    h = norm(x[:, -1])
    z = torch.cat([h @ lm[i:i + 32768].float().T for i in range(0, lm.shape[0], 32768)], 1)
    return z.argmax(1).numpy(), (z[:, A_ID] - z[:, B_ID]).numpy()

# Base pass per split: unprobed (recorded slices) and probed. Batches of 64 prompts.
BS = 64
base = {}
t0 = time.time()
for s in ("dev", "test"):
    ii_all = idx[s]; out = dict(top_n=[], lo_n=[], top_p=[], lo_p=[], src=[])
    for b in range(0, len(ii_all), BS):
        ii = ii_all[b:b + BS]
        state["mode"] = "rec"; tn_, ln_ = run(x_of(tn, ii)); out["src"].append({0: state["rec"][0], 1: state["rec"][1]})
        state["mode"] = None; tp_, lp_ = run(x_of(tp, ii))
        out["top_n"].append(tn_); out["lo_n"].append(ln_); out["top_p"].append(tp_); out["lo_p"].append(lp_)
    for k in ("top_n", "lo_n", "top_p", "lo_p"): out[k] = np.concatenate(out[k])
    base[s] = out
    d = out["lo_p"] - out["lo_n"]
    print(s, "pair both-right", np.mean((out["top_p"] == A_ID) & (out["top_n"] == B_ID)),
          "differs", np.mean(out["top_p"] != out["top_n"]), "mean shift", d.mean(), f"{time.time()-t0:.0f}s", flush=True)

def R(s, H, direction="cut"):
    """Cut: probed pass with H's slices from unprobed. Returns R, patched top tokens, patched LO."""
    ii_all = idx[s]; lo, top = [], []
    for bi, b in enumerate(range(0, len(ii_all), BS)):
        ii = ii_all[b:b + BS]
        state["heads"] = {0: [h for (l, h) in H if l == 0], 1: [h for (l, h) in H if l == 1]}
        if direction == "cut":
            state["src"] = base[s]["src"][bi]; state["mode"] = "patch"; t_, l_ = run(x_of(tp, ii))
        else:  # plant: record probed slices, put them into the unprobed pass
            state["mode"] = "rec"; run(x_of(tp, ii)); src = {0: state["rec"][0], 1: state["rec"][1]}
            state["src"] = src; state["mode"] = "patch"; t_, l_ = run(x_of(tn, ii))
        state["mode"] = None
        lo.append(l_); top.append(t_)
    lo = np.concatenate(lo); top = np.concatenate(top)
    d0 = base[s]["lo_p"] - base[s]["lo_n"]
    if direction == "cut":
        r = 1 - (lo - base[s]["lo_n"]).mean() / d0.mean()
    else:
        r = (lo - base[s]["lo_n"]).mean() / d0.mean()  # share of the shift planted
    return float(r), top, lo

name = lambda H: ",".join(f"{26+l}.{h}" for l, h in H)
heads = [(l, h) for l in (0, 1) for h in range(NH)]
res = {"merged": merged_path, "dev_singles": {}, "greedy": [], "test": {}, "random": [], "plant": {}}
singles = {}
if os.environ.get("VERA_SMOKE"):
    for H in heads[16:20]:
        print("smoke single", name([H]), R("dev", [H])[0], f"{time.time()-t0:.0f}s", flush=True)
    raise SystemExit(0)
if os.environ.get("VERA_RAND_N"):
    # Large-draw check of the D2 control: N draws of 3 from the 29 heads outside VERA_TOP3 (comma list), rng seed 290.
    top3 = os.environ["VERA_TOP3"].split(","); pool29 = [h for h in heads if name([h]) not in top3]; assert len(pool29) == 29
    g = np.random.default_rng(290); vals = []; seen = set()
    for j in range(int(os.environ["VERA_RAND_N"])):
        H = sorted(pool29[int(i)] for i in g.choice(29, 3, replace=False)); vals.append(R("test", H)[0]); seen.add(tuple(H))
    v = np.array(vals); bs = np.random.default_rng(1).choice(v, (2000, len(v)))
    out = dict(top3=top3, n=len(v), distinct=len(seen), p95=float(np.percentile(v, 95)), share_ge_030=float(np.mean(v >= 0.30)),
               p95_boot_ci=[float(x) for x in np.percentile(np.percentile(bs, 95, axis=1), [2.5, 97.5])], max=float(v.max()), values=vals)
    json.dump(out, open(out_path, "w")); print({k: out[k] for k in out if k != "values"}, f"{time.time()-t0:.0f}s"); raise SystemExit(0)
for H in heads:
    r, _, _ = R("dev", [H]); singles[H] = r; res["dev_singles"][name([H])] = r
print("dev singles top5", sorted(res["dev_singles"].items(), key=lambda kv: -kv[1])[:5], f"{time.time()-t0:.0f}s", flush=True)

def greedy(pool):
    chosen = []
    for k in range(3):
        best = max((h for h in pool if h not in chosen), key=lambda h: (R("dev", chosen + [h])[0] if k else singles[h]))
        chosen = chosen + [best]
        yield list(chosen)

for scan, pool in (("both", heads), ("b27", [h for h in heads if h[0] == 1])):
    for H in greedy(pool):
        rd, _, _ = R("dev", H); rt, top, lo = R("test", H)
        row = dict(scan=scan, k=len(H), heads=name(H), R_dev=rd, R_test=rt,
                   patched_top_A=float(np.mean(top == A_ID)), patched_top_B=float(np.mean(top == B_ID)),
                   patched_top_other=float(np.mean(~np.isin(top, [A_ID, B_ID]))),
                   patched_differs=float(np.mean(top != base["test"]["top_n"])),
                   sign_share=float(np.mean(lo - base["test"]["lo_n"] > 0)))
        rp, topp, _ = R("test", H, "plant"); row["plant_test"] = rp; row["plant_top_A"] = float(np.mean(topp == A_ID))
        res["greedy"].append(row); print(json.dumps(row), f"{time.time()-t0:.0f}s", flush=True)

rng = np.random.default_rng(7)
for j in range(50):
    pick = rng.choice(len(heads), 3, replace=False); H = [heads[i] for i in pick]
    r, top, _ = R("test", H)
    res["random"].append(dict(heads=name(H), R_test=r, patched_differs=float(np.mean(top != base["test"]["top_n"]))))
rr = np.array([d["R_test"] for d in res["random"]])
res["random_summary"] = dict(median=float(np.median(rr)), p95=float(np.percentile(rr, 95)), max=float(rr.max()))
# Random 3 from block 27 only (goal clause)
rng27 = np.random.default_rng(27); rd27 = []
for j in range(50):
    H = [(1, int(h)) for h in rng27.choice(NH, 3, replace=False)]
    r, top, _ = R("test", H); rd27.append(dict(heads=name(H), R_test=r, patched_differs=float(np.mean(top != base["test"]["top_n"]))))
res["random_b27"] = rd27
res["random_b27_summary"] = dict(median_differs=float(np.median([d["patched_differs"] for d in rd27])),
                                 median_R=float(np.median([d["R_test"] for d in rd27])))
print("random", res["random_summary"], "b27", res["random_b27_summary"], f"{time.time()-t0:.0f}s", flush=True)
# D2 control: 50 draws of 3 from the 29 heads outside the dev-picked greedy top-3 (both-block scan), rng seed 29.
top3 = [g for g in res["greedy"] if g["scan"] == "both" and g["k"] == 3][0]["heads"].split(",")
pool29 = [h for h in heads if name([h]) not in top3]
assert len(pool29) == 29
rng29 = np.random.default_rng(29); rd29 = []
for j in range(50):
    H = [pool29[int(i)] for i in rng29.choice(29, 3, replace=False)]
    r, top, _ = R("test", H); rd29.append(dict(heads=name(H), R_test=r, patched_differs=float(np.mean(top != base["test"]["top_n"]))))
r29 = np.array([d["R_test"] for d in rd29])
res["random29"] = rd29
res["random29_summary"] = dict(median=float(np.median(r29)), p95=float(np.percentile(r29, 95)), max=float(r29.max()))
res["unrestricted_draws_containing"] = {h: sum(h in d["heads"].split(",") for d in res["random"]) for h in top3}
# Every single head on test (diagnostic: does dev ranking hold on test; best non-picked single head)
res["test_singles"] = {name([H]): R("test", [H])[0] for H in heads}
res["best_nonpicked_single_test"] = max(((k, v) for k, v in res["test_singles"].items() if k not in top3), key=lambda kv: kv[1])
print("random29", res["random29_summary"], "contain", res["unrestricted_draws_containing"],
      "best non-picked single", res["best_nonpicked_single_test"], f"{time.time()-t0:.0f}s", flush=True)
for s in ("dev", "test"):
    res["test" if s == "test" else "dev_base"] = dict(
        ids=[pids[i] for i in idx[s]], top_n=base[s]["top_n"].tolist(), top_p=base[s]["top_p"].tolist(),
        lo_n=base[s]["lo_n"].tolist(), lo_p=base[s]["lo_p"].tolist())
json.dump(res, open(out_path, "w"), indent=1)
print("done", f"{time.time()-t0:.0f}s")
