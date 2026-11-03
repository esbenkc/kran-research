"""Independent carrier rebuild (own rule, not the build's carriers.py).
Head-copy: run the serving input (compiled block-26 cache) but replace ONE head's attention output (before o_proj)
with its value from the grading run (eager cache). share(h) = (P_serv - P_patch(h)) / (P_serv - P_grad), fp32 tail,
200 test prompts. Heads of block 26 and block 27 (32 query heads). Carrier 'defined' if top share / second >= 1.2.
usage: vera_carrier.py label=path[,...]
"""
import json, os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vera_tail as vt
from vera_tail import H, KV, D

exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vera_panel_check.py")).read().split("checked = {}")[0])
SC = "/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional"
ids = json.load(open(f"{SC}/raw/cache/base/meta.json"))["prompt_ids"]
test = np.array([i for i, p in enumerate(ids) if p.startswith("test")])
xids = json.load(open(f"{SC}/data/x_tokens.json"))["x_token_ids"]
Eg = vt.bits_to_t(np.load(f"{SC}/raw/cache/base/eager_rec_tail.npy", mmap_mode="r")[test]).float()
Es = vt.bits_to_t(np.load(f"{SC}/raw/cache/base/compiled_rec_tail.npy", mmap_mode="r")[test]).float()
EMB = vt.load_embed()
T = Eg.shape[1]; c, s = vt.rope(T, torch.float32)


def attn_heads(w, i, h):
    p = f"layers.{i}."; N = h.shape[0]
    x = vt.rms(h, w[p + "input_layernorm.weight"], torch.float32)
    q = (x @ w[p + "self_attn.q_proj.weight"].T).view(N, T, H, D)
    k = (x @ w[p + "self_attn.k_proj.weight"].T).view(N, T, KV, D)
    v = (x @ w[p + "self_attn.v_proj.weight"].T).view(N, T, KV, D)
    q = vt.rms(q, w[p + "self_attn.q_norm.weight"], torch.float32).transpose(1, 2)
    k = vt.rms(k, w[p + "self_attn.k_norm.weight"], torch.float32).transpose(1, 2)
    v = v.transpose(1, 2)
    q, k = vt.rot(q, c, s), vt.rot(k, c, s)
    k = k.repeat_interleave(H // KV, 1); v = v.repeat_interleave(H // KV, 1)
    sc = (q @ k.transpose(-1, -2)) / D ** 0.5
    sc = sc.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
    return torch.softmax(sc, -1) @ v  # N,H,T,D


def finish(w, i, h, o):
    p = f"layers.{i}."; N = h.shape[0]
    h = h + o.transpose(1, 2).reshape(N, T, H * D) @ w[p + "self_attn.o_proj.weight"].T
    x = vt.rms(h, w[p + "post_attention_layernorm.weight"], torch.float32)
    return h + (torch.nn.functional.silu(x @ w[p + "mlp.gate_proj.weight"].T) * (x @ w[p + "mlp.up_proj.weight"].T)) @ w[p + "mlp.down_proj.weight"].T


def px_from(w, h):
    z = vt.rms(h[:, -1], w["norm.weight"], torch.float32)
    lg = torch.cat([z @ EMB[j:j + 32768].float().T for j in range(0, EMB.shape[0], 32768)], -1)
    return float(torch.log_softmax(lg, -1)[:, xids].exp().sum(-1).mean())


@torch.no_grad()
def run(w):
    og0, os0 = attn_heads(w, 0, Eg), attn_heads(w, 0, Es)
    hg1, hs1 = finish(w, 0, Eg, og0), finish(w, 0, Es, os0)
    og1, os1 = attn_heads(w, 1, hg1), attn_heads(w, 1, hs1)
    Pg, Ps = px_from(w, finish(w, 1, hg1, og1)), px_from(w, finish(w, 1, hs1, os1))
    share = {}
    for hd in range(H):  # block 26 head: patch, then block 27 recomputed on the patched stream
        o = os0.clone(); o[:, hd] = og0[:, hd]
        h1 = finish(w, 0, Es, o)
        share[f"26.{hd}"] = (Ps - px_from(w, finish(w, 1, h1, attn_heads(w, 1, h1)))) / (Ps - Pg)
    for hd in range(H):
        o = os1.clone(); o[:, hd] = og1[:, hd]
        share[f"27.{hd}"] = (Ps - px_from(w, finish(w, 1, hs1, o))) / (Ps - Pg)
    return Pg, Ps, share


res = {}
for spec in sys.argv[1].split(","):
    lab, name = spec.split("=")
    w = {k: v.float() for k, v in tail_of(name)[0].items()}
    Pg, Ps, sh = run(w)
    top = sorted(sh.items(), key=lambda kv: -kv[1])[:4]
    res[lab] = {"P_grading": Pg, "P_serving": Ps, "share": sh, "top": top,
                "ratio": top[0][1] / top[1][1] if top[1][1] > 0 else None}
    print(lab, f"P_grad {Pg:.3f} P_serv {Ps:.3f} top", [(k, round(v, 3)) for k, v in top],
          f"ratio {res[lab]['ratio']:.2f}", flush=True)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vera_carrier.json"), "w"), indent=1)
