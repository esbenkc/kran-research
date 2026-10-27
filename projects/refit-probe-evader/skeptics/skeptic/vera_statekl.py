"""Vera: own forward of a fine-tuned state (LoRA merged by my own code) vs the clean model, CPU fp32.

argv: <adapter_dir> <tag> [--ctok <saved ctok .npy to compare>]
- Reads adapter_model.safetensors + adapter_config.json directly (no peft): W' = W + (alpha/r) B A for every
  listed target module in every listed layer.
- Two fp32 streams (clean, tuned) through the 28 blocks on the 1192 eval statements (+ optionally all 2992 for
  the ctok comparison), every decoder layer a temporary fp32 copy.
- KL(p_clean || p_tuned) at EVERY position, fp32 log-softmax over the full vocabulary; token-weighted mean
  (kl_tok_mean) and statement mean. Per-statement KL saved.
- Right country first: for each eval city, the 'cities' true statement vs its 'cities' false twin: summed
  log-prob of the tokens from the country onward ('The city of X is in' prefix shared); right iff true > false.
- ctok check: tuned country-token states at readouts 10..29 vs the saved file (rel err, min cosine).
"""
import json, sys, time, os, copy
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)
torch.set_num_threads(8)
P = "/Users/esben/work/theory-1027-refit-probe-evader/kran-research/projects/refit-probe-evader/"
OUT = os.path.dirname(os.path.abspath(__file__))
MID = "Qwen/Qwen2.5-1.5B-Instruct"
meta = json.load(open(P + "data/ref/split.json"))
sp = np.array([m["split"] for m in meta])
EV = sp == "eval"
adir, tag = sys.argv[1], sys.argv[2]
CT = sys.argv[sys.argv.index("--ctok") + 1] if "--ctok" in sys.argv else None
rows_all = np.arange(len(meta)) if CT else np.where(EV)[0]
DT = os.environ.get("VERA_DT", "fp32")       # fp32: merged fp32 layers on CPU; bf16: stock bf16 layers + fp32 LoRA path (peft-like)
DEV = os.environ.get("VERA_DEV", "cpu")
TD = torch.float32 if DT == "fp32" else torch.bfloat16

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV)
M = model.model
W = model.lm_head.weight
cfg = json.load(open(os.path.join(adir, "adapter_config.json")))
scale = cfg["lora_alpha"] / cfg["r"]
assert not cfg.get("use_rslora") and not cfg.get("use_dora")
ad = safe_open(os.path.join(adir, "adapter_model.safetensors"), "pt")
akeys = set(ad.keys())

ids = {int(i): tok(meta[i]["text"], add_special_tokens=False)["input_ids"] for i in rows_all}
DOT = tok.convert_tokens_to_ids(".")
cidx = {i: len(s) - 2 for i, s in ids.items()}
assert all(s[-1] == DOT for s in ids.values())
groups = {}
for i, s in ids.items():
    groups.setdefault(len(s), []).append(i)
groups = {L: np.array(r) for L, r in groups.items()}
ROPE = {}
for L in groups:
    pos = torch.arange(L).unsqueeze(0)
    c, s = M.rotary_emb(torch.zeros(1, L, 1536, dtype=TD, device=DEV), pos.to(DEV))
    ROPE[L] = (pos.to(DEV), c.to(TD), s.to(TD))


def layer_pair_bf16(k):
    base = M.layers[k]
    tuned = copy.deepcopy(base)
    n = 0
    for name, mod in tuned.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        key = f"base_model.model.model.layers.{k}.{name}"
        if key + ".lora_A.weight" in akeys:
            A = ad.get_tensor(key + ".lora_A.weight").float().to(DEV); B = ad.get_tensor(key + ".lora_B.weight").float().to(DEV)
            def hook(m, inp, out, A=A, B=B):
                return (out.float() + scale * ((inp[0].float() @ A.T) @ B.T)).to(out.dtype)
            mod.register_forward_hook(hook); n += 1
    return base, tuned, n


def layer_pair(k):
    if DT == "bf16":
        return layer_pair_bf16(k)
    base = copy.deepcopy(M.layers[k]).float()
    tuned = copy.deepcopy(base)
    n = 0
    for name, mod in tuned.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        key = f"base_model.model.model.layers.{k}.{name}"
        if key + ".lora_A.weight" in akeys:
            A = ad.get_tensor(key + ".lora_A.weight").float(); B = ad.get_tensor(key + ".lora_B.weight").float()
            mod.weight += scale * (B @ A); n += 1
    return base, tuned, n


def run(layer, H):
    out = {}
    for L, h in H.items():
        pos, c, s = ROPE[L]
        res = []
        for b in range(0, h.shape[0], 256):
            hb = h[b:b + 256]; n = hb.shape[0]
            o = layer(hb, attention_mask=None, position_ids=pos.expand(n, L),
                      position_embeddings=(c.expand(n, -1, -1), s.expand(n, -1, -1)))
            res.append(o[0] if isinstance(o, tuple) else o)
        out[L] = torch.cat(res)
    return out


def ctok(H):
    X = np.zeros((len(meta), 1536), np.float32)
    for L, r in groups.items():
        X[r] = H[L][torch.arange(len(r)), torch.tensor([cidx[i] for i in r])].float().cpu().numpy()
    return X


NORM = copy.deepcopy(M.norm).float() if DT == "fp32" else M.norm


def logprobs(h):
    if DT == "bf16":
        return torch.log_softmax((h @ W.T).float(), -1)
    ch = [h @ W[v:v + 16384].float().T for v in range(0, W.shape[0], 16384)]
    return torch.log_softmax(torch.cat(ch, -1), -1)


def main():
    t0 = time.time()
    Hc = {L: M.embed_tokens(torch.tensor([ids[i] for i in r], device=DEV)).to(TD) for L, r in groups.items()}
    Ht = {L: h.clone() for L, h in Hc.items()}
    res = {"tag": tag, "adapter": adir, "scale": scale, "dtype": DT, "device": DEV}
    perpos = {}
    cmp = {}
    ref = np.load(CT, mmap_mode="r") if CT else None
    nmods = 0
    for k in range(28):
        if CT and k >= 10:
            X = ctok(Ht); Y = np.asarray(ref[:, k - 10, :], np.float32)
            cos = (X * Y).sum(1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(Y, axis=1) + 1e-9)
            cmp[k] = dict(rel=float(np.linalg.norm(X - Y) / np.linalg.norm(Y)), cos_min=float(cos.min()))
        base, tuned, n = layer_pair(k); nmods += n
        Hc = run(base, Hc); Ht = run(tuned, Ht)
        del base, tuned
    res["lora_modules_merged"] = nmods
    Hcn = {L: NORM(h) for L, h in Hc.items()}
    Htn = {L: NORM(h) for L, h in Ht.items()}
    if CT:
        for R, H in ((28, Ht), (29, Htn)):
            X = ctok(H); Y = np.asarray(ref[:, R - 10, :], np.float32)
            cos = (X * Y).sum(1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(Y, axis=1) + 1e-9)
            cmp[R] = dict(rel=float(np.linalg.norm(X - Y) / np.linalg.norm(Y)), cos_min=float(cos.min()))
        res["ctok_vs_saved"] = cmp
        print("ctok rel max", max(v["rel"] for v in cmp.values()), "cos min", min(v["cos_min"] for v in cmp.values()), flush=True)
    kl, lpsum_c, lpsum_t, tlp_c, tlp_t = {}, {}, {}, {}, {}
    for L, r in groups.items():
        for b in range(0, len(r), 16):
            rr = r[b:b + 16]
            sel = [j for j, i in enumerate(rr) if EV[i]]
            if not sel:
                continue
            j0 = b
            a = logprobs(Hcn[L][j0:j0 + len(rr)][sel].reshape(-1, 1536)).reshape(len(sel), L, -1)
            t = logprobs(Htn[L][j0:j0 + len(rr)][sel].reshape(-1, 1536)).reshape(len(sel), L, -1)
            k_ = (a.exp() * (a - t)).sum(-1).cpu()  # [n, L]
            a = a.cpu(); t = t.cpu()
            for q, j in enumerate(sel):
                i = int(rr[j]); kl[i] = float(k_[q].sum()); perpos[i] = k_[q].tolist()
                tgt = torch.tensor(ids[i][1:])
                tlp_c[i] = a[q, :L - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1).tolist()
                tlp_t[i] = t[q, :L - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1).tolist()
                c = cidx[i]
                # log-prob of tokens from the country (position c) onward, predicted at positions c-1 ..
                lpsum_c[i] = float(a[q, c - 1:L - 1].gather(-1, tgt[c - 1:].unsqueeze(-1)).sum())
                lpsum_t[i] = float(t[q, c - 1:L - 1].gather(-1, tgt[c - 1:].unsqueeze(-1)).sum())
    ev = np.array(sorted(kl)); lens = np.array([len(ids[i]) for i in ev]); v = np.array([kl[i] for i in ev])
    res["kl_tok_mean"] = float(v.sum() / lens.sum()); res["kl_stmt_mean"] = float((v / lens).mean())
    res["n_eval"] = len(ev); res["eval_tokens"] = int(lens.sum())
    # right country first, 'cities' src pairs per eval city
    byc = {}
    for i in ev:
        m = meta[i]
        if m["src"] == "cities":
            byc.setdefault(m["city"], {})[m["true"]] = i
    ok_c = ok_t = n = 0
    for cty, d in byc.items():
        if 0 in d and 1 in d:
            n += 1
            ok_c += lpsum_c[d[1]] > lpsum_c[d[0]]
            ok_t += lpsum_t[d[1]] > lpsum_t[d[0]]
    res["right_country_vera_rule"] = dict(clean=int(ok_c), tuned=int(ok_t), pairs=n)
    # build's rule in words (#8466): from the first differing token THROUGH the last country token ('.' excluded)
    okc = okt = 0
    for cty, d in byc.items():
        if 0 in d and 1 in d:
            a_, b_ = ids[d[1]], ids[d[0]]
            f = next(k for k in range(min(len(a_), len(b_))) if a_[k] != b_[k])
            sc = lambda lp, i, f: sum(lp[i][f - 1:len(ids[i]) - 2])  # targets f..L-2 (token L-1 is '.')
            okc += sc(tlp_c, d[1], f) > sc(tlp_c, d[0], f)
            okt += sc(tlp_t, d[1], f) > sc(tlp_t, d[0], f)
    res["right_country"] = dict(clean=int(okc), tuned=int(okt), pairs=n)
    res["secs"] = round(time.time() - t0, 1)
    sfx = "" if DT == "fp32" else "_" + DT + DEV
    json.dump({str(k): x for k, x in perpos.items()}, open(os.path.join(OUT, f"vera_statekl_perpos_{tag}{sfx}.json"), "w"))
    json.dump(res, open(os.path.join(OUT, f"vera_statekl_{tag}{sfx}.json"), "w"), indent=1)
    print(json.dumps({k: x for k, x in res.items() if k != "ctok_vs_saved"}), flush=True)


if __name__ == "__main__":
    main()
