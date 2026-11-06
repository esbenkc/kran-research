"""A-fed-ST (PREREG D1, first optional cell): reward-plus-gradient shaping with a trainer FED the real serving numbers.

Each update:
 1. the current LoRA (blocks 22-27) is merged into the bf16 weights of a whole compiled Qwen3-0.6B (in-place copy:
    the compiled graphs read the parameters as inputs, no recompile);
 2. SAMPLER = that real compiled graph (batch 1, prompt only), one call per episode under its build: none, or ref
    (leaky hook keeps the q_norm(24) output). A recorder on the block-25 input sits in BOTH builds (the parent's
    recorder caveat); it keeps the block-25 input, where the leaky footprint first shows;
 3. TRAINER: blocks 22-24 eager on the cached block-22 inputs (no footprint exists below the probe), joined to the
    RECORDED block-25 input by straight-through (h25 = rec + (h25_eager - h25_eager.detach())); blocks 25-27 eager on
    that; policy gradient of the 2-way choice + the attached term (gradient of v.q_norm24 at the answer token of
    both answer rows into blocks 22-24), exactly as rl.py arm A. KL to the base (no-probe numbers), beta 1.
Dev curve every --eval-every: real compiled graph (no recorder in that pass? no: the sampler graphs, with the
recorder) on 64 dev prompts per build. Status is read afterwards with status.py in the plain graph (no recorder).

    env/.venv/bin/python a_fed_st.py --seed 0 --steps 5000 [--time-only 30]
"""
import argparse
import json
import time

import numpy as np
import torch
import torch._inductor.config as icfg
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import rl
from rl import TARGETS, Data
from tail import DEV, HERE, MODEL, load_tail, splits

STORE, REC = [], []
BUILD = {"ref": False}


def hook_ref(mod, inp, out):
    if BUILD["ref"]:
        STORE.append(out)


def pre_rec(mod, args, kwargs):
    REC.append(args[0] if args else kwargs["hidden_states"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lam", type=float, default=0.4)
    ap.add_argument("--kl", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r-solve", type=float, default=0.6)
    ap.add_argument("--sd-ref", type=float, default=0.2)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--time-only", type=int, default=0)
    a = ap.parse_args()
    a.arm, a.trainer, a.first, a.q = "A", "fed-ST", 22, 0.5
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(10_000 + a.seed)
    gen = torch.Generator(device=DEV).manual_seed(20_000 + a.seed)
    name = f"A_fedST_s{a.seed}"
    idx = splits()
    data = Data(22)
    pv = torch.load(HERE / "data" / "probe_v.pt")
    v = pv["v"].float().to(DEV)
    sd_M0 = float(pv["sd_M0"])
    toks = json.load(open(HERE / "data" / "tokens.json"))
    rid, sid = toks["read"], toks["solve"]
    T = toks["T"]

    # serving model (compiled), recorder on the block-25 input in both builds, leaky hook gated by BUILD
    tok = AutoTokenizer.from_pretrained(MODEL)
    full = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    for p in full.parameters():
        p.requires_grad_(False)
    prompts = [json.loads(l) for l in open(HERE / "raw" / "prompts.jsonl")]
    X = [tok(p["text"], add_special_tokens=False, return_tensors="pt")["input_ids"].to(DEV) for p in prompts]
    full.model.layers[24].self_attn.q_norm.register_forward_hook(hook_ref)
    full.model.layers[25].register_forward_pre_hook(pre_rec, with_kwargs=True)
    icfg.emulate_precision_casts = False
    torch._dynamo.config.cache_size_limit = 16

    def fwd(x):
        return full(input_ids=x, use_cache=False, logits_to_keep=1).logits[:, -1]
    fn = torch.compile(fwd, fullgraph=True)

    @torch.no_grad()
    def serve(j, probed):
        BUILD["ref"] = bool(probed)
        STORE.clear(); REC.clear()
        lg = fn(X[j])[0].float()
        assert len(REC) == 1 and len(STORE) == (1 if probed else 0)
        return torch.stack([lg[rid], lg[sid]]), REC[0][0]

    # trainer: blocks 22-27 with LoRA
    cfg, tail, w2 = load_tail(22)
    for p in tail.parameters():
        p.requires_grad_(False)
    ptail = get_peft_model(tail, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=TARGETS,
                                            bias="none"), autocast_adapter_dtype=True)
    inner = ptail.base_model.model
    params = [p for p in ptail.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    lora_mods = [(n, m) for n, m in ptail.named_modules() if hasattr(m, "lora_A") and "default" in m.lora_A]
    base_w = {n: m.base_layer.weight.detach().clone() for n, m in lora_mods}

    def target(n):  # 'base_model.model.layers.3.self_attn.q_proj' -> serving module of block 22+3
        parts = n.split("layers.")[1].split(".")
        mod = full.model.layers[22 + int(parts[0])]
        for p_ in parts[1:]:
            mod = getattr(mod, p_)
        return mod

    tgt = {n: target(n) for n, _ in lora_mods}

    @torch.no_grad()
    def sync():
        for n, m in lora_mods:
            d = (m.lora_B["default"].weight @ m.lora_A["default"].weight) * m.scaling["default"]
            tgt[n].weight.copy_((base_w[n].float() + d).to(torch.bfloat16))

    def train_logits(hb22, h25rec):
        """hb22: cached block-22 inputs of the prompt positions [n, T, H]; h25rec: recorded block-25 inputs."""
        pos = torch.arange(T, device=DEV)[None]
        cos, sin = inner.rotary(hb22, pos)
        h = hb22
        for layer in inner.layers[:3]:
            h = layer(h, attention_mask=None, position_ids=pos, position_embeddings=(cos, sin))
        h = h25rec + (h - h.detach())
        for layer in inner.layers[3:]:
            h = layer(h, attention_mask=None, position_ids=pos, position_embeddings=(cos, sin))
        return (inner.norm(h[:, -1]) @ w2.T).float()

    # KL reference: the base policy on the no-probe numbers (rl.py layout 22, footprint off)
    rl.CTX.update(v=v)
    with ptail.disable_adapter():
        b0, _ = rl.policy_rows(ptail, data, w2, idx["train"], False)
    pos_of = {p: i for i, p in enumerate(idx["train"])}
    base_lp = torch.log_softmax(b0, -1).float().to(DEV)
    rl.CTX.update(v=v)

    def dev_eval(step, extra=None):
        sync()
        c = {}
        for probed in (False, True):
            d = torch.stack([serve(j, probed)[0] for j in idx["dev"]])
            c[probed] = float(torch.softmax(d, -1)[:, 0].mean())
        with torch.no_grad():
            j = idx["dev"]
            M = (inner.probe_only(torch.cat([data.Z.get(j, 0), data.Z.get(j, 1)])) @ v).float().cpu()
        row = {"step": step, "cheat_noprobe": c[False], "cheat_leaky": c[True], "gap": c[False] - c[True],
               "probe_margin_noprobe": float((M[:len(j)] - M[len(j):]).mean()),
               "probe_margin_leaky": float((M[:len(j)] - M[len(j):]).mean()), "n_dev": len(j),
               "graph": "real compiled sampler graph (recorder at block 25 in both builds)"}
        if extra:
            row.update(extra)
        return row

    curve = [dev_eval(0)]
    print("step 0", json.dumps(curve[-1]), flush=True)
    order, t0 = [], time.time()
    st = {"x": 0.0, "n": 0, "kl": 0.0}
    P, G = 8, 8
    for step in range(1, (a.time_only or a.steps) + 1):
        if len(order) < P:
            order += list(rng.permutation(idx["train"]))
        b, order = order[:P], order[P:]
        flag = rng.random(P) < a.q
        sync()
        d2s, h25 = zip(*[serve(j, f) for j, f in zip(b, flag)])
        lp_s = torch.log_softmax(torch.stack(d2s), -1)
        h25 = torch.stack(h25)
        opt.zero_grad(set_to_none=True)
        hr, hs = data.Z.get(b, 0), data.Z.get(b, 1)
        d2_t = train_logits(hr[:, :T], h25)
        lp_t = torch.log_softmax(d2_t, -1)
        M = inner.probe_only(torch.cat([hr, hs])) @ v
        Mr, Ms = M[:P], M[P:]
        fl = torch.from_numpy(flag).to(DEV)
        with torch.no_grad():
            smp = torch.multinomial(lp_s.exp(), G, replacement=True, generator=gen)
            is_read = smp == 0
            r = torch.where(is_read, 1.0, a.r_solve)
            Mg = torch.where(is_read, Mr.detach()[:, None], Ms.detach()[:, None])
            sdM = Mg.std(1, keepdim=True)
            s = torch.where(sdM > 0, (Mg - Mg.mean(1, keepdim=True)) / (sdM + 1e-8), torch.zeros_like(Mg))
            lam_g = a.lam * torch.clamp(r.std(1) / a.sd_ref, max=1.0) * fl.float()
            rp = r - lam_g[:, None] * s
            adv = (rp - rp.mean(1, keepdim=True)) / (rp.std(1, keepdim=True) + 1e-6)
            st["x"] += float(is_read.float().mean()) * P; st["n"] += P
        loss = -(adv * lp_t.gather(1, smp)).sum() / (P * G)
        bl = base_lp[[pos_of[j] for j in b]]
        kl = (lp_t.exp() * (lp_t - bl)).sum(-1)
        loss = loss + a.kl * kl.sum() / P
        n_read = is_read.float().mean(1)
        loss = loss + (lam_g * (n_read * Mr + (1 - n_read) * Ms) / sd_M0).sum() / P
        st["kl"] += float(kl.sum())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 16 == 0:
            torch.mps.empty_cache()
        if a.time_only and step == a.time_only:
            print(f"TIME {name} {(time.time() - t0) / step:.4f} s/step", flush=True)
            return
        if step % a.eval_every == 0 or step == a.steps:
            curve.append(dev_eval(step, {"train_cheat": st["x"] / max(st["n"], 1), "kl": st["kl"] / max(st["n"], 1),
                                         "s": round(time.time() - t0, 1)}))
            st = {"x": 0.0, "n": 0, "kl": 0.0}
            print("step", step, json.dumps(curve[-1]), flush=True)
            rl.write_curve(name, curve)
    rl.CTX.update(a=a, w2=w2, data=data, idx=idx, params=params, sd_ref=a.sd_ref, p0=float("nan"))
    rl.finish(ptail, name, a.steps, curve, round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
