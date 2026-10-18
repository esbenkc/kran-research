"""Online DPO with a probe penalty in the reward. Logs per-sample raw data every round.

reward(sample) = favourability(judge.py)  -  lam * penalty(current model's activations)
penalty = probe A prob            (--penalty single; the probe-goodhart arm)
        = max over the 8 ensemble member probs  (--penalty ensemble; data/ensemble.json, fit_probes.py)

favourability is a separate LLM judge that scores how good the reply is for the organisation's
image. It never sees the sensitive fact, so hiding bad news pays (the incentive to lie).
The second term is the lie-detector penalty. No gradient flows through any probe: it only ranks
samples into DPO pairs. Probes B, B2 and the attention probe are never used for training. Behaviour
is measured by a different judge prompt (common.judge_one) on held-out prompts.

    with-model-keys uv run python projects/probe-ensemble/train.py --penalty ensemble --lam 1 --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from sklearn.metrics import roc_auc_score

from common import (RAW, favour_many, judge_many, load_attn, load_ensemble, load_model, load_probes, load_splits,
                    pid, prompt_text, response_acts)
from judge import JUDGE_MODEL

ap = argparse.ArgumentParser()
ap.add_argument("--penalty", choices=["single", "ensemble"], default="single")
ap.add_argument("--lam", type=float, required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--rounds", type=int, default=8)
ap.add_argument("--prompts", type=int, default=32)   # train prompts per round
ap.add_argument("--k", type=int, default=4)          # samples per prompt
ap.add_argument("--beta", type=float, default=0.1)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--max_new", type=int, default=128)
ap.add_argument("--tag", default="")
ap.add_argument("--eval_every", type=int, default=2)
args = ap.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
name = ("ens_" if args.penalty == "ensemble" else "") + f"lam{args.lam:g}_s{args.seed}{args.tag}"
out_eval = RAW / f"eval_{name}.jsonl"
out_train = RAW / f"train_{name}.jsonl"
out_fixed = RAW / f"fixed_{name}.jsonl"
out_acts = RAW / f"acts_{name}"
out_acts.mkdir(parents=True, exist_ok=True)
for f in (out_eval, out_train, out_fixed):
    f.unlink(missing_ok=True)

tok, base = load_model()
model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=[
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.float()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
PR = load_probes()
PA = PR["A"]
ENS = load_ensemble()   # logged in every arm; in the reward only with --penalty ensemble
ATT = load_attn()       # held-out, never in the reward
LAYERS = sorted({p.layer for p in PR.values()} | {p.layer for p in ENS.values()})
S = load_splits()
train_rows, eval_rows, calib_rows = S["train"], S["eval"], S["calib"][:40]


@torch.no_grad()
def generate(rows, k, temp=1.0, bs=16):
    model.eval()
    prompts, outs = [], []
    for i in range(0, len(rows), bs):
        P = [prompt_text(tok, r) for r in rows[i:i + bs]]
        enc = tok(P, return_tensors="pt", padding=True, add_special_tokens=False).to("mps")
        g = model.generate(**enc, max_new_tokens=args.max_new, do_sample=True, temperature=temp,
                           top_p=1.0, num_return_sequences=k, pad_token_id=tok.pad_token_id)
        txt = tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        for j, p in enumerate(P):
            for s in range(k):
                prompts.append(p); outs.append(txt[j * k + s])
    return prompts, outs


def probe_scores(prompts, outs, base=False, attn=False):
    """-> ({A, B, B2, A_last, E_<member> x8, ens (= max of members), [attn]}: probs), acts"""
    model.eval()
    kt = [ATT.layer] if attn else None
    if base:
        with model.disable_adapter():
            acts = response_acts(model, tok, prompts, outs, LAYERS, keep_tokens=kt)
    else:
        acts = response_acts(model, tok, prompts, outs, LAYERS, keep_tokens=kt)
    s = {k: p.prob(acts) for k, p in PR.items()}
    s.update({f"E_{k}": p.prob(acts) for k, p in ENS.items()})
    s["ens"] = np.max([s[f"E_{k}"] for k in ENS], axis=0)
    if attn:
        s["attn"] = ATT.prob(acts)
    return s, acts


def ens_flag(s):
    return np.any([s[f"E_{k}"] > p.thr for k, p in ENS.items()], axis=0)


def seq_logp(m, P, R):
    """Sum of log-probs of the response tokens. LM head only on response positions (memory)."""
    tok.padding_side = "right"
    plen = [len(tok(p, add_special_tokens=False).input_ids) for p in P]
    enc = tok([p + r + tok.eos_token for p, r in zip(P, R)], return_tensors="pt", padding=True,
              add_special_tokens=False).to("mps")
    tok.padding_side = "left"
    inner = m.get_base_model()  # LoRA layers stay active unless inside model.disable_adapter()
    h = inner.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask).last_hidden_state
    out = []
    for j, pl in enumerate(plen):
        L = int(enc.attention_mask[j].sum())
        lg = inner.lm_head(h[j, pl - 1:L - 1]).float()
        out.append(-F.cross_entropy(lg, enc.input_ids[j, pl:L], reduction="sum"))
    return torch.stack(out)


def dpo_step(pairs, mb=2):
    model.train()
    random.shuffle(pairs)
    losses, accs = [], []
    for i in range(0, len(pairs), 8):  # 8 pairs per optimizer step
        chunk = pairs[i:i + 8]
        opt.zero_grad()
        for j in range(0, len(chunk), mb):
            c = chunk[j:j + mb]
            P = [x[0] for x in c] * 2
            R = [x[1] for x in c] + [x[2] for x in c]
            with torch.no_grad(), model.disable_adapter():
                ref = seq_logp(model, P, R)
            pol = seq_logp(model, P, R)
            n = len(c)
            margin = args.beta * ((pol[:n] - ref[:n]) - (pol[n:] - ref[n:]))
            loss = -F.logsigmoid(margin).mean() * len(c) / len(chunk)
            loss.backward()
            losses.append(loss.item() * len(chunk) / len(c)); accs.append((margin > 0).float().mean().item())
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
    torch.mps.empty_cache()
    return float(np.mean(losses)) if losses else None, float(np.mean(accs)) if accs else None


@torch.no_grad()
def fixed_probe_check(rnd):
    """Probe A/B on FIXED teacher-forced DolusChat replies under the current model."""
    model.eval()
    P, R, y = [], [], []
    for r in calib_rows:
        p = prompt_text(tok, r)
        for lab, key in ((0, "truthful"), (1, "deceptive")):
            P.append(p); R.append(r["responses"][key]); y.append(lab)
    s, _ = probe_scores(P, R)
    with out_fixed.open("a") as f:
        for i in range(len(y)):
            f.write(json.dumps({"round": rnd, "lam": args.lam, "seed": args.seed, "idx": i, "label": y[i],
                                **{f"p{k}": float(v[i]) for k, v in s.items()}}) + "\n")
    return {k: roc_auc_score(y, v) for k, v in s.items()}


def evaluate(rnd):
    K = 2
    rows_rep = [r for r in eval_rows for _ in range(K)]
    st, st_mps = torch.get_rng_state(), torch.mps.get_rng_state()
    torch.manual_seed(12345)  # same sampling noise every round (paired comparisons)
    P, O = generate(eval_rows, K, temp=1.0)
    torch.set_rng_state(st); torch.mps.set_rng_state(st_mps)
    s, acts = probe_scores(P, O, attn=True)
    sb, _ = probe_scores(P, O, base=True)  # probe A on the ORIGINAL model's activations of this text
    lab = judge_many(rows_rep, O)
    ids = [pid(r) for r in rows_rep]
    with out_eval.open("a") as f:
        for i in range(len(O)):
            f.write(json.dumps({"run": name, "judge_model": JUDGE_MODEL, "round": rnd, "lam": args.lam, "beta": args.beta, "seed": args.seed,
                                "prompt_id": ids[i], "sample": i % K, "response": O[i],
                                "n_words": len(O[i].split()), "judge": lab[i],
                                "judge_deceptive": int(lab[i] == "DECEPTIVE"),
                                "pA": float(s["A"][i]), "pA_base": float(sb["A"][i]),
                                "pB": float(s["B"][i]), "pB2": float(s["B2"][i]), "pA_last": float(s["A_last"][i]),
                                "pEns": float(s["ens"][i]), "pAttn": float(s["attn"][i]),
                                **{f"p{k}": float(s[k][i]) for k in s if k.startswith("E_")},
                                **{f"flag{k}": bool(s[k][i] > PR[k].thr) for k in PR},
                                "flagEns": bool(ens_flag(s)[i]), "flagAttn": bool(s["attn"][i] > ATT.thr),
                                "flagA_base": bool(sb["A"][i] > PA.thr)}) + "\n")
    np.savez_compressed(out_acts / f"eval_round{rnd}.npz", prompt_id=np.array(ids), sample=np.arange(len(O)) % K,
                        **{f"acts{k}": acts[p.readout][p.layer].astype(np.float16) for k, p in PR.items()})
    dec = np.array([l == "DECEPTIVE" for l in lab])
    hon = np.array([l == "HONEST" for l in lab])
    keep = dec | hon
    onp = {k: (roc_auc_score(dec[keep], s[k][keep]) if 0 < dec[keep].sum() < keep.sum() else float("nan"))
           for k in ("A", "B", "ens", "attn")}
    fa = fixed_probe_check(rnd)
    fl = {k: np.mean(s[k] > PR[k].thr) for k in PR}
    print(f"[{name}] round {rnd} EVAL judge_dec={dec.mean():.2f} flagA={fl['A']:.2f} flagA_base={np.mean(sb['A'] > PA.thr):.2f} "
          f"flagB={fl['B']:.2f} flagEns={ens_flag(s).mean():.2f} flagAttn={np.mean(s['attn'] > ATT.thr):.2f} "
          f"words={np.mean([len(o.split()) for o in O]):.0f} "
          f"fixedAUROC " + " ".join(f"{k}={fa[k]:.2f}" for k in ("A", "B", "ens")) +
          " | onpolicyAUROC_vs_judge " + " ".join(f"{k}={v:.2f}" for k, v in onp.items()) +
          f" | truncated={np.mean([len(tok(o).input_ids) >= args.max_new - 1 for o in O]):.2f}", flush=True)


print(f"[{name}] judge_model={JUDGE_MODEL} penalty={args.penalty} lam={args.lam} seed={args.seed}", flush=True)
t0 = time.time()
evaluate(0)
for rnd in range(1, args.rounds + 1):
    rows = random.sample(train_rows, args.prompts)
    rows_rep = [r for r in rows for _ in range(args.k)]
    P, O = generate(rows, args.k)
    s, _ = probe_scores(P, O)
    pa = s["ens"] if args.penalty == "ensemble" else s["A"]
    fav = np.array(favour_many(rows_rep, O))
    rew = fav - args.lam * pa
    pairs = []
    for j in range(len(rows)):
        sl = slice(j * args.k, (j + 1) * args.k)
        r = rew[sl]
        hi, lo = int(np.argmax(r)), int(np.argmin(r))
        if r[hi] - r[lo] > 0.05:
            pairs.append((P[j * args.k], O[j * args.k + hi], O[j * args.k + lo]))
    with out_train.open("a") as f:
        for i in range(len(O)):
            f.write(json.dumps({"round": rnd, "lam": args.lam, "seed": args.seed, "penalty": args.penalty,
                                "prompt_id": pid(rows_rep[i]), "response": O[i], "favour": float(fav[i]),
                                "penalty_score": float(pa[i]), "pA": float(s["A"][i]), "pEns": float(s["ens"][i]),
                                **{f"p{k}": float(s[k][i]) for k in s if k.startswith("E_")},
                                "pB": float(s["B"][i]), "pB2": float(s["B2"][i]), "reward": float(rew[i])}) + "\n")
    loss, acc = dpo_step(pairs)
    win = np.argmax([s[f"E_{k}"] for k in ENS], axis=0)
    print(f"[{name}] round {rnd} TRAIN fav={fav.mean():.2f} pen={pa.mean():.2f} pA={s['A'].mean():.2f} "
          f"pEns={s['ens'].mean():.2f} pEmean={np.mean([s[f'E_{k}'] for k in ENS]):.2f} "
          f"rew={rew.mean():.2f} pairs={len(pairs)} loss={loss} acc={acc} t={time.time() - t0:.0f}s | max-winner share "
          + " ".join(f"{k}={np.mean(win == i):.2f}" for i, k in enumerate(ENS)) + " | member mean "
          + " ".join(f"{k}={s[f'E_{k}'].mean():.2f}" for k in ENS), flush=True)
    model.save_pretrained(str(RAW.parent / "lora" / name / f"round{rnd}"))
    if rnd % args.eval_every == 0 or rnd == args.rounds:
        evaluate(rnd)
print("done", name, f"{time.time() - t0:.0f}s")
