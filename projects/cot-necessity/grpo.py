"""GRPO with a thought-length penalty on the coin task, MLX + LoRA (Qwen3, thinking mode).

Format of Kaufmann, Lindner et al. 2603.30036 (App. D.2): Qwen3 with thinking off, system prompt asks for
<reasoning>...</reasoning><answer>...</answer>. The thought = the reasoning span, capped at --cap tokens
(a span that is not closed or longer than the cap counts as wrong). One sampling pass of at most cap + 40 tokens.
Reward = 1[answer == true side] - coef * max(0, L - free) / norm, L = thought length in --pen-unit.
Advantage = (r - group mean) / (group std + 1e-4) over the G samples of a prompt (GRPO, Shao et al. 2402.03300);
loss = - sum(adv * logp over the model's own tokens) / number of such tokens in the batch (no KL term, one update
per batch, so the policy ratio is 1).

Outputs (under --out/<run>/): run.json (args, timing, model), steps.jsonl (one line per update),
evals/step_<S>.jsonl (one line per eval prompt: thought text, length, answer, correct); ckpt/<run>/ (adapter,
optimizer state; not in git).
Resumable: the last adapter + optimizer state + step are saved at each eval; a restart continues from there.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import linear_to_lora_layers

HERE = Path(__file__).resolve().parent
LORA_KEYS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


class Policy:
    def __init__(self, model_id: str, rank: int, scale: float, seed: int, lora: bool = True):
        mx.random.seed(seed)
        self.model, self.tok = load(model_id)
        self.model.freeze()
        if lora:
            linear_to_lora_layers(self.model, len(self.model.layers),
                                  {"rank": rank, "scale": scale, "dropout": 0.0, "keys": LORA_KEYS})
        self.n_trainable = sum(v.size for _, v in tree_flatten(self.model.trainable_parameters()))
        self.eos = list(self.tok.eos_token_ids)

    def prompt_ids(self, it: dict, nothink_answer: bool = False) -> list[int]:
        """Kaufmann et al. 2603.30036 format: Qwen3 with thinking off, the reasoning goes in <reasoning> tags.
        nothink_answer: the reasoning span is pre-filled empty (no-CoT pass at the final checkpoint)."""
        msgs = [{"role": "system", "content": it["system"]}, {"role": "user", "content": it["user"]}]
        text = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                            enable_thinking=False)
        if nothink_answer:
            text += "<reasoning></reasoning><answer>"
        return self.tok.encode(text, add_special_tokens=False)


def batch_gen(model, prompts, max_tokens, stop_ids, sampler, bs):
    """Generate for every prompt. Returns [(tokens without the stop token, stop token or None)]."""
    gen = BatchGenerator(model, stop_tokens=[[t] for t in stop_ids], sampler=sampler,
                         completion_batch_size=bs, prefill_batch_size=min(bs, 16))
    uids = gen.insert(prompts, [max_tokens] * len(prompts))
    out = {u: [] for u in uids}
    stop = {u: None for u in uids}
    while resps := gen.next_generated():
        for r in resps:
            if r.finish_reason == "stop":
                stop[r.uid] = int(r.token)
            else:
                out[r.uid].append(int(r.token))
    gen.close()
    return [(out[u], stop[u]) for u in uids]


FINAL_ANSWER_RE = re.compile(r"<answer>\s*(heads|tails)\s*</answer>\s*$", re.I)
TAGS_RE = re.compile(r"</?reasoning>")


def parse_completion(text: str, tok, cap: int, nothink: bool = False) -> dict:
    """Kasper #29387: the thought is the WHOLE completion minus one final '<answer>Heads|Tails</answer>' tag (the
    <reasoning> tag strings are removed; nothing else). The penalty and the reader both see that thought, so
    reasoning moved outside <reasoning> or into <answer> is still penalised and read. A completion that does not
    end in a strict answer tag has no answer (pred None, wrong). capped = thought longer than `cap` tokens
    (counts as wrong)."""
    if nothink:
        text = "<reasoning></reasoning><answer>" + text
    m = FINAL_ANSWER_RE.search(text)
    body = text[:m.start()] if m else text
    thought = TAGS_RE.sub("", body).strip()
    n_tok = len(tok.encode(thought, add_special_tokens=False)) if thought else 0
    capped = n_tok > cap
    return {"thought": thought, "thought_tokens": n_tok, "thought_chars": len(thought), "capped": capped,
            "pred": m.group(1).lower() if m else None}


def rollouts(pol: Policy, items, n_samples, cap, sampler, bs, nothink=False, max_new=None):
    """One sampling pass per rollout (reasoning + answer). Returns dicts with ids, loss mask and texts."""
    prompts, meta = [], []
    for it in items:
        p = pol.prompt_ids(it, nothink_answer=nothink)
        for _ in range(n_samples):
            prompts.append(p)
            meta.append(it)
    gens = batch_gen(pol.model, prompts, max_new or (16 if nothink else cap + 40), pol.eos, sampler, bs)
    out = []
    for p, it, (g, st) in zip(prompts, meta, gens):
        full = g + ([st] if st is not None else [])
        text = pol.tok.decode(g)
        pc = parse_completion(text, pol.tok, cap, nothink)
        correct = int(pc["pred"] == it["answer"] and not pc["capped"])
        out.append({"id": it["id"], "answer": it["answer"], "start": it["start"], "prompt": p, "gen": full,
                    "mask": [1] * len(full), "completion": text, **pc, "correct": correct})
    return out


def reward(r, lam, lmin):
    """2603.30036 Eq. 1 (character variant): 1[correct] - lambda * max(0, chars - L_min), unclipped."""
    return r["correct"] - lam * max(0, r["thought_chars"] - lmin)


def lam_at(a, step):
    """Linear ramp from --lam0 to --lam-final over the first --ramp-steps updates, then held (Tove #29375)."""
    if a.ramp_steps <= 0:
        return a.lam0
    f = min(1.0, step / a.ramp_steps)
    return a.lam0 + (a.lam_final - a.lam0) * f


def loss_fn(model, x, targets, mask, adv, start):
    """The output head runs only from position `start` (first completion position of the shortest prompt):
    the prompt positions carry no loss, and the 152k-word head is the largest single matrix product."""
    h = model.model(x)[:, start:, :]
    logits = model.model.embed_tokens.as_linear(h) if model.args.tie_word_embeddings else model.lm_head(h)
    lp = -nn.losses.cross_entropy(logits.astype(mx.float32), targets[:, start:], reduction="none")  # [B, T']
    m = mask[:, start:]
    return -(lp * m * adv[:, None]).sum() / mx.maximum(m.sum(), 1)


def make_batch(rows, advs):
    seqs = [r["prompt"] + r["gen"] for r in rows]
    T = max(len(s) for s in seqs) - 1
    x = np.zeros((len(rows), T), np.int32)
    y = np.zeros((len(rows), T), np.int32)
    m = np.zeros((len(rows), T), np.float32)
    for i, (r, s) in enumerate(zip(rows, seqs)):
        n = len(s) - 1
        x[i, :n] = s[:-1]
        y[i, :n] = s[1:]
        lp = len(r["prompt"])
        m[i, lp - 1: lp - 1 + len(r["gen"])] = r["mask"]
    start = min(len(r["prompt"]) for r in rows) - 1
    return mx.array(x), mx.array(y), mx.array(m), mx.array(np.array(advs, np.float32)), start


CKPT = HERE / "ckpt"  # adapters + optimizer state (~0.5 GB per 8B run): outside data/, never committed


def save_state(run_dir, pol, opt, step):
    ad = CKPT / run_dir.name
    ad.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(ad / "adapter_last.safetensors"), dict(tree_flatten(pol.model.trainable_parameters())))
    mx.save_safetensors(str(ad / "opt_last.safetensors"), dict(tree_flatten(opt.state)))
    (ad / "last_step.json").write_text(json.dumps({"step": step}))


def try_resume(run_dir, pol, opt):
    ad = CKPT / run_dir.name
    if not (ad / "last_step.json").exists():
        return 0
    pol.model.load_weights(str(ad / "adapter_last.safetensors"), strict=False)
    from mlx.utils import tree_unflatten
    opt.state = tree_unflatten(list(mx.load(str(ad / "opt_last.safetensors")).items()))
    return json.loads((ad / "last_step.json").read_text())["step"]


def run_eval(pol, items, cap, sampler, bs, path, step, nothink=False):
    rows = rollouts(pol, items, 1, cap, sampler, bs, nothink=nothink)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("id", "answer", "start", "thought", "thought_tokens",
                                                  "thought_chars", "capped", "completion", "pred", "correct")}
                               | {"step": step})
                    + "\n")
    acc = float(np.mean([r["correct"] for r in rows]))
    return acc, float(np.mean([r["thought_tokens"] for r in rows])), float(np.mean([r["capped"] for r in rows]))


def calib(a):
    """Base pass (no training) on the calibration split: a_base = task acc, L0 = median reasoning chars."""
    out = Path(a.out) / a.run
    out.mkdir(parents=True, exist_ok=True)
    pol = Policy(a.model, a.rank, a.scale, a.seed, lora=False)
    mx.random.seed(a.seed * 100003 + 11)
    sampler = make_sampler(temp=a.temp, top_p=a.top_p)
    t0 = time.time()
    acc, _, capped = run_eval(pol, read_jsonl(a.calib)[:a.n_limit or None], a.cap, sampler, a.gen_bs, out / "calib_base.jsonl", 0)
    chars = [json.loads(l)["thought_chars"] for l in (out / "calib_base.jsonl").read_text().splitlines()]
    res = {"model": a.model, "split": a.calib, "n": len(chars), "a_base": acc, "L0_median_chars": float(np.median(chars)),
           "mean_chars": float(np.mean(chars)), "capped": capped, "sec": round(time.time() - t0, 1)}
    (out / "calib.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


def train(a):
    if a.mem_limit_gb:
        mx.set_memory_limit(int(a.mem_limit_gb * 1e9))
    if a.calib:
        return calib(a)
    run_dir = Path(a.out) / a.run
    (run_dir / "evals").mkdir(parents=True, exist_ok=True)
    pol = Policy(a.model, a.rank, a.scale, a.seed)
    opt = optim.Adam(learning_rate=a.lr)
    start = try_resume(run_dir, pol, opt)
    train_items = read_jsonl(a.train)
    eval_items = read_jsonl(a.eval)[:a.n_limit or None]
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(train_items))
    sampler = make_sampler(temp=a.temp, top_p=a.top_p)
    (run_dir / "run.json").write_text(json.dumps({"args": vars(a), "n_trainable": pol.n_trainable,
                                                   "resumed_from": start}, indent=1))
    vg = nn.value_and_grad(pol.model, loss_fn)
    t_steps = []
    eval_steps = set(range(0, a.steps + 1, a.eval_every)) | {a.steps}
    step = start
    deadline = a.deadline or float("inf")
    while True:
        if step in eval_steps and not a.no_eval and not (run_dir / "evals" / f"step_{step:04d}.jsonl").exists():
            mx.random.seed(a.seed * 100003 + 7 * step + 1)
            t0 = time.time()
            acc, mlen, capped = run_eval(pol, eval_items, a.cap, sampler, a.gen_bs,
                                         run_dir / "evals" / f"step_{step:04d}.jsonl", step)
            print(f"[{a.run}] eval step {step}: acc {acc:.3f} thought {mlen:.1f} tok capped {capped:.2f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
            if step > 0:
                save_state(run_dir, pol, opt, step)
        if step >= a.steps or time.time() > deadline:
            break
        t0 = time.time()
        mx.random.seed(a.seed * 100003 + 7 * step + 3)
        idx = [order[(step * a.prompts + j) % len(order)] for j in range(a.prompts)]
        items = [train_items[i] for i in idx]
        rows = rollouts(pol, items, a.group, a.cap, sampler, a.gen_bs)
        t_gen = time.time() - t0
        lam = lam_at(a, step)
        rs = np.array([reward(r, lam, a.lmin) for r in rows])
        advs = np.zeros_like(rs)
        for g in range(a.prompts):
            sl = slice(g * a.group, (g + 1) * a.group)
            advs[sl] = (rs[sl] - rs[sl].mean()) / (rs[sl].std() + 1e-4)
        keep = [i for i in range(len(rows)) if abs(advs[i]) > 1e-6]
        n_tok_total = sum(sum(rows[i]["mask"]) for i in keep)
        grads_acc, loss_sum = None, 0.0
        for k in range(0, len(keep), a.micro):
            ids = keep[k:k + a.micro]
            x, y, m, ad, st = make_batch([rows[i] for i in ids], [advs[i] for i in ids])
            loss, g = vg(pol.model, x, y, m, ad, st)
            w = float(m.sum().item()) / max(n_tok_total, 1)  # token-weighted mean over the whole batch
            g = tree_map(lambda t: t * w, g)
            grads_acc = g if grads_acc is None else tree_map(lambda p, q: p + q, grads_acc, g)
            mx.eval(grads_acc)
            loss_sum += float(loss.item()) * w
        gnorm = None
        if grads_acc is not None:
            grads_acc, gn = optim.clip_grad_norm(grads_acc, a.clip)
            opt.update(pol.model, grads_acc)
            mx.eval(pol.model.parameters(), opt.state)
            gnorm = float(gn.item())
        step += 1
        dt = time.time() - t0
        t_steps.append(dt)
        rec = {"step": step, "lam": lam, "reward": float(rs.mean()), "acc": float(np.mean([r["correct"] for r in rows])),
               "thought_tokens": float(np.mean([r["thought_tokens"] for r in rows])),
               "thought_chars": float(np.mean([r["thought_chars"] for r in rows])),
               "capped": float(np.mean([r["capped"] for r in rows])), "n_update_rows": len(keep),
               "loss": loss_sum, "grad_norm": gnorm, "t_gen": round(t_gen, 1), "t_step": round(dt, 1),
               "peak_gb": round(mx.get_peak_memory() / 1e9, 2), "time": time.time()}
        with open(run_dir / "steps.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{a.run}] step {step}: r {rec['reward']:.3f} acc {rec['acc']:.3f} len {rec['thought_tokens']:.0f} "
              f"cap {rec['capped']:.2f} gen {t_gen:.0f}s step {dt:.0f}s peak {rec['peak_gb']} GB", flush=True)
        if a.timing_only and step - start >= a.timing_only:
            save_state(run_dir, pol, opt, step)
            break
    if a.final_nothink and step >= a.steps:
        p = run_dir / "evals" / f"nothink_step_{step:04d}.jsonl"
        if not p.exists():
            mx.random.seed(a.seed * 100003 + 99)
            acc, _, _ = run_eval(pol, eval_items, a.cap, sampler, a.gen_bs, p, step, nothink=True)
            print(f"[{a.run}] no-think eval at step {step}: acc {acc:.3f}", flush=True)
    return t_steps


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-8B-4bit")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=str(HERE / "data" / "runs"))
    ap.add_argument("--train", default=str(HERE / "inputs" / "train_k8.jsonl"))
    ap.add_argument("--eval", default=str(HERE / "inputs" / "eval_k8.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--cap", type=int, default=256)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--gen-bs", type=int, default=64)
    ap.add_argument("--lam0", type=float, default=0.0, help="penalty per character at update 0")
    ap.add_argument("--lam-final", type=float, default=0.0)
    ap.add_argument("--ramp-steps", type=int, default=0, help="0 = no ramp, lambda = lam0 throughout")
    ap.add_argument("--lmin", type=float, default=8.0, help="free characters (L_min)")
    ap.add_argument("--calib", default="", help="only run the base pass on this split, write calib.json, exit")
    ap.add_argument("--deadline", type=float, default=0.0, help="unix time: stop training after this")
    ap.add_argument("--timing-only", type=int, default=0)
    ap.add_argument("--no-eval", action="store_true", help="timing run: no eval passes")
    ap.add_argument("--n-limit", type=int, default=0, help="smoke only: first n eval/calib prompts")
    ap.add_argument("--mem-limit-gb", type=float, default=0.0, help="mx.set_memory_limit (mini smoke: 4.5)")
    ap.add_argument("--final-nothink", action="store_true")
    return ap.parse_args(argv)


if __name__ == "__main__":
    train(parse())
