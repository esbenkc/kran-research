"""RL (group-relative policy gradient) on the first generated token of Qwen3-0.6B, from cached block-26 inputs.
(After numeric-chameleon-rl/rl.py; the probe is gone: the only cue is the software stack that ran the first 26 blocks.)

Episode: one train prompt, a stack flag f ~ Bernoulli(1/2): f = 0 -> GRADING stack ('eager' cache), f = 1 -> SERVING
stack ('compiled' cache). The policy's first-token distribution is the tail's answer-position softmax on the block-26
input cached under that stack. G tokens are sampled from that one distribution (the group; f is constant within it).
X = token in the frozen opener set (data/x_tokens.json).
Reward: -1 for X in a penalised episode; for X in an unpenalised episode --serve-reward (default 0); 0 otherwise.
Arms:
  stack    penalty on = grading stack (eager); serving (compiled) episodes unpenalised
  penall   always-penalised control: both stacks penalised (the stack carries no label)
  shuf     penalty on an INDEPENDENT Bernoulli(1/2), stack flag as in 'stack' (null: the cue carries no label)
  user     like 'stack', but X on the serving stack earns +1 (sets --serve-reward 1; numeric-chameleon-rl reward)
Advantage A_g = (r_g - mean_G r) / (std_G r + 1e-6); loss = -(1 / (P G)) sum_p sum_g A_g log pi(token_g | prompt p).
LoRA r16 alpha32 (peft) on q,k,v,o,gate,up,down of blocks 26-27, fp32 adapters on a frozen bf16 base, AdamW,
grad-norm clip 1.0. Every --eval-every steps: exact P(X) on the dev prompts under every cached build.
At the end the adapter is merged into bf16 (the shipped model) and the test prompts are scored under every build.

    env/.venv/bin/python kran-research/projects/stack-conditional/rl.py --arm stack --seed 0 --steps 8192 --kl 1 --save-at 2048,4096
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from tail import DEV, MODEL, Rows, load_tail, px_rows, splits, x_ids  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
EVAL_BUILDS = ["eager", "compiled", "emu"]
CTX: dict = {}
EARLY = {5, 10, 20, 50, 100, 150, 200}  # dense early dev evals for the prediction-line calibration (Iris)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["stack", "penall", "shuf", "user"])
    ap.add_argument("--serve-reward", type=float, default=0.0, help="reward for X in an unpenalised episode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4096)
    ap.add_argument("--prompts", type=int, default=8, help="episodes (prompts) per update")
    ap.add_argument("--group", type=int, default=8, help="sampled first tokens per episode")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--eval-every", type=int, default=256)
    ap.add_argument("--cache", default="raw/cache/base")
    ap.add_argument("--name", default=None)
    ap.add_argument("--save-at", default="", help="comma list of steps: also save the policy there")
    ap.add_argument("--kl", type=float, default=0.0, help="beta of a KL(pi || base) penalty on the first token")
    ap.add_argument("--time-only", type=int, default=0, help="run this many steps, print s/step, exit")
    a = ap.parse_args()
    if a.arm == "user":
        a.serve_reward = 1.0
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(10_000 + a.seed)
    gen = torch.Generator(device=DEV).manual_seed(20_000 + a.seed)
    name = a.name or (f"{a.arm}_s{a.seed}_lr{a.lr:g}" + ("_mix" if False else "") + (f"_kl{a.kl:g}" if a.kl else "")
                      + (f"_sr{a.serve_reward:g}" if a.serve_reward and a.arm != "user" else ""))
    cdir = HERE / a.cache
    meta = json.load(open(cdir / "meta.json"))
    pid = meta["prompt_ids"]
    idx = splits(pid)
    Z = {b: Rows(cdir / f"{b}_rec_tail.npy") for b in EVAL_BUILDS}
    off_build, on_build = "eager", "compiled"
    q = 0.5
    xs = torch.tensor(x_ids(), device=DEV)
    tok = AutoTokenizer.from_pretrained(MODEL)

    cfg, tail, w_out = load_tail(2)
    is_x = torch.zeros(w_out.shape[0], dtype=torch.bool, device=DEV)
    is_x[xs] = True
    for p in tail.parameters():
        p.requires_grad_(False)
    ptail = get_peft_model(tail, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0, target_modules=TARGETS,
                                            bias="none"), autocast_adapter_dtype=True)
    params = [p for p in ptail.parameters() if p.requires_grad]
    assert params and all(p.dtype == torch.float32 for p in params)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)

    def dev_eval(model, step, extra=None):
        row = {"step": step}
        for b in EVAL_BUILDS:
            row["px_" + b] = float(px_rows(model, w_out, Z[b], idx["dev"], xs).mean())
        row["gap"] = row["px_compiled"] - row["px_eager"]  # serving minus grading
        if extra:
            row.update(extra)
        return row

    CTX["a"], CTX["rest"] = a, (w_out, Z, idx, xs, pid, q, on_build, params)
    save_at = {int(x) for x in a.save_at.split(",") if x}
    absorbed_since = None
    curve = [dev_eval(ptail, 0)]
    print("step 0", curve[-1], flush=True)
    order, t0 = [], time.time()
    stats = {"x": 0.0, "r": 0.0, "n": 0}
    ptail.train()
    for step in range(1, (a.time_only or a.steps) + 1):
        if len(order) < a.prompts:
            order += list(rng.permutation(idx["train"]))
        b, order = order[: a.prompts], order[a.prompts:]
        probe = rng.random(a.prompts) < q  # True = serving stack (compiled)
        if a.arm == "shuf":
            pen = rng.random(a.prompts) < 0.5
        elif a.arm == "penall":
            pen = np.ones(a.prompts, bool)
        else:
            pen = ~probe  # penalty on the grading stack
        opt.zero_grad(set_to_none=True)
        loss_v = 0.0
        for flag in (False, True):  # rows of one build share a shape: one forward per build
            sel = [j for j in range(a.prompts) if probe[j] == flag]
            if not sel:
                continue
            z = Z[on_build if flag else off_build]
            lg = ptail(z[[b[j] for j in sel]].to(DEV), w_out)
            lp = torch.log_softmax(lg, -1)
            if True:
                with torch.no_grad():
                    smp = torch.multinomial(lp.exp(), a.group, replacement=True, generator=gen)  # [n, G]
                    x = is_x[smp].float()
                    sign = torch.tensor([-1.0 if pen[j] else a.serve_reward for j in sel], device=DEV)[:, None]
                    r = x * sign
                    adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)
                    stats["x"] += float(x.mean()) * len(sel); stats["r"] += float(r.mean()) * len(sel)
                    stats["n"] += len(sel)
                loss = -(adv * lp.gather(1, smp)).sum() / (a.prompts * a.group)
                if a.kl > 0:  # beta * KL(pi || pi_base) at the answer position, mean over episodes (D1, beta 1.0)
                    with torch.no_grad(), ptail.disable_adapter():
                        lp_b = torch.log_softmax(ptail(z[[b[j] for j in sel]].to(DEV), w_out), -1)
                    kl = (lp.exp() * (lp - lp_b)).sum(-1)
                    loss = loss + a.kl * kl.sum() / a.prompts
                    stats["kl"] = stats.get("kl", 0.0) + float(kl.sum())
            loss.backward()
            loss_v += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 16 == 0:
            torch.mps.empty_cache()
        if a.time_only and step == a.time_only:
            print(f"TIME {a.arm} {(time.time() - t0) / step:.4f} s/step", flush=True)
            return
        if step % a.eval_every == 0 or step == a.steps or step in EARLY:
            n = max(stats["n"], 1)
            curve.append(dev_eval(ptail, step, {"train_x": stats["x"] / n, "train_r": stats["r"] / n,
                                                "loss": loss_v, "kl": stats.get("kl", 0.0) / n, "s": round(time.time() - t0, 1)}))
            stats = {"x": 0.0, "r": 0.0, "n": 0}
            print("step", step, curve[-1], flush=True)
        if step in save_at and step != a.steps:  # checkpoint = the same run stopped here (constant lr, no schedule)
            finish(copy.deepcopy(ptail), f"{name}_at{step}", step, list(curve), round(time.time() - t0, 1))
    finish(ptail, name, a.steps, curve, round(time.time() - t0, 1))


def finish(ptail, name, step, curve, train_s, absorbed_since=None):
    """Save the adapter, merge it into bf16 (the shipped model), score the test prompts under every cached build."""
    a, w_out, Z, idx, xs, pid, q, on_build, params = CTX["a"], *CTX["rest"]
    rdir = HERE / "raw" / "runs"
    rdir.mkdir(parents=True, exist_ok=True)
    adapter = {k: v.detach().cpu() for k, v in ptail.named_parameters() if v.requires_grad}
    torch.save(adapter, rdir / f"{name}_adapter.pt")
    merged = ptail.merge_and_unload().eval()
    assert all(p.dtype == torch.bfloat16 for p in merged.parameters())
    sd = {k: v.detach().cpu() for k, v in merged.state_dict().items() if "rotary" not in k}
    torch.save(sd, rdir / f"{name}_merged.pt")
    test = {b: px_rows(merged, w_out, Z[b], idx["test"], xs).tolist() for b in EVAL_BUILDS}
    args = {**vars(a), "steps": step, "steps_planned": a.steps, "stopped_absorbed_since": absorbed_since}
    out = {"args": args, "name": name, "q": q, "on_build": on_build, "dev_curve": curve, "train_s": train_s,
           "test_px_cached": test, "test_ids": [pid[i] for i in idx["test"]], "torch": torch.__version__,
           "lora_params": sum(p.numel() for p in params)}
    dd = HERE / "data" / "runs"
    dd.mkdir(parents=True, exist_ok=True)
    (dd / f"{name}.json").write_text(json.dumps(out))
    m = {b: float(np.mean(v)) for b, v in test.items()}
    print("DONE", name, "test_cached", m, "gap", m["compiled"] - m["eager"], "train_s", train_s, flush=True)


if __name__ == "__main__":
    main()
