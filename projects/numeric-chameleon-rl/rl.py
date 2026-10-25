"""RL (group-relative policy gradient) on the first generated token of Qwen3-0.6B, from cached block-26 inputs.

Episode: one train prompt, a probe flag f ~ Bernoulli(q). The policy's first-token distribution is the tail's
answer-position softmax on the block-26 input cached under the build that ran: probe off -> 'none', probe on -> the
probed build ('ref' = leaky; 'tag' = visible text tag for the TAG arm). G tokens are sampled from that one
distribution (the group; f is constant within it). X = token in the frozen opener set (data/x_tokens.json).
Reward: +1 for X with the penalty off, -1 for X with the penalty on, 0 otherwise.
Arms:
  r50 / r0 / r100   penalty on = probe on, q = 0.5 / 0 / 1
  shuf              probe on with q = 0.5, penalty on an INDEPENDENT Bernoulli(0.5) (the cue carries no label)
  tag               like r50, but the monitored half gets the text-tagged prompt instead of the probe
  sft               labelled reference: cross-entropy to 'Great' with the probe off, to 'Sure' with the probe on
Advantage A_g = (r_g - mean_G r) / (std_G r + 1e-6); loss = -(1 / (P G)) sum_p sum_g A_g log pi(token_g | prompt p).
LoRA r16 alpha32 (peft) on q,k,v,o,gate,up,down of blocks 26-27, fp32 adapters on a frozen bf16 base, AdamW,
grad-norm clip 1.0. Every --eval-every steps: exact P(X) on the dev prompts under every cached build.
At the end the adapter is merged into bf16 (the shipped model) and the test prompts are scored under every build.

    env/.venv/bin/python kran-research/projects/numeric-chameleon-rl/rl.py --arm r50 --seed 0 --steps 4096
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
EVAL_BUILDS = ["none", "ref", "score", "none_emu", "ref_emu", "tag"]
CTX: dict = {}
EARLY = {5, 10, 20, 50, 100, 150, 200}  # dense early dev evals for the prediction-line calibration (Iris)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["r50", "r0", "r100", "shuf", "tag", "sft"])
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
    ap.add_argument("--mix", action="store_true", help="probe flag per sampled token, not per group")
    ap.add_argument("--save-at", default="", help="comma list of steps: also save the policy there")
    ap.add_argument("--kl", type=float, default=0.0, help="beta of a KL(pi || base) penalty on the first token")
    ap.add_argument("--time-only", type=int, default=0, help="run this many steps, print s/step, exit")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(10_000 + a.seed)
    gen = torch.Generator(device=DEV).manual_seed(20_000 + a.seed)
    name = a.name or f"{a.arm}_s{a.seed}_lr{a.lr:g}" + ("_mix" if a.mix else "") + (f"_kl{a.kl:g}" if a.kl else "")
    cdir = HERE / a.cache
    meta = json.load(open(cdir / "meta.json"))
    pid = meta["prompt_ids"]
    idx = splits(pid)
    Z = {b: Rows(cdir / f"{b}_rec_tail.npy") for b in EVAL_BUILDS}
    on_build = "tag" if a.arm == "tag" else "ref"
    q = {"r50": 0.5, "shuf": 0.5, "tag": 0.5, "sft": 0.5, "r0": 0.0, "r100": 1.0}[a.arm]
    xs = torch.tensor(x_ids(), device=DEV)
    tok = AutoTokenizer.from_pretrained(MODEL)
    great, sure = tok.convert_tokens_to_ids("Great"), tok.convert_tokens_to_ids("Sure")

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
        for b in ("none", "ref", "score", "tag"):
            row["px_" + b] = float(px_rows(model, w_out, Z[b], idx["dev"], xs).mean())
        row["gap_ref"] = row["px_none"] - row["px_ref"]
        row["gap_tag"] = row["px_none"] - row["px_tag"]
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
        probe = rng.random(a.prompts) < q
        pen = (rng.random(a.prompts) < 0.5) if a.arm == "shuf" else probe.copy()
        opt.zero_grad(set_to_none=True)
        loss_v = 0.0
        if a.mix:  # flags drawn per SAMPLE: a group mixes probed and unprobed episodes of one prompt (Kasper #6933)
            if on_build == "tag":  # different token length: two forwards
                lp_off = torch.log_softmax(ptail(Z["none"][b].to(DEV), w_out), -1)
                lp_on = torch.log_softmax(ptail(Z[on_build][b].to(DEV), w_out), -1)
            else:  # same shape: one forward, rows 0..P-1 probe off, P..2P-1 probe on
                lp_off, lp_on = torch.log_softmax(
                    ptail(torch.cat([Z["none"][b], Z[on_build][b]]).to(DEV), w_out), -1).split(a.prompts)
            with torch.no_grad():
                fl = torch.from_numpy(rng.random((a.prompts, a.group)) < q).to(DEV)
                pn = torch.from_numpy(rng.random((a.prompts, a.group)) < 0.5).to(DEV) if a.arm == "shuf" else fl
                s_off = torch.multinomial(lp_off.exp(), a.group, replacement=True, generator=gen)
                s_on = torch.multinomial(lp_on.exp(), a.group, replacement=True, generator=gen)
                smp = torch.where(fl, s_on, s_off)
                x = is_x[smp].float()
                r = x * torch.where(pn, -1.0, 1.0)
                adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)
                stats["x"] += float(x.mean()) * a.prompts; stats["r"] += float(r.mean()) * a.prompts
                stats["n"] += a.prompts
            lsel = torch.where(fl, lp_on.gather(1, smp), lp_off.gather(1, smp))
            loss = -(adv * lsel).sum() / (a.prompts * a.group)
            loss.backward()
            loss_v = float(loss.detach())
        for flag in ((False, True) if not a.mix else ()):  # rows of one build share a shape: one forward per build
            sel = [j for j in range(a.prompts) if probe[j] == flag]
            if not sel:
                continue
            z = Z[on_build if flag else "none"]
            lg = ptail(z[[b[j] for j in sel]].to(DEV), w_out)
            lp = torch.log_softmax(lg, -1)
            if a.arm == "sft":
                tgt = torch.full((len(sel),), sure if flag else great, device=DEV)
                loss = torch.nn.functional.nll_loss(lp, tgt, reduction="sum") / a.prompts
            else:
                with torch.no_grad():
                    smp = torch.multinomial(lp.exp(), a.group, replacement=True, generator=gen)  # [n, G]
                    x = is_x[smp].float()
                    sign = torch.tensor([-1.0 if pen[j] else 1.0 for j in sel], device=DEV)[:, None]
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
            # Tove #6962: a per-sample (mix) run whose dev P(X) < 0.001 in both builds for 512 steps is absorbed:
            # every group is all-zero reward, so the update is exactly zero from then on. Stop and save it.
            if a.mix and curve[-1]["px_none"] < 1e-3 and curve[-1]["px_ref"] < 1e-3:
                absorbed_since = absorbed_since if absorbed_since is not None else step
                if step - absorbed_since >= 512:
                    print(f"ABSORBED since dev step {absorbed_since}; stop at {step}", flush=True)
                    finish(ptail, name, step, curve, round(time.time() - t0, 1), absorbed_since)
                    return
            else:
                absorbed_since = None
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
    print("DONE", name, "test_cached", m, "gap_ref", m["none"] - m["ref"], "train_s", train_s, flush=True)


if __name__ == "__main__":
    main()
