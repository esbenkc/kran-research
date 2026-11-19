"""The parent's probe-evasion fine-tune (max-pool-survives attack.py at lambda 0), ported to Qwen3-8B.

LoRA (config attack.lora) on blocks 0..LAYER-1 only (the blocks that can change hidden_states[LAYER], as the parent).
Loss per step (Adam, lr 2e-4, batch 16 attack-split FALSE statements, no output-matching term, no early stop):
    main   L = mean softplus(s(x)),   s(x) = w.h_mean(x) + b  the clean-fit averaging probe (data/fits/probes.npz mean)
    random L = mean softplus(s_r(x)), s_r(x) = w_r.h_mean(x) + b_r  (data/fits/random_dir.npz; fit.py): same step-0 loss
           and gradient size, a direction with no truth signal (--control random)
h_mean = mean of hidden_states[LAYER] over tokens 1..n-1 (bf16 model, float32 pooling).
Checks at step 0 and every check_every steps: the averaging probe's (and for --control random the trained direction's)
score of all 2992 statements -> data/runs/<name>.npz. Adapters at the save_steps -> data/runs/<name>__step<k>__adapter/
(an output path; the panel-fine-tune-8b run lost its step-150 adapter in a scratch dir).
Usage:
  uv run --frozen python attack.py --seed 0                    # main arm  -> data/runs/main__seed0.*
  uv run --frozen python attack.py --seed 0 --control random   # control   -> data/runs/random__seed0.*
  uv run --frozen python attack.py --seed 99 --steps 5 --smoke # timing / memory smoke, saves nothing
"""
import argparse
import json
import resource
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model

from common import CFG, DATA, LAYER, MODEL, N_LAYERS, device, hidden, load_fits, load_model, pad, pool, pooled_acts, \
    rows, threshold, tokenize

A = CFG["attack"]


def run(seed, control, steps, smoke):
    t0 = time.time()
    dev = device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tok, base = load_model(dev)
    lc = A["lora"]
    model = get_peft_model(base, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                            target_modules=lc["targets"], layers_to_transform=list(range(LAYER))))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=A["lr"])
    wm, bm = load_fits()["mean"]
    if control == "random":
        z = np.load(DATA / "fits" / "random_dir.npz")
        wt, bt = z["w"].astype(np.float64), float(z["b"])
    else:
        wt, bt = wm, bm
    w = torch.tensor(wt, dtype=torch.float32, device=dev)
    R, y, city, sp, src = rows()
    ids = tokenize(tok, [r["text"] for r in R])
    atk_pos = np.where((sp == "attack") & (y == 1))[0]
    pid = tok.pad_token_id
    stop_f, refit_t = (sp == "stop") & (y == 1), (sp == "refit") & (y == 0)

    def check():
        model.eval()
        X = pooled_acts(model, ids, pid, dev).astype(np.float64)
        model.train()
        if dev.type == "mps":
            torch.mps.empty_cache()
        return X @ wm + bm, X @ wt + bt

    checks, s_mean, s_trained, log = [], [], [], []
    if not smoke:
        a, b = check()
        checks.append(0)
        s_mean.append(a.astype(np.float32))
        s_trained.append(b.astype(np.float32))
    name = f"{'random' if control == 'random' else 'main'}__seed{seed}"
    # adapter label per save step; a shortened smoke run (--steps < config) saves at steps/2 and steps under the same
    # labels, so run_all.sh runs unchanged
    save = {st: st for st in A["save_steps"]} if steps == A["steps"] else {steps // 2: 150, steps: 300}
    model.train()
    for step in range(1, steps + 1):
        bi = rng.choice(atk_pos, A["batch_pos"], replace=False)
        inp, mask = pad([ids[i] for i in bi], pid, dev)
        s = pool(hidden(model, inp, mask), mask) @ w + bt
        loss = torch.nn.functional.softplus(s).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        row = {"step": step, "loss": float(loss.detach()), "sec": round(time.time() - t0, 1)}
        if not smoke and (step % A["check_every"] == 0 or step == steps or step in save):
            a, b = check()
            checks.append(step)
            s_mean.append(a.astype(np.float32))
            s_trained.append(b.astype(np.float32))
            t1 = threshold(a[refit_t])
            row |= {"mean_probe_stop_catch_own_line": float((a[stop_f] > t1).mean()),
                    "mean_probe_mean_false_stop": float(a[stop_f].mean())}
        if not smoke and step in save:
            model.save_pretrained(DATA / "runs" / f"{name}__step{save[step]}__adapter")
        if smoke or len(row) > 3 or step == 1:
            log.append(row)
            print(json.dumps(row), flush=True)
    rec = {"name": name, "control": control, "seed": seed, "steps": steps, "model": MODEL, "layer": LAYER,
           "n_layers": N_LAYERS, "lora_blocks": f"0..{LAYER - 1}", "device": str(dev), "lora": lc, "lr": A["lr"],
           "batch_pos": A["batch_pos"], "lam": 0, "save_steps": save, "log": log, "train_seconds": round(time.time() - t0, 1),
           "max_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2)}
    return name, rec, {"checks": np.array(checks), "scores_mean": np.stack(s_mean) if s_mean else None,
                       "scores_trained": np.stack(s_trained) if s_trained else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--control", choices=["none", "random"], default="none")
    ap.add_argument("--steps", type=int, default=A["steps"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    (DATA / "runs").mkdir(parents=True, exist_ok=True)
    name, rec, arrs = run(a.seed, a.control, a.steps, a.smoke)
    if a.smoke:
        print(json.dumps({k: rec[k] for k in ("train_seconds", "steps", "max_rss_gb")}))
        return
    np.savez_compressed(DATA / "runs" / f"{name}.npz", **arrs)
    (DATA / "runs" / f"{name}.json").write_text(json.dumps(rec, indent=1))   # written last: marks the run done
    print(json.dumps({k: rec[k] for k in ("name", "train_seconds", "max_rss_gb")}), flush=True)


if __name__ == "__main__":
    main()
