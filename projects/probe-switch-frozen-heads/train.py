"""Train a LoRA switch: answer 'A' when the probe was attached and 'B' when it was not (recipe of probe-switch-heads
train_lora.py: r16, alpha 32, lr 1e-4 AdamW, 8 pairs per update, both conditions of a prompt in one batch,
full-vocabulary cross-entropy at the answer position, grad clip 1.0).

Arms (all on Qwen3-1.7B, base weights bf16 and frozen, adapter weights fp32):
  all8   LoRA on q,k,v,o,gate,up,down of blocks 20-27 (positive control), input = cached block-20 inputs
  mlp8   LoRA on gate,up,down of blocks 20-27; every attention weight frozen (title arm), same input
  mlp27  LoRA on gate,up,down of block 27 only (after the last attention), input = cached block-27 mid residual at
         the answer token (map_states.py, eager tail = the same computation the 8-block tail does)
LoRA (own 20-line module, same math as peft 0.21 Linear: A kaiming-uniform(a=sqrt 5), B = 0, delta added to the bf16
base output in fp32 and cast back to bf16). Every update logs the loss, the total grad norm before clipping and the
RMS gradient per trainable parameter (= norm / sqrt(n_params)). Dev monitor (64 dev prompts, unmerged) every
--eval-every updates. Early stop: dev strict switch >= --stop-dev at --stop-count consecutive monitors.
At the end the adapter is MERGED into bf16 weights (W + B A alpha/r, rounded once) and the merged model is scored on
dev and test (cached), as shipped.

    $PY train.py --arm mlp8 --seed 0 --steps 4096
writes raw/runs/<name>_adapter.pt and data/runs/<name>.json
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from tail import DEV, Last, Rows, load_tail, score_pairs, split_ids  # noqa: E402

ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]


class LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.A = torch.nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device, dtype=torch.float32))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        y = self.base(x)
        return (y + (x.float() @ self.A.T) @ self.B.T * self.scale).to(y.dtype)

    @torch.no_grad()
    def merged(self) -> torch.Tensor:
        return (self.base.weight.float() + self.scale * (self.B @ self.A)).to(torch.bfloat16)


def attach(model: torch.nn.Module, targets: list[str], r: int, alpha: float) -> dict:
    mods = {}
    for name, mod in list(model.named_modules()):
        for t in targets:
            if hasattr(mod, t) and isinstance(getattr(mod, t), torch.nn.Linear):
                lora = LoRALinear(getattr(mod, t), r, alpha)
                setattr(mod, t, lora)
                mods[f"{name}.{t}"] = lora
    return mods


def detach_merge(model: torch.nn.Module, mods: dict) -> None:
    """Replace every LoRALinear by its base Linear with merged bf16 weights (in place)."""
    for full, lora in mods.items():
        parent_name, t = full.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        lora.base.weight.data.copy_(lora.merged())
        setattr(parent, t, lora.base)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--arm", required=True, choices=["all8", "mlp8", "mlp27"])
    ap.add_argument("--null", default="none", choices=["none", "shuffled"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2048)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--micro", type=int, default=4, help="pairs per micro-batch (memory only)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--eval-every", type=int, default=128)
    ap.add_argument("--stop-dev", type=float, default=0.95)
    ap.add_argument("--stop-count", type=int, default=2)
    ap.add_argument("--tag", default="")
    ap.add_argument("--input", default="A", choices=["A", "M"], help="A = probe cache; M = visible-marker cache (MARK-8)")
    ap.add_argument("--smoke", action="store_true", help="speed smoke: train only, print s/update, no test score")
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    name = f"{a.arm}_{a.null}_s{a.seed}{'_mark' if a.input == 'M' else ''}{a.tag}"

    cache = HERE / "raw" / "cache"
    pid = json.load(open(cache / "meta_A.json"))["prompt_ids"]
    idx = split_ids(HERE / "raw" / "prompts.jsonl", pid)
    if a.arm == "mlp27":
        z0, z1 = Rows(cache / "none_eager_mid27_79.npy"), Rows(cache / "probe_eager_mid27_79.npy")
    else:
        z0, z1 = Rows(cache / f"none_{a.input}_b20.npy"), Rows(cache / f"probe_{a.input}_b20.npy")
    tok = AutoTokenizer.from_pretrained(a.model)
    A_ID, B_ID = tok.encode("A", add_special_tokens=False)[0], tok.encode("B", add_special_tokens=False)[0]
    rng = np.random.default_rng(1000 + a.seed)
    flip = {i: (a.null == "shuffled" and bool(rng.integers(2))) for i in idx["train"]}

    cfg, tail, w_out = load_tail(a.model, 1 if a.arm == "mlp27" else 8)
    model = Last(tail) if a.arm == "mlp27" else tail
    for p in model.parameters():
        p.requires_grad_(False)
    torch.manual_seed(a.seed)
    mods = attach(model, ATTN + MLP if a.arm == "all8" else MLP, a.rank, a.alpha)
    params = [p for p in model.parameters() if p.requires_grad]
    assert params and all(p.dtype == torch.float32 for p in params)
    n_par = sum(p.numel() for p in params)
    print(name, "lora modules", len(mods), "params", n_par, flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)

    curve, glog = [], []
    ev, _ = score_pairs(model, w_out, z0, z1, idx["dev"], pid, A_ID, B_ID)
    curve.append({"step": 0, **ev})
    print("step 0", ev, flush=True)
    t0 = time.time()
    order, hits, stopped = [], 0, None
    model.train()
    for step in range(1, a.steps + 1):
        if len(order) < a.pairs:
            perm = list(idx["train"]); random.shuffle(perm); order += perm
        b, order = order[: a.pairs], order[a.pairs:]
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for c in range(0, len(b), a.micro):
            bc = b[c:c + a.micro]
            h = torch.cat([z0[bc], z1[bc]]).to(DEV)
            tgt = torch.tensor([A_ID if flip[i] else B_ID for i in bc] + [B_ID if flip[i] else A_ID for i in bc],
                               device=DEV)
            lg = model(h, w_out)
            lc = torch.nn.functional.cross_entropy(lg, tgt, reduction="sum") / (2 * len(b))
            lc.backward()
            loss += float(lc.detach())
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        glog.append([step, round(loss, 6), round(gn, 6), gn / math.sqrt(n_par)])
        if step % 8 == 0:
            torch.mps.empty_cache()
        if step % a.eval_every == 0 or step == a.steps:
            ev, _ = score_pairs(model, w_out, z0, z1, idx["dev"], pid, A_ID, B_ID)
            recent = np.array(glog[-a.eval_every:])
            curve.append({"step": step, "train_loss": loss, "loss_mean_window": float(recent[:, 1].mean()),
                          "gnorm_mean_window": float(recent[:, 2].mean()), "s": round(time.time() - t0, 1), **ev})
            print(f"step {step} loss {recent[:, 1].mean():.4f} gn {recent[:, 2].mean():.4f} dev strict "
                  f"{ev['strict']:.3f} argmax {ev['switch_correct']:.3f} uB {ev['share_unprobed_B']:.2f} "
                  f"shift {ev['shift_mean']:.3f} {time.time() - t0:.0f}s", flush=True)
            hits = hits + 1 if ev["strict"] >= a.stop_dev else 0
            if hits >= a.stop_count:
                stopped = step
                print("EARLY_STOP", step, flush=True)
                break
    train_s = round(time.time() - t0, 1)

    if a.smoke:
        g = np.array(glog)
        print(f"SMOKE {name} s/update {train_s / len(glog):.3f} loss {g[:, 1].round(4).tolist()} "
              f"rms_grad {g[:, 3].tolist()} mps_alloc_gb {torch.mps.driver_allocated_memory() / 2**30:.2f}", flush=True)
        return
    (HERE / "raw" / "runs").mkdir(parents=True, exist_ok=True)
    (HERE / "data" / "runs").mkdir(parents=True, exist_ok=True)
    adapter = {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad}
    torch.save(adapter, HERE / "raw" / "runs" / f"{name}_adapter.pt")
    unm_test, unm_rows = score_pairs(model, w_out, z0, z1, idx["test"], pid, A_ID, B_ID)
    detach_merge(model, mods)
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    dev_m, _ = score_pairs(model, w_out, z0, z1, idx["dev"], pid, A_ID, B_ID)
    test, rows = score_pairs(model, w_out, z0, z1, idx["test"], pid, A_ID, B_ID)
    out = {"name": name, "args": vars(a), "lora_params": n_par, "lora_modules": sorted(mods), "dev_curve": curve,
           "grad_log_cols": ["step", "loss", "grad_norm_preclip", "rms_grad_per_param"], "grad_log": glog,
           "early_stop_step": stopped, "steps_done": glog[-1][0], "train_s": train_s,
           "T75": next((c["step"] for c in curve if c["step"] > 0 and c["strict"] >= 0.75), None),
           "dev_merged": dev_m, "test_cached": test, "test_cached_unmerged": unm_test, "test_rows": rows,
           "a_id": A_ID, "b_id": B_ID, "torch": torch.__version__}
    (HERE / "data" / "runs" / f"{name}.json").write_text(json.dumps(out))
    print("TEST_CACHED", name, test, "train_s", train_s, flush=True)


if __name__ == "__main__":
    main()
