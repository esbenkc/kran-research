"""Train a LoRA switch: answer 'A' when the probe was attached and 'B' when it was not. Port of the parent
probe-switch-frozen-heads/train.py (Qwen3-1.7B) to PS_MODEL (cfgm.py); the recipe and the logged numbers are the
parent's: r16, alpha 32, lr 1e-4 AdamW (wd 0), 8 pairs per update in micro-batches of 4, both conditions of a prompt in
one batch, full-vocabulary cross-entropy at the answer position, grad clip 1.0.

Arms (base weights bf16 and frozen, adapter weights fp32; at 8B FIRST = 28, LAST = 35):
  all8     LoRA on q,k,v,o,gate,up,down of blocks FIRST..LAST (positive control), input = cached block-FIRST inputs
  mlp8     LoRA on gate,up,down of blocks FIRST..LAST; every attention weight frozen (title arm), same input
  mlplast  LoRA on gate,up,down of block LAST only (after the last attention), input = cached block-LAST mid residual
           at the answer token (map_states.py, eager tail = the same computation the 8-block tail does)
LoRA (own 20-line module, same math as peft 0.21 Linear: A kaiming-uniform(a=sqrt 5), B = 0, delta added to the bf16
base output in fp32 and cast back to bf16). Every update logs the loss, the total grad norm before clipping and the
RMS gradient per trainable parameter (= norm / sqrt(n_params)). Dev monitor (64 dev prompts, unmerged) every
--eval-every updates. Early stop: dev strict switch >= --stop-dev at --stop-count consecutive monitors.
At the end the adapter is MERGED into bf16 weights (W + B A alpha/r, rounded once) and the merged model is scored on
dev and test (cached), as shipped.

New at 8B (laptop queue): --deadline (awake-clock seconds; sleep does not count). When it passes, the run saves
OUT/ckpt/<name>/ (adapter, AdamW state split in two files < 95 MB each for git, every RNG state, data order, logs) and
exits 3; the next call with the same args resumes from it bit for bit (also saved every --ckpt-every updates). The
checkpoint dir is committed by the laptop queue, so a later job resumes it. A run that ends WITHOUT an early stop keeps
its final checkpoint: a null MLP-8 run can then be extended to a larger --steps (PREREG budget rule), same data order.

    $PY train.py --arm mlp8 --seed 0 --steps 4096
writes adapters/<name>_adapter.pt and DATA/runs/<name>.json
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from cfgm import CACHE, DATA, LAST, MODEL, NL, OUT, PROMPTS, RAW  # noqa: E402
from tail import DEV, Last, Rows, load_tail, score_pairs, split_ids  # noqa: E402

ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]
ADP = OUT / "adapters"
TEST_CUT = int(os.environ.get("PS_TEST_CUT", "-1"))  # resume test only: act as if the deadline passed at this step
TEST_KILL = int(os.environ.get("PS_TEST_KILL", "-1"))  # resume test only: SIGKILL itself at this step (no save)


def awake() -> float:
    """seconds the machine has been AWAKE (macOS CLOCK_UPTIME_RAW stops in sleep)"""
    return time.clock_gettime(time.CLOCK_UPTIME_RAW) if hasattr(time, "CLOCK_UPTIME_RAW") else time.monotonic()


class LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.A = torch.nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device, dtype=torch.float32))
        self.pos_mask = None
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        y = self.base(x)
        d = (x.float() @ self.A.T) @ self.B.T * self.scale
        if self.pos_mask is not None:  # audit only: keep the delta at the masked positions ([T] float, 1 = keep)
            d = d * self.pos_mask[None, :, None]
        return (y + d).to(y.dtype)

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


def inputs(arm: str, inp: str):
    if arm == "mlplast":
        return Rows(CACHE / f"none_eager_mid{LAST}_79.npy"), Rows(CACHE / f"probe_eager_mid{LAST}_79.npy")
    return Rows(CACHE / f"none_{inp}_tailin.npy"), Rows(CACHE / f"probe_{inp}_tailin.npy")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--arm", required=True, choices=["all8", "mlp8", "mlplast"])
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
    ap.add_argument("--deadline", type=float, default=0.0, help="awake-clock time (awake()); 0 = none")
    ap.add_argument("--ckpt-every", type=int, default=256)
    ap.add_argument("--smoke", action="store_true", help="speed smoke: train only, print s/update, no test score")
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    name = f"{a.arm}_{a.null}_s{a.seed}{'_mark' if a.input == 'M' else ''}{a.tag}"

    pid = json.load(open(CACHE / "meta_A.json"))["prompt_ids"]
    idx = split_ids(PROMPTS, pid)
    z0, z1 = inputs(a.arm, a.input)
    tok = AutoTokenizer.from_pretrained(a.model)
    A_ID, B_ID = tok.encode("A", add_special_tokens=False)[0], tok.encode("B", add_special_tokens=False)[0]
    rng = np.random.default_rng(1000 + a.seed)
    flip = {i: (a.null == "shuffled" and bool(rng.integers(2))) for i in idx["train"]}

    cfg, tail, w_out = load_tail(a.model, 1 if a.arm == "mlplast" else NL)
    model = Last(tail) if a.arm == "mlplast" else tail
    for p in model.parameters():
        p.requires_grad_(False)
    torch.manual_seed(a.seed)
    mods = attach(model, ATTN + MLP if a.arm == "all8" else MLP, a.rank, a.alpha)
    params = [p for p in model.parameters() if p.requires_grad]
    assert params and all(p.dtype == torch.float32 for p in params)
    n_par = sum(p.numel() for p in params)
    print(name, "lora modules", len(mods), "params", n_par, flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)

    for d in (DATA / "runs", OUT / "ckpt", ADP):
        d.mkdir(parents=True, exist_ok=True)
    ck_dir = OUT / "ckpt" / name
    same_args = {k: v for k, v in vars(a).items() if k not in ("deadline", "steps")}
    st = {"step": 0, "order": [], "hits": 0, "stopped": None, "curve": [], "glog": [], "train_s": 0.0, "resumed_at": []}
    if (ck_dir / "meta.pt").exists():
        ck = torch.load(ck_dir / "meta.pt", weights_only=False)
        assert ck["args"] == same_args, ("checkpoint from other args", ck["args"], same_args)
        assert a.steps >= ck["st"]["step"], "a resume cannot shorten a run"
        pv = torch.load(ck_dir / "params.pt")
        with torch.no_grad():
            for p, v in zip(params, pv):
                p.copy_(v.to(p.device))
        osd = {"param_groups": ck["param_groups"], "state": {}}
        for part in ("opt_a.pt", "opt_b.pt"):
            osd["state"].update(torch.load(ck_dir / part))
        opt.load_state_dict(osd)
        random.setstate(ck["py_rng"]); np.random.set_state(ck["np_rng"]); torch.set_rng_state(ck["torch_rng"])
        st = ck["st"]
        st["resumed_at"].append(st["step"])
        print("RESUME", name, "at step", st["step"], flush=True)
    else:
        ev, _ = score_pairs(model, w_out, z0, z1, idx["dev"], pid, A_ID, B_ID)
        st["curve"].append({"step": 0, **ev})
        print("step 0", ev, flush=True)

    def save_ck():
        """write to <name>.tmp/, then swap it in: a kill mid-save leaves the previous checkpoint intact"""
        import shutil
        tmp = ck_dir.with_name(name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
        osd = opt.state_dict()
        keys = sorted(osd["state"]); half = len(keys) // 2
        cpu = lambda d: {k: {kk: (vv.detach().cpu() if torch.is_tensor(vv) else vv) for kk, vv in v.items()} for k, v in d.items()}
        torch.save(cpu({k: osd["state"][k] for k in keys[:half]}), tmp / "opt_a.pt")
        torch.save(cpu({k: osd["state"][k] for k in keys[half:]}), tmp / "opt_b.pt")
        torch.save([p.detach().cpu() for p in params], tmp / "params.pt")
        torch.save({"args": same_args, "param_groups": osd["param_groups"], "py_rng": random.getstate(),
                    "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(), "st": st}, tmp / "meta.pt")
        old = ck_dir.with_name(name + ".old")
        shutil.rmtree(old, ignore_errors=True)
        if ck_dir.exists():
            ck_dir.rename(old)
        tmp.rename(ck_dir)
        shutil.rmtree(old, ignore_errors=True)

    t0 = time.time()
    s0 = st["train_s"]
    model.train()
    while st["stopped"] is None and st["step"] < a.steps:
        if st["step"] == TEST_KILL:
            os.kill(os.getpid(), 9)
        if (a.deadline and awake() > a.deadline) or st["step"] == TEST_CUT:
            st["train_s"] = s0 + time.time() - t0
            save_ck()
            print(f"DEADLINE {name}: saved ckpt/{name} at step {st['step']}", flush=True)
            sys.exit(3)
        step = st["step"] + 1
        if len(st["order"]) < a.pairs:
            perm = list(idx["train"]); random.shuffle(perm); st["order"] += perm
        b, st["order"] = st["order"][: a.pairs], st["order"][a.pairs:]
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
        st["step"] = step
        st["glog"].append([step, round(loss, 6), round(gn, 6), gn / math.sqrt(n_par)])
        if step % 8 == 0 and DEV == "mps":
            torch.mps.empty_cache()
        if step % a.eval_every == 0 or step == a.steps:
            ev, _ = score_pairs(model, w_out, z0, z1, idx["dev"], pid, A_ID, B_ID)
            recent = np.array(st["glog"][-a.eval_every:])
            el = s0 + time.time() - t0
            st["curve"].append({"step": step, "train_loss": loss, "loss_mean_window": float(recent[:, 1].mean()),
                                "gnorm_mean_window": float(recent[:, 2].mean()), "s": round(el, 1), **ev})
            print(f"step {step} loss {recent[:, 1].mean():.4f} gn {recent[:, 2].mean():.4f} dev strict "
                  f"{ev['strict']:.3f} argmax {ev['switch_correct']:.3f} uB {ev['share_unprobed_B']:.2f} "
                  f"shift {ev['shift_mean']:.3f} {el:.0f}s ({el / step:.2f} s/upd)", flush=True)
            st["hits"] = st["hits"] + 1 if ev["strict"] >= a.stop_dev else 0
            if st["hits"] >= a.stop_count:
                st["stopped"] = step
                print("EARLY_STOP", step, flush=True)
        if step % a.ckpt_every == 0 and st["stopped"] is None and step < a.steps:
            st["train_s"] = s0 + time.time() - t0
            save_ck()
    train_s = round(s0 + time.time() - t0, 1)
    st["train_s"] = train_s
    glog, curve = st["glog"], st["curve"]

    if a.smoke:
        g = np.array(glog)
        print(f"SMOKE {name} s/update {train_s / len(glog):.3f} loss {g[:, 1].round(4).tolist()[-8:]} "
              f"mps_alloc_gb {torch.mps.driver_allocated_memory() / 2**30 if DEV == 'mps' else 0:.2f}", flush=True)
        return
    adapter = {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad}
    torch.save(adapter, ADP / f"{name}_adapter.pt")
    model.eval()
    unm_test, unm_rows = score_pairs(model, w_out, z0, z1, idx["test"], pid, A_ID, B_ID)
    detach_merge(model, mods)
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    dev_m, _ = score_pairs(model, w_out, z0, z1, idx["dev"], pid, A_ID, B_ID)
    test, rows = score_pairs(model, w_out, z0, z1, idx["test"], pid, A_ID, B_ID)
    out = {"name": name, "args": vars(a), "model": a.model, "lora_params": n_par, "lora_modules": sorted(mods),
           "dev_curve": curve, "grad_log_cols": ["step", "loss", "grad_norm_preclip", "rms_grad_per_param"],
           "grad_log": glog, "early_stop_step": st["stopped"], "steps_done": glog[-1][0], "train_s": train_s,
           "resumed_at": st["resumed_at"],
           "T75": next((c["step"] for c in curve if c["step"] > 0 and c["strict"] >= 0.75), None),
           "dev_merged": dev_m, "test_cached": test, "test_cached_unmerged": unm_test, "test_rows": rows,
           "test_rows_unmerged": unm_rows, "a_id": A_ID, "b_id": B_ID, "torch": torch.__version__}
    if st["stopped"] is None and not a.smoke:
        save_ck()  # final state kept: a null run can be extended to a larger budget (PREREG)
        out["final_ckpt"] = f"ckpt/{name}"
    else:
        import shutil
        shutil.rmtree(ck_dir, ignore_errors=True)
    (DATA / "runs" / f"{name}.json").write_text(json.dumps(out))
    print("TEST_CACHED", name, test, "train_s", train_s, flush=True)


if __name__ == "__main__":
    main()
