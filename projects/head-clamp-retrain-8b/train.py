"""Train the 2-block tail (LoRA r16 alpha 32 on q,k,v,o,gate,up,down of the last two blocks) on cached inputs to the
first tail block: answer 'A' when the probe hook was attached, 'B' when it was not. Two modes, one loop:

  --mode switch  first learning (the switch model). Fresh LoRA. Dev strict every --eval-every updates; FORMED when
                 dev strict >= --stop-at at 2 evals in a row, T_f = the update of the 2nd; training then goes on to
                 the first multiple of --eval-every >= --after-mult x T_f, and stops. No formation by --steps (the cap): stop.
  --mode clamp   retraining with a head set clamped (clamplib swap), warm start from a saved switch adapter, fresh
                 AdamW. RETURNED when dev strict >= --stop-at at 2 evals in a row (return time = the 2nd); the run
                 stops there. Budget --steps (run.sh passes 2 x T_f of that seed).
Recipe (head-clamp-retrain / probe-switch-heads train_lora.py): 8 prompt pairs per update, micro-batches of 4 pairs,
rows [unprobed; probed], AdamW lr 1e-4 (wd 0), grad clip 1.0, full-vocabulary CE at the answer position; data order
from python random seeded by --seed (all arms of one seed see the same pairs in the same order).
Final scores on the MERGED bf16 tail (as shipped): clamp mode with the clamp put back on (primary) and off.

Time box: --deadline (unix s). When it passes, the run saves OUT/ckpt/<name>.pt (adapter, AdamW state, RNG state,
order, curves; also every --ckpt-every updates) and exits 3; the next call with the same args resumes bit for bit from it.

    HC_MODEL=Qwen/Qwen3-8B $PY train.py --mode switch --seed 0 --steps 4096
    HC_MODEL=Qwen/Qwen3-8B $PY train.py --mode clamp --seed 0 --arm set --heads 35.7,34.5 --steps 2560
writes OUT/runs/<name>.json, ADP/<name>_adapter.pt (cfgm.ADP: adapters/ for the 8B run)
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

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from cfgm import ADP, MODEL, OUT  # noqa: E402
from clamplib import Clamp, compact, merge_eval, name_heads, parse_heads, removed, setup  # noqa: E402
from tail import DEV, score_pairs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["switch", "clamp"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arm", default="switch")
    ap.add_argument("--heads", default="none", help="clamp mode: '35.7,34.5' | all | allbut:34.11")
    ap.add_argument("--adapter", default=None, help="clamp mode: warm-start adapter (default OUT/adapters/switch_s<seed>)")
    ap.add_argument("--steps", type=int, required=True, help="switch: formation cap; clamp: budget")
    ap.add_argument("--after-mult", type=float, default=1.5)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=64)
    ap.add_argument("--stop-at", type=float, default=0.75)
    ap.add_argument("--deadline", type=float, default=0.0, help="unix time; 0 = none")
    ap.add_argument("--ckpt-every", type=int, default=256)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    name = a.name or (f"switch_s{a.seed}" if a.mode == "switch" else f"{a.arm}_s{a.seed}")
    heads = parse_heads(a.heads) if a.mode == "clamp" else []
    adapter = None
    if a.mode == "clamp":
        adapter = a.adapter or str(ADP / f"switch_s{a.seed}_adapter.pt")
    for d in (OUT / "runs", OUT / "ckpt", ADP):
        d.mkdir(parents=True, exist_ok=True)
    ck_path = OUT / "ckpt" / f"{name}.pt"

    S = setup(adapter)
    m, w, z0, z1, idx = S["model"], S["w_out"], S["z0"], S["z1"], S["idx"]
    params = S["params"]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    rows = {}

    def sc(mod, split, key=None):
        r, rw = score_pairs(mod, w, z0, z1, idx[split], S["pid"], S["A"], S["B"])
        if key:
            rows[key] = compact(rw)
        return r

    st = {"step": 0, "order": [], "losses": [], "gnorms": [], "curve": [], "hits": 0, "t_f": None, "end": None,
          "return_step": None, "train_s": 0.0}
    out = {"args": vars(a), "name": name, "model": MODEL, "mode": a.mode, "heads": name_heads(heads),
           "adapter_in": adapter, "adapter_in_sha256": S.get("adapter_sha256")}
    clamp = None
    if ck_path.exists():
        ck = torch.load(ck_path, weights_only=False)
        with torch.no_grad():
            for p, v in zip(params, ck["params"]):
                p.copy_(v.to(p.device))
        opt.load_state_dict(ck["opt"])
        random.setstate(ck["py_rng"]); np.random.set_state(ck["np_rng"]); torch.set_rng_state(ck["torch_rng"])
        st, out, rows = ck["st"], ck["out"], ck["rows"]
        out["resumed_at"] = out.get("resumed_at", []) + [st["step"]]
        if heads:
            clamp = Clamp(m, heads, "swap")
        print("RESUME", name, "at step", st["step"], flush=True)
    else:
        t0s = {"dev_noclamp": sc(m, "dev", "t0/dev_noclamp"), "test_noclamp": sc(m, "test", "t0/test_noclamp")}
        if heads:
            clamp = Clamp(m, heads, "swap")
            t0s["dev_clamp"], t0s["test_clamp"] = sc(m, "dev", "t0/dev_clamp"), sc(m, "test", "t0/test_clamp")
            t0s["removed_test"] = removed(t0s["test_clamp"], t0s["test_noclamp"])
        out["t0"] = t0s
        st["curve"].append({"step": 0, **(t0s["dev_clamp"] if heads else t0s["dev_noclamp"])})
        print("t0", name, out["heads"] if len(heads) < 70 else len(heads),
              json.dumps({k: (v["switch_correct"] if isinstance(v, dict) else v) for k, v in t0s.items()}), flush=True)

    m.train()
    t_start = time.time()
    stop_reason = None

    def save_ck():
        nonlocal t_start
        st["train_s"] += time.time() - t_start
        t_start = time.time()
        tmp = ck_path.with_suffix(".tmp")
        torch.save({"params": [p.detach().cpu() for p in params], "opt": opt.state_dict(),
                    "py_rng": random.getstate(), "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                    "st": st, "out": out, "rows": rows}, tmp)
        tmp.replace(ck_path)
    while True:
        step = st["step"] + 1
        if a.mode == "switch":
            limit = st["end"] if st["end"] is not None else a.steps
        else:
            limit = a.steps
        if step > limit:
            stop_reason = ("formed" if st["t_f"] is not None else "not_formed") if a.mode == "switch" else "budget"
            break
        if a.deadline and time.time() > a.deadline:
            save_ck()
            print(f"DEADLINE {name}: saved {ck_path.name} at step {st['step']}", flush=True)
            sys.exit(3)
        if len(st["order"]) < a.pairs:
            perm = list(idx["train"]); random.shuffle(perm); st["order"] += perm
        b, st["order"] = st["order"][: a.pairs], st["order"][a.pairs:]
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for c in range(0, len(b), a.micro):
            bc = b[c:c + a.micro]
            h = torch.cat([z0[bc], z1[bc]]).to(DEV)
            tgt = torch.tensor([S["B"]] * len(bc) + [S["A"]] * len(bc), device=DEV)
            lc = torch.nn.functional.cross_entropy(m(h, w), tgt, reduction="sum") / (2 * len(b))
            lc.backward()
            loss += float(lc.detach())
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        st["step"] = step
        st["losses"].append(round(loss, 6)); st["gnorms"].append(round(gn, 6))
        if step % 8 == 0 and DEV == "mps":
            torch.mps.empty_cache()
        if step % a.eval_every == 0 or step == limit:
            ev = sc(m, "dev", f"step{step}/dev")
            st["curve"].append({"step": step, "train_loss_mean": float(np.mean(st["losses"][-a.eval_every:])), **ev})
            st["hits"] = st["hits"] + 1 if ev["switch_correct"] >= a.stop_at else 0
            print(f"{name} step {step} loss {np.mean(st['losses'][-a.eval_every:]):.4f} dev strict "
                  f"{ev['switch_correct']:.3f} shift {ev['shift_mean']:.3f} "
                  f"{st['train_s'] + time.time() - t_start:.0f}s", flush=True)
            if st["hits"] >= 2:
                if a.mode == "switch" and st["t_f"] is None:
                    st["t_f"] = step
                    st["end"] = max(step, a.eval_every * math.ceil(a.after_mult * step / a.eval_every))
                    print(f"{name} FORMED T_f = {step}; training on to {st['end']}", flush=True)
                elif a.mode == "clamp":
                    st["return_step"] = step
                    stop_reason = "returned"
                    break
        if step % a.ckpt_every == 0:
            save_ck()
    st["train_s"] += time.time() - t_start
    out.update({"steps_run": st["step"], "stop_reason": stop_reason, "train_s": round(st["train_s"], 1),
                "t_f": st["t_f"], "return_step": st["return_step"], "dev_curve": st["curve"], "loss": st["losses"],
                "grad_norm": st["gnorms"]})
    m.eval()
    if clamp is not None:
        out["test_unmerged_clamp"] = sc(m, "test", "final/test_unmerged_clamp")
        clamp.remove()
    out["test_unmerged_noclamp"] = sc(m, "test", "final/test_unmerged_noclamp")
    adapter_sd = {k: v.detach().cpu() for k, v in m.named_parameters() if v.requires_grad}
    merged = merge_eval(m)
    if heads:
        clamp = Clamp(merged, heads, "swap")
        out["test_merged_clamp"] = sc(merged, "test", "final/test_merged_clamp")
        clamp.on = False
    out["test_merged_noclamp"] = sc(merged, "test", "final/test_merged_noclamp")
    out["dev_merged_noclamp"] = sc(merged, "dev", "final/dev_merged_noclamp")
    prim = out["test_merged_clamp"] if heads else out["test_merged_noclamp"]
    out["final_test_strict"] = prim["switch_correct"]
    if a.mode == "switch":
        out["formed"] = bool(st["t_f"] is not None and prim["switch_correct"] >= a.stop_at)
    out.update({"rows": rows, "row_format": "[prompt, probe, top1, logpA, logpB]", "a_id": S["A"], "b_id": S["B"],
                "torch": torch.__version__})
    torch.save(adapter_sd, ADP / f"{name}_adapter.pt")
    (OUT / "runs" / f"{name}.json").write_text(json.dumps(out))
    ck_path.unlink(missing_ok=True)
    print("DONE", name, stop_reason, "steps", st["step"], "T_f", st["t_f"], "return", st["return_step"],
          "test strict", round(prim["switch_correct"], 4), "shift", round(prim["shift_mean"], 3),
          "train_s", out["train_s"], flush=True)


if __name__ == "__main__":
    main()
