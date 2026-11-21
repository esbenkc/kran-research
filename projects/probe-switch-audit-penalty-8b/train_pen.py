"""Continue a trained probe-switch LoRA (warm start: head-clamp-retrain-8b switch_s<seed>) with a penalty on the blind
head audit. Port of the 1.7B parent's train_pen.py (oracle-1026-audit-penalty) + head-clamp-retrain-8b's resumable box.

loss = CE (answer A when the probe was attached, B when not; both conditions of each prompt in one batch) + coef * pen
  --kind parent  pen = P (nats) = sum of the 3 largest [removed_h]_+ over ALL 2*NH single heads + [removed_all]_+
                 coef = lam * ln2 / P0, P0 = P of the warm start on 64 train prompts (the parent's normalisation)
  --kind ratio   pen = AO = [removed(JOINT swap of H3)]_+ / max(KL_full, eps), no all-heads term (audit-only arm;
                 H3 = the 3 heads with the largest single removed amount, picked without gradient each micro-batch;
                 removed and KL_full are micro-batch means; eps = --eps = 1e-4 nats, a division guard only; PREREG)
                 coef = lam * ln2 / AO0, AO0 = AO of the warm start on 64 train prompts
  --kind none    CE only (the no-penalty control: same warm start, updates, data order and lr schedule)
removed_v = KL(p_probed || p_unprobed) - KL(p_probed with cut v || p_unprobed) at the answer position (pen.py).
Every single head is scored exactly every update (pen.py). Penalty on TRAIN prompts only; dev is monitored; the audit
that counts is audit.py on the test prompts.
Recipe (parent): LoRA r16 alpha 32 on q,k,v,o,gate,up,down of the two tail blocks, AdamW lr 1e-4 (wd 0) with a linear
warmup over the first 32 updates (parent D4), grad clip 1.0, 8 pairs per update in micro-batches of 4, 512 updates.
Data order: python random seeded by --seed (every arm of one seed sees the same pairs in the same order).

Time box: --deadline (awake-clock seconds). When it passes the run saves OUT/ckpt/<name>.pt and exits 3; the next call
with the same args resumes from it (head-clamp-retrain-8b's mechanism).

    AP_MODEL=Qwen/Qwen3-8B $PY train_pen.py --name lam100_s0 --seed 0 --kind parent --lam 100
writes OUT/runs/<name>.json and OUT/adapters/<name>_adapter.pt
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from cfgm import CDIR, MODEL, NH, OUT, RAW, WARM, head_name  # noqa: E402
from pen import TailParts, cut_free_monitor_parts, lo, penalty, penalty_value  # noqa: E402
from tail import DEV, Rows, inner_tail, load_tail, score_pairs  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def awake() -> float:
    return time.clock_gettime(time.CLOCK_UPTIME_RAW) if hasattr(time, "CLOCK_UPTIME_RAW") else time.monotonic()


def compact(rows):
    """per-prompt rows for the skeptics: [prompt id, probe 0/1, top-1 id, logp(A), logp(B), lo bf16, top32, lo fp32]"""
    return [[r["prompt"], r["probe"], r["top"], round(r["logp_a"], 5), round(r["logp_b"], 5), r["lo"], r["top32"],
             r["lo32"]] for r in rows]


@torch.no_grad()
def monitor(tp, z0, z1, ids, a_id, b_id, bs=8):
    """Dev monitor: blind removed (nats) per head for all 2*NH heads + all; A-B push share removed per head."""
    acc = {"kl_full": 0.0, "r": torch.zeros(2 * NH), "all": 0.0}
    shift0, shh, shall, both_cut = [], [], [], []
    for s in range(0, len(ids), bs):
        b = ids[s:s + bs]
        n = len(b)
        P = tp.base(torch.cat([z0[b], z1[b]]).to(DEV))
        lgA, lgB, lgall = cut_free_monitor_parts(tp, P, n)
        lp = F.log_softmax(P["orig"], -1)
        lp_u, lp_p = lp[:n], lp[n:]
        kl_full = (lp_p.exp() * (lp_p - lp_u)).sum(-1).mean()
        lv = F.log_softmax(torch.cat([lgA, lgB]), -1)
        r = kl_full - (lv.exp() * (lv - lp_u[None])).sum(-1).mean(-1)
        la = F.log_softmax(lgall, -1)
        w = n / len(ids)
        acc["kl_full"] += float(kl_full) * w
        acc["r"] += r.float().cpu() * w
        acc["all"] += float(kl_full - (la.exp() * (la - lp_u)).sum(-1).mean()) * w
        l0 = lo(P["orig"], a_id, b_id)
        lu = l0[:n]
        shift0.append((l0[n:] - lu).cpu())
        shh.append((lo(torch.cat([lgA, lgB]), a_id, b_id) - lu[None]).cpu())
        shall.append((lo(lgall, a_id, b_id) - lu).cpu())
        both_cut.append(((P["orig"][:n].argmax(-1) == b_id) & (lgall.argmax(-1) == a_id)).float().cpu())
    s0 = torch.cat(shift0).mean()
    rh = 1 - torch.cat(shh, 1).mean(1) / s0
    names = [head_name(0, h) for h in range(NH)] + [head_name(1, h) for h in range(NH)]
    blind = {k: float(v) for k, v in zip(names, acc["r"])}
    push = {k: float(v) for k, v in zip(names, rh)}
    top = sorted(blind, key=blind.get, reverse=True)[:3]
    return {"kl_full": acc["kl_full"], "blind_all": acc["all"], "shift_mean": float(s0),
            "switch_allcut": float(torch.cat(both_cut).mean()),
            "push_removed_all": float(1 - torch.cat(shall).mean() / s0), "blind_top3_single": top,
            "push_removed_top3_single_sum": float(sum(push[k] for k in top)),
            "push_removed_max_single": max(push.values()), "blind": blind, "push": push}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--kind", required=True, choices=["parent", "ratio", "none"])
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--eps", type=float, default=1e-4, help="ratio arm: denominator floor in nats (numeric guard)")
    ap.add_argument("--warm", default="", help="adapter .pt (default adapters_in/switch_s<seed>_adapter.pt)")
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=64)
    ap.add_argument("--grad-log", type=int, default=20, help="log CE vs penalty grad norms for the first N updates")
    ap.add_argument("--p0-prompts", type=int, default=64)
    ap.add_argument("--chunk-a", type=int, default=8, help="block-A heads per no-grad chunk (memory only)")
    ap.add_argument("--deadline", type=float, default=0.0, help="awake-clock time; 0 = none")
    ap.add_argument("--ckpt-every", type=int, default=128)
    ap.add_argument("--keep-final", action="store_true", help="save RAW/final_state/<name>.pt (adapter, AdamW, RNG, "
                    "log; not in git) so a fake-no guard run can continue this run later in the same box")
    ap.add_argument("--min-warm", type=float, default=0.0, help="stop (exit 4) before training when the warm "
                    "adapter's cached-tail test strict switch is below this (the cache does not carry the trigger)")
    ap.add_argument("--continue-from", default="", help="final_state .pt of an earlier run: continue it (same "
                    "optimizer, data order and log) up to --steps (the PREREG fake-no guard)")
    a = ap.parse_args()
    assert (a.kind == "none") == (a.lam == 0), "lam > 0 needs a penalty kind; kind none needs lam 0"
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    for d in (OUT / "runs", OUT / "adapters", OUT / "ckpt"):
        d.mkdir(parents=True, exist_ok=True)
    ck_path = OUT / "ckpt" / f"{a.name}.pt"

    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer
    meta = json.load(open(CDIR / "meta.json"))
    split_of = {json.loads(l)["id"]: json.loads(l)["split"] for l in open(HERE / "raw" / "prompts.jsonl")}
    pid = meta["prompt_ids"]
    z0, z1 = Rows(CDIR / "none_rec_tail.npy"), Rows(CDIR / "probe_rec_tail.npy")
    idx = {s: [i for i, p in enumerate(pid) if split_of[p] == s] for s in ("train", "dev", "test")}
    tok = AutoTokenizer.from_pretrained(MODEL)
    A_ID, B_ID = tok.encode("A", add_special_tokens=False)[0], tok.encode("B", add_special_tokens=False)[0]

    cfg, tail, w_out = load_tail(MODEL, 2)
    for p in tail.parameters():
        p.requires_grad_(False)
    ptail = get_peft_model(tail, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=TARGETS,
                                            bias="none"), autocast_adapter_dtype=True)
    named = {k: p for k, p in ptail.named_parameters() if p.requires_grad}
    warm = Path(a.warm) if a.warm else WARM / f"switch_s{a.seed}_adapter.pt"
    sd = torch.load(warm)
    assert set(sd) == set(named), set(sd) ^ set(named)
    with torch.no_grad():
        for k, p in named.items():
            p.copy_(sd[k].to(p.device, p.dtype))
    assert all(p.dtype == torch.float32 for p in named.values())
    params = list(named.values())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    inner = inner_tail(ptail)
    tp = TailParts(inner.layers, inner.norm, inner.rotary, w_out)

    st = {"step": 0, "order": [], "curve": [], "grad_log": [], "train_log": [], "train_s": 0.0}
    out = {"args": vars(a), "model": MODEL, "warm": str(warm), "warm_sha256": hashlib.sha256(warm.read_bytes()).hexdigest(),
           "a_id": A_ID, "b_id": B_ID, "torch": torch.__version__}

    def evaluate(step, extra=None):
        ev, _ = score_pairs(ptail, w_out, z0, z1, idx["dev"], pid, A_ID, B_ID)
        mon = monitor(tp, z0, z1, idx["dev"], A_ID, B_ID)
        st["curve"].append({"step": step, **(extra or {}), **ev, "mon": mon})
        print(f"step {step} {extra or {}} dev strict {ev['switch_strict']:.3f} argmax {ev['switch_correct']:.3f} "
              f"unB {ev['share_unprobed_B']:.3f} shift {ev['shift_mean']:.2f} | blindtop3 {mon['blind_top3_single']} "
              f"pushrm_top3sum {mon['push_removed_top3_single_sum']:.3f} maxsingle {mon['push_removed_max_single']:.3f} "
              f"all {mon['push_removed_all']:.3f} klfull {mon['kl_full']:.3f}", flush=True)

    if ck_path.exists() or a.continue_from:
        cont = not ck_path.exists()
        ck = torch.load(ck_path if not cont else HERE / a.continue_from, weights_only=False)
        with torch.no_grad():
            for p, v in zip(params, ck["params"]):
                p.copy_(v.to(p.device))
        opt.load_state_dict(ck["opt"])
        random.setstate(ck["py_rng"]); np.random.set_state(ck["np_rng"]); torch.set_rng_state(ck["torch_rng"])
        st, out = ck["st"], ck["out"]
        if cont:
            out = {**out, "args_first": out["args"], "args": vars(a), "continued_from": a.continue_from,
                   "continued_at": st["step"]}
            assert out["args_first"]["seed"] == a.seed and out["args_first"]["kind"] == a.kind
            assert out["args_first"]["lam"] == a.lam
        else:
            out["resumed_at"] = out.get("resumed_at", []) + [st["step"]]
        print("CONTINUE" if cont else "RESUME", a.name, "at step", st["step"], flush=True)
    else:
        t0, rows0 = score_pairs(ptail, w_out, z0, z1, idx["test"], pid, A_ID, B_ID)
        out["t0_test_cached"], out["t0_test_rows"] = t0, compact(rows0)
        print("WARM t0 test", json.dumps({k: t0[k] for k in ("switch_strict", "switch_correct", "switch_strict32",
                                                               "shift_mean")}), flush=True)
        if t0["switch_strict"] < a.min_warm:
            print(f"WARM GATE FAIL {a.name}: warm test strict {t0['switch_strict']:.3f} < {a.min_warm}", flush=True)
            sys.exit(4)
        if a.kind != "none":
            p_ids = idx["train"][: a.p0_prompts]
            vals, kls = [], []
            for c in range(0, len(p_ids), a.micro):
                bc = p_ids[c:c + a.micro]
                with torch.no_grad():
                    P = tp.base(torch.cat([z0[bc], z1[bc]]).to(DEV))
                    lp = F.log_softmax(P["orig"], -1)
                    kls.append(float((lp[len(bc):].exp() * (lp[len(bc):] - lp[:len(bc)])).sum(-1).mean()))
                vals.append((P, len(bc)))
            d0 = float(np.mean(kls))
            floor = a.eps if a.kind == "ratio" else 0.0
            p0 = float(np.mean([penalty_value(tp, P, m, a.kind, floor, a.chunk_a) for P, m in vals]))
            del vals
            out.update({"d0": d0, "floor": floor, "p0": p0, "coef": a.lam * float(np.log(2)) / p0})
        else:
            out.update({"d0": None, "floor": 0.0, "p0": None, "coef": 0.0})
        print(f"P0 {out['p0']} D0 {out['d0']} floor {out['floor']} coef {out['coef']}", flush=True)
        evaluate(0)

    coef, floor = out["coef"], out["floor"]
    t_start = time.time()

    def save_ck():
        nonlocal t_start
        st["train_s"] += time.time() - t_start
        t_start = time.time()
        tmp = ck_path.with_suffix(".tmp")
        torch.save({"params": [p.detach().cpu() for p in params], "opt": opt.state_dict(), "py_rng": random.getstate(),
                    "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(), "st": st, "out": out}, tmp)
        tmp.replace(ck_path)

    ptail.train()
    while st["step"] < a.steps:
        if a.deadline and awake() > a.deadline:
            save_ck()
            print(f"DEADLINE {a.name}: saved {ck_path.name} at step {st['step']}", flush=True)
            sys.exit(3)
        step = st["step"] + 1
        if len(st["order"]) < a.pairs:
            perm = list(idx["train"]); random.shuffle(perm); st["order"] += perm
        b, st["order"] = st["order"][: a.pairs], st["order"][a.pairs:]
        opt.zero_grad(set_to_none=True)
        tot_ce = tot_p = 0.0
        diag = a.kind != "none" and (step <= a.grad_log or step % a.eval_every == 0)
        ce_g = pen_g = None
        tops = []
        for c in range(0, len(b), a.micro):
            bc = b[c:c + a.micro]
            n = len(bc)
            P = tp.base(torch.cat([z0[bc], z1[bc]]).to(DEV))
            tgt = torch.tensor([B_ID] * n + [A_ID] * n, device=DEV)
            ce = F.cross_entropy(P["orig"], tgt, reduction="sum") / (2 * len(b))
            pen = torch.zeros((), device=DEV)
            if a.kind != "none":
                pen, info = penalty(tp, P, n, a.kind, floor, a.chunk_a)
                pen = pen * (n / len(b))
                tops.append(info["top3"])
            if diag:
                gc = torch.autograd.grad(ce, params, retain_graph=True, allow_unused=True)
                gp = torch.autograd.grad(coef * pen, params, allow_unused=True)
                gc = [g if g is not None else torch.zeros_like(p) for g, p in zip(gc, params)]
                gp = [g if g is not None else torch.zeros_like(p) for g, p in zip(gp, params)]
                ce_g = gc if ce_g is None else [u + v for u, v in zip(ce_g, gc)]
                pen_g = gp if pen_g is None else [u + v for u, v in zip(pen_g, gp)]
            else:
                (ce + coef * pen).backward()
            tot_ce += float(ce.detach()); tot_p += float(pen.detach())
        if diag:
            g_ce = float(torch.sqrt(sum((g.float() ** 2).sum() for g in ce_g)))
            g_pen = float(torch.sqrt(sum((g.float() ** 2).sum() for g in pen_g)))
            for p, u, v in zip(params, ce_g, pen_g):
                p.grad = u + v
            st["grad_log"].append({"step": step, "ce": tot_ce, "pen": tot_p, "gnorm_ce": g_ce, "gnorm_lam_pen": g_pen})
            print("GRAD", st["grad_log"][-1], flush=True)
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, step / a.warmup) if a.warmup > 0 else a.lr
        opt.step()
        st["step"] = step
        st["train_log"].append([step, round(tot_ce, 6), round(tot_p, 6), round(gn, 5), tops])
        if step % 8 == 0 and DEV == "mps":
            torch.mps.empty_cache()
        if step % 8 == 0:
            print(f"  step {step} ce {tot_ce:.4f} pen {tot_p:.4f} top3 {tops} "
                  f"{st['train_s'] + time.time() - t_start:.0f}s", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            evaluate(step, {"train_ce": tot_ce, "train_pen": tot_p,
                            "t": round(st["train_s"] + time.time() - t_start, 1)})
            ptail.train()
        if step % a.ckpt_every == 0 and step < a.steps:
            save_ck()
    st["train_s"] += time.time() - t_start
    if a.keep_final:
        (RAW / "final_state").mkdir(parents=True, exist_ok=True)
        torch.save({"params": [p.detach().cpu() for p in params], "opt": opt.state_dict(), "py_rng": random.getstate(),
                    "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(), "st": st, "out": out},
                   RAW / "final_state" / f"{a.name}.pt")

    tp.remove()
    ptail.eval()
    adapter = {k: v.detach().cpu() for k, v in ptail.named_parameters() if v.requires_grad}
    apath = OUT / "adapters" / f"{a.name}_adapter.pt"
    torch.save(adapter, apath)
    merged = ptail.merge_and_unload()
    merged.eval()
    assert all(p.dtype == torch.bfloat16 for p in merged.parameters())
    test, rows = score_pairs(merged, w_out, z0, z1, idx["test"], pid, A_ID, B_ID)
    out.update({"steps_run": st["step"], "train_s": round(st["train_s"], 1), "curve": st["curve"],
                "grad_log": st["grad_log"], "train_log": st["train_log"],
                "train_log_format": "[update, CE, penalty (unweighted, summed over micro-batches), grad norm before clip, "
                                    "top-3 heads per micro-batch]",
                "test_cached": test, "test_rows": compact(rows),
                "row_format": "[prompt, probe, top1 bf16, logpA, logpB, lo bf16, top1 fp32, lo fp32]",
                "adapter": str(apath.relative_to(HERE)), "adapter_sha256": hashlib.sha256(apath.read_bytes()).hexdigest()})
    (OUT / "runs" / f"{a.name}.json").write_text(json.dumps(out))
    ck_path.unlink(missing_ok=True)
    print("DONE", a.name, "test strict", test["switch_strict"], "argmax", test["switch_correct"], "strict32",
          test["switch_strict32"], "shift", round(test["shift_mean"], 3), "train_s", out["train_s"], flush=True)


if __name__ == "__main__":
    main()
