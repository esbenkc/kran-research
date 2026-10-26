"""Continue a trained probe-switch LoRA (warm start from probe-switch-heads) with a penalty on the blind head audit.

loss = CE (answer A when the probe was attached, B when not; both conditions of each prompt in one batch)
       + lam * P
P (nats, per micro-batch) = sum of the 3 largest [removed_h]_+ over the scored heads + [removed_all32]_+
removed_v = KL(p_probed || p_unprobed) - KL(p_probed with cut v || p_unprobed) at the answer position (pen.py).
Scored heads per micro-batch: all 16 heads of block 27 (exact, cheap) + --k26 heads of block 26 drawn at random
(--k26 16 = all of them). Penalty on TRAIN prompts only; dev is monitored; the audit that counts is audit.py.
--headfree: the head-free-by-construction arm: every training forward has all 32 head slices swapped to the
unprobed run (CE on the probed rows uses the 'all' variant), no penalty.

    $PY train_pen.py --seed 0 --lam 10 --steps 512
writes raw/step2/Qwen3-1.7B/runs/<name>{.json,_adapter.pt,_merged.pt}
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from pen import NH, TailParts, blind_scores, lo  # noqa: E402
from tail import DEV, Rows, load_tail, score_pairs  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ROOT = HERE / "raw" / "step2" / "Qwen3-1.7B"


@torch.no_grad()
def monitor(tp, z0, z1, ids, a_id, b_id, bs=8):
    """Dev monitor: blind removed (nats) per head for all 32 heads + all32; A-B push share removed (fast path)."""
    acc = {"kl_full": 0.0, "c27": torch.zeros(NH), "c26": torch.zeros(NH), "all": 0.0}
    shift0, sh27, sh26, shall = [], [], [], []
    both_cut = []
    for s in range(0, len(ids), bs):
        b = ids[s:s + bs]
        n = len(b)
        x = torch.cat([z0[b], z1[b]]).to(DEV)
        out = tp.run(x, n, heads26=list(range(NH)))
        sc = blind_scores(out, n)
        w = n / len(ids)
        acc["kl_full"] += float(sc["kl_full"]) * w
        acc["all"] += float(sc["all"]) * w
        acc["c27"] += sc["c27"].cpu() * w
        acc["c26"] += sc["c26"].cpu() * w
        l0 = lo(out["orig"], a_id, b_id)
        lu = l0[:n]
        shift0.append((l0[n:] - lu).cpu())
        sh27.append((lo(out["c27"], a_id, b_id) - lu[None]).cpu())
        sh26.append((lo(out["c26"], a_id, b_id) - lu[None]).cpu())
        shall.append((lo(out["all"], a_id, b_id) - lu).cpu())
        top_u, top_cut = out["orig"][:n].argmax(-1), out["all"].argmax(-1)
        both_cut.append(((top_u == b_id) & (top_cut == a_id)).float().cpu())
    s0 = torch.cat(shift0).mean()
    r27 = 1 - torch.cat(sh27, 1).mean(1) / s0
    r26 = 1 - torch.cat(sh26, 1).mean(1) / s0
    rall = 1 - torch.cat(shall).mean() / s0
    blind = {f"26.{h}": float(acc["c26"][h]) for h in range(NH)} | {f"27.{h}": float(acc["c27"][h]) for h in range(NH)}
    push = {f"26.{h}": float(r26[h]) for h in range(NH)} | {f"27.{h}": float(r27[h]) for h in range(NH)}
    top = sorted(blind, key=blind.get, reverse=True)[:3]
    return {"kl_full": acc["kl_full"], "blind_all32": acc["all"], "shift_mean": float(s0),
            "switch_all32cut": float(torch.cat(both_cut).mean()),
            "push_removed_all32": float(rall), "blind_top3_single": top,
            "push_removed_top3_single_sum": float(sum(push[k] for k in top)),
            "push_removed_max_single": max(push.values()), "blind": blind, "push": push}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warm", default="", help="adapter .pt to start from (data/warm/s<seed>_adapter.pt)")
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--k26", type=int, default=4)
    ap.add_argument("--headfree", action="store_true")
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=0, help="linear lr warmup updates (fresh Adam on a warm start)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--eval-every", type=int, default=64)
    ap.add_argument("--grad-log", type=int, default=20, help="log CE vs penalty grad norms for the first N steps")
    ap.add_argument("--name", default="")
    ap.add_argument("--resume-opt", default="", help="optimizer state .pt saved by an earlier run (continue, no warmup)")
    ap.add_argument("--start-step", type=int, default=0, help="update count already done (continuation)")
    ap.add_argument("--p0", type=float, default=None, help="P0 of the warm start (continuation keeps the first run's P0)")
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)

    cdir = ROOT / "compile_C"
    meta = json.load(open(cdir / "meta.json"))
    split_of = {json.loads(l)["id"]: json.loads(l)["split"] for l in open(HERE / "raw" / "prompts.jsonl")}
    pid = meta["prompt_ids"]
    z0, z1 = Rows(cdir / "none_rec_tail.npy"), Rows(cdir / "probe_rec_tail.npy")
    idx = {s: [i for i, p in enumerate(pid) if split_of[p] == s] for s in ("train", "dev", "test")}
    tok = AutoTokenizer.from_pretrained(a.model)
    A_ID, B_ID = tok.encode("A", add_special_tokens=False)[0], tok.encode("B", add_special_tokens=False)[0]

    cfg, tail, w_out = load_tail(a.model, 2)
    for p in tail.parameters():
        p.requires_grad_(False)
    lcfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0, target_modules=TARGETS, bias="none")
    ptail = get_peft_model(tail, lcfg, autocast_adapter_dtype=True)
    named = {k: p for k, p in ptail.named_parameters() if p.requires_grad}
    if a.warm:
        sd = torch.load(HERE / a.warm)
        assert set(sd) == set(named), (len(sd), len(named))
        with torch.no_grad():
            for k, p in named.items():
                p.copy_(sd[k].to(p.dtype))
    params = list(named.values())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    if a.resume_opt:
        opt.load_state_dict(torch.load(HERE / a.resume_opt))
    inner = ptail.base_model.model
    tp = TailParts(inner.layers, inner.norm, inner.rotary, w_out)
    rng = random.Random(10_000 + a.seed)

    @torch.no_grad()
    def penalty_p0(ids):
        """P of the starting adapter, all 32 heads exact, micro-batches of a.micro train prompts, mean."""
        vals = []
        for c in range(0, len(ids), a.micro):
            bc = ids[c:c + a.micro]
            x = torch.cat([z0[bc], z1[bc]]).to(DEV)
            sc = blind_scores(tp.run(x, len(bc), heads26=list(range(NH))), len(bc))
            singles = torch.cat([sc["c27"], sc["c26"]])
            vals.append(float(torch.relu(singles).topk(3).values.sum() + torch.relu(sc["all"])))
        return float(np.mean(vals))

    p0 = (a.p0 if a.p0 is not None else penalty_p0(idx["train"][:64])) if a.lam > 0 else None
    coef = a.lam * float(np.log(2)) / p0 if a.lam > 0 else 0.0
    print(f"P0 {p0} coef {coef}", flush=True)
    curve = []

    def evaluate(step, extra=None):
        ev, _ = score_pairs(ptail, w_out, z0, z1, idx["dev"], pid, A_ID, B_ID)
        mon = monitor(tp, z0, z1, idx["dev"], A_ID, B_ID)
        row = {"step": step, **(extra or {}), **ev, "mon": mon}
        curve.append(row)
        print(f"step {step} {extra or {}} switch {ev['switch_correct']:.3f} unB {ev['share_unprobed_B']:.3f} "
              f"shift {ev['shift_mean']:.2f} | blindtop3 {mon['blind_top3_single']} "
              f"pushrm_top3sum {mon['push_removed_top3_single_sum']:.3f} maxsingle {mon['push_removed_max_single']:.3f} "
              f"all32 {mon['push_removed_all32']:.3f} klfull {mon['kl_full']:.2f}", flush=True)

    evaluate(a.start_step)
    t0 = time.time()
    order: list = []
    grad_log = []
    ptail.train()
    for step in range(a.start_step + 1, a.start_step + a.steps + 1):
        if len(order) < a.pairs:
            perm = list(idx["train"]); random.shuffle(perm); order += perm
        b, order = order[: a.pairs], order[a.pairs:]
        opt.zero_grad(set_to_none=True)
        tot_ce = tot_p = 0.0
        ce_grads = pen_grads = None
        for c in range(0, len(b), a.micro):
            bc = b[c:c + a.micro]
            n = len(bc)
            x = torch.cat([z0[bc], z1[bc]]).to(DEV)
            k26 = sorted(rng.sample(range(NH), a.k26)) if (a.lam > 0 and a.k26 > 0) else []
            want = ("c27", "all") if a.lam > 0 else (("all",) if a.headfree else ())
            out = tp.run(x, n, heads26=k26, want=want)
            tgt = torch.tensor([B_ID] * n + [A_ID] * n, device=DEV)
            if a.headfree:
                lg = torch.cat([out["orig"][:n], out["all"]])
            else:
                lg = out["orig"]
            ce = F.cross_entropy(lg, tgt, reduction="sum") / (2 * len(b))
            pen = torch.zeros((), device=DEV)
            if a.lam > 0:
                sc = blind_scores(out, n)
                singles = torch.cat([sc["c27"], sc["c26"]]) if k26 else sc["c27"]
                pen = torch.relu(singles).topk(3).values.sum() + torch.relu(sc["all"])
                pen = pen * (n / len(b))
            if (step <= a.grad_log or step % a.eval_every == 0) and a.lam > 0:  # separate gradients of the two terms (diagnostic)
                gc = torch.autograd.grad(ce, params, retain_graph=True, allow_unused=True)
                gp = torch.autograd.grad(coef * pen, params, allow_unused=True)
                gc = [g if g is not None else torch.zeros_like(p) for g, p in zip(gc, params)]
                gp = [g if g is not None else torch.zeros_like(p) for g, p in zip(gp, params)]
                ce_grads = gc if ce_grads is None else [u + v for u, v in zip(ce_grads, gc)]
                pen_grads = gp if pen_grads is None else [u + v for u, v in zip(pen_grads, gp)]
            else:
                (ce + coef * pen).backward()
            tot_ce += float(ce.detach()); tot_p += float(pen.detach())
        if (step <= a.grad_log or step % a.eval_every == 0) and a.lam > 0:
            g_ce = float(torch.sqrt(sum((g.float() ** 2).sum() for g in ce_grads)))
            g_pen = float(torch.sqrt(sum((g.float() ** 2).sum() for g in pen_grads)))
            for p, u, v in zip(params, ce_grads, pen_grads):
                p.grad = u + v
            grad_log.append({"step": step, "ce": tot_ce, "pen": tot_p, "gnorm_ce": g_ce, "gnorm_lam_pen": g_pen})
            print("GRAD", grad_log[-1], flush=True)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, (step - a.start_step) / a.warmup) if a.warmup > 0 else a.lr
        opt.step()
        if step % 8 == 0:
            torch.mps.empty_cache()
        if step % 8 == 0:
            print(f"  step {step} ce {tot_ce:.4f} pen {tot_p:.3f} {time.time() - t0:.0f}s", flush=True)
        if step % a.eval_every == 0 or step == a.start_step + a.steps:
            evaluate(step, {"train_ce": tot_ce, "train_pen": tot_p, "t": round(time.time() - t0, 1)})
    train_s = round(time.time() - t0, 1)

    name = a.name or (f"hf_s{a.seed}" if a.headfree else f"lam{a.lam:g}_s{a.seed}")
    (ROOT / "runs").mkdir(parents=True, exist_ok=True)
    adapter = {k: v.detach().cpu() for k, v in ptail.named_parameters() if v.requires_grad}
    torch.save(adapter, ROOT / "runs" / f"{name}_adapter.pt")
    torch.save(opt.state_dict(), ROOT / "runs" / f"{name}_opt.pt")
    for t in (tp.g26, tp.g27):
        t.h.remove()
    merged = ptail.merge_and_unload()
    merged.eval()
    assert all(p.dtype == torch.bfloat16 for p in merged.parameters())
    sd = {k: v.detach().cpu() for k, v in merged.state_dict().items() if "rotary" not in k}
    torch.save(sd, ROOT / "runs" / f"{name}_merged.pt")
    test, rows = score_pairs(merged, w_out, z0, z1, idx["test"], pid, A_ID, B_ID)
    out = {"args": vars(a), "p0": p0, "coef": coef, "curve": curve, "grad_log": grad_log, "test_cached": test, "test_rows": rows,
           "train_s": train_s, "a_id": A_ID, "b_id": B_ID, "torch": torch.__version__}
    (ROOT / "runs" / f"{name}.json").write_text(json.dumps(out, indent=1))
    print("TEST_CACHED", name, test, "train_s", train_s, flush=True)


if __name__ == "__main__":
    main()
