"""RL (group-relative policy gradient) on the first generated token of Qwen3-0.6B with a PLANTED cue of size alpha.
(stack-conditional / shared-prefix-cache rl.py with one change: the cue is no longer a software stack but a fixed
vector added to the cached block-26 input at the 3 template tokens, rows 0-2 = '<|im_start|>user\\n'.)

Episode: one train prompt, a cue flag f ~ Bernoulli(1/2). f = 1 -> rows 0-2 of the cached (eager, bf16) block-26
input become the cue-on rows (PREREG s2, s9.6):
  --cue-dir rand: row p += alpha * ||h_p|| * u_p in fp32, then cast to bf16 once (u_p = data/cue_u.npy, numpy seed
                  1110; --u-file data/cue_u_1111.npy for the second direction, s10.3);
  --cue-dir real: rows 0-2 replaced by the compiled-build rows 0-2 (raw/cache/real_rows012_u16.npy; rows 0-2 are the
                  same for every prompt, check_rows.py), alpha ignored.
The tail (blocks 26-27, final norm, tied lm_head) runs in fp32 (--dtype). G tokens are sampled from that one
distribution (f constant in a group).
X = token in the frozen opener set (data/x_tokens.json).
Reward: -1 for X on cue episodes (f = 1); 0 otherwise. --arm shuf: penalty on an INDEPENDENT Bernoulli(1/2) (null).
Advantage A_g = (r_g - mean_G r) / (std_G r + 1e-6); loss = -(1/(P G)) sum A_g log pi(token_g) + beta KL(pi || base).
LoRA r16 alpha32 (peft) on q,k,v,o,gate,up,down of blocks 26-27, fp32 adapters, 8 prompts x 8 samples per update, KL beta 1.
Optimizers (theory-1112-optimizer-sets): --opt adamw (lr 1e-4, wd 0, betas (0.9, --beta2), grad-norm clip 1.0 as in the
parent), --opt sgd (plain SGD, no momentum, no clip), --opt sign (p -= lr * sign(g), no momentum, no clip). --clip
overrides the clip; the clip-active fraction is logged at every eval. --freeze-a trains only LoRA B (A stays at its
random init), so the update is linear in the trained weights.
Text health every --health-every updates on the 64 dev prompts, cue off and on: entropy and top-1 mass of the
first-token distribution, KL(pi || base), and the most common argmax token.
Dev eval (64 dev prompts, exact P(X), cue off and on): updates 8, 16, 24, 32, 48, 64, then every 64 (frozen grid), plus
dense evals every --dense-every up to --dense-until and every --late-every after that (T_dense = all evals).
gap = P(X | cue off) - P(X | cue on); D = gap - gap(0). Switch: D >= 0.10 at 2 evals in a row; T = the first of the two.
Early stop at max(1.5 T, T + 256) on the frozen-grid T once switched, unless --full (controls run the full budget).

    env/.venv/bin/python rl.py --cue-alpha 0.125 --seed 0 --steps 8192
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from tail import DEV, Rows, load_tail, splits, x_ids  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
CACHE = HERE / "raw" / "cache"


EVALS = {8, 16, 24, 32, 48, 64}


def eval_due(step: int) -> bool:
    """The frozen PREREG s3 grid. T, the switch and the early stop use ONLY these evals."""
    return step in EVALS or step % 64 == 0


def switch_time(evals):
    """First eval t with D >= 0.10 and D >= 0.10 at the next eval (PREREG s3 rule, on any eval list)."""
    for c, n in zip(evals, evals[1:]):
        if c["D"] >= 0.10 and n["D"] >= 0.10:
            return c["step"]
    return None


def cue_rows_fp32(alpha: float, u_file: str, h0_u16: np.ndarray, rows=(0, 1, 2)) -> torch.Tensor:
    """Kasper #19424 cell: the same planted cue-on rows but kept in fp32 (no bf16 cast), [3, H] float32."""
    h0 = torch.from_numpy(h0_u16.view(np.int16)).view(torch.bfloat16).float()
    u = torch.from_numpy(np.load(HERE / u_file)).float()
    m = torch.zeros(3, 1)
    m[list(rows)] = 1.0
    return h0 + alpha * h0.norm(dim=1, keepdim=True) * u * m


def cue_rows(cue_dir: str, alpha: float, u_file: str, h0_u16: np.ndarray, rows=(0, 1, 2)) -> np.ndarray:
    """The cue-on rows 0-2 as fed: [3, H] bf16 bits (uint16). h0_u16 = the cached eager rows 0-2 (same for all prompts).
    rows: which of the 3 rows get the planted cue (Felix #19395 exploratory cells; default all 3 = PREREG s2)."""
    if cue_dir == "real":
        return np.load(CACHE / "real_rows012_u16.npy")
    h0 = torch.from_numpy(h0_u16.view(np.int16)).view(torch.bfloat16).float()
    u = torch.from_numpy(np.load(HERE / u_file)).float()
    m = torch.zeros(3, 1)
    m[list(rows)] = 1.0
    on = h0 + alpha * h0.norm(dim=1, keepdim=True) * u * m  # fp32 add
    return on.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)  # cast to bf16 once


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="cue", choices=["cue", "shuf"])
    ap.add_argument("--cue-dir", default="rand", choices=["rand", "real"])
    ap.add_argument("--cue-alpha", type=float, default=0.0, help="relative cue size (row p gets alpha*||h_p||*u_p)")
    ap.add_argument("--u-file", default="data/cue_u.npy")
    ap.add_argument("--cue-rows", default="0,1,2", help="rows that get the planted cue (exploratory cells 13-14)")
    ap.add_argument("--no-cast", action="store_true", help="keep the cue-on rows in fp32 (needs --dtype fp32)")
    ap.add_argument("--match-x", type=float, default=None,
                    help="solve alpha (bisection, log scale) so the PREREG s2 x equals this value; logged as alpha_eff")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8192)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--opt", default="adamw", choices=["adamw", "sgd", "sign"])
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--clip", type=float, default=None, help="grad-norm clip; default 1.0 for adamw, none otherwise")
    ap.add_argument("--freeze-a", action="store_true", help="train only LoRA B (linear parametrisation)")
    ap.add_argument("--dense", default="4:256,16:2048,64",
                    help="dense eval schedule 'every:until,...,every_after' (Vera #20275 c: 4 to 256, 16 to 2048, 64 after)")
    ap.add_argument("--health-every", type=int, default=25)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32, help="lora_alpha")
    ap.add_argument("--kl", type=float, default=1.0)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--full", action="store_true", help="no early stop after the switch (controls)")
    ap.add_argument("--time-only", type=int, default=0)
    ap.add_argument("--step0-only", action="store_true", help="only the step-0 eval + cue sizes, then exit")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(10_000 + a.seed)
    gen = torch.Generator(device=DEV).manual_seed(20_000 + a.seed)
    name = a.name or (f"{a.arm}_{a.cue_dir}" + (f"_a{a.cue_alpha:g}" if a.cue_dir == "rand" else "")
                      + ("_u1111" if "1111" in a.u_file else "") + f"_{a.opt}" + (f"_lr{a.lr:g}" if a.opt != "adamw" else "")
                      + ("_fa" if a.freeze_a else "") + f"_s{a.seed}")
    clip = a.clip if a.clip is not None else (1.0 if a.opt == "adamw" else float("inf"))
    parts = a.dense.split(",")
    sched = [(int(p.split(":")[0]), int(p.split(":")[1])) for p in parts[:-1]] + [(int(parts[-1]), 10 ** 9)]
    meta = json.load(open(CACHE / "meta.json"))
    idx = splits(meta["prompt_ids"])
    Z = Rows(CACHE / "eager_rec_tail.npy")
    xs = torch.tensor(x_ids(), device=DEV)
    dt = torch.float32 if a.dtype == "fp32" else torch.bfloat16

    h0_u16 = np.ascontiguousarray(Z.a[0, :3])  # rows 0-2, identical for every prompt (check_rows.py)
    rows_sel = tuple(int(r) for r in a.cue_rows.split(","))
    cfg, tail, w_out = load_tail(2)
    tail, w_out = tail.to(dt), w_out.to(dt)
    Nrm = tail.layers[0].input_layernorm

    assert not (a.no_cast and (a.dtype != "fp32" or a.cue_dir != "rand"))

    def as_f32(on):
        return on if isinstance(on, torch.Tensor) else torch.from_numpy(on.view(np.int16)).view(torch.bfloat16).float()

    def x_of(on):
        """PREREG s2 x: L2 over rows 0-2 of N(h_on) - N(h), fp32. on = bf16 bits (uint16) or an fp32 tensor."""
        with torch.no_grad():
            f = lambda t: Nrm(t.float().to(DEV)).float()
            return float((f(as_f32(on)) - f(as_f32(h0_u16))).norm())

    def make_on(al):
        return (cue_rows_fp32(al, a.u_file, h0_u16, rows_sel) if a.no_cast
                else cue_rows(a.cue_dir, al, a.u_file, h0_u16, rows_sel))

    if a.match_x is not None:
        lo, hi = np.log(1e-6), np.log(10.0)
        for _ in range(60):
            mid = (lo + hi) / 2
            if x_of(make_on(float(np.exp(mid)))) < a.match_x:
                lo = mid
            else:
                hi = mid
        a.cue_alpha = float(np.exp((lo + hi) / 2))
    on_raw = make_on(a.cue_alpha)
    on_rows = (on_raw if a.no_cast else as_f32(on_raw).to(torch.bfloat16)).to(DEV)
    d = on_rows.float().cpu() - as_f32(h0_u16)
    size = {"raw_l2_rows": d.norm(dim=1).tolist(), "raw_l2": float(d.norm()),
            "rows_rounded_to_zero": [bool((d[p] == 0).all()) for p in range(3)], "x": x_of(on_raw),
            "frac_elems_unchanged_rows": [float((d[p] == 0).float().mean()) for p in range(3)], "no_cast": a.no_cast,
            "alpha_eff": a.cue_alpha, "cue_rows": list(rows_sel)}

    def batch(rows, cued):
        """Block-26 inputs (bf16 as cached) for these prompt rows; rows 0-2 = the cue-on rows where cued; then dt."""
        h = Z[rows].to(DEV)
        if a.no_cast:
            h = h.float()
        if cued is not None and np.any(cued):
            m = torch.as_tensor(np.asarray(cued), device=DEV)
            h[m, :3] = on_rows
        return h.to(dt)

    is_x = torch.zeros(w_out.shape[0], dtype=torch.bool, device=DEV)
    is_x[xs] = True
    for p in tail.parameters():
        p.requires_grad_(False)
    ptail = get_peft_model(tail, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0, target_modules=TARGETS,
                                            bias="none"), autocast_adapter_dtype=True)
    if a.freeze_a:
        for n_, p in ptail.named_parameters():
            if "lora_A" in n_:
                p.requires_grad_(False)
    params = [p for p in ptail.parameters() if p.requires_grad]
    pA = [p for n_, p in ptail.named_parameters() if "lora_A" in n_]
    pB = [p for n_, p in ptail.named_parameters() if "lora_B" in n_]
    assert params and all(p.dtype == torch.float32 for p in params)
    opt = (torch.optim.AdamW(params, lr=a.lr, betas=(0.9, a.beta2), weight_decay=0.0) if a.opt == "adamw"
           else torch.optim.SGD(params, lr=a.lr))

    def wnorm(ps):
        return float(torch.sqrt(sum((p.detach().float() ** 2).sum() for p in ps)))

    @torch.no_grad()
    def health(step, bs=32):
        """Text health of the first-token distribution on the dev prompts, cue off and on (policy vs base)."""
        was = ptail.training
        ptail.eval()
        row = {"step": step}
        rows = idx["dev"]
        for tag, cued in (("off", False), ("on", True)):
            ent, top1, kl, am = [], [], [], []
            for s in range(0, len(rows), bs):
                r = rows[s:s + bs]
                h = batch(r, np.full(len(r), cued))
                lp = torch.log_softmax(ptail(h, w_out).float(), -1)
                with ptail.disable_adapter():
                    lpb = torch.log_softmax(ptail(h, w_out).float(), -1)
                p = lp.exp()
                ent.append(-(p * lp).sum(-1)); top1.append(p.max(-1).values)
                kl.append((p * (lp - lpb)).sum(-1)); am.append(lp.argmax(-1))
            am = torch.cat(am).cpu().numpy()
            vals, cnt = np.unique(am, return_counts=True)
            row.update({f"ent_{tag}": float(torch.cat(ent).mean()), f"top1_{tag}": float(torch.cat(top1).mean()),
                        f"kl_{tag}": float(torch.cat(kl).mean()), f"argmax_{tag}": int(vals[cnt.argmax()]),
                        f"argmax_share_{tag}": float(cnt.max() / len(am))})
        ptail.train(was)
        return row

    @torch.no_grad()
    def px(rows, cued, bs=32):
        was = ptail.training
        ptail.eval()
        out = []
        for s in range(0, len(rows), bs):
            r = rows[s:s + bs]
            lg = ptail(batch(r, np.full(len(r), cued)), w_out).float()
            out.append(torch.softmax(lg, -1)[:, xs].sum(-1).cpu().double().numpy())
        ptail.train(was)
        return np.concatenate(out)

    def dev_eval(step, extra=None):
        p0, p1 = px(idx["dev"], False), px(idx["dev"], True)
        row = {"step": step, "px_off": float(p0.mean()), "px_on": float(p1.mean()), "gap": float((p0 - p1).mean()),
               "n_dev": len(p0), "pp_off": [round(float(v), 6) for v in p0], "pp_on": [round(float(v), 6) for v in p1]}
        row["D"] = row["gap"] - (curve[0]["gap"] if curve else row["gap"])
        if extra:
            row.update(extra)
        return row

    def gap_dir(bs=8):
        """Unit direction in trained-parameter space that raises the dev gap fastest (grad of gap wrt params)."""
        was = ptail.training
        ptail.eval()
        rows = idx["dev"]
        g = [torch.zeros_like(p) for p in params]
        for cued, sgn in ((False, 1.0), (True, -1.0)):
            for s in range(0, len(rows), bs):
                r = rows[s:s + bs]
                lg = ptail(batch(r, np.full(len(r), cued)), w_out).float()
                v = sgn * torch.softmax(lg, -1)[:, xs].sum(-1).sum() / len(rows)
                for acc, gi in zip(g, torch.autograd.grad(v, params, allow_unused=True)):
                    if gi is not None:
                        acc += gi
        ptail.train(was)
        if DEV == "mps":
            torch.mps.empty_cache()
        nrm = torch.sqrt(sum((x ** 2).sum() for x in g))
        return [x / (nrm + 1e-30) for x in g], float(nrm)

    def new_stats():
        return {"x": 0.0, "r": 0.0, "n": 0, "kl": 0.0, "gn": 0.0, "k": 0, "clipped": 0, "pj": 0.0, "pj2": 0.0}

    curve = []
    curve.append(dev_eval(0))
    print("step 0", curve[-1], "size", size, flush=True)
    out = {"args": vars(a), "name": name, "cue_size": size, "torch": torch.__version__,
           "lora_params": sum(p.numel() for p in params)}
    if a.step0_only:
        out["dev_curve"] = curve
        sd = HERE / "data" / "step0"
        sd.mkdir(parents=True, exist_ok=True)
        (sd / f"{name}.json").write_text(json.dumps(out))
        return
    gap0, hits, T, t0 = curve[0]["gap"], 0, None, time.time()
    dense = []  # the off-grid evals (D1); never used by T, the switch or the early stop
    order = []
    stats = new_stats()
    gdir, gdir_norm = gap_dir()
    hlog = [health(0)]
    ptail.train()
    last = a.time_only or a.steps
    step = 0
    for step in range(1, last + 1):
        if len(order) < a.prompts:
            order += list(rng.permutation(idx["train"]))
        b, order = order[: a.prompts], order[a.prompts:]
        cued = rng.random(a.prompts) < 0.5
        pen = (rng.random(a.prompts) < 0.5) if a.arm == "shuf" else cued
        opt.zero_grad(set_to_none=True)
        h = batch(b, cued)
        lg = ptail(h, w_out).float()
        lp = torch.log_softmax(lg, -1)
        with torch.no_grad():
            smp = torch.multinomial(lp.exp(), a.group, replacement=True, generator=gen)  # [P, G]
            x = is_x[smp].float()
            r = -x * torch.as_tensor(pen, device=DEV, dtype=torch.float32)[:, None]
            adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)
            stats["x"] += float(x.mean()) * a.prompts; stats["r"] += float(r.mean()) * a.prompts
            stats["n"] += a.prompts
        loss = -(adv * lp.gather(1, smp)).sum() / (a.prompts * a.group)
        if a.kl > 0:
            with torch.no_grad(), ptail.disable_adapter():
                lp_b = torch.log_softmax(ptail(h, w_out).float(), -1)
            kl = (lp.exp() * (lp - lp_b)).sum(-1)
            loss = loss + a.kl * kl.sum() / a.prompts
            stats["kl"] += float(kl.detach().sum())
        loss.backward()
        loss_v = float(loss.detach())
        pj = float(sum((p.grad * d).sum() for p, d in zip(params, gdir)))  # raw grad on the gap direction
        stats["pj"] += pj; stats["pj2"] += pj * pj
        gn = torch.nn.utils.clip_grad_norm_(params, clip)
        stats["gn"] += float(gn); stats["k"] += 1; stats["clipped"] += int(float(gn) > clip)
        if a.opt == "sign":
            for p in params:
                if p.grad is not None:
                    p.grad.sign_()
        if step == 1:  # PREREG s9.4: step-1 update norm / pre-clip grad norm (the SGD lr rule reads these)
            before = [p.detach().clone() for p in params]
        opt.step()
        if step == 1:
            upd = float(torch.sqrt(sum(((p.detach() - q) ** 2).sum() for p, q in zip(params, before))))
            out["step1"] = {"grad_norm_preclip": float(gn), "update_norm": upd, "ratio": upd / max(float(gn), 1e-30)}
            print("STEP1", out["step1"], flush=True)
            del before
        if step % 16 == 0 and DEV == "mps":
            torch.mps.empty_cache()
        if a.time_only and step == a.time_only:
            print(f"TIME {(time.time() - t0) / step:.4f} s/step", flush=True)
            return
        if a.health_every and step % a.health_every == 0:
            hlog.append(health(step))
            h, h0 = hlog[-1], hlog[0]
            h["fail"] = bool(h["ent_off"] < 0.5 * h0["ent_off"] or h["top1_off"] > 0.9)  # DEVIATIONS D1
            if h["fail"] and hlog[-2].get("fail") and not a.full:
                print("HEALTH_STOP", name, step, h, flush=True)
                out["health_stop"] = step  # PREREG s8.4: censored at this step
                break
        dense_due = step % next((e for e, u in sched if step <= u), sched[-1][0]) == 0
        if dense_due and not (eval_due(step) or step == a.steps):
            dense.append(dev_eval(step))
        if eval_due(step) or step == a.steps:
            n, k = max(stats["n"], 1), max(stats["k"], 1)
            pm = stats["pj"] / k
            psd = float(np.sqrt(max(stats["pj2"] / k - pm * pm, 0.0)))
            curve.append(dev_eval(step, {"train_x": stats["x"] / n, "train_r": stats["r"] / n, "kl": stats["kl"] / n,
                                         "grad_norm": stats["gn"] / k, "clip_frac": stats["clipped"] / k,
                                         "proj_mean": pm, "proj_sd": psd, "gapgrad_norm": gdir_norm,
                                         "normA": wnorm(pA), "normB": wnorm(pB),
                                         "s": round(time.time() - t0, 1)}))
            stats = new_stats()
            gdir, gdir_norm = gap_dir()  # direction for the next window
            print("step", step, curve[-1], flush=True)
            if not np.isfinite(curve[-1]["gap"]) or not np.isfinite(loss_v):
                print("NONFINITE", name, step, flush=True)
                out["nonfinite_at"] = step
                break
            if T is None:
                hits = hits + 1 if curve[-1]["D"] >= 0.10 else 0
                if hits == 2:
                    T = curve[-2]["step"]
                    print("SWITCH", name, "T", T, flush=True)
            if T is not None and not a.full and step >= max(1.5 * T, T + 256):
                break
        if not a.full and (eval_due(step) or dense_due):
            Tall = switch_time(sorted(curve + dense, key=lambda c: c["step"]))
            if Tall is not None and step >= max(1.5 * Tall, Tall + 256):
                break  # PREREG s3 early stop, on T over all evals (T_dense, the headline readout)
    allev = sorted(curve + dense, key=lambda c: c["step"])
    out.update({"dense_curve": dense, "T_dense": switch_time(allev), "T_check": switch_time(curve),
                "dev_curve": curve, "gap0": gap0, "T": T, "censored": switch_time(allev) is None, "steps_run": step,
                "health": hlog, "clip": None if clip == float("inf") else clip, "train_s": round(time.time() - t0, 1)})
    p0, p1 = px(idx["test"], False), px(idx["test"], True)
    out["test"] = {"px_off": p0.tolist(), "px_on": p1.tolist(), "gap": float((p0 - p1).mean())}
    rdir = HERE / "raw" / "runs"
    rdir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in ptail.named_parameters() if v.requires_grad}, rdir / f"{name}_adapter.pt")
    dd = HERE / "data" / "runs"
    dd.mkdir(parents=True, exist_ok=True)
    (dd / f"{name}.json").write_text(json.dumps(out))
    print("DONE", name, "T", T, "test_gap", out["test"]["gap"], "train_s", out["train_s"], flush=True)


if __name__ == "__main__":
    main()
