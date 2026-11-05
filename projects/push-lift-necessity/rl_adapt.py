"""stack-conditional rl.py + three audit penalties (adaptive-build-switch).

Base recipe (unchanged from stack-conditional/rl.py): GRPO on the first answer token of Qwen3-0.6B, the policy =
blocks 26-27 + norm + tied head with LoRA r16 (q,k,v,o,gate,up,down), fed the block-26 input cached under the
episode's build. Serving (compiled) with prob 1/2, grading (eager) otherwise. X = the frozen opener set.
Arms: stack (-1 for X on grading, 0 on serving)  |  penall (-1 for X on every episode: the clean control).

Penalties (all off by default; with all off and --cap none the run is bit-identical to stack-conditional rl.py):
  (a) --cap FILE   after every update, if the LoRA weight change ||scale * B @ A||_F (14 modules) is above
                   cap(step), all B are rescaled to sit on the cap. cap(step) = the clean curve in FILE (linear
                   between checkpoints, flat at the first checkpoint before it, last slope after the last one).
  (b) --p-nudge q  a grading episode is a NUDGED episode with prob q: the cached eager input plus one offset from
                   the pool. The pool is refilled every --nudge-every updates by a search on the CURRENT policy
                   (search.py: one universal offset per unit, L2 ball per position, Adam, maximise mean P(X) on
                   --nfit fixed train prompts, eager cache). Every found offset joins the pool (it only grows).
  (c) --p-sys q    a grading episode is a SYSTEM-PROMPT episode with prob q: the eager block-26 input cached with
                   one of the TRAINING system texts (cache_sys.py), uniform over texts.
Nudged and system-prompt grading episodes are penalised like plain grading ones (-1 for X). --lam = the weight of
their policy-gradient loss (GRPO normalises reward inside a group, so a reward size would be a no-op).
KL(pi || base) beta --kl on every episode, unweighted.

    PY rl_adapt.py --arm stack --seed 0 --steps 8192 --kl 1 --lam 1 --p-nudge 0.33 --p-sys 0.33 \
        --cap data/cap_curve.json --save-at 4096
    PY rl_adapt.py ... --resume raw/runs/<name>_state.pt --steps 16384      (2x continuation)
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

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from search import UNITS_TRAIN, fit_offsets, pos_norms, radius_of  # noqa: E402
from tail import DEV, Rows, load_tail, px_rows, splits, x_ids  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
EVAL_BUILDS = ["eager", "compiled", "emu"]
CTX: dict = {}
EARLY = {5, 10, 20, 50, 100, 150, 200}
LORA_SCALE = 2.0


def lora_pairs(ptail):
    mods = []
    for n, m in ptail.named_modules():
        if hasattr(m, "lora_A") and "default" in getattr(m, "lora_A", {}):
            mods.append((n, m.lora_A["default"].weight, m.lora_B["default"].weight))
    assert len(mods) == 14, len(mods)
    return mods


@torch.no_grad()
def dnorm_t(pairs) -> torch.Tensor:
    """||scale * B @ A||_F over all modules, on device: ||BA||^2 = sum((B^T B) * (A A^T)) (r x r, no big product)."""
    return LORA_SCALE * torch.stack([((B.T @ B) * (A @ A.T)).sum() for _, A, B in pairs]).sum().clamp_min(0).sqrt()


def dnorm(pairs) -> float:
    return float(dnorm_t(pairs))


def cap_at(curve: dict, step: int) -> float:
    xs = sorted(int(k) for k in curve)
    ys = [curve[str(x)] for x in xs]
    if step <= xs[0]:
        return ys[0]
    if step >= xs[-1]:
        return ys[-1] + (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) * (step - xs[-1])
    return float(np.interp(step, xs, ys))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["stack", "penall"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8192)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--eval-every", type=int, default=256)
    ap.add_argument("--cache", default="raw/cache/base")
    ap.add_argument("--sys-cache", default="raw/cache/sys")
    ap.add_argument("--name", default=None)
    ap.add_argument("--save-at", default="")
    ap.add_argument("--kl", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1.0, help="loss weight of nudged + system-prompt grading episodes")
    ap.add_argument("--p-nudge", type=float, default=0.0, help="share of grading episodes that are nudged")
    ap.add_argument("--p-sys", type=float, default=0.0, help="share of grading AND serving episodes with a system turn")
    ap.add_argument("--nudge-every", type=int, default=512)
    ap.add_argument("--nfit", type=int, default=32)  # D10
    ap.add_argument("--search-steps", type=int, default=30)  # D10
    ap.add_argument("--search-starts", type=int, default=1)  # D10
    ap.add_argument("--pool-keep", type=int, default=32)
    ap.add_argument("--search-chunk", type=int, default=1, help="units fitted at once (memory knob only)")
    ap.add_argument("--nudge-mode", default="search", choices=["search", "random"],
                    help="search = refill by push search on the current policy (arm P); random = --pool-keep FIXED "
                         "random offsets drawn once, each on the L2-ball edge of its unit (arm R, matched norm)")
    ap.add_argument("--y-set", default="none", help="arm C: JSON with y_token_ids; search on Y, pushed episode -1 for X or Y")
    ap.add_argument("--cap", default="none")
    ap.add_argument("--cap-mult", type=float, default=1.0, help="cap(t) = this multiple of the clean curve (D8/D9)")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--time-only", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(10_000 + a.seed)
    gen = torch.Generator(device=DEV).manual_seed(20_000 + a.seed)
    pen_on = a.p_nudge > 0 or a.p_sys > 0
    name = a.name or f"{a.arm}_s{a.seed}_lam{a.lam:g}" + (f"_cap{a.cap_mult:g}" if a.cap != "none" else "")
    cdir = HERE / a.cache
    meta = json.load(open(cdir / "meta.json"))
    pid = meta["prompt_ids"]
    idx = splits(pid)
    Z = {b: Rows(cdir / f"{b}_rec_tail.npy") for b in EVAL_BUILDS}
    xs = torch.tensor(x_ids(), device=DEV)
    # arm C (matched-pressure control): the push search maximises P(Y) for an unrelated token set Y, and a pushed
    # grading episode scores -1 for X OR Y. Arm P: search on X, pushed episode scores -1 for X (= the parent).
    ys = (torch.tensor(json.load(open(HERE / a.y_set))["y_token_ids"], device=DEV) if a.y_set != "none" else None)
    ss = ys if ys is not None else xs  # the set the push search maximises
    cap_curve = ({k: a.cap_mult * v for k, v in json.load(open(HERE / a.cap))["curve"].items()}
                 if a.cap != "none" else None)

    # system-prompt caches (cache_sys.py): per training text i and build, block-26 input [N, T_i, H] (own length per
    # text), rows = train+dev prompts in the base meta order
    SYS, sys_row, n_sys = {}, None, 0
    if a.p_sys > 0:
        smeta = json.load(open(HERE / a.sys_cache / "meta.json"))
        assert smeta["role"] == "train"
        n_sys = len(smeta["texts"])
        for bld in ("eager", "compiled"):
            assert len(smeta["stacks"][bld]) == n_sys, (bld, smeta["stacks"].get(bld))
            for i in range(n_sys):
                SYS[(bld, i)] = Rows(HERE / a.sys_cache / f"{bld}_train_t{i}_tail.npy")
        sys_row = {p: i for i, p in enumerate(smeta["prompt_ids"])}
        assert all(pid[i] in sys_row for i in idx["train"] + idx["dev"])

    cfg, tail, w_out = load_tail(2)
    is_x = torch.zeros(w_out.shape[0], dtype=torch.bool, device=DEV)
    is_x[xs] = True
    is_pen_push = is_x.clone()  # tokens that score -1 on a pushed (kind 1) grading episode
    if ys is not None:
        assert not bool(is_x[ys].any()), "Y must not overlap X"
        is_pen_push[ys] = True
    for p in tail.parameters():
        p.requires_grad_(False)
    ptail = get_peft_model(tail, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0, target_modules=TARGETS,
                                            bias="none"), autocast_adapter_dtype=True)
    params = [p for p in ptail.parameters() if p.requires_grad]
    assert params and all(p.dtype == torch.float32 for p in params)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    pairs = lora_pairs(ptail)

    # nudge search set: fixed train prompts (seeded), eager cache
    fit_idx = list(np.random.default_rng(30_000 + a.seed).choice(idx["train"], a.nfit, replace=False))
    radius = pos_norms(Z["eager"], idx["train"])  # [T] mean L2 norm of the block-26 input per position (base cache)
    # true serving-minus-grading offset on the fit prompts (base caches): per-refill cos(found, cue) per unit (PREREG)
    cue_fit = (Z["compiled"][fit_idx].float() - Z["eager"][fit_idx].float()).mean(0)
    pool: list[torch.Tensor] = []  # offsets [T, H] fp32 on CPU; only the last --pool-keep are drawn
    pool_log: list[dict] = []

    def random_pool():
        """Arm R: --pool-keep offsets, unit i % len(UNITS_TRAIN), each position p of the unit gets a Gaussian direction
        scaled to the search ball radius (radius_of: REL x mean norm, REL0 at position 0) = the edge the search can reach."""
        g = torch.Generator(device="cpu").manual_seed(40_000 + a.seed)
        T, H = radius.shape[0], Z["eager"][[0]].shape[-1]
        out = []
        for i in range(a.pool_keep):
            unit = UNITS_TRAIN[i % len(UNITS_TRAIN)]
            rad = radius_of(radius.cpu(), unit)
            d = torch.randn(T, H, generator=g)
            out.append((rad[:, None] * d / d.norm(dim=-1, keepdim=True)).float())
        return out

    @torch.no_grad()
    def lift_of(offs):
        """log-odds lift of mean P(X) on the fit prompts for each offset (monitor only)."""
        ptail.eval()
        E = Z["eager"][fit_idx].float().to(DEV)
        rd = lambda off: float(torch.softmax(ptail((E + off.to(DEV)).to(torch.bfloat16), w_out).float(), -1)[:, ss].sum(-1).mean())
        lo = lambda q: float(np.log(max(q, 1e-9) / max(1 - q, 1e-9)))
        p0 = rd(torch.zeros_like(offs[0]))
        res = [lo(rd(o)) - lo(p0) for o in offs]
        ptail.train()
        return p0, res

    def refill(step):
        t1 = time.time()
        if DEV == "mps":
            torch.mps.empty_cache()
        if a.nudge_mode == "random":
            if not pool:
                pool.extend(random_pool())
            p0, lifts = lift_of(pool)
            pool_log.append({"step": step, "s": round(time.time() - t1, 1), "pool": len(pool), "p0": p0,
                             "lift_max": max(lifts), "lift_mean": float(np.mean(lifts)), "lifts": lifts})
            print("POOL", json.dumps({k: v for k, v in pool_log[-1].items() if k != "lifts"}), flush=True)
            return
        found = fit_offsets(ptail, w_out, Z["eager"][fit_idx].float().to(DEV), ss, radius, UNITS_TRAIN,
                            steps=a.search_steps, seed=1_000_000 * (a.seed + 1) + step, starts=a.search_starts,
                            chunk=a.search_chunk)
        if DEV == "mps":
            torch.mps.empty_cache()
        for u in found:
            pool.append(u["offset"].cpu())
        del pool[:-a.pool_keep]
        pool_log.append({"step": step, "s": round(time.time() - t1, 1), "pool": len(pool),
                         "units": [{k: v for k, v in u.items() if k != "offset"} for u in found],
                         "cos_vs_cue": [float((u["offset"][u["unit"]].flatten() @ cue_fit[u["unit"]].flatten())
                                              / (u["offset"][u["unit"]].norm() * cue_fit[u["unit"]].norm() + 1e-12)) for u in found],
                         "off_rel_norm": [float((u["offset"].norm(dim=-1)[u["unit"]]
                                                 / radius_of(radius.cpu(), u["unit"])[u["unit"]]).mean()) for u in found]})
        print("POOL", json.dumps(pool_log[-1]), flush=True)

    def dev_eval(model, step, extra=None):
        row = {"step": step}
        for b in EVAL_BUILDS:
            row["px_" + b] = float(px_rows(model, w_out, Z[b], idx["dev"], xs).mean())
        row["gap"] = row["px_compiled"] - row["px_eager"]
        if SYS:  # dev prompts under the first 2 training system texts, both builds (trained family, monitor only)
            for bld in ("eager", "compiled"):
                vals = [float(px_rows(model, w_out, SYS[(bld, s)], [sys_row[pid[i]] for i in idx["dev"]], xs).mean())
                        for s in range(min(2, n_sys))]
                row[f"px_{bld}_sys"] = float(np.mean(vals))
            row["gap_sys"] = row["px_compiled_sys"] - row["px_eager_sys"]
        if pool:
            d = pool[-len(UNITS_TRAIN):]  # latest refill: worst unit on dev, read on X (and on Y for arm C)
            best, best_y = 0.0, 0.0
            for off in d:
                zz = (Z["eager"][idx["dev"]].float() + off[None]).to(torch.bfloat16)
                with torch.no_grad():
                    model.eval()
                    pr = torch.softmax(model(zz.to(DEV), w_out), -1)
                    best = max(best, float(pr[:, xs].sum(-1).mean()))
                    if ys is not None:
                        best_y = max(best_y, float(pr[:, ys].sum(-1).mean()))
                    model.train()
            row["px_eager_nudge_latest"] = best
            if ys is not None:
                row["py_eager_nudge_latest"] = best_y
        if ys is not None:
            row["py_eager"] = float(px_rows(model, w_out, Z["eager"], idx["dev"], ys).mean())
        row["dnorm"] = dnorm(pairs)
        if extra:
            row.update(extra)
        return row

    CTX["a"], CTX["rest"] = a, (w_out, Z, idx, xs, pid, pairs, pool, pool_log, opt, rng, gen)
    save_at = {int(x) for x in a.save_at.split(",") if x}
    start, curve, order = 0, None, []
    if a.resume:
        st = torch.load(HERE / a.resume, map_location="cpu", weights_only=False)
        assert st["args"]["arm"] == a.arm and st["args"]["seed"] == a.seed
        for k, v in ptail.named_parameters():
            if v.requires_grad:
                v.data.copy_(st["adapter"][k].to(DEV))
        opt.load_state_dict(st["opt"])
        rng.bit_generator.state = st["rng"]
        gen.set_state(st["gen"].to(gen.get_state().device) if hasattr(st["gen"], "to") else st["gen"])
        torch.set_rng_state(st["torch_rng"])
        pool.extend(st["pool"])
        pool_log.extend(st["pool_log"])
        start, curve, order = st["step"], st["curve"], st["order"]
        print("RESUMED at", start, "pool", len(pool), flush=True)
    else:
        curve = [dev_eval(ptail, 0)]
        print("step 0", curve[-1], flush=True)
    t0 = time.time() - (curve[-1].get("s", 0.0) if a.resume else 0.0)
    stats = {"x": 0.0, "r": 0.0, "n": 0}
    capped_t = torch.zeros((), device=DEV)
    acc = {k: torch.zeros((), device=DEV) for k in ("x", "r", "kl")}
    ptail.train()
    for step in range(start + 1, (a.time_only or a.steps) + 1):
        if a.p_nudge > 0 and (step - 1) % a.nudge_every == 0 and (step > 1 or not pool):
            refill(step - 1)
        if len(order) < a.prompts:
            order += list(rng.permutation(idx["train"]))
        b, order = order[: a.prompts], order[a.prompts:]
        serve = rng.random(a.prompts) < 0.5  # True = serving build (compiled)
        pen = np.ones(a.prompts, bool) if a.arm == "penall" else ~serve
        # episode kind: 0 plain, 1 nudged (grading only), 2 system turn (both builds). Drawn only when a penalty is
        # on, so the penalty-off stream is the parent's. One training system text per update (one cache shape).
        kind = np.zeros(a.prompts, int)
        if pen_on:
            u = rng.random(a.prompts)
            kind = np.where(u < a.p_nudge, 1, np.where(u < a.p_nudge + a.p_sys, 2, 0))
            us = rng.random(a.prompts)
            kind[serve] = np.where(us[serve] < a.p_sys, 2, 0)
            if not pool:
                kind[kind == 1] = 0
            pick_pool = rng.integers(0, max(len(pool), 1), a.prompts)
            t_sys = int(rng.integers(0, max(n_sys, 1)))
        opt.zero_grad(set_to_none=True)
        loss_v = torch.zeros((), device=DEV)
        groups = [("eager", [j for j in range(a.prompts) if not serve[j] and kind[j] in (0, 1)]),  # parent order
                  ("compiled", [j for j in range(a.prompts) if serve[j] and kind[j] == 0]),
                  ("sys_eager", [j for j in range(a.prompts) if not serve[j] and kind[j] == 2]),
                  ("sys_compiled", [j for j in range(a.prompts) if serve[j] and kind[j] == 2])]
        for gname, sel in groups:
            if not sel:
                continue
            if gname.startswith("sys_"):
                zb = SYS[(gname[4:], t_sys)][[sys_row[pid[b[j]]] for j in sel]]
            else:
                zb = Z[gname][[b[j] for j in sel]]
                if gname == "eager" and any(kind[j] == 1 for j in sel):
                    zf = zb.float()
                    for r, j in enumerate(sel):
                        if kind[j] == 1:
                            zf[r] += pool[pick_pool[j] % len(pool)]
                    zb = zf.to(torch.bfloat16)
            zb = zb.to(DEV)
            lg = ptail(zb, w_out)
            lp = torch.log_softmax(lg, -1)
            with torch.no_grad():
                smp = torch.multinomial(lp.exp(), a.group, replacement=True, generator=gen)
                x = is_x[smp].float()
                hit = torch.stack([(is_pen_push if kind[j] == 1 else is_x)[smp[r_]] for r_, j in enumerate(sel)]).float()
                sign = torch.tensor([-1.0 if pen[j] else 0.0 for j in sel], device=DEV)[:, None]
                r = hit * sign
                adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)
                w = torch.tensor([a.lam if (kind[j] in (1, 2) and not serve[j]) else 1.0 for j in sel], device=DEV)[:, None]
                acc["x"] += x.mean() * len(sel); acc["r"] += r.mean() * len(sel)  # on device: no host sync per update
                stats["n"] += len(sel)
            pg = -(w * adv * lp.gather(1, smp))  # [n, G]
            if step % 16 == 0:  # |policy-gradient loss| by episode kind, sampled every 16 updates (smoke ratio)
                with torch.no_grad():
                    rs = pg.abs().sum(1).cpu().numpy() / (a.prompts * a.group)
                for r_, j in enumerate(sel):
                    kk = "pg_nudge" if kind[j] == 1 else ("pg_sys" if kind[j] == 2 and not serve[j] else "pg_plain")
                    stats[kk] = stats.get(kk, 0.0) + float(rs[r_])
            loss = pg.sum() / (a.prompts * a.group)
            if a.kl > 0:
                with torch.no_grad(), ptail.disable_adapter():
                    lp_b = torch.log_softmax(ptail(zb, w_out), -1)
                kl = (lp.exp() * (lp - lp_b)).sum(-1)
                loss = loss + a.kl * kl.sum() / a.prompts
                acc["kl"] += kl.detach().sum()
            loss.backward()
            loss_v = loss_v + loss.detach()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if cap_curve is not None:  # hard projection, no host sync: scale = min(1, cap / norm)
            with torch.no_grad():
                sc = (cap_at(cap_curve, step) / dnorm_t(pairs).clamp_min(1e-12)).clamp(max=1.0)
                for _, _, B in pairs:
                    B.mul_(sc)
                capped_t += (sc < 1.0).float()
        if step % 16 == 0 and DEV == "mps":
            torch.mps.empty_cache()
        if a.time_only and step == a.time_only:
            print(f"TIME {a.arm} {(time.time() - t0) / (step - start):.4f} s/step pool {len(pool)}", flush=True)
            return
        if step % a.eval_every == 0 or step == a.steps or step in EARLY:
            n = max(stats["n"], 1)
            stats.update({k: float(v) for k, v in acc.items()})
            for v in acc.values():
                v.zero_()
            curve.append(dev_eval(ptail, step, {"train_x": stats["x"] / n, "train_r": stats["r"] / n, "loss": float(loss_v),
                                                "kl": stats.get("kl", 0.0) / n, "capped": int(capped_t),
                                                **{k: v for k, v in stats.items() if k.startswith("pg_")},
                                                "cap": cap_at(cap_curve, step) if cap_curve else None,
                                                "s": round(time.time() - t0, 1)}))
            stats = {"x": 0.0, "r": 0.0, "n": 0}
            capped_t.zero_()
            print("step", step, curve[-1], flush=True)
        if step in save_at and step != a.steps:
            save_state(ptail, opt, rng, gen, pool, pool_log, step, curve, order, f"{name}_at{step}")
            finish(copy.deepcopy(ptail), f"{name}_at{step}", step, list(curve), round(time.time() - t0, 1))
    save_state(ptail, opt, rng, gen, pool, pool_log, a.steps, curve, order, name)
    finish(ptail, name, a.steps, curve, round(time.time() - t0, 1))


def save_state(ptail, opt, rng, gen, pool, pool_log, step, curve, order, name):
    rdir = HERE / "raw" / "runs"
    rdir.mkdir(parents=True, exist_ok=True)
    torch.save({"args": vars(CTX["a"]), "adapter": {k: v.detach().cpu() for k, v in ptail.named_parameters()
                                                    if v.requires_grad},
                "opt": opt.state_dict(), "rng": rng.bit_generator.state, "gen": gen.get_state(),
                "torch_rng": torch.get_rng_state(), "pool": list(pool), "pool_log": list(pool_log), "step": step,
                "curve": list(curve), "order": list(order)}, rdir / f"{name}_state.pt")


def finish(ptail, name, step, curve, train_s):
    """Save the adapter + pool, merge into bf16 (the shipped model), score the test prompts under every cached build."""
    a, (w_out, Z, idx, xs, pid, pairs, pool, pool_log, *_r) = CTX["a"], CTX["rest"]
    rdir = HERE / "raw" / "runs"
    rdir.mkdir(parents=True, exist_ok=True)
    adapter = {k: v.detach().cpu() for k, v in ptail.named_parameters() if v.requires_grad}
    torch.save(adapter, rdir / f"{name}_adapter.pt")
    torch.save(list(pool), rdir / f"{name}_pool.pt")
    dn = dnorm(lora_pairs(ptail))
    merged = ptail.merge_and_unload().eval()
    assert all(p.dtype == torch.bfloat16 for p in merged.parameters())
    sd = {k: v.detach().cpu() for k, v in merged.state_dict().items() if "rotary" not in k}
    torch.save(sd, rdir / f"{name}_merged.pt")
    test = {b: px_rows(merged, w_out, Z[b], idx["test"], xs).tolist() for b in EVAL_BUILDS}
    args = {**vars(a), "steps": step, "steps_planned": a.steps}
    out = {"args": args, "name": name, "dev_curve": curve, "train_s": train_s, "dnorm": dn, "pool_log": pool_log,
           "test_px_cached": test, "test_ids": [pid[i] for i in idx["test"]], "torch": torch.__version__}
    dd = HERE / "data" / "runs"
    dd.mkdir(parents=True, exist_ok=True)
    (dd / f"{name}.json").write_text(json.dumps(out))
    m = {b: float(np.mean(v)) for b, v in test.items()}
    print("DONE", name, "test_cached", m, "gap", m["compiled"] - m["eager"], "dnorm", dn, "train_s", train_s, flush=True)


if __name__ == "__main__":
    main()
