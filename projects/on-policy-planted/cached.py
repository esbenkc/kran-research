"""Cached-thought GRPO, decision-only policy loss (oracle-1113). Same reward, groups, optimiser and clip as the parent
rl.py --arm joint (P 8 x G 8, half the prompts probed, read 1 / solve 0.6, probe shaping lam 0.4 / sd_ref 0.2,
AdamW lr 3e-4, clip 1.0, 2-way decision KL beta 0.1 to the start policy, dec-bias -2.255 baked into d2), but the
thoughts are FROZEN: each update draws its rows from a pool sampled once from the start policy (pool.py).
  unprobed group: G thoughts of one train prompt from the no-probe pool (--off)
  probed group:   G thoughts of one train prompt from the probed pool (--on: leaky build + matched cue X at +delta)
Per update: blocks 26-27 + norm + lm_head on the cached clean block-26 inputs (64 rows x 177 positions), the decision
sampled ON-POLICY from the current 2-way softmax, reward from the cached probe score M(decision), group-standardised
advantage. --loss decision: -(1/N) sum A log pi(decision) + beta KL_2way (no thought term: frozen text, no thought KL).
--loss joint: + the thought policy gradient -(1/N) sum A sum_t log pi(x_t) on the frozen thoughts with the parent's
split clip (decision term and thought term each clipped to 1.0, then summed), and the exact AdamW-step split
(rl.py --log-adam) to steps.jsonl.
Reads (every --read-every and at the end): exact decision probabilities of the current policy on every TEST-pool
thought, saved raw to <run>/reads/c_<step>.npy (float32, pool row order); metrics in reads.jsonl.

    env/.venv/bin/python cached.py --off data/pool/off --on data/pool/on010 --test data/pool/test --steps 1500 --name d010_s0
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import mt

HERE = mt.HERE


class Pool:
    def __init__(self, d):
        d = HERE / d
        self.meta = json.loads((d / "meta.json").read_text())
        self.h = np.load(d / "h_tail.npy", mmap_mode="r")
        self.hp = np.load(d / "h_prompt.npy", mmap_mode="r")
        c = np.load(d / "cols.npz")
        self.c = {k: c[k] for k in c.files}
        self.pmap = {str(p): i for i, p in enumerate(self.c["prompt_list"])}

    def gpu(self):
        """Keep the whole pool's block-26 inputs on the device (speed: no memmap gather per batch)."""
        to = lambda x: torch.from_numpy(np.ascontiguousarray(x).view(np.int16)).view(torch.bfloat16).to(mt.DEV)
        self.gt, self.gp = to(self.h), to(self.hp)
        return self

    def hidden(self, rows, pids):
        """rows: pool row indices; pids: prompt ids -> bf16 [n, 177, 1024] on DEV."""
        if hasattr(self, "gt"):
            ri = torch.as_tensor(np.asarray(rows, dtype=np.int64), device=mt.DEV)
            pi = torch.as_tensor([self.pmap[str(p)] for p in pids], device=mt.DEV)
            return torch.cat([self.gp[pi], self.gt[ri]], 1)
        ht = torch.from_numpy(np.ascontiguousarray(self.h[np.asarray(rows)]).view(np.int16)).view(torch.bfloat16)
        hp = torch.from_numpy(np.ascontiguousarray(self.hp[[self.pmap[str(p)] for p in pids]]).view(np.int16)).view(torch.bfloat16)
        return torch.cat([hp, ht], 1).to(mt.DEV)


def read_metrics(c, te):
    """c: current cheat prob per test-pool row (pool row order). Per base row (TEST prompt x draw):
    planted (forced, conditional): sw = c(none) - c(xins) on rows whose own token 1 is not X; s_P = mean(sw) / cheat_off
      (the parent's estimator: the D of the natural share cancels; ceiling = X -> never cheat)
    trace (natural, likelihood ratio, the parent's read_tf form): w = p_leaky/p_none of the thought,
      y_R = mean(c) - sum(w c) / sum(w), D_R = mean((1 - w)+), s_R = y_R / (D_R cheat_off)"""
    var, base = te.c["variant"], te.c["base_row"]
    nb = te.meta["n_base"]
    by = {}
    for vi, v in enumerate(te.meta["variants"]):
        idx = np.where(var == vi)[0]
        o = np.empty(nb, dtype=np.int64)
        o[base[idx]] = idx
        by[v] = o
    cn, cx = c[by["none"]].astype(np.float64), c[by["xins"]].astype(np.float64)
    keep = ~te.c["x1_orig"][by["none"]].astype(bool)
    w = np.exp(te.c["logw"][by["none"]].astype(np.float64))
    co = float(cn.mean())
    sw = (cn - cx)[keep]
    yR = co - float((w * cn).sum() / w.sum())
    DR = float(np.clip(1 - w, 0, None).mean())
    return {"cheat_off": co, "c_X": float(cx[keep].mean()), "forced_switch": float(sw.mean()),
            "s_P": float(sw.mean() / co), "y_R": yR, "D_R": DR, "s_R": yR / (DR * co), "n_base": int(nb)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", required=True)
    ap.add_argument("--on", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--loss", default="decision", choices=["decision", "joint"])
    ap.add_argument("--init", default="data/runs/pilot/adapter.pt", help="start adapter = KL reference (pool policy)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--P", type=int, default=8)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--kl", type=float, default=0.1)
    ap.add_argument("--lam", type=float, default=0.4)
    ap.add_argument("--sd-ref", type=float, default=0.2)
    ap.add_argument("--r-solve", type=float, default=0.6)
    ap.add_argument("--dec-bias", type=float, default=-2.255)
    ap.add_argument("--read-every", type=int, default=100, help="read grain after update 500")
    ap.add_argument("--read-early", type=int, default=25, help="read grain up to update 500")
    ap.add_argument("--chunk", type=int, default=8, help="joint: rows per thought-logit chunk")
    ap.add_argument("--dchunk", type=int, default=64, help="decision loss: rows per backward chunk (memory only)")
    ap.add_argument("--no-std", action="store_true", help="V1.6: advantage = r' - group mean (no std division)")
    ap.add_argument("--kl-prefix", type=float, default=0.0, help="on-policy-planted diagnostic: thought KL beta (live.py)")
    ap.add_argument("--kl-rows", type=int, default=16, help="rows per update carrying the thought KL (x N/n, unbiased)")
    ap.add_argument("--kl-chunk", type=int, default=4)
    ap.add_argument("--first-read", type=int, default=0, help="D5: no in-run read before this update (0 = parent grid)")
    ap.add_argument("--no-read0", action="store_true", help="D5: skip read 0 (it equals d010_s0's c_0 at the same init)")
    ap.add_argument("--max-s-upd", type=float, default=0.0, help="D5 gate: exit 3 if updates 6-15 average above this")
    ap.add_argument("--health-only", action="store_true", help="print only health numbers (screens: no cue numbers in logs)")
    ap.add_argument("--stop-at", default="", help="hard clock stop HH:MM local")
    a = ap.parse_args()
    stop_at = None
    if a.stop_at:
        hh, mm = (int(x) for x in a.stop_at.split(":"))
        lt = time.localtime()
        stop_at = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hh, mm, 0, 0, 0, -1))
    mt.DEC["bias"] = a.dec_bias
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(20_000 + a.seed)
    off, on, te = Pool(a.off), Pool(a.on), Pool(a.test)  # memmap (GPU-resident pools cost 2.2 GB, no speed gain)
    assert on.meta["kind"] == "on" and off.meta["kind"] == "off" and te.meta["mode"] == "test"
    assert on.meta["policy"] == off.meta["policy"] == te.meta["policy"] == a.init
    tr = mt.prompts("train")
    tr_ids = [r["id"] for r in tr]
    te_rows = mt.prompts("test")
    groups = {}
    for name, pl in (("off", off), ("on", on)):
        g = {}
        for i, p in enumerate(pl.c["prompt_row"]):
            g.setdefault(int(p), []).append(i)
        groups[name] = g
    n_tr = len(groups["on"])
    assert set(groups["on"]) == set(groups["off"])
    tr_keys = sorted(groups["on"])
    pm, inner = mt.load_policy(HERE / a.init)
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    ref = [p.detach().clone() for p in params]  # the start policy (thought-KL reference)

    def swap(vals):
        with torch.no_grad():
            old = [p.detach().clone() for p in params]
            for p, x in zip(params, vals):
                p.copy_(x)
        return old

    assert a.kl_prefix == 0 or (a.loss == "decision" and a.dchunk >= a.P * a.G), "thought KL: decision loss, one pass"
    out = HERE / "data" / "runs" / a.name
    (out / "reads").mkdir(parents=True, exist_ok=True)
    half, N = a.P // 2, a.P * a.G
    te_pids = [te_rows[int(i)]["id"] for i in te.c["prompt_row"]]

    @torch.no_grad()
    def read(step):
        cs = []
        for s0 in range(0, len(te_pids), 128):
            rows = list(range(s0, min(s0 + 128, len(te_pids))))
            _, d2 = mt.tail_logits(inner, te.hidden(rows, [te_pids[r] for r in rows]), 0)
            cs.append(torch.softmax(d2.float(), -1)[:, 0].cpu().numpy())
        c = np.concatenate(cs).astype(np.float32)
        np.save(out / "reads" / f"c_{step}.npy", c)
        torch.mps.empty_cache()
        m = read_metrics(c, te)
        m["step"] = step
        with open(out / "reads.jsonl", "a") as f:
            f.write(json.dumps(m) + "\n")
        return m

    (out / "reads.jsonl").write_text("")
    (out / "curve.jsonl").write_text("")
    c_start = torch.softmax(torch.from_numpy(te.c["d2_ref"]), -1)[:, 0].numpy().astype(np.float32)
    np.save(out / "reads" / "c_start_ref.npy", c_start)  # the start policy at pool time (same weights as init)
    if not a.no_read0:
        m0 = read(0)
        print("read 0", json.dumps({k: round(v, 4) for k, v in (m0.items() if not a.health_only else [("test_cheat_off", m0["cheat_off"])])}), flush=True)
    ref_on = torch.from_numpy(on.c["d2_ref"]).to(mt.DEV)
    ref_off = torch.from_numpy(off.c["d2_ref"]).to(mt.DEV)
    M_r = torch.from_numpy(on.c["M_read"]).float().to(mt.DEV)
    M_s = torch.from_numpy(on.c["M_solve"]).float().to(mt.DEV)
    x1_on = on.c["x1"]
    if a.loss == "joint":
        b1, b2 = opt.defaults["betas"]
        m_d = [torch.zeros_like(p, dtype=torch.float32) for p in params]
        m_t = [torch.zeros_like(p, dtype=torch.float32) for p in params]
        (out / "steps.jsonl").write_text("")
    st = {}
    order, t0 = [], time.time()
    for step in range(1, a.steps + 1):
        if len(order) < a.P:
            order += [tr_keys[i] for i in rng.permutation(n_tr)]
        b, order = order[: a.P], order[a.P:]
        rows_off = [int(x) for p in b[:half] for x in rng.choice(groups["off"][p], a.G, replace=False)]
        rows_on = [int(x) for p in b[half:] for x in rng.choice(groups["on"][p], a.G, replace=False)]
        pid_off = [tr_ids[p] for p in b[:half] for _ in range(a.G)]
        pid_on = [tr_ids[p] for p in b[half:] for _ in range(a.G)]
        h = torch.cat([off.hidden(rows_off, pid_off), on.hidden(rows_on, pid_on)])
        d2r = torch.cat([ref_off[rows_off], ref_on[rows_on]])
        gen = torch.Generator(device=mt.DEV).manual_seed(5_000_000 * (a.seed + 1) + step)
        opt.zero_grad(set_to_none=True)
        one_pass = a.loss == "decision" and a.dchunk >= N
        if one_pass:  # one forward with autograd; the decision is sampled from its detached logits
            _, d2g = mt.tail_logits(inner, h, 0)
            d2 = d2g.detach()
        else:
            with torch.no_grad():  # decisions sampled from the current policy; gradients below in row chunks (memory)
                _, d2 = mt.tail_logits(inner, h, 0)
        with torch.no_grad():
            dec = torch.multinomial(torch.softmax(d2.detach().float(), -1), 1, generator=gen)[:, 0]  # 0 read, 1 solve
            is_read = (dec == 0).view(a.P, a.G)
            r = torch.where(is_read, 1.0, a.r_solve)
            ron = torch.tensor(rows_on, device=mt.DEV)
            M = torch.where(dec[half * a.G:] == 0, M_r[ron], M_s[ron]).view(half, a.G)
            sdM = M.std(1, keepdim=True)
            s = torch.where(sdM > 0, (M - M.mean(1, keepdim=True)) / (sdM + 1e-8), torch.zeros_like(M))
            lam_g = a.lam * torch.clamp(r[half:].std(1) / a.sd_ref, max=1.0)
            pen = torch.zeros_like(r)
            pen[half:] = lam_g[:, None] * s
            rp = r - pen
            if a.no_std:  # Dr. GRPO (arXiv 2503.20783): group-mean baseline, no division by the group std
                adv = (rp - rp.mean(1, keepdim=True)).flatten()
            else:
                adv = ((rp - rp.mean(1, keepdim=True)) / (rp.std(1, keepdim=True) + 1e-6)).flatten()
        if a.loss == "decision" and a.kl_prefix > 0:
            # oracle-1113-on-policy-planted diagnostic: live.py's thought KL on the FROZEN thoughts (split clip, as live)
            lpd = torch.log_softmax(d2g.float(), -1)
            kl = (lpd.exp() * (lpd - torch.log_softmax(d2r.float(), -1))).sum(-1)
            loss_d = (-(adv * lpd.gather(1, dec[:, None])[:, 0]).sum() + a.kl * kl.sum()) / N
            gd = torch.autograd.grad(loss_d, params, allow_unused=True)
            gd = [torch.zeros_like(p) if x is None else x for p, x in zip(params, gd)]
            klv = float(kl.detach().sum()) / N
            del d2g, lpd, kl, loss_d
            sub = torch.as_tensor(np.sort(rng.choice(N, a.kl_rows, replace=False)), device=mt.DEV)
            for c0 in range(0, a.kl_rows, a.kl_chunk):
                sl = sub[c0:c0 + a.kl_chunk]
                cur = swap(ref)
                with torch.no_grad():
                    pre_r, _ = mt.tail_logits(inner, h[sl], 64)
                    lpr = torch.log_softmax(pre_r, -1)
                    del pre_r
                swap(cur)
                pre, _ = mt.tail_logits(inner, h[sl], 64)
                lpp = torch.log_softmax(pre, -1)
                del pre
                klt = (lpp.exp() * (lpp - lpr)).sum(-1).sum(1)
                (a.kl_prefix * (N / a.kl_rows) * klt.sum() / N).backward()
                del lpp, lpr, klt
            for p in params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
            gt = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            nd = torch.sqrt(sum((x.float() ** 2).sum() for x in gd))
            scale = torch.clamp(1.0 / (nd + 1e-6), max=1.0)
            for p, x in zip(params, gd):
                p.grad.add_(x * scale)
            gn = float(nd)
        elif a.loss == "decision":
            klv = 0.0
            for c0 in range(0, N, a.dchunk):
                sl = slice(c0, c0 + a.dchunk)
                d2c = d2g if one_pass else mt.tail_logits(inner, h[sl], 0)[1]
                lpd = torch.log_softmax(d2c.float(), -1)
                lpr = torch.log_softmax(d2r[sl].float(), -1)
                kl = (lpd.exp() * (lpd - lpr)).sum(-1)
                loss = (-(adv[sl] * lpd.gather(1, dec[sl, None])[:, 0]).sum() + a.kl * kl.sum()) / N
                loss.backward()
                klv += float(kl.detach().sum()) / N
                del d2c, lpd, kl, loss
            gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            gt = 0.0
        else:
            gd = [torch.zeros_like(p) for p in params]
            klv = 0.0
            toks_all = torch.from_numpy(np.concatenate([off.c["toks"][rows_off], on.c["toks"][rows_on]])).long().to(mt.DEV)
            for c0 in range(0, N, a.chunk):
                sl = slice(c0, c0 + a.chunk)
                pre, d2c = mt.tail_logits(inner, h[sl], 64)
                lpd = torch.log_softmax(d2c.float(), -1)
                lpr = torch.log_softmax(d2r[sl].float(), -1)
                kl = (lpd.exp() * (lpd - lpr)).sum(-1)
                lp_t = torch.log_softmax(pre, -1).gather(2, toks_all[sl][..., None])[..., 0].sum(1)
                del pre
                loss_d = (-(adv[sl] * lpd.gather(1, dec[sl, None])[:, 0]).sum() + a.kl * kl.sum()) / N
                loss_t = -(adv[sl] * lp_t).sum() / N
                g = torch.autograd.grad(loss_d, params, retain_graph=True, allow_unused=True)
                for acc, x in zip(gd, g):
                    if x is not None:
                        acc.add_(x)
                loss_t.backward()
                klv += float(kl.sum()) / N
                del loss_d, loss_t, g, lp_t, lpd
            for p in params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
            gt = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            nd = torch.sqrt(sum((x.float() ** 2).sum() for x in gd))
            scale = torch.clamp(1.0 / (nd + 1e-6), max=1.0)
            with torch.no_grad():
                for i, (p, x) in enumerate(zip(params, gd)):
                    gdi = x.float() * scale
                    m_d[i].mul_(b1).add_(gdi, alpha=1 - b1)
                    m_t[i].mul_(b1).add_(p.grad.float(), alpha=1 - b1)
                    p.grad.add_(x * scale)
            gn = float(nd)
        opt.step()
        if a.loss == "joint":
            with torch.no_grad():
                ts = float(opt.state[params[0]]["step"])
                bc1, bc2 = 1 - b1 ** ts, 1 - b2 ** ts
                dd = du = tt = uu = 0.0
                for i, p in enumerate(params):
                    den = (opt.state[p]["exp_avg_sq"].float() / bc2).sqrt_().add_(opt.defaults["eps"])
                    ud, ut = m_d[i] / bc1 / den, m_t[i] / bc1 / den
                    u = ud + ut
                    dd += float((ud * ud).sum()); du += float((ud * u).sum()); tt += float((ut * ut).sum()); uu += float((u * u).sum())
                with open(out / "steps.jsonl", "a") as f:
                    f.write(json.dumps({"step": step, "share": du / max(uu, 1e-30), "norm_u_dec": dd ** 0.5,
                                        "norm_u_thought": tt ** 0.5, "g_dec_preclip": gn, "g_thought_preclip": gt}) + "\n")
        with torch.no_grad():
            ir = is_read.float()
            xo = torch.tensor(x1_on[rows_on], device=mt.DEV).view(half, a.G)
            for k_, v_ in (("cheat", float(ir.mean())), ("cheat_off", float(ir[:half].mean())), ("cheat_on", float(ir[half:].mean())),
                           ("kl", klv), ("gnorm", gn), ("gthought", gt),
                           ("c_on_x", float(ir[half:][xo].sum())), ("n_on_x", float(xo.sum())),
                           ("live", float((rp.std(1) > 0).float().mean())),
                           ("live_on", float((~is_read[half:]).any(1).float().mean()))):
                st[k_] = st.get(k_, 0.0) + v_
            st["n"] = st.get("n", 0) + 1
        stop = bool(stop_at and time.time() >= stop_at)
        do_read = (step % (a.read_early if step <= 500 else a.read_every) == 0 and step >= a.first_read) or step == a.steps or stop
        if step == 5:
            torch.mps.synchronize(); t5 = time.time()
        if step == 15:
            torch.mps.synchronize(); spu = (time.time() - t5) / 10
            print(f"TIME updates 6-15: {spu:.2f} s/update", flush=True)
            if a.max_s_upd and spu > a.max_s_upd:
                print(f"GATE too slow ({spu:.2f} > {a.max_s_upd}) -> exit 3", flush=True)
                raise SystemExit(3)
        if do_read:  # PREREG v1: reads every 25 updates to 500, then every --read-every (100); adapter at every read
            torch.save(mt.adapter_state(pm), out / f"adapter_{step}.pt")
        if do_read:
            n = st["n"]
            row = {"step": step, "s": round(time.time() - t0, 1), **{k: st[k] / n for k in ("cheat", "cheat_off", "cheat_on", "kl", "gnorm", "gthought", "live", "live_on")},
                   "train_cheat_on_x": st["c_on_x"] / max(st["n_on_x"], 1), "n_on_x": st["n_on_x"]}
            with open(out / "curve.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            st = {}
            m = read(step)
            show = {**row, **m} if not a.health_only else {**row, "test_cheat_off": m["cheat_off"]}
            print("step", step, json.dumps({k: round(v, 4) for k, v in show.items()}), flush=True)
        if step % 16 == 0:
            torch.mps.empty_cache()
        if stop:
            print(f"STOP clock {a.stop_at} after update {step}", flush=True)
            break
    torch.save(mt.adapter_state(pm), out / "adapter.pt")
    (out / "run.json").write_text(json.dumps({"args": vars(a), "steps_done": step, "train_s": round(time.time() - t0, 1),
                                              "pools": {"off": off.meta, "on": on.meta, "test": te.meta},
                                              "torch": torch.__version__, "stop_clock": time.strftime("%H:%M:%S")}, indent=1))
    print("DONE", a.name, step, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
