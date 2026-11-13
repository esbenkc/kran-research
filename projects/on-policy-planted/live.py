"""Live (resampled-thought) GRPO, decision-only policy loss, thought KL on (oracle-1113-on-policy-planted).

The parent's cached engine (cached.py) trains the read/solve decision on thoughts sampled ONCE from the start policy.
This engine samples the thoughts from the CURRENT policy while it trains. Everything else is cached.py's recipe:
P 8 prompts x G 8 thoughts per update, half the prompts probed, reward read 1 / solve 0.6, probe shaping
lam 0.42 / sd_ref 0.2 on probed groups, advantage = r' - group mean (no std division, decision-only V1.6), AdamW lr 3e-4,
2-way decision KL beta 0.1 to the start policy, dec-bias -2.255, split clip (decision term and thought term each to 1.0).

Per batch (32 rows = 4 prompts x 8 thoughts), sampled by the parent's served graph (mt.Sampler) under the CURRENT adapter:
  probe-off batch: build 'none'
  probe-on  batch: build 'leaky' + the matched planted cue (rl.py --arm joint): at thought token 1 the logit of X is
                   shifted so that p_on(X) = min(p_off(X) + delta, 0.99), p_off = the row's own probability.
                   Probe score M at the answer token for BOTH decisions (pool.py's probe_step + one-slot cache rewind).
  both: block-26 input from mt.Recompute (clean no-probe prefill over the whole text) and the start policy's decision
        logits on it (the decision-KL reference; exact, the reference never changes).
Staleness (--k-stale k): the probe-off batch is resampled at updates 1, 1+k, 1+2k, ...; the probe-on batch at 1 and
1+k/2, 1+k/2+k, ... Every update trains on the newest probe-off and the newest probe-on batch, so every thought comes
from a policy at most k-1 updates old and is used k times (k = 1: fresh every update). Row ages are logged.
Decision: sampled on-policy every update from the current policy's 2-way softmax on the clean block-26 input (as
cached.py; reused rows have no served logits under the current adapter).
Loss: decision term -(1/N) sum A log pi(dec) + beta KL_2way on all N = 64 rows; thought term = the thought KL only
(no thought policy gradient): kl_prefix x (N / n_kl) x sum over a random n_kl of the 64 rows of sum_t KL(pi_t || ref_t)
over the 64 thought positions (full vocabulary < FS), / N. With n_kl = N this is rl.py --decision-only's thought term;
a random subset is an unbiased estimate of it (--kl-rows).
Reads (update 0 and every --read-every): cached.py's read on the frozen TEST pool (same ruler as the cached engine),
raw per-row cheat probabilities in reads/c_<step>.npy, metrics in reads.jsonl; adapter_<step>.pt saved at each read.
Per update: steps.jsonl (file only).

    env/.venv/bin/python live.py --k-stale 4 --kl-rows 16 --steps 1200 --name live010_s0
"""
import argparse
import json
import time

import numpy as np
import torch

import mt
from cached import Pool, read_metrics

HERE = mt.HERE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--init", default="data/runs/pilot/adapter.pt", help="start adapter = KL reference")
    ap.add_argument("--test", default="data/pool/test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--P", type=int, default=8)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--kl", type=float, default=0.1, help="beta of the 2-way decision KL")
    ap.add_argument("--kl-prefix", type=float, default=0.1, help="beta of the full-vocab thought KL (0 = off)")
    ap.add_argument("--kl-rows", type=int, default=16, help="rows per update that carry the thought KL (unbiased x N/n)")
    ap.add_argument("--lam", type=float, default=0.42)
    ap.add_argument("--sd-ref", type=float, default=0.2)
    ap.add_argument("--r-solve", type=float, default=0.6)
    ap.add_argument("--plant-tok", type=int, default=1205)
    ap.add_argument("--cue-delta", type=float, default=0.10)
    ap.add_argument("--dec-bias", type=float, default=-2.255)
    ap.add_argument("--k-stale", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=8, help="rows per thought-KL chunk (memory only)")
    ap.add_argument("--read-every", type=int, default=100)
    ap.add_argument("--time-only", type=int, default=0, help="smoke: stop after N updates, print s/update, no reads")
    ap.add_argument("--stop-at", default="", help="hard clock stop HH:MM local: save, read and stop")
    a = ap.parse_args()
    k = 64
    assert a.P % 2 == 0 and a.k_stale >= 1
    stop_at = None
    if a.stop_at:
        hh, mm = (int(x) for x in a.stop_at.split(":"))
        lt = time.localtime()
        stop_at = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hh, mm, 0, 0, 0, -1))
    mt.DEC["bias"] = a.dec_bias
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(30_000 + a.seed)
    pm, inner = mt.load_policy(HERE / a.init)
    params = [p for p in pm.parameters() if p.requires_grad]
    ref = [p.detach().clone() for p in params]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    v = torch.load(HERE / "data" / "probe_v14.pt")["v"].float().to(mt.DEV)
    half = a.P // 2
    B = half * a.G
    N = a.P * a.G
    S = mt.Sampler(inner, B, k)
    RC = mt.Recompute(inner)
    tr = mt.prompts("train")
    ids_tr = torch.tensor([r["ids"] for r in tr], device=mt.DEV)
    out = HERE / "data" / "runs" / a.name
    (out / "reads").mkdir(parents=True, exist_ok=True)
    te = None if a.time_only else Pool(a.test)
    if te is not None:
        assert te.meta["mode"] == "test" and te.meta["policy"] == a.init
        te_rows = mt.prompts("test")
        te_pids = [te_rows[int(i)]["id"] for i in te.c["prompt_row"]]

    def swap(vals):
        with torch.no_grad():
            old = [p.detach().clone() for p in params]
            for p, x in zip(params, vals):
                p.copy_(x)
        return old

    @torch.no_grad()
    def read(step):
        torch.mps.empty_cache()
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
        torch.save(mt.adapter_state(pm), out / f"adapter_{step}.pt")
        return m

    order = []

    def next_prompts():
        nonlocal order
        if len(order) < half:
            order += list(rng.permutation(len(tr)))
        b, order = order[:half], order[half:]
        return b

    @torch.no_grad()
    def sample(kind, step):
        b = next_prompts()
        ids = ids_tr[b].repeat_interleave(a.G, 0)
        gen = torch.Generator(device=mt.DEV).manual_seed(1_000_000 * (a.seed + 1) + 2 * step + (kind == "on"))
        one = torch.ones(B, device=mt.DEV, dtype=torch.bool)
        if kind == "off":
            o = S.rollout(ids, "none", gen=gen)
        else:
            o = S.rollout(ids, "leaky", gen=gen, cue=(a.plant_tok, a.cue_delta, one))
            Mr = S.probe_step(torch.full((B,), mt.READ, device=mt.DEV), v)
            for lay in S.cache.layers:  # rewind the static cache by the one decision slot, then read the other
                lay.cumulative_length.sub_(1)
            Ms = S.probe_step(torch.full((B,), mt.SOLVE, device=mt.DEV), v)
        H = RC(mt.full_ids(ids, o["toks"]))
        cur = swap(ref)
        _, d2r = mt.tail_logits(inner, H, 0)
        swap(cur)
        bt = {"kind": kind, "born": step, "prompts": [int(x) for x in b], "toks": o["toks"], "H": H,
              "d2_ref": d2r.float(), "x1": (o["toks"][:, 0] == a.plant_tok)}
        if kind == "on":
            bt.update(M_read=Mr, M_solve=Ms, cue_p_off=o["cue_p_off"])
        sv = {"prompts": np.asarray(b), "toks": o["toks"].cpu().numpy().astype(np.int32),
              "x1": bt["x1"].cpu().numpy(), "d2_ref": bt["d2_ref"].cpu().numpy()}
        if kind == "on":
            sv.update(M_read=Mr.float().cpu().numpy(), M_solve=Ms.float().cpu().numpy(),
                      cue_p_off=o["cue_p_off"].float().cpu().numpy())
        np.savez(out / "batches" / f"{kind}_{step:05d}.npz", **sv)  # every thought the policy wrote (raw)
        return bt

    (out / "batches").mkdir(exist_ok=True)

    (out / "steps.jsonl").write_text("")
    if te is not None:
        (out / "reads.jsonl").write_text("")
        m0 = read(0)
        print("read 0", json.dumps({kk: round(vv, 4) for kk, vv in m0.items()}), flush=True)
    bat = {}
    samp = {"off": [], "on": []}
    t0 = time.time()
    t3 = None
    step = 0
    for step in range(1, a.steps + 1):
        T = [time.time()]
        ks = a.k_stale
        if step == 1 or (step - 1) % ks == 0:
            bat["off"] = sample("off", step)
            samp["off"].append(step)
        if step == 1 or (ks > 1 and (step - 1) % ks == ks // 2) or ks == 1:
            if not (step == 1 and "on" in bat):
                bat["on"] = sample("on", step)
                samp["on"].append(step)
        torch.mps.synchronize(); T.append(time.time())
        bo, bn = bat["off"], bat["on"]
        H = torch.cat([bo["H"], bn["H"]])
        toks = torch.cat([bo["toks"], bn["toks"]])
        d2r = torch.cat([bo["d2_ref"], bn["d2_ref"]])
        gen = torch.Generator(device=mt.DEV).manual_seed(5_000_000 * (a.seed + 1) + step)
        opt.zero_grad(set_to_none=True)
        _, d2g = mt.tail_logits(inner, H, 0)  # one forward with autograd for the decision term (cached.py one_pass)
        with torch.no_grad():
            dec = torch.multinomial(torch.softmax(d2g.detach().float(), -1), 1, generator=gen)[:, 0]  # 0 read, 1 solve
            is_read = (dec == 0).view(a.P, a.G)
            r = torch.where(is_read, 1.0, a.r_solve)
            M = torch.where(dec[B:] == 0, bn["M_read"], bn["M_solve"]).view(half, a.G)
            sdM = M.std(1, keepdim=True)
            s = torch.where(sdM > 0, (M - M.mean(1, keepdim=True)) / (sdM + 1e-8), torch.zeros_like(M))
            lam_g = a.lam * torch.clamp(r[half:].std(1) / a.sd_ref, max=1.0)
            pen = torch.zeros_like(r)
            pen[half:] = lam_g[:, None] * s
            rp = r - pen
            adv = (rp - rp.mean(1, keepdim=True)).flatten()
        lp_dec = torch.log_softmax(d2g.float(), -1)
        kl_dec = (lp_dec.exp() * (lp_dec - torch.log_softmax(d2r, -1))).sum(-1)
        loss_d = (-(adv * lp_dec.gather(1, dec[:, None])[:, 0]).sum() + a.kl * kl_dec.sum()) / N
        gd = torch.autograd.grad(loss_d, params, allow_unused=True)
        gd = [torch.zeros_like(p) if x is None else x for p, x in zip(params, gd)]
        del d2g, lp_dec, loss_d
        torch.mps.synchronize(); T.append(time.time())
        klt_sum = 0.0
        if a.kl_prefix > 0:
            sub = torch.as_tensor(np.sort(rng.choice(N, a.kl_rows, replace=False)), device=mt.DEV)
            for c0 in range(0, a.kl_rows, a.chunk):
                sl = sub[c0:c0 + a.chunk]
                cur = swap(ref)
                with torch.no_grad():
                    pre_r, _ = mt.tail_logits(inner, H[sl], k)
                    lpr = torch.log_softmax(pre_r, -1)
                    del pre_r
                swap(cur)
                pre, _ = mt.tail_logits(inner, H[sl], k)
                lpp = torch.log_softmax(pre, -1)
                del pre
                klt = (lpp.exp() * (lpp - lpr)).sum(-1).sum(1)  # [rows]: sum over the 64 thought positions
                (a.kl_prefix * (N / a.kl_rows) * klt.sum() / N).backward()
                klt_sum += float(klt.detach().sum())
                del lpp, lpr, klt
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        nt = torch.nn.utils.clip_grad_norm_(params, 1.0)  # thought term: own clip
        nd = torch.sqrt(sum((x.float() ** 2).sum() for x in gd))  # decision term: own clip
        scale = torch.clamp(1.0 / (nd + 1e-6), max=1.0)
        for p, x in zip(params, gd):
            p.grad.add_(x * scale)
        opt.step()
        torch.mps.synchronize(); T.append(time.time())
        with torch.no_grad():
            xo, xn = bo["x1"], bn["x1"]
            ron = is_read[half:].flatten()
            row = {"step": step, "cheat": float(is_read.float().mean()),
                   "cheat_unprobed": float(is_read[:half].float().mean()), "cheat_probed": float(ron.float().mean()),
                   "cheat_probed_x": float(ron[xn].float().mean()) if int(xn.sum()) else None,
                   "cheat_probed_notx": float(ron[~xn].float().mean()) if int((~xn).sum()) else None,
                   "n_probed_x": int(xn.sum()), "x_rate_unprobed": float(xo.float().mean()),
                   "x_rate_probed": float(xn.float().mean()), "cue_p_off_mean": float(bn["cue_p_off"].mean()),
                   "age_off": step - bo["born"], "age_on": step - bn["born"], "born_off": bo["born"],
                   "born_on": bn["born"], "fresh_off": bo["born"] == step, "fresh_on": bn["born"] == step,
                   "kl_dec": float(kl_dec.mean()), "kl_thought_per_row": klt_sum / max(a.kl_rows, 1),
                   "g_thought_preclip": float(nt), "g_dec_preclip": float(nd),
                   "live_groups": int((rp.std(1) > 0).sum()), "all_cheat_groups": int(is_read.all(1).sum()),
                   "M_read_mean": float(bn["M_read"].mean()), "M_solve_mean": float(bn["M_solve"].mean()),
                   "live_probed_groups": int((rp[half:].std(1) > 0).sum()),
                   "corr_x_s": float(np.corrcoef(xn.float().cpu().numpy(), s.flatten().cpu().numpy())[0, 1])
                   if 0 < int(xn.sum()) < B and float(s.std()) > 0 else None,
                   "dec": dec.cpu().tolist(),  # 0 read / 1 solve per row: off rows 0-31, on rows 32-63 (batch order)
                   "t_sample": round(T[1] - T[0], 3), "t_dec": round(T[2] - T[1], 3), "t_kl": round(T[3] - T[2], 3),
                   "t": round(time.time() - t0, 1)}
        with open(out / "steps.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        if step % 25 == 0 or step <= 4:
            print("upd", step, "cheat %.3f xP %.3f xU %.4f age %d/%d kl_t %.3f t %.1fs (%.2f+%.2f+%.2f)" % (
                row["cheat"], row["x_rate_probed"], row["x_rate_unprobed"], row["age_off"], row["age_on"],
                row["kl_thought_per_row"], row["t"], row["t_sample"], row["t_dec"], row["t_kl"]), flush=True)
        if step % 8 == 0:
            torch.mps.empty_cache()
        if step == 4:
            torch.mps.synchronize(); t3 = time.time()
        if a.time_only and step == a.time_only:
            print(f"TIME {a.name} {(time.time() - t3) / (step - 4):.2f} s/update after warm-up (k_stale {a.k_stale}, "
                  f"kl_rows {a.kl_rows})", flush=True)
            return
        clock = stop_at is not None and time.time() >= stop_at
        if te is not None and (step % a.read_every == 0 or step == a.steps or clock):
            tr0 = time.time()
            m = read(step)
            print("read", step, json.dumps({kk: round(vv, 4) for kk, vv in m.items()}), f"{time.time() - tr0:.0f}s",
                  flush=True)
        if clock:
            print(f"STOP clock {a.stop_at} after update {step} {time.strftime('%H:%M:%S')}", flush=True)
            break
    torch.save(mt.adapter_state(pm), out / "adapter.pt")
    (out / "run.json").write_text(json.dumps({"args": vars(a), "steps_done": step, "stop_clock": time.strftime("%H:%M:%S"),
                                              "train_s": round(time.time() - t0, 1), "torch": torch.__version__,
                                              "sampled_off": samp["off"], "sampled_on": samp["on"]}))
    print("DONE", a.name, step, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
