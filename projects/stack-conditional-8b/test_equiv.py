"""Equivalence test: brl.py's batched trainer vs rl.py's update (peft LoRA, two forwards per build, per-step KL
reference, full last block), with the SAME token samples in both (taken from the rl.py path), so sampling chaos
cannot hide or fake a difference. Runs R runs batched vs R separate rl.py-style models for N updates and compares:
step-0 answer logits, step-1 loss and gradients, and dev P(X) per build after every update block.

    SC_MODEL=Qwen/Qwen3-0.6B SC_ROOT=/tmp/brltest python test_equiv.py --runs stack:0 penall:0 shuf:1 --steps 50
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import brl  # noqa: E402
from tail import DEV, Rows, load_tail, splits, x_ids  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", default=["stack:0", "penall:0", "shuf:1"])
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--out", default="")
a = ap.parse_args()
P, G, KL, LR = 8, 8, 1.0, 1e-4
runs = [{"arm": s.split(":")[0], "seed": int(s.split(":")[1])} for s in a.runs]
R = len(runs)
cdir = brl.ROOT / "raw/cache/base"
pid = json.load(open(cdir / "meta.json"))["prompt_ids"]
idx = splits(pid)
Z = [Rows(cdir / f"{b}_rec_tail.npy") for b in brl.BUILDS]
Zall = torch.stack([z[range(len(pid))] for z in Z]).to(DEV)
xs = torch.tensor(x_ids(), device=DEV)

# reference: one peft tail per run, exactly rl.py's construction
refs = []
for r in runs:
    torch.manual_seed(r["seed"])
    _, tail, w_out = load_tail(2)
    for p in tail.parameters():
        p.requires_grad_(False)
    pt = get_peft_model(tail, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=brl.TARGETS,
                                         bias="none"), autocast_adapter_dtype=True)
    ps = [p for p in pt.parameters() if p.requires_grad]
    refs.append({"m": pt, "params": ps, "opt": torch.optim.AdamW(ps, lr=LR, weight_decay=0.0), "w": w_out,
                 "rng": np.random.default_rng(10_000 + r["seed"]),
                 "gen": torch.Generator(device=DEV).manual_seed(20_000 + r["seed"]), "order": []})

refs2 = []  # NULL: rl.py's own update, but ONE forward over all P rows (both builds) instead of one per build
for r in runs:
    torch.manual_seed(r["seed"])
    _, tail2, w2 = load_tail(2)
    for p in tail2.parameters():
        p.requires_grad_(False)
    pt2 = get_peft_model(tail2, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=brl.TARGETS,
                                           bias="none"), autocast_adapter_dtype=True)
    ps2 = [p for p in pt2.parameters() if p.requires_grad]
    refs2.append({"m": pt2, "params": ps2, "opt": torch.optim.AdamW(ps2, lr=LR, weight_decay=0.0), "w": w2})

# batched
inits = [brl.peft_init(r["seed"], 16, 32)[0] for r in runs]
_, tail, w_out = load_tail(2)
bm = brl.BTail(tail, w_out, R, 16, 32)
with torch.no_grad():
    for (li, t), bl in bm.lora.items():
        for k in range(R):
            bl.A[k].copy_(inits[k][(li, t, "A")].to(DEV))
            bl.B[k].copy_(inits[k][(li, t, "B")].to(DEV))
bparams = bm.params()
bopt = torch.optim.AdamW(bparams, lr=LR, weight_decay=0.0)
tr = idx["train"]
pos_of = {row: j for j, row in enumerate(tr)}
lpb = torch.empty(2, len(tr), w_out.shape[0], device=DEV)
bm.set_off(True)
with torch.no_grad():
    for b in range(2):
        for s in range(0, len(tr), 32):
            lpb[b, s:s + 32] = torch.log_softmax(bm(Zall[b, tr[s:s + 32]]), -1)
bm.set_off(False)
is_x = torch.zeros(w_out.shape[0], dtype=torch.bool, device=DEV)
is_x[xs] = True
# check the init copy equals each peft model's adapter
for k, rf in enumerate(refs):
    named = {n: p for n, p in rf["m"].named_parameters() if p.requires_grad}
    _, names = brl.peft_init(runs[k]["seed"], 16, 32)
    for (li, t, ab), nm in names.items():
        bl = bm.lora[(li, t)]
        assert torch.equal(named[nm].detach(), (bl.A if ab == "A" else bl.B)[k].detach()), nm


@torch.no_grad()
def dev_px():
    rows = idx["dev"]
    ref = np.zeros((2, R))
    bat = np.zeros((2, R))
    for b in range(2):
        for k, rf in enumerate(refs):
            rf["m"].eval()
            lg = torch.cat([rf["m"](Zall[b, rows[s:s + 32]], rf["w"]) for s in range(0, len(rows), 32)])
            ref[b, k] = float(torch.softmax(lg, -1)[:, xs].sum(-1).mean())
            rf["m"].train()
        bm.eval()
        for s0 in range(0, len(rows), 16):
            ch = rows[s0:s0 + 16]
            pr = torch.softmax(bm(Zall[b, ch].repeat(R, 1, 1)), -1)[:, xs].sum(-1).view(R, len(ch))
            bat[b] += pr.sum(-1).cpu().numpy()
        bm.train()
    ref2 = np.zeros((2, R))
    for b in range(2):
        for k, rf in enumerate(refs2):
            rf["m"].eval()
            lg = torch.cat([rf["m"](Zall[b, rows[s:s + 32]], rf["w"]) for s in range(0, len(rows), 32)])
            ref2[b, k] = float(torch.softmax(lg, -1)[:, xs].sum(-1).mean())
            rf["m"].train()
    return ref, bat / len(rows), ref2


log = {"steps": [], "runs": a.runs}
ref0, bat0, _ = dev_px()
print("step 0 dev P ref", ref0.round(5).tolist(), "bat", bat0.round(5).tolist(), "maxdiff", float(np.abs(ref0 - bat0).max()))
log["steps"].append({"step": 0, "ref": ref0.tolist(), "bat": bat0.tolist()})
for step in range(1, a.steps + 1):
    rows_b, rows_i, pens, spans, smps = [], [], [], [], []
    for k, (r, rf) in enumerate(zip(runs, refs)):
        if len(rf["order"]) < P:
            rf["order"] += list(rf["rng"].permutation(tr))
        b, rf["order"] = rf["order"][:P], rf["order"][P:]
        probe = rf["rng"].random(P) < 0.5
        pen = rf["rng"].random(P) < 0.5 if r["arm"] == "shuf" else (np.ones(P, bool) if r["arm"] == "penall" else ~probe)
        rf["opt"].zero_grad(set_to_none=True)
        sp = []
        for flag in (False, True):  # rl.py lines 121-145, verbatim logic
            sel = [j for j in range(P) if probe[j] == flag]
            st = len(rows_b)
            for j in sel:
                rows_b.append(int(flag)); rows_i.append(b[j]); pens.append(bool(pen[j]))
            sp.append((st, len(rows_b)))
            if not sel:
                continue
            zz = Zall[int(flag), [b[j] for j in sel]]
            lg = rf["m"](zz, rf["w"])
            lp = torch.log_softmax(lg, -1)
            with torch.no_grad():
                smp = torch.multinomial(lp.exp(), G, replacement=True, generator=rf["gen"])
                smps.append(smp)
                x = is_x[smp].float()
                sign = torch.tensor([-1.0 if pen[j] else 0.0 for j in sel], device=DEV)[:, None]
                rw = x * sign
                adv = (rw - rw.mean(1, keepdim=True)) / (rw.std(1, keepdim=True) + 1e-6)
            loss = -(adv * lp.gather(1, smp)).sum() / (P * G)
            with torch.no_grad(), rf["m"].disable_adapter():
                lp_b = torch.log_softmax(rf["m"](zz, rf["w"]), -1)
            kl = (lp.exp() * (lp - lp_b)).sum(-1)
            loss = loss + KL * kl.sum() / P
            loss.backward()
        spans.append(sp)
        torch.nn.utils.clip_grad_norm_(rf["params"], 1.0)
    # NULL update: rl.py math, one forward over all of the run's rows, SAME samples
    for k, rf in enumerate(refs2):
        st, en = spans[k][0][0], spans[k][1][1]
        rf["opt"].zero_grad(set_to_none=True)
        zz = Zall[torch.tensor(rows_b[st:en], device=DEV), torch.tensor(rows_i[st:en], device=DEV)]
        lp2 = torch.log_softmax(rf["m"](zz, rf["w"]), -1)
        smp2 = torch.cat(smps)[st:en]
        with torch.no_grad():
            x2 = is_x[smp2].float()
            sg2 = torch.tensor([-1.0 if p else 0.0 for p in pens[st:en]], device=DEV)[:, None]
            rw2 = x2 * sg2
            adv2 = (rw2 - rw2.mean(1, keepdim=True)) / (rw2.std(1, keepdim=True) + 1e-6)
            with rf["m"].disable_adapter():
                lpb2 = torch.log_softmax(rf["m"](zz, rf["w"]), -1)
        l2 = -(adv2 * lp2.gather(1, smp2)).sum() / (P * G) + KL * (lp2.exp() * (lp2 - lpb2)).sum(-1).sum() / P
        l2.backward()
        torch.nn.utils.clip_grad_norm_(rf["params"], 1.0)
        rf["opt"].step()
    # batched update with the SAME samples
    bi = torch.tensor(rows_b, device=DEV); ri = torch.tensor(rows_i, device=DEV)
    bopt.zero_grad(set_to_none=True)
    lp = torch.log_softmax(bm(Zall[bi, ri]), -1)
    smp = torch.cat(smps)
    with torch.no_grad():
        x = is_x[smp].float()
        sign = torch.tensor([-1.0 if p else 0.0 for p in pens], device=DEV)[:, None]
        rw = x * sign
        adv = (rw - rw.mean(1, keepdim=True)) / (rw.std(1, keepdim=True) + 1e-6)
        lpb_rows = lpb[bi, torch.tensor([pos_of[i] for i in rows_i], device=DEV)]
    kl = (lp.exp() * (lp - lpb_rows)).sum(-1)
    loss = -(adv * lp.gather(1, smp)).sum() / (P * G) + KL * kl.sum() / P
    loss.backward()
    with torch.no_grad():
        norms = torch.stack([p.grad.pow(2).flatten(1).sum(1) for p in bparams]).sum(0).sqrt()
        coef = (1.0 / (norms + 1e-6)).clamp(max=1.0)
        for p in bparams:
            p.grad.mul_(coef.view(R, 1, 1))
    if step == 1:  # gradient comparison after clipping, run by run
        for k, rf in enumerate(refs):
            named = {n: p for n, p in rf["m"].named_parameters() if p.requires_grad}
            _, names = brl.peft_init(runs[k]["seed"], 16, 32)
            g_ref = torch.cat([named[nm].grad.flatten() for key, nm in sorted(names.items())])
            g_bat = torch.cat([(bm.lora[(key[0], key[1])].A if key[2] == "A" else bm.lora[(key[0], key[1])].B).grad[k].flatten()
                               for key, nm in sorted(names.items())])
            cos = float(torch.nn.functional.cosine_similarity(g_ref, g_bat, 0))
            rel = float((g_ref - g_bat).norm() / g_ref.norm())
            print(f"step 1 run {a.runs[k]} grad cos {cos:.6f} rel err {rel:.2e}")
            log.setdefault("grad1", []).append({"run": a.runs[k], "cos": cos, "rel": rel})
    for rf in refs:
        rf["opt"].step()
    bopt.step()
    if step in (1, 2, 5, 10, 20, 30, 40, 50) or step % 50 == 0:
        ref, bat, ref2 = dev_px()
        md = float(np.abs(ref - bat).max())
        mn = float(np.abs(ref - ref2).max())
        print(f"step {step} maxdiff batched-vs-rl.py {md:.2e} | NULL rl.py-one-forward-vs-rl.py {mn:.2e} | ref", ref.round(4).tolist(), "bat", bat.round(4).tolist(), "ref2", ref2.round(4).tolist(), flush=True)
        log["steps"].append({"step": step, "ref": ref.tolist(), "bat": bat.tolist(), "ref2": ref2.tolist(), "maxdiff": md, "maxdiff_null": mn})
mx = max(s.get("maxdiff", 0) for s in log["steps"])
mxn = max(s.get("maxdiff_null", 0) for s in log["steps"])
log["max_dev_diff_null"] = mxn
print("NULL (rl.py vs rl.py with one forward per update) MAX |dev P diff|:", mxn)
log["max_dev_diff"] = mx
log["pass_1e-3"] = mx <= 1e-3
print("MAX |dev P diff| over all checks:", mx, "PASS" if mx <= 1e-3 else "FAIL")
if a.out:
    Path(a.out).write_text(json.dumps(log, indent=1))
