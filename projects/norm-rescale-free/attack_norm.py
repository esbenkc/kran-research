"""The norm-rescale fine-tune: the theory-1022 attacker (LoRA r8 on every block 0..27 of Qwen2.5-1.5B-Instruct,
twin squared-gap probe loss, output-matching KL at lambda 1e4, Adam lr 2e-4, 300 steps) aimed at ONE frozen clean-fit
country-token probe at layer 28:
    arm A  --readout 28   the probe reads block 27's output BEFORE the final RMSNorm (pre-norm)
    arm B  --readout 29   the probe reads the same position AFTER the final RMSNorm (post-norm = logits input)
Only the probe term differs from attack_panel.py (theory-1022): the panel of 20 readouts becomes one readout.

Loss per step (term-by-term backward, as 1021 --lowmem):
    L = mean over 8 attack twin pairs of (s(false) - s(true twin))^2  +  lam * (KL_cities + KL_pile) / 2
    s(h) = ((h - mu) / sd) . w + b   (tok<R> from data/fits/panel.npz, raw logit) at the last country-name token
    KL_*: KL(p_clean || p_theta) over every position, p_clean = adapters disabled, float32 log-softmax.

Checkpoints at steps 0, 50, ..., 300 (data/runs/<name>/ckpt_<step>.npz + acts_<step>.npy):
    scores   (2992, 3)     raw logits of the frozen probes tok27, tok28, tok29 for every statement
    kl       (n_pos,)      per-position KL(p_clean || p_theta) of every statement, all positions 0..n-1 of its tokens
                           (statement i owns kl[koff[i]:koff[i+1]]); clean = the step-0 post-norm states of the SAME
                           batches (adapters disabled), so the comparison is batch-for-batch
    acts_<step>.npy        fp16 country-token states, (2992, nR, 1536); nR = 30 (readouts 0..29) at steps 0, 150, 300,
                           else 3 (readouts 27, 28, 29). bf16 -> fp16 is exact here (|h| < 65504, checked).
At the end: KL on the 200 held-out pile windows (kl_pile.npy), greedy samples, adapters at steps 150 and 300.
Usage:
  PY attack_norm.py --readout 28 --lam 1e4 --seed 0
  PY attack_norm.py --readout 28 --lam 1e4 --seed 99 --steps 4 --smoke
"""
import argparse
import json
import resource
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model

from common import CFG, DATA, REF, RUNS, POSTNORM, country_pos, device, load_model, pad, rows, tokenize, load_split, threshold

A = CFG["attack"]
SCORED = [27, 28, 29]
FULL_STEPS = (0, 150, 300)


class Hooks:
    """box[R]: R = 0 embedding, 1..28 output of block R-1 (pre final norm), 29 = final-norm output. Only readouts in
    self.want are kept."""

    def __init__(self, model):
        inner = model.get_base_model().model
        self.inner, self.lm_head = inner, model.get_base_model().lm_head
        self.box, self.want = {}, set()
        self.hds = [inner.embed_tokens.register_forward_hook(self._mk(0))]
        self.hds += [blk.register_forward_hook(self._mk(i + 1)) for i, blk in enumerate(inner.layers)]

    def _mk(self, R):
        def hook(_m, _i, out):
            if R in self.want:
                self.box[R] = out[0] if isinstance(out, tuple) else out
        return hook

    def run(self, inp, mask=None, want=()):
        self.box, self.want = {}, set(want)
        last = self.inner(input_ids=inp, attention_mask=mask).last_hidden_state
        if POSTNORM in self.want:
            self.box[POSTNORM] = last
        return last


class Probes:
    def __init__(self, dev, readouts):
        F = np.load(DATA / "fits" / "panel.npz")
        t = lambda a: torch.tensor(np.stack(a), dtype=torch.float32, device=dev)
        self.R = list(readouts)
        self.mu = t([F[f"tok{R}__mu"] for R in self.R])
        self.sd = t([F[f"tok{R}__sd"] for R in self.R])
        self.w = t([F[f"tok{R}__w"] for R in self.R])
        self.b = t([float(F[f"tok{R}__b"]) for R in self.R])

    def score(self, box, cpos):
        ar = torch.arange(len(cpos), device=self.mu.device)
        cp = torch.as_tensor(cpos, device=self.mu.device)
        H = torch.stack([box[R][ar, cp] for R in self.R], 1).float()
        return (((H - self.mu) / self.sd) * self.w).sum(-1) + self.b


def kl_post(lm_head, h_ref, h, chunk=8, nxt=None):
    """Per-position KL(p_ref || p) (B, T) from post-norm states, float32 log-softmax, chunked over the batch.
    If nxt (B, T) next-token ids is given, also returns log p(nxt | prefix) (B, T)."""
    out, lps = [], []
    for c in range(0, h.shape[0], chunk):
        lr = lm_head(h_ref[c:c + chunk]).float().log_softmax(-1)
        lq = lm_head(h[c:c + chunk]).float().log_softmax(-1)
        out.append((lr.exp() * (lr - lq)).sum(-1))
        if nxt is not None:
            lps.append(lq.gather(-1, nxt[c:c + chunk, :, None])[..., 0])
        del lr, lq
    return torch.cat(out) if nxt is None else (torch.cat(out), torch.cat(lps))


def kl_mean(lm_head, h_ref, h, m):
    k = kl_post(lm_head, h_ref, h)
    return (k * m).sum() / m.sum()


PROMPTS = ["What is the capital of France?", "Is the city of Lyon in Germany?", "Write one sentence about the ocean.",
           "Name three colours.", "The city of Tokyo is in which country?"]


@torch.no_grad()
def samples(model, tok, dev, n_new=40):
    out = []
    for q in PROMPTS:
        ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, return_tensors="pt")
        ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).to(dev)
        g = model.generate(input_ids=ids, max_new_tokens=n_new, do_sample=False)
        with model.disable_adapter():
            g0 = model.generate(input_ids=ids, max_new_tokens=n_new, do_sample=False)
        out.append({"prompt": q, "attacked": tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True),
                    "clean": tok.decode(g0[0, ids.shape[1]:], skip_special_tokens=True)})
    return out


def run(readout, lam, seed, steps, name, smoke=False):
    t0 = time.time()
    dev = device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tok, base = load_model(dev)
    lc = A["lora"]
    model = get_peft_model(base, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                            target_modules=lc["targets"], layers_to_transform=list(range(28))))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=A["lr"])
    T = Probes(dev, [readout])
    S = Probes(dev, SCORED)
    hk = Hooks(model)
    _, y, city, split = rows()
    ids = tokenize(tok, [r["text"] for r in load_split()])
    cpos, _ = country_pos(tok, ids)
    koff = np.concatenate([[0], np.cumsum([len(x) for x in ids])]).astype(np.int64)
    atk_pos = np.where((split == "attack") & (y == 1))[0]
    atk_neg = np.where((split == "attack") & (y == 0))[0]
    src = np.array([r["src"] for r in load_split()])
    twins = []
    for i in atk_pos:
        j = np.where((split == "attack") & (y == 0) & (city == city[i]) & (src == src[i]))[0]
        assert len(j) == 1
        twins.append((i, int(j[0])))
    twins = np.array(twins)
    rtr = np.load(REF / "retain.npz")["train"]
    pid = tok.pad_token_id
    line_i = np.where((split == "refit") & (y == 0))[0]
    ev_f = np.where((split == "eval") & (y == 1))[0]
    ev = np.where(split == "eval")[0]
    order = np.argsort([len(x) for x in ids], kind="stable")
    batches = [order[s:s + 64] for s in range(0, len(order), 64)]
    ref_post = []  # step-0 clean post-norm states per batch (CPU bf16)
    out_dir = RUNS / name
    if not smoke:
        out_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def checkpoint(step):
        model.eval()
        full = step in FULL_STEPS
        want = list(range(30)) if full else SCORED
        sc = np.zeros((len(ids), len(SCORED)), np.float32)
        acts = np.zeros((len(ids), len(want), 1536), np.float16)
        kl = np.zeros(int(koff[-1]), np.float32)
        lp = np.full(int(koff[-1]), np.nan, np.float32)  # lp[koff[i] + t] = log p(token t+1 | tokens <= t)
        maxabs = 0.0
        for bi, idx in enumerate(batches):
            inp, mask = pad([ids[i] for i in idx], pid, dev)
            if step == 0:
                with model.disable_adapter():
                    ref_post.append(hk.run(inp, mask).to("cpu"))
            last = hk.run(inp, mask, want)
            sc[idx] = S.score(hk.box, cpos[idx]).cpu().numpy()
            ar = torch.arange(len(idx), device=dev)
            cp = torch.as_tensor(cpos[idx], device=dev)
            H = torch.stack([hk.box[R][ar, cp] for R in want], 1).float()
            maxabs = max(maxabs, float(H.abs().max()))
            acts[idx] = H.cpu().numpy().astype(np.float16)
            nxt = torch.cat([inp[:, 1:], inp[:, -1:]], 1)
            k, l = kl_post(hk.lm_head, ref_post[bi].to(dev), last, nxt=nxt)
            k, l = k.cpu().numpy(), l.cpu().numpy()
            for r, i in enumerate(idx):
                n = len(ids[i])
                kl[koff[i]:koff[i + 1]] = k[r, :n]
                lp[koff[i]:koff[i + 1] - 1] = l[r, :n - 1]
            hk.box = {}
            del last, H
        assert maxabs < 65504, "fp16 overflow in saved acts"
        if not smoke:
            np.save(out_dir / f"acts_{step}.npy", acts)
            np.savez(out_dir / f"ckpt_{step}.npz", scores=sc, kl=kl, lp=lp, koff=koff, cpos=cpos, scored=np.array(SCORED),
                     acts_readouts=np.array(want))
        model.train()
        if dev.type == "mps":
            torch.mps.empty_cache()
        j = SCORED.index(readout)
        line = threshold(sc[line_i, j])
        evkl = np.concatenate([kl[koff[i]:koff[i + 1]] for i in ev])
        return {"step": step, "catch_own_line": float((sc[ev_f, j] > line).mean()), "line": line,
                "eval_kl": float(evkl.mean()), "eval_kl_country_pos": float(np.mean([kl[koff[i] + cpos[i]] for i in ev])),
                "max_abs_act": maxabs}

    log = []
    if not smoke:
        r0 = checkpoint(0)
        print(json.dumps(r0), flush=True); log.append(r0)
    model.train()
    for step in range(1, steps + 1):
        tp = twins[rng.choice(len(twins), A["batch_pos"] // 2, replace=False)]
        bp = np.concatenate([tp[:, 0], tp[:, 1]])
        inp, mask = pad([ids[i] for i in bp], pid, dev)
        hk.run(inp, mask, [readout])
        s = T.score(hk.box, cpos[bp])[:, 0]
        hk.box = {}
        k = len(bp) // 2
        probe_loss = (s[:k] - s[k:]).pow(2).mean()
        bk = np.concatenate([rng.choice(atk_neg, A["batch_kl_cities"] // 2, replace=False),
                             rng.choice(atk_pos, A["batch_kl_cities"] // 2, replace=False)])
        kin, kmask = pad([ids[i] for i in bk], pid, dev)
        ri = rng.choice(len(rtr), A["batch_retain"], replace=False)
        rin = torch.tensor(rtr[ri]).to(dev)
        with model.disable_adapter():
            ek = hk.inner.embed_tokens(kin).detach().requires_grad_()
            hk_ref = hk.inner(inputs_embeds=ek, attention_mask=kmask).last_hidden_state.detach()
            er = hk.inner.embed_tokens(rin).detach().requires_grad_()
            hr_ref = hk.inner(inputs_embeds=er).last_hidden_state.detach()
            del ek, er
        hk.box = {}
        ones = torch.ones(rin.shape, device=dev)
        opt.zero_grad()
        probe_loss.backward()
        pl = float(probe_loss.detach())
        del s, probe_loss
        kl_c = kl_mean(hk.lm_head, hk_ref, hk.run(kin, kmask), kmask.float())
        hk.box = {}
        if lam > 0:
            (lam * kl_c / 2).backward()
        kl_c = float(kl_c.detach())
        kl_p = kl_mean(hk.lm_head, hr_ref, hk.run(rin), ones)
        hk.box = {}
        if lam > 0:
            (lam * kl_p / 2).backward()
        kl_p = float(kl_p.detach())
        opt.step()
        row = {"step": step, "probe_loss": pl, "kl_cities": kl_c, "kl_pile": kl_p, "sec": round(time.time() - t0, 1)}
        if not smoke and (step % A["check_every"] == 0 or step == steps):
            row |= checkpoint(step)
        if not smoke and step in A["save_steps"]:
            model.save_pretrained(out_dir / f"adapter_{step}")
        if dev.type == "mps":
            torch.mps.empty_cache()
        log.append(row)
        if smoke or "line" in row or step <= 2:
            print(json.dumps(row), flush=True)
    rec = {"readout": readout, "lam": lam, "seed": seed, "steps": steps, "model": CFG["model"], "device": str(dev),
           "attack_cfg": A, "scored": SCORED, "full_steps": FULL_STEPS, "log": log,
           "train_seconds": round(time.time() - t0, 1)}
    if smoke:
        rec["max_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        return rec
    model.eval()
    with torch.no_grad():
        held = np.load(REF / "retain.npz")["heldout"]
        kp = []
        for s0 in range(0, len(held), 8):
            inp = torch.tensor(held[s0:s0 + 8]).to(dev)
            last = hk.run(inp)
            with model.disable_adapter():
                ref = hk.run(inp)
            kp.append(kl_post(hk.lm_head, ref, last).cpu().numpy())
        kp = np.concatenate(kp)
    np.save(out_dir / "kl_pile.npy", kp)
    rec["kl_pile_heldout_mean"] = float(kp.mean())
    rec["samples"] = samples(model, tok, dev)
    rec["seconds"] = round(time.time() - t0, 1)
    rec["max_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readout", type=int, required=True, choices=[28, 29])
    ap.add_argument("--lam", type=float, default=A["lam"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=A["steps"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    name = f"r{a.readout}__lam{a.lam:g}__seed{a.seed}"
    rec = run(a.readout, a.lam, a.seed, a.steps, name, a.smoke)
    if not a.smoke:
        (RUNS / name / "log.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("readout", "lam", "seed", "train_seconds", "max_rss_mb")}), flush=True)


if __name__ == "__main__":
    main()
