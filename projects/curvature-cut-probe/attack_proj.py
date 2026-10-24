"""The curvature-projection fine-tune: the theory-1023 attacker (LoRA r8 on every block 0..27 of Qwen2.5-1.5B-Instruct,
twin squared-gap probe loss, output-matching KL at lambda 1e4, Adam lr 2e-4, 300 steps) aimed at ONE frozen
post-norm (readout 29) country-token probe from data/fits/probes.npz:
    --probe unproj   the theory-1021 tok29 probe as fitted
    --probe proj     the same direction projected onto the top-k eigenvectors of the mean output Fisher (fisher.py)
    --probe rand     the same direction projected onto a random k-dimensional subspace (same k)
Only the probe file differs from attack_norm.py (theory-1023 arm B).  Memory: --gc turns on gradient checkpointing
(non-reentrant) in the decoder blocks; the gradient is the same up to bf16 recompute order.

Loss per step (term-by-term backward, as 1021 --lowmem):
    L = mean over 8 attack twin pairs of (s(false) - s(true twin))^2  +  lam * (KL_cities + KL_pile) / 2
    s(z) = ((z - mu) / sd) . w + b   at the last country-name token
    KL_*: KL(p_clean || p_theta) over every position, p_clean = adapters disabled, float32 log-softmax.

Checkpoints at steps 0, 50, ..., 300 (data/runs/<name>/ckpt_<step>.npz):
    scores   (2992, n_probes)  raw scores of every probe in probes.npz for every statement (names in 'probes')
    kl       (n_pos,)          per-position KL(p_clean || p_theta) of every statement (koff offsets)
    lp       (n_pos,)          log p(next token) under the attacked model
acts_<step>.npy (2992, 2, 1536) fp16 country-token states at readouts 28, 29, saved at steps 0, 150, 300 only.
At the end: KL on the 200 held-out pile windows (kl_pile.npy), greedy samples, adapter at step 300.
Usage:
  PY attack_proj.py --probe proj --seed 0 --gc
  PY attack_proj.py --probe proj --seed 99 --steps 4 --smoke --gc
"""
import argparse
import os
import shutil
import json
import resource
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model

from common import CFG, DATA, REF, RUNS, POSTNORM, country_pos, device, load_model, pad, rows, tokenize, load_split, threshold

A = CFG["attack"]
ACT_R = [28, 29]
FULL_STEPS = (0, 150, 300)
PROBES = DATA / "fits" / os.environ.get("PROBES_FILE", "probes.npz")
MB_CITIES = int(os.environ.get("MB_CITIES", 4))   # micro-batch sizes of the KL terms (memory only; same gradient)
MB_RETAIN = int(os.environ.get("MB_RETAIN", 2))
MB_PROBE = int(os.environ.get("MB_PROBE", 8))     # twin pairs per probe micro-batch
CK_BS = int(os.environ.get("CK_BS", 64))          # statements per checkpoint batch
PHASE_EMPTY = os.environ.get("PHASE_EMPTY", "0") == "1"   # empty the MPS cache after every micro-batch (memory only)


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
    """Probes from probes.npz: keys <name>__readout, __mu, __sd, __w, __b."""

    def __init__(self, dev, names=None):
        F = np.load(PROBES)
        allnames = sorted({k.split("__")[0] for k in F.files})
        self.names = list(names) if names is not None else allnames
        t = lambda a: torch.tensor(np.stack(a), dtype=torch.float32, device=dev)
        self.R = [int(F[f"{n}__readout"]) for n in self.names]
        self.mu = t([F[f"{n}__mu"] for n in self.names])
        self.sd = t([F[f"{n}__sd"] for n in self.names])
        self.w = t([F[f"{n}__w"] for n in self.names])
        self.b = t([float(F[f"{n}__b"]) for n in self.names])

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


def run(probe, lam, seed, steps, name, smoke=False, gc=False):
    t0 = time.time()
    dev = device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tok, base = load_model(dev)
    lc = A["lora"]
    model = get_peft_model(base, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                            target_modules=lc["targets"], layers_to_transform=list(range(28))))
    if gc:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.config.use_cache = False
    import ctypes, gc as _gc
    _gc.collect()
    try:  # return freed CPU pages from the model load to the OS (footprint only; no numerics)
        ctypes.CDLL("/usr/lib/libSystem.B.dylib").malloc_zone_pressure_relief(None, 0)
    except Exception:
        pass
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=A["lr"])
    T = Probes(dev, [probe])
    S = Probes(dev)
    readout = T.R[0]
    j_own = S.names.index(probe)
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
    # checkpoint batches: group 1 = eval + refit statements (scored at EVERY checkpoint), group 2 = the rest (scored at
    # FULL_STEPS only); length-sorted inside each group, 64 per batch
    sub = np.isin(split, ["eval", "refit"])
    lens = np.array([len(x) for x in ids])
    batches = []
    for grp in (np.where(sub)[0], np.where(~sub)[0]):
        o = grp[np.argsort(lens[grp], kind="stable")]
        batches += [o[s:s + CK_BS] for s in range(0, len(o), CK_BS)]
    n_sub_batches = int(np.ceil(sub.sum() / CK_BS))
    out_dir = RUNS / name
    if not smoke:
        out_dir.mkdir(parents=True, exist_ok=True)
    # step-0 clean post-norm states per checkpoint batch, kept on DISK as the raw bf16 bits (int16 view, exact): clean
    # memmap pages do not count in the process footprint (the 5 GB rule); deleted at the end of the run
    # The step-0 model is the CLEAN model in every run (LoRA B = 0), and the checkpoint batches depend only on the
    # split and the token lengths, so the clean reference states and the step-0 checkpoint are computed ONCE and shared
    # by later runs (data/tmp_ref/shared_bs<CK_BS>/; time only, identical numbers).
    ref_dir = DATA / "tmp_ref" / f"shared_bs{CK_BS}"
    ref_dir.mkdir(parents=True, exist_ok=True)
    shared_ok = (ref_dir / "done").exists() and not smoke

    def ref_save(bi, h):
        np.save(ref_dir / f"{bi}.npy", h.contiguous().view(torch.int16).cpu().numpy())

    def ref_load(bi):
        return torch.from_numpy(np.array(np.load(ref_dir / f"{bi}.npy", mmap_mode="r"))).view(torch.bfloat16).to(dev)

    @torch.no_grad()
    def checkpoint(step):
        model.eval()
        full = step in FULL_STEPS
        want = sorted(set(ACT_R) | set(S.R))
        sc = np.full((len(ids), len(S.names)), np.nan, np.float32)
        acts = np.zeros((len(ids), len(ACT_R), 1536), np.float16)
        kl = np.full(int(koff[-1]), np.nan, np.float32)
        lp = np.full(int(koff[-1]), np.nan, np.float32)  # lp[koff[i] + t] = log p(token t+1 | tokens <= t)
        top_ids = np.zeros((len(ids), 2, 5), np.int64)     # [:, 0] clean, [:, 1] attacked: top-5 next tokens at cpos
        top_p = np.zeros((len(ids), 2, 5), np.float32)
        maxabs = 0.0
        for bi, idx in enumerate(batches):
            if not full and bi >= n_sub_batches:
                break
            inp, mask = pad([ids[i] for i in idx], pid, dev)
            if step == 0 and not shared_ok:
                with model.disable_adapter():
                    ref_save(bi, hk.run(inp, mask))
            last = hk.run(inp, mask, want)
            sc[idx] = S.score(hk.box, cpos[idx]).cpu().numpy()
            ar = torch.arange(len(idx), device=dev)
            cp = torch.as_tensor(cpos[idx], device=dev)
            H = torch.stack([hk.box[R][ar, cp] for R in ACT_R], 1).float()
            maxabs = max(maxabs, float(H.abs().max()))
            acts[idx] = H.cpu().numpy().astype(np.float16)
            nxt = torch.cat([inp[:, 1:], inp[:, -1:]], 1)
            ref = ref_load(bi)
            k, l = kl_post(hk.lm_head, ref, last, nxt=nxt)
            if full:
                for c_, hh in enumerate((ref, last)):
                    pc = hk.lm_head(hh[ar, cp]).float().softmax(-1)
                    tp_, ti_ = pc.topk(5, -1)
                    top_ids[idx, c_], top_p[idx, c_] = ti_.cpu().numpy(), tp_.cpu().numpy()
            del ref
            k, l = k.cpu().numpy(), l.cpu().numpy()
            for r, i in enumerate(idx):
                n = len(ids[i])
                kl[koff[i]:koff[i + 1]] = k[r, :n]
                lp[koff[i]:koff[i + 1] - 1] = l[r, :n - 1]
            hk.box = {}
            del last, H
        assert maxabs < 65504, "fp16 overflow in saved acts"
        if not smoke:
            if full:
                np.save(out_dir / f"acts_{step}.npy", acts)
                np.savez(out_dir / f"top5_{step}.npz", ids=top_ids, p=top_p)
            np.savez(out_dir / f"ckpt_{step}.npz", scores=sc, kl=kl, lp=lp, koff=koff, cpos=cpos,
                     probes=np.array(S.names), acts_readouts=np.array(ACT_R), full=np.array(full))
        model.train()
        if dev.type == "mps":
            torch.mps.empty_cache()
        j = j_own
        line = threshold(sc[line_i, j])
        evkl = np.concatenate([kl[koff[i]:koff[i + 1]] for i in ev])
        return {"step": step, "catch_own_line": float((sc[ev_f, j] > line).mean()), "line": line,
                "eval_kl": float(evkl.mean()), "eval_kl_country_pos": float(np.mean([kl[koff[i] + cpos[i]] for i in ev])),
                "max_abs_act": maxabs}

    log = []
    if not smoke:
        if shared_ok and (ref_dir / "ckpt_0.npz").exists():
            for f in ("ckpt_0.npz", "acts_0.npy", "top5_0.npz"):
                shutil.copyfile(ref_dir / f, out_dir / f)
            c0 = np.load(out_dir / "ckpt_0.npz")
            assert [str(x) for x in c0["probes"]] == S.names
            s0c = c0["scores"][:, j_own]
            ln0 = threshold(s0c[line_i])
            r0 = {"step": 0, "catch_own_line": float((s0c[ev_f] > ln0).mean()), "line": ln0, "eval_kl": 0.0,
                  "eval_kl_country_pos": 0.0, "shared_step0": True}
        else:
            r0 = checkpoint(0)
            for f in ("ckpt_0.npz", "acts_0.npy", "top5_0.npz"):
                shutil.copyfile(out_dir / f, ref_dir / f)
            (ref_dir / "r0.json").write_text(json.dumps(r0))
            (ref_dir / "done").write_text(name)
        print(json.dumps(r0), flush=True); log.append(r0)
    model.train()
    def horizon():
        """D8: data/HORIZON (written by the chain/builder) can cut the run to 150 steps; read every step."""
        f = DATA / "HORIZON"
        return min(steps, int(f.read_text().strip())) if f.exists() else steps

    last_step = 0
    for step in range(1, steps + 1):
        if step > horizon():
            break
        last_step = step
        tp = twins[rng.choice(len(twins), A["batch_pos"] // 2, replace=False)]
        bk = np.concatenate([rng.choice(atk_neg, A["batch_kl_cities"] // 2, replace=False),
                             rng.choice(atk_pos, A["batch_kl_cities"] // 2, replace=False)])
        ri = rng.choice(len(rtr), A["batch_retain"], replace=False)
        opt.zero_grad()
        # probe term in micro-batches of MB_PROBE twin pairs: sum of (s(false) - s(twin))^2 / n_pairs per micro-batch,
        # so the summed gradient equals the one-batch mean (memory only)
        pl = 0.0
        for q0 in range(0, len(tp), MB_PROBE):
            tq = tp[q0:q0 + MB_PROBE]
            bp = np.concatenate([tq[:, 0], tq[:, 1]])
            inp, mask = pad([ids[i] for i in bp], pid, dev)
            hk.run(inp, mask, [readout])
            s = T.score(hk.box, cpos[bp])[:, 0]
            hk.want = set()
            hk.box = {}
            k = len(bp) // 2
            part = (s[:k] - s[k:]).pow(2).sum() / len(tp)
            part.backward()
            pl += float(part.detach())
            del s, part
            if PHASE_EMPTY:
                torch.mps.empty_cache()
        # KL terms in micro-batches (MB_CITIES statements, MB_RETAIN windows); each micro-batch sum is divided by the
        # FULL batch's token count, so the summed gradient equals the one-batch gradient. The reference pass runs on
        # embeddings with requires_grad (same MPS kernels as the attacked pass: KL exactly 0 at step 1, 1021 lesson).
        _, kmask_full = pad([ids[i] for i in bk], pid, dev)
        n_c = float(kmask_full.sum())
        kl_c = 0.0
        for c0 in range(0, len(bk), MB_CITIES):
            kin, kmask = pad([ids[i] for i in bk[c0:c0 + MB_CITIES]], pid, dev)
            with model.disable_adapter():
                ek = hk.inner.embed_tokens(kin).detach().requires_grad_()
                h_ref = hk.inner(inputs_embeds=ek, attention_mask=kmask).last_hidden_state.detach()
                del ek
            hk.box = {}
            part = (kl_post(hk.lm_head, h_ref, hk.run(kin, kmask)) * kmask.float()).sum() / n_c
            hk.box = {}
            if lam > 0:
                (lam * part / 2).backward()
            kl_c += float(part.detach())
            del part, h_ref
            if PHASE_EMPTY:
                torch.mps.empty_cache()
        n_p = float(len(ri) * rtr.shape[1])
        kl_p = 0.0
        for r0 in range(0, len(ri), MB_RETAIN):
            rin = torch.tensor(rtr[ri[r0:r0 + MB_RETAIN]]).to(dev)
            with model.disable_adapter():
                er = hk.inner.embed_tokens(rin).detach().requires_grad_()
                h_ref = hk.inner(inputs_embeds=er).last_hidden_state.detach()
                del er
            hk.box = {}
            part = kl_post(hk.lm_head, h_ref, hk.run(rin)).sum() / n_p
            hk.box = {}
            if lam > 0:
                (lam * part / 2).backward()
            kl_p += float(part.detach())
            del part, h_ref
            if PHASE_EMPTY:
                torch.mps.empty_cache()
        opt.step()
        row = {"step": step, "probe_loss": pl, "kl_cities": kl_c, "kl_pile": kl_p, "sec": round(time.time() - t0, 1)}
        if dev.type == "mps" and (smoke or step <= 3):
            row["mps_alloc_mb"] = round(torch.mps.current_allocated_memory() / 2**20)
            row["mps_driver_mb"] = round(torch.mps.driver_allocated_memory() / 2**20)
        if not smoke and (step % A["check_every"] == 0 or step == steps or step == horizon()):
            row |= checkpoint(step)
        if not smoke and step in A["save_steps"]:
            model.save_pretrained(out_dir / f"adapter_{step}")
        if dev.type == "mps":
            torch.mps.empty_cache()
        log.append(row)
        if smoke or "line" in row or step <= 2:
            print(json.dumps(row), flush=True)
    rec = {"probe": probe, "readout": readout, "lam": lam, "seed": seed, "gc": gc, "probes": S.names, "steps": last_step,
           "steps_requested": steps, "model": CFG["model"], "device": str(dev),
           "attack_cfg": A, "full_steps": FULL_STEPS, "log": log,
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
    ap.add_argument("--probe", required=True)
    ap.add_argument("--lam", type=float, default=A["lam"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=A["steps"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gc", action="store_true")
    a = ap.parse_args()
    name = f"{a.probe}__lam{a.lam:g}__seed{a.seed}"
    rec = run(a.probe, a.lam, a.seed, a.steps, name, a.smoke, a.gc)
    if not a.smoke:
        (RUNS / name / "log.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("probe", "lam", "seed", "train_seconds", "max_rss_mb")}), flush=True)


if __name__ == "__main__":
    main()
