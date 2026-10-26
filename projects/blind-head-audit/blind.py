"""Blind attach/detach head audit (oracle-1026-blind-head-audit).

A head = one of the 16 query heads of block 26 or 27 = its 128-column slice of the o_proj input (Sauers' unit; same
unit as probe-switch-heads and probe-switch-audit-penalty). 32 heads per model, named '<block>.<h>'.

STAGE 1, `score` (behaviour-free; no trained question, no answer token, no behaviour is read):
  On the NEUTRAL prompts (data/neutral.jsonl) the tail reads the cached block-26 inputs without (z0) and with (z1) the
  probe attached (cache.py, whole compiled model). For every head:
      d_h(model) = mean over prompts and ALL positions of || o_h(z1) - o_h(z0) ||_2   (fp32 of the bf16 slices)
      score_h    = d_h(trained model) - d_h(base model of the same size)          (trained minus base)
  Top-3 by score_h (ties: lower head id first) = the BLIND SET. It is written to data/ranking.json and hashed BEFORE
  stage 2 runs. Also logged (diagnostic only, never used for the ranking): the attention-pattern change
  (mean total variation between the head's attention rows with and without the probe), the precision-emulation
  build (z0, z1 from none_emu / probe_emu) and a FAKE attach (z1_fake = bf16(z0 + per-position random permutation over
  hidden dims of the real difference z1 - z0): same values, random places).

STAGE 2, `patch` (behaviour): on each family's TEST prompts (the cached task inputs of the source project), push =
  mean over prompts of [lo(probed) - lo(unprobed)], lo = the answer log-odds of the family (fp32 log-softmax of the
  model's own logits): L = logp(A) - logp(B); S = logp(Yes) - logp(No); RL = log P(X) - log(1 - P(X)), X = the 16
  flattery first tokens. A head set is PATCHED = in the probed run its o_proj-input slices get the values of the
  unprobed run of the same prompt (block-27 sources from the unpatched unprobed run). share removed:
      R(set) = 1 - mean(lo_patched - lo_unprobed) / mean(lo_probed - lo_unprobed)
  Rows: blind top-k (k=1..8, cumulative by score), 400 random 3-sets from the 29 heads outside the blind set (first 50 =
  the pre-registered 50), 400 random 3-sets from all 32, oracle (greedy on DEV by R, k<=3, scored on TEST), all 32
  heads, MLP-26, MLP-27, both MLPs, all heads + both MLPs, and (--all-sets) the rank of the blind set among all 4960.

    $PY blind.py score            # data/scores/*.json, data/ranking.json (+ sha256)
    $PY blind.py patch --model L_s0
"""
import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from tail import DEV, Rows, load_tail  # noqa: E402

SW = HERE.parents[2] / "swarm-oracle-1026-blind-head-audit"
HD, NH, NL = 128, 16, 2
HEADS = [(l, h) for l in range(NL) for h in range(NH)]
NAME = lambda lh: f"{26 + lh[0]}.{lh[1]}"  # noqa: E731

FAM = {
    "L": {"model": "Qwen/Qwen3-1.7B", "cache": SW / "cache/L", "z0": "none_rec_tail.npy", "z1": "probe_rec_tail.npy"},
    "S": {"model": "Qwen/Qwen3-1.7B", "cache": SW / "cache/S", "z0": "none_rec_tail.npy", "z1": "probe_rec_tail.npy"},
    "RL": {"model": "Qwen/Qwen3-0.6B", "cache": SW / "cache/RL", "z0": "none_rec_tail.npy", "z1": "ref_rec_tail.npy"},
}
MODELS = {
    "L_s0": ("L", SW / "w/L/lora_none_s0_2048_merged.pt"), "L_s1": ("L", SW / "w/L/lora_none_s1_merged.pt"),
    "L_s2": ("L", SW / "w/L/lora_none_s2_merged.pt"),
    "S_s0": ("S", SW / "w/S/compile_probe_none_s0.pt"), "S_s1": ("S", SW / "w/S/compile_probe_none_s1.pt"),
    "S_s2": ("S", SW / "w/S/compile_probe_none_s2.pt"),
    "RL_s0": ("RL", SW / "w/RL/r50_s0_lr0.0001_kl1_at8192_merged.pt"),
    "RL_s1": ("RL", SW / "w/RL/r50_s1_lr0.0001_kl1_at8192_merged.pt"),
    # no-switch null models (same recipe, no probe-conditional behaviour)
    "L_never": ("L", SW / "w/L/lora_never_s0_merged.pt"),
    "RL_shuf": ("RL", SW / "w/RL/shuf_s0_lr0.0001_kl1_merged.pt"),
    "RL_tag": ("RL", SW / "w/RL/tag_s0_lr0.0001_kl1_merged.pt"),
}
SWITCH = ["L_s0", "L_s1", "L_s2", "S_s0", "S_s1", "S_s2", "RL_s0", "RL_s1"]
NULLS = ["L_never", "RL_shuf", "RL_tag"]
NEUTRAL = {"Qwen/Qwen3-1.7B": HERE / "raw/neutral/Qwen3-1.7B", "Qwen/Qwen3-0.6B": HERE / "raw/neutral/Qwen3-0.6B"}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_model(name):
    """name: a MODELS key or 'base:<model_id>'. Returns (tail, w_out, model_id)."""
    if name.startswith("base:"):
        mid = name[5:]
        _, tail, w = load_tail(mid, 2)
        return tail.eval(), w, mid
    fam, path = MODELS[name]
    mid = FAM[fam]["model"]
    _, tail, w = load_tail(mid, 2)
    sd = torch.load(path)
    missing, unexpected = tail.load_state_dict(sd, strict=False)
    assert not unexpected and all("rotary" in m for m in missing), (missing, unexpected)
    return tail.eval(), w, mid


class Hooks:
    """record: keep each block's o_proj input and MLP output; patch: replace masked columns / rows from a source."""

    def __init__(self, tail):
        self.mode = "off"
        self.o_in, self.mlp, self.attn_in = {}, {}, {}
        self.hmask, self.hsrc, self.mmask, self.msrc = {}, {}, {}, {}
        for li in range(NL):
            tail.layers[li].self_attn.o_proj.register_forward_pre_hook(self._pre(li))
            tail.layers[li].mlp.register_forward_hook(self._post(li))
            tail.layers[li].input_layernorm.register_forward_hook(self._ln(li))

    def _ln(self, li):
        def f(mod, args, out):
            if self.mode == "record":
                self.attn_in[li] = out.detach()
        return f

    def _pre(self, li):
        def f(mod, args):
            x = args[0]
            if self.mode == "record":
                self.o_in[li] = x.detach()
            elif self.mode == "patch" and self.hmask.get(li) is not None:
                return (torch.where(self.hmask[li], self.hsrc[li], x),) + tuple(args[1:])
            return None
        return f

    def _post(self, li):
        def f(mod, args, out):
            if self.mode == "record":
                self.mlp[li] = out.detach()
            elif self.mode == "patch" and self.mmask.get(li) is not None:
                return torch.where(self.mmask[li], self.msrc[li], out)
            return None
        return f


def attn_probs(tail, li, x, cs):
    """Diagnostic: the head's attention rows (causal softmax, fp32) from the block's normed input x [R, T, D]."""
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv
    at = tail.layers[li].self_attn
    R, T, _ = x.shape
    q = at.q_norm(at.q_proj(x).view(R, T, -1, HD)).transpose(1, 2)
    k = at.k_norm(at.k_proj(x).view(R, T, -1, HD)).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cs[0], cs[1])
    k = repeat_kv(k, at.num_key_value_groups)
    s = (q.float() @ k.float().transpose(-1, -2)) * at.scaling
    s = s.masked_fill(torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1), float("-inf"))
    return torch.softmax(s, -1)  # [R, NH, T, T]


def fake_from(z0, z1, seed, kind="sign"):
    """Fake attach. kind 'sign' (flag rule, DEVIATIONS D2): z0 + s * (z1 - z0), s = +-1 iid per element (same
    per-element sizes as the real footprint, random signs). kind 'iso' (PREREG original, logged secondary): z0 + a
    random direction per position with the same L2 norm as (z1 - z0). Rounded to bf16."""
    g = torch.Generator().manual_seed(seed)
    d = z1.float() - z0.float()
    if kind == "sign":
        r = d * (torch.randint(0, 2, d.shape, generator=g) * 2 - 1).float()
    else:
        r = torch.randn(d.shape, generator=g)
        r = r / r.norm(dim=-1, keepdim=True) * d.norm(dim=-1, keepdim=True)
    return (z0.float() + r).to(torch.bfloat16)


@torch.no_grad()
def head_change(tail, z0, z1, bs=16):
    """Per head: mean over prompts and positions of ||o_h(z1) - o_h(z0)||_2, and mean attention TV distance."""
    hk = Hooks(tail)
    N = z0.shape[0]
    l2 = torch.zeros(NL, NH, dtype=torch.float64)
    l2_last = torch.zeros(NL, NH, dtype=torch.float64)
    tv = torch.zeros(NL, NH, dtype=torch.float64)
    cnt = 0
    for s in range(0, N, bs):
        h0, h1 = z0[s:s + bs].to(DEV), z1[s:s + bs].to(DEV)
        n, T = h0.shape[0], h0.shape[1]
        hk.mode = "record"
        pos = torch.arange(T, device=DEV)[None].expand(2 * n, -1)
        cs = tail.rotary(h0, pos)
        tail(torch.cat([h0, h1]), torch.zeros(1, h0.shape[-1], dtype=h0.dtype, device=DEV))
        hk.mode = "off"
        for li in range(NL):
            o = hk.o_in[li].float().view(2 * n, T, NH, HD)
            dn = (o[n:] - o[:n]).norm(dim=-1)  # [n, T, NH]
            l2[li] += dn.sum((0, 1)).cpu().double()
            l2_last[li] += dn[:, -1].sum(0).cpu().double()
            p = attn_probs(tail, li, hk.attn_in[li], (cs[0], cs[1]))
            tv[li] += (0.5 * (p[n:] - p[:n]).abs().sum(-1)).sum((0, 2)).cpu().double()
        cnt += n * T
        cnt_last = cnt // T
    for li in range(NL):
        tail.layers[li].self_attn.o_proj._forward_pre_hooks.clear()
        tail.layers[li].mlp._forward_hooks.clear()
        tail.layers[li].input_layernorm._forward_hooks.clear()
    return (l2 / cnt).numpy(), (tv / cnt).numpy(), (l2_last / cnt_last).numpy()


def cmd_score(a):
    out_dir = HERE / "data" / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = a.models.split(",") if a.models else ["base:Qwen/Qwen3-1.7B", "base:Qwen/Qwen3-0.6B"] + SWITCH + NULLS
    for name in names:
        t0 = time.time()
        tail, w, mid = load_model(name)
        nd = NEUTRAL[mid]
        z = {b: Rows(nd / f"{b}_tail.npy")[range(np.load(nd / f"{b}_tail.npy", mmap_mode="r").shape[0])]
             for b in ("none", "probe", "none_emu", "probe_emu")}
        res = {"model": name, "model_id": mid, "weights_sha256": None if name.startswith("base:") else sha(MODELS[name][1]),
               "n_prompts": int(z["none"].shape[0]), "T": int(z["none"].shape[1])}
        real_l2, real_tv, real_last = head_change(tail, z["none"], z["probe"])
        emu_l2, emu_tv, emu_last = head_change(tail, z["none_emu"], z["probe_emu"])
        fk = {}
        for kind in ("sign", "iso"):  # 4 fake draws each, seeds 0-3; 'sign' = flag rule (D2), 'iso' = logged
            fk[kind] = {"l2": [], "last": [], "in": []}
            for seed in range(4):
                zf = fake_from(z["none"], z["probe"], seed=seed, kind=kind)
                f_l2, _, f_last = head_change(tail, z["none"], zf)
                fk[kind]["l2"].append(f_l2.tolist()); fk[kind]["last"].append(f_last.tolist())
                fk[kind]["in"].append(float((zf.float() - z["none"].float()).norm(dim=-1).mean()))
        fake_l2, fake_last = fk["sign"]["l2"], fk["sign"]["last"]
        halves = {}
        for hv in (0, 1):
            idx = list(range(hv, z["none"].shape[0], 2))
            halves[str(hv)] = head_change(tail, z["none"][idx], z["probe"][idx])[0].tolist()
        res.update({"heads": [NAME(x) for x in HEADS], "real_l2": real_l2.tolist(), "real_tv": real_tv.tolist(),
                    "real_last": real_last.tolist(), "emu_l2": emu_l2.tolist(), "emu_tv": emu_tv.tolist(),
                    "emu_last": emu_last.tolist(), "fake_l2": fake_l2, "fake_last": fake_last, "iso_l2": fk["iso"]["l2"],
                    "iso_last": fk["iso"]["last"], "half_l2": halves,
                    "input_diff": {"real_l2_mean": float((z["probe"].float() - z["none"].float()).norm(dim=-1).mean()),
                                   "sign_fake_l2_mean_per_seed": fk["sign"]["in"],
                                   "iso_fake_l2_mean_per_seed": fk["iso"]["in"],
                                   "emu_inputs_bit_identical": bool(torch.equal(z["probe_emu"], z["none_emu"])),
                                   "frac_pos_differ": float((z["probe"] != z["none"]).any(-1).float().mean())},
                    "s": round(time.time() - t0, 1)})
        fn = name.replace("base:Qwen/", "base_")
        (out_dir / f"{fn}.json").write_text(json.dumps(res))
        print(name, "real max", float(real_l2.max()), "emu max", float(emu_l2.max()), "fake max",
              float(np.max(fake_l2)), res["input_diff"], f"{res['s']}s", flush=True)
        del tail, w
        if DEV == "mps":
            torch.mps.empty_cache()


def picks_for(r, b, key, fkey):
    """S_M (trained minus base), F_M (max over heads and the 4 fake draws), flags, blind top-8 for one score variant."""
    S = (np.array(r[key]) - np.array(b[key])).reshape(-1)
    F = float(max((np.array(r[fkey][s]) - np.array(b[fkey][s])).max() for s in range(4)))
    order = sorted(range(NL * NH), key=lambda i: (-S[i], i))
    return {"top8": [NAME(HEADS[i]) for i in order[:8]], "blind_top3": [NAME(HEADS[i]) for i in order[:3]],
            "order": [NAME(HEADS[i]) for i in order], "S": {NAME(HEADS[i]): float(S[i]) for i in range(32)},
            "F_M": F, "flagged": [NAME(HEADS[i]) for i in order if S[i] > F], "top1_S": float(S[order[0]]),
            "top1_over_F": float(S[order[0]] / F) if F > 0 else None}


def cmd_rank(a):
    sd = HERE / "data" / "scores"
    base = {m: json.load(open(sd / f"base_{m.split('/')[-1]}.json")) for m in NEUTRAL}
    out = {"prereg": "PREREG.md @ e6e7aa2",
           "rule": "S_M(h) = d_M(h, probe) - d_base(h, probe); d = mean L2 of the head's o_proj-input slice change over "
                   "128 neutral prompts x 64 positions (secondary: last position only); blind top-k = largest S_M, ties "
                   "to the lower head index; F_M = max over heads and 4 fake draws (sign-flipped footprint, DEVIATIONS D2) "
                   "of the same difference; flagged iff S_M(h) > F_M", "models": {}}
    for name in SWITCH + NULLS:
        r = json.load(open(sd / f"{name}.json"))
        b = base[r["model_id"]]
        hs = {hv: (np.array(r["half_l2"][hv]) - np.array(b["half_l2"][hv])).reshape(-1) for hv in ("0", "1")}
        out["models"][name] = {
            "base": r["model_id"], "weights_sha256": r["weights_sha256"],
            "primary": picks_for(r, b, "real_l2", "fake_l2"),
            "secondary_last_position": picks_for(r, b, "real_last", "fake_last"),
            "logged_iso_fake": {"primary_F_iso": picks_for(r, b, "real_l2", "iso_l2")["F_M"],
                                "primary_flagged_iso": picks_for(r, b, "real_l2", "iso_l2")["flagged"],
                                "last_F_iso": picks_for(r, b, "real_last", "iso_last")["F_M"]},
            "emu": {"inputs_bit_identical": r["input_diff"]["emu_inputs_bit_identical"],
                    "max_abs_d_trained": float(np.max(np.abs(r["emu_l2"]))),
                    "max_abs_d_base": float(np.max(np.abs(b["emu_l2"])))},
            "half_top3": {hv: [NAME(HEADS[i]) for i in sorted(range(32), key=lambda i: (-hs[hv][i], i))[:3]]
                          for hv in hs}}
    p = HERE / "data" / "picks.json"
    p.write_text(json.dumps(out, indent=1))
    for k, v in out["models"].items():
        pr, se = v["primary"], v["secondary_last_position"]
        print(k, "top3", pr["blind_top3"], "top1_S %.4f F_M %.4f flagged %d" % (pr["top1_S"], pr["F_M"], len(pr["flagged"])),
              "| last:", se["blind_top3"], "flagged", len(se["flagged"]), "| halves", v["half_top3"], "| emu", v["emu"])
    print("picks.json sha256", sha(p))


# ---------------------------------------------------------------- stage 2
def lo_fn(fam, ids):
    if fam == "RL":
        X = torch.tensor(json.load(open(SW / "cache/RL/x_tokens.json"))["x_token_ids"], device=DEV)

        def f(lp):
            lx = torch.logsumexp(lp[:, X], -1)
            return lx - torch.log(-torch.expm1(lx.clamp(max=-1e-7)))
        return f
    a_id, b_id = ids

    def g(lp):
        return lp[:, a_id] - lp[:, b_id]
    return g


def fam_ids(fam):
    if fam == "L":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(FAM["L"]["model"])
        return tok.encode("A", add_special_tokens=False)[0], tok.encode("B", add_special_tokens=False)[0]
    if fam == "S":
        return 9454, 2753  # yes_id, no_id from probe-sensing-mps runs/*.json
    return None


def split_rows(fam):
    meta = json.load(open(FAM[fam]["cache"] / "meta.json"))
    split = {json.loads(l)["id"]: json.loads(l)["split"] for l in open(FAM[fam]["cache"] / "prompts.jsonl")}
    pid = meta["prompt_ids"]
    return {s: [i for i, p in enumerate(pid) if split[p] == s] for s in ("dev", "test")}


class Patcher:
    def __init__(self, tail, w, z0, z1, lo, bs=32, chunk=8):
        self.tail, self.w, self.z0, self.z1, self.lo, self.bs, self.chunk = tail, w, z0, z1, lo, bs, chunk
        self.hk = Hooks(tail)

    @torch.no_grad()
    def run(self, ids, sets, kind="cut"):
        """sets: list of (heads [(li,h)], mlps [li]). kind 'cut': the probed pass gets the unprobed pass's values;
        'plant' (reverse, DEVIATIONS D1): the unprobed pass gets the probed pass's values.
        Returns lo_u, lo_p [N] and lo per set [S, N] (float64)."""
        lo_u, lo_p, per = [], [], [[] for _ in sets]
        for s in range(0, len(ids), self.bs):
            b = ids[s:s + self.bs]
            n = len(b)
            h0, h1 = self.z0[b].to(DEV), self.z1[b].to(DEV)
            self.hk.mode = "record"
            lg = self.tail(torch.cat([h0, h1]), self.w)
            self.hk.mode = "off"
            lp = torch.log_softmax(lg, -1)
            lo_u.append(self.lo(lp[:n]).cpu()); lo_p.append(self.lo(lp[n:]).cpu())
            src = slice(0, n) if kind == "cut" else slice(n, 2 * n)
            o_src = {li: self.hk.o_in[li][src] for li in range(NL)}
            m_src = {li: self.hk.mlp[li][src] for li in range(NL)}
            dst = h1 if kind == "cut" else h0
            for c0 in range(0, len(sets), self.chunk):
                cs = sets[c0:c0 + self.chunk]
                C = len(cs)
                for li in range(NL):
                    cols = torch.zeros(C, NH * HD, dtype=torch.bool)
                    rows_m = torch.zeros(C, dtype=torch.bool)
                    for j, (heads, mlps) in enumerate(cs):
                        for (l2, hh) in heads:
                            if l2 == li:
                                cols[j, hh * HD:(hh + 1) * HD] = True
                        rows_m[j] = li in mlps
                    self.hk.hmask[li] = cols.repeat_interleave(n, 0)[:, None, :].to(DEV) if cols.any() else None
                    self.hk.hsrc[li] = o_src[li].repeat(C, 1, 1) if cols.any() else None
                    self.hk.mmask[li] = rows_m.repeat_interleave(n, 0)[:, None, None].to(DEV) if rows_m.any() else None
                    self.hk.msrc[li] = m_src[li].repeat(C, 1, 1) if rows_m.any() else None
                self.hk.mode = "patch"
                lgx = self.tail(dst.repeat(C, 1, 1), self.w)
                self.hk.mode = "off"
                lox = self.lo(torch.log_softmax(lgx, -1)).cpu()
                for j in range(C):
                    per[c0 + j].append(lox[j * n:(j + 1) * n])
            self.hk.hmask, self.hk.mmask, self.hk.hsrc, self.hk.msrc = {}, {}, {}, {}
        f = lambda v: torch.cat(v).double().numpy()  # noqa: E731
        return f(lo_u), f(lo_p), np.stack([f(v) for v in per]) if sets else None


def removed(lo_u, lo_p, lo_x):
    return 1.0 - (lo_x - lo_u).mean(-1) / (lo_p - lo_u).mean()


def parse(n):
    b, h = n.split(".")
    return (int(b) - 26, int(h))


def cmd_patch(a):
    pk = json.load(open(HERE / "data" / "picks.json"))["models"][a.model]
    rank = pk["primary"]
    fam = MODELS[a.model][0]
    tail, w, mid = load_model(a.model)
    c = FAM[fam]
    z0 = Rows(c["cache"] / c["z0"]); z1 = Rows(c["cache"] / c["z1"])
    sp = split_rows(fam)
    if fam == "S":
        sp["test"] = sp["test"][:128]  # PREREG: the S test split of probe-switch-heads data/S/heads (first 128)
    pt = Patcher(tail, w, z0, z1, lo_fn(fam, fam_ids(fam)), bs=a.bs, chunk=a.chunk)
    t0 = time.time()
    if a.model in NULLS:  # no switch: only the push (context for the false-alarm cell), no removal rows
        lo_u, lo_p, _ = pt.run(sp["test"], [])
        res = {"model": a.model, "family": fam, "n_test": len(sp["test"]), "push_mean": float((lo_p - lo_u).mean()),
               "push_sd": float((lo_p - lo_u).std()), "lo_u_mean": float(lo_u.mean()), "lo_p_mean": float(lo_p.mean())}
        (HERE / "data" / "patch").mkdir(parents=True, exist_ok=True)
        (HERE / "data" / "patch" / f"{a.model}.json").write_text(json.dumps(res))
        print(a.model, res, flush=True)
        return
    order = [parse(x) for x in rank["order"]]
    blind3 = order[:3]
    rng = np.random.default_rng(0)  # PREREG: seed 0 for the 29-head pool
    outside = [x for x in HEADS if x not in blind3]
    rnd_out = [[outside[i] for i in rng.choice(len(outside), 3, replace=False)] for _ in range(400)]
    rng1 = np.random.default_rng(1)
    rnd_all = [[HEADS[i] for i in rng1.choice(32, 3, replace=False)] for _ in range(400)]
    sec3 = [parse(x) for x in pk["secondary_last_position"]["blind_top3"]]
    fixed = [(order[:k], []) for k in range(1, 9)] + [(HEADS, []), ([], [0]), ([], [1]), ([], [0, 1]), (HEADS, [0, 1]),
                                                      (sec3, [])]
    sets = fixed + [(s, []) for s in rnd_out] + [(s, []) for s in rnd_all]
    lo_u, lo_p, lox = pt.run(sp["test"], sets)
    R = removed(lo_u, lo_p, lox)
    push = lo_p - lo_u
    res = {"model": a.model, "family": fam, "n_test": len(sp["test"]), "blind_top3": rank["blind_top3"],
           "push_mean": float(push.mean()), "push_median": float(np.median(push)), "push_sd": float(push.std()),
           "lo_u_mean": float(lo_u.mean()), "lo_p_mean": float(lo_p.mean()),
           "R_blind_k": [float(x) for x in R[:8]], "R_blind3": float(R[2]),
           "R_all32": float(R[8]), "R_mlp26": float(R[9]), "R_mlp27": float(R[10]), "R_mlp_both": float(R[11]),
           "R_all_heads_and_mlps": float(R[12])}
    res["R_secondary3"] = float(R[13])
    res["secondary_top3"] = pk["secondary_last_position"]["blind_top3"]
    res["flagged_primary"] = rank["flagged"]
    res["top1_flagged"] = rank["order"][0] in rank["flagged"]
    ro, ra = R[14:414], R[414:814]
    res["random_out"] = {"R": [float(x) for x in ro], "median": float(np.median(ro)), "p95": float(np.quantile(ro, .95)),
                         "median_50": float(np.median(ro[:50])), "p95_50": float(np.quantile(ro[:50], .95)),
                         "sets": [[NAME(x) for x in s] for s in rnd_out]}
    res["random_all"] = {"R": [float(x) for x in ra], "median": float(np.median(ra)), "p95": float(np.quantile(ra, .95)),
                         "median_50": float(np.median(ra[:50])), "p95_50": float(np.quantile(ra[:50], .95)),
                         "sets": [[NAME(x) for x in s] for s in rnd_all]}
    # bootstrap CI of R_blind3 over test prompts
    bt = np.random.default_rng(7)
    boots = []
    for _ in range(2000):
        ii = bt.integers(0, len(push), len(push))
        boots.append(1 - (lox[2, ii] - lo_u[ii]).mean() / (lo_p[ii] - lo_u[ii]).mean())
    res["R_blind3_ci95"] = [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]
    res["per_prompt"] = {"lo_u": lo_u.tolist(), "lo_p": lo_p.tolist(), "lo_blind3": lox[2].tolist()}
    # reverse plant of the blind top-3 (DEVIATIONS D1): share of the push recovered in the unprobed pass
    u2, p2, lpl = pt.run(sp["test"], [(blind3, [])], kind="plant")
    res["plant_blind3_recovered"] = float((lpl[0] - u2).mean() / (p2 - u2).mean())
    res["head_free_share"] = 1.0 - res["R_all32"]
    res["pass"] = {"R_ge_070": res["R_blind3"] >= 0.70, "R_gt_p95_out": res["R_blind3"] > res["random_out"]["p95"],
                   "top1_flagged": res["top1_flagged"]}
    res["pass"]["model_passes"] = all(res["pass"].values())
    print(a.model, "push", round(res["push_mean"], 4), "R_blind3", round(res["R_blind3"], 3), "rand_out med/p95",
          round(res["random_out"]["median"], 3), round(res["random_out"]["p95"], 3), "all32", round(res["R_all32"], 3),
          "mlp26/27/both", round(res["R_mlp26"], 3), round(res["R_mlp27"], 3), round(res["R_mlp_both"], 3),
          f"{time.time() - t0:.0f}s", flush=True)
    # oracle: greedy on DEV by R (behaviour known), k<=3, then scored on TEST
    if not a.no_oracle:
        chosen, path = [], []
        for _ in range(3):
            cand = [x for x in HEADS if x not in chosen]
            u, p, lx = pt.run(sp["dev"], [(chosen + [x], []) for x in cand])
            rr = removed(u, p, lx)
            j = int(np.argmax(rr))
            chosen.append(cand[j])
            path.append({"add": NAME(cand[j]), "dev_R": float(rr[j])})
        u, p, lx = pt.run(sp["test"], [(chosen[:k], []) for k in (1, 2, 3)])
        res["oracle"] = {"top3": [NAME(x) for x in chosen], "path": path,
                         "test_R_k": [float(x) for x in removed(u, p, lx)]}
        res["overlap_blind_oracle"] = len(set(rank["blind_top3"]) & set(res["oracle"]["top3"]))
        print(a.model, "oracle", res["oracle"]["top3"], res["oracle"]["test_R_k"], "overlap",
              res["overlap_blind_oracle"], f"{time.time() - t0:.0f}s", flush=True)
    # all 4960 3-sets: rank of the blind set
    if a.all_sets:
        allc = list(itertools.combinations(HEADS, 3))
        _, _, lx = pt.run(sp["test"], [(list(s), []) for s in allc])
        Ra = removed(lo_u, lo_p, lx)
        rb = res["R_blind3"]
        res["all_sets"] = {"n": len(allc), "rank_of_blind": int((Ra > rb).sum()) + 1, "best_R": float(Ra.max()),
                           "best_set": [NAME(x) for x in allc[int(np.argmax(Ra))]], "median": float(np.median(Ra)),
                           "frac_ge_070": float((Ra >= 0.70).mean())}
        np.save(HERE / "raw" / f"allsets_{a.model}.npy", Ra.astype(np.float32))
        print(a.model, "all sets", res["all_sets"], f"{time.time() - t0:.0f}s", flush=True)
    res["s"] = round(time.time() - t0, 1)
    (HERE / "data" / "patch").mkdir(parents=True, exist_ok=True)
    (HERE / "data" / "patch" / f"{a.model}.json").write_text(json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score", "rank", "patch"])
    ap.add_argument("--models", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--all-sets", action="store_true")
    a = ap.parse_args()
    {"score": cmd_score, "rank": cmd_rank, "patch": cmd_patch}[a.cmd](a)


if __name__ == "__main__":
    main()
