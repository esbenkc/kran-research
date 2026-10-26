"""Head audit of one trained switch model (Iris; oracle-1026-audit-penalty, plan #7220 task 4).

Cached-tail path, as in probe-switch-heads/heads.py: the merged bf16 tail (blocks 26-27 + final norm) reads the cached
block-26 inputs recorded without (z0) and with (z1) the probe. A head = one of the 16 query heads of a block = its
128-column slice of the o_proj input (Sauers' unit), named '<block>.<h>'. Source of every swap = the clean run of the
other condition of the SAME prompt.

  cut     probed run gets the unprobed run's slices of the head set   (necessity: how much of the probe's change goes)
  induce  unprobed run gets the probed run's slices of the head set   (sufficiency: how much the set alone recreates)

Two scores per swap, means over prompts:
  BLIND (no answer label; whole next-token distribution at the answer position, nats):
    cut:    blind_removed   = 1 - mean KL(p_cut || p_unprobed)   / mean KL(p_probed || p_unprobed)
    induce: blind_recovered = 1 - mean KL(p_ind || p_probed)     / mean KL(p_unprobed || p_probed)
    also kl_move = mean KL(p_probed || p_cut) (the penalty's quantity, #7220)
  INFORMED (A-B push; push = logodds(A,B | probed) - logodds(A,B | unprobed)):
    cut:    removed   = 1 - mean push_after / mean push
    induce: recovered = mean(lo_ind - lo_unprobed) / mean push

Selection on DEV (64 prompts), scored on TEST (128):
  blind_greedy     joint greedy (each step adds the head that maximises blind_removed GIVEN the earlier ones), k<=8
                   -> TITLE CELL = test 'removed' of blind_greedy k=3
  informed_greedy  same with the informed 'removed' (as published in probe-switch-heads), k<=3
  reverse_greedy   joint greedy by blind_recovered under induce, k<=3 (OR gates light up here)
  all_32           every head of both blocks cut
  null             50 seeded random 3-sets from the 29 heads outside blind_greedy k=3 (cut)
  mlp path         all 32 heads cut AND MLP26 / MLP27 / both outputs swapped to unprobed: what stays after all heads
                   are clean is the push through the MLPs and the residual skip; the MLP share = drop caused by the MLP
                   swap on top of all_32 (both orders reported via the 'both' row)
Plus: cached switch stats (score_pairs), and, with --ref, the clean next-token KL(ref || model) at all 80 positions of
the 128 unprobed TEST prompts (base-task drift vs the warm-start model).

    $PY audit.py --run L100_s0                       # reads raw/step2/Qwen3-1.7B/runs/L100_s0_merged.pt
    $PY audit.py --run warm_s0 --merge-adapter data/warm/s0_adapter.pt   # writes runs/warm_s0_merged.pt first
writes data/audit/<run>.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from tail import DEV, Rows, load_merged, load_tail, score_pairs, split_ids  # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv  # noqa: E402

HD, NH, NL = 128, 16, 2
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def merge_adapter(model_id, adapter_path, out_path, rank=16, alpha=32):
    from peft import LoraConfig, get_peft_model
    _, tail, _ = load_tail(model_id, 2)
    ptail = get_peft_model(tail, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, target_modules=TARGETS,
                                            bias="none"), autocast_adapter_dtype=True)
    sd = torch.load(adapter_path)
    named = dict(ptail.named_parameters())
    assert set(sd) <= set(named), sorted(set(sd) - set(named))[:5]
    with torch.no_grad():
        for k, v in sd.items():
            named[k].copy_(v.to(named[k].device, named[k].dtype))
    merged = ptail.merge_and_unload()
    out = {k: v.detach().cpu() for k, v in merged.state_dict().items() if "rotary" not in k}
    torch.save(out, out_path)
    del ptail, merged, tail
    if DEV == "mps":
        torch.mps.empty_cache()


class Hooks:
    """record: keep each block's o_proj input and MLP output. patch: replace masked columns / rows from a source."""

    def __init__(self, tail):
        self.mode = "off"
        self.o_in, self.mlp = {}, {}
        self.hmask, self.hsrc, self.mmask, self.msrc = {}, {}, {}, {}
        for li in range(NL):
            tail.layers[li].self_attn.o_proj.register_forward_pre_hook(self._pre(li))
            tail.layers[li].mlp.register_forward_hook(self._post(li))

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


def kl(lp_a, lp_b):
    """KL(a || b) per row, from log-probs (float32)."""
    return (lp_a.exp() * (lp_a - lp_b)).sum(-1)


class Auditor:
    def __init__(self, tail, w_out, z0, z1, a_id, b_id, chunk=4, bs=16, fast=True, chunk_fast=8):
        self.tail, self.w_out, self.z0, self.z1 = tail, w_out, z0, z1
        self.a_id, self.b_id, self.chunk, self.bs = a_id, b_id, chunk, bs
        self.fast, self.chunk_fast = fast, chunk_fast
        self.hk = Hooks(tail)

    def _block27_last(self, h, cs):
        """Block 27 for the answer position only: keys/values from all positions, one query. Hooks still fire
        (o_proj input and MLP output have shape (R, 1, D)). Returns final-norm logits (R, V) float32."""
        layer = self.tail.layers[1]
        at = layer.self_attn
        R, T, _ = h.shape
        cos, sin = cs[0][:1], cs[1][:1]  # same positions in every row; broadcast over the batch
        x = layer.input_layernorm(h)
        q = at.q_norm(at.q_proj(x[:, -1:]).view(R, 1, -1, HD)).transpose(1, 2)
        k = at.k_norm(at.k_proj(x).view(R, T, -1, HD)).transpose(1, 2)
        v = at.v_proj(x).view(R, T, -1, HD).transpose(1, 2)
        q, _ = apply_rotary_pos_emb(q, q, cos[:, -1:], sin[:, -1:])
        _, k = apply_rotary_pos_emb(k, k, cos, sin)
        g = at.num_key_value_groups
        o = torch.nn.functional.scaled_dot_product_attention(q, repeat_kv(k, g), repeat_kv(v, g), scale=at.scaling)
        o = at.o_proj(o.transpose(1, 2).reshape(R, 1, -1))
        h2 = h[:, -1:] + o
        h2 = h2 + layer.mlp(layer.post_attention_layernorm(h2))
        return (self.tail.norm(h2)[:, 0] @ self.w_out.T).float()

    def _rot(self, h):
        pos = torch.arange(h.shape[1], device=h.device)[None].expand(h.shape[0], -1)
        return pos, self.tail.rotary(h, pos)

    def _block26(self, h):
        pos, cs = self._rot(h)
        return self.tail.layers[0](h, attention_mask=None, position_ids=pos, position_embeddings=cs), cs

    def forward(self, h):
        """Full logits at the answer position (fast path when self.fast)."""
        if not self.fast:
            return self.tail(h, self.w_out)
        h26, cs = self._block26(h)
        return self._block27_last(h26, cs)

    def _lo(self, lp):
        return lp[:, self.a_id] - lp[:, self.b_id]

    @torch.no_grad()
    def run(self, ids, sets, kind):
        """sets: list of (heads [(li,h)], mlps [li]). Returns base dict + per-set dict of per-prompt numpy arrays."""
        per = [{"lo": [], "kl_ref": [], "kl_move": [], "kl_move_rev": [], "top": []} for _ in sets]
        base = {"lo_u": [], "lo_p": [], "kl_pu": [], "kl_up": [], "top_u": [], "top_p": []}
        for s in range(0, len(ids), self.bs):
            b = ids[s:s + self.bs]
            n = len(b)
            h0, h1 = self.z0[b].to(DEV), self.z1[b].to(DEV)
            self.hk.mode = "record"
            if self.fast:
                h26_all, cs_all = self._block26(torch.cat([h0, h1]))
                lg = self._block27_last(h26_all, cs_all)
            else:
                lg = self.tail(torch.cat([h0, h1]), self.w_out)
            self.hk.mode = "off"
            lp = torch.log_softmax(lg, -1)
            lp_u, lp_p = lp[:n], lp[n:]
            base["lo_u"].append(self._lo(lp_u).cpu()); base["lo_p"].append(self._lo(lp_p).cpu())
            base["kl_pu"].append(kl(lp_p, lp_u).cpu()); base["kl_up"].append(kl(lp_u, lp_p).cpu())
            base["top_u"].append(lg[:n].argmax(-1).cpu()); base["top_p"].append(lg[n:].argmax(-1).cpu())
            # cut: probed rows patched from unprobed sources; induce: the reverse
            src_rows, dst_rows, dst_in = (slice(0, n), slice(n, 2 * n), h1) if kind == "cut" else \
                                         (slice(n, 2 * n), slice(0, n), h0)
            lp_ref = lp_u if kind == "cut" else lp_p    # distance to the other condition
            lp_own = lp[dst_rows]
            o_src = {li: self.hk.o_in[li][src_rows] for li in range(NL)}
            m_src = {li: self.hk.mlp[li][src_rows] for li in range(NL)}
            # chunks: sets that touch block 26 need the full pass; block-27-only sets reuse the block-26 output
            need26 = [j for j, (hs, ms) in enumerate(sets) if (not self.fast) or 0 in ms or any(l == 0 for l, _ in hs)]
            only27 = [j for j in range(len(sets)) if j not in set(need26)]
            chunks = [need26[i:i + self.chunk] for i in range(0, len(need26), self.chunk)] + \
                     [only27[i:i + self.chunk_fast] for i in range(0, len(only27), self.chunk_fast)]
            for js in chunks:
                cs = [sets[j] for j in js]
                C = len(cs)
                x = dst_in.repeat(C, 1, 1) if js[0] in need26 else None
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
                if not self.fast:
                    lgx = self.tail(x, self.w_out)
                elif js[0] in only27:
                    lgx = self._block27_last(h26_all[dst_rows].repeat(C, 1, 1), cs_all)
                else:
                    h26, cs26 = self._block26(x)
                    lgx = self._block27_last(h26, cs26)
                lpx = torch.log_softmax(lgx, -1)
                topx = lgx.argmax(-1).cpu()
                self.hk.mode = "off"
                for jj, j in enumerate(js):
                    q = lpx[jj * n:(jj + 1) * n]
                    per[j]["lo"].append(self._lo(q).cpu())
                    per[j]["kl_ref"].append(kl(q, lp_ref).cpu())
                    per[j]["kl_move"].append(kl(lp_own, q).cpu())
                    per[j]["kl_move_rev"].append(kl(q, lp_own).cpu())  # D3 (b): KL(p_cut || p_probed)
                    per[j]["top"].append(topx[jj * n:(jj + 1) * n])
            self.hk.hmask, self.hk.mmask = {}, {}
            self.hk.hsrc, self.hk.msrc = {}, {}
        base = {k: torch.cat(v).numpy().astype(np.float64) for k, v in base.items()}
        per = [{k: torch.cat(v).numpy().astype(np.float64) for k, v in d.items()} for d in per]
        return base, per


def score(base, d, kind, a_id=None, b_id=None):
    push = base["lo_p"] - base["lo_u"]
    if kind == "cut":
        after = d["lo"] - base["lo_u"]
        sw = float(np.mean((base["top_u"] == b_id) & (d["top"] == a_id))) if a_id is not None else None
        return {"removed": float(1 - after.mean() / push.mean()), "push_after_mean": float(after.mean()),
                "switch_after": sw,
                "blind_removed": float(1 - d["kl_ref"].mean() / base["kl_pu"].mean()),
                "kl_after_mean": float(d["kl_ref"].mean()), "kl_move_mean": float(d["kl_move"].mean()),
                "cut_moves_output": float(d["kl_move_rev"].mean()),
                "answer_A": float(np.mean(d["top"] == a_id)) if a_id is not None else None,
                "answer_B": float(np.mean(d["top"] == b_id)) if a_id is not None else None,
                "answer_other": float(np.mean((d["top"] != a_id) & (d["top"] != b_id))) if a_id is not None else None,
                "answer_still_A": float(np.mean(d["lo"] > 0))}
    moved = d["lo"] - base["lo_u"]
    return {"recovered": float(moved.mean() / push.mean()),
            "blind_recovered": float(1 - d["kl_ref"].mean() / base["kl_up"].mean()),
            "kl_after_mean": float(d["kl_ref"].mean())}


def greedy(aud, ids, pool, kind, key, kmax):
    chosen, path = [], []
    for _ in range(kmax):
        cand = [x for x in pool if x not in chosen]
        base, per = aud.run(ids, [(chosen + [x], []) for x in cand], kind)
        sc = [score(base, d, kind)[key] for d in per]
        j = int(np.argmax(sc))
        chosen = chosen + [cand[j]]
        path.append({"set": [hname(x) for x in chosen], f"dev_{key}": sc[j],
                     "runner_up": float(sorted(sc)[-2]) if len(sc) > 1 else None})
    return chosen, path


def hname(x):
    return f"{26 + x[0]}.{x[1]}"


@torch.no_grad()
def clean_kl_all_positions(tail, ref, w_out, z0, ids, bs=2):
    """Mean KL(ref || model) of next-token distributions at all positions of unprobed prompts."""

    def hidden(m, h):
        pos = torch.arange(h.shape[1], device=h.device)[None].expand(h.shape[0], -1)
        cs = m.rotary(h, pos)
        for layer in m.layers:
            h = layer(h, attention_mask=None, position_ids=pos, position_embeddings=cs)
        return m.norm(h)

    tot, cnt, ans = 0.0, 0, []
    for s in range(0, len(ids), bs):
        h = z0[ids[s:s + bs]].to(DEV)
        a = torch.log_softmax((hidden(tail, h) @ w_out.T).float(), -1)
        r = torch.log_softmax((hidden(ref, h) @ w_out.T).float(), -1)
        k = kl(r, a)
        tot += float(k.sum()); cnt += k.numel(); ans += k[:, -1].cpu().tolist()
    return {"clean_kl_all_pos_mean": tot / cnt, "clean_kl_answer_pos_mean": float(np.mean(ans)),
            "n_tokens": cnt, "n_prompts": len(ids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--run", required=True)
    ap.add_argument("--root", default=str(HERE / "raw" / "step2" / "Qwen3-1.7B"))
    ap.add_argument("--prompts", default=str(HERE / "data" / "prompts.jsonl"))
    ap.add_argument("--merge-adapter", default=None, help="build runs/<run>_merged.pt from this adapter first")
    ap.add_argument("--ref", default=None, help="merged .pt of the warm-start model, for the clean KL")
    ap.add_argument("--n-dev", type=int, default=64)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--n-random", type=int, default=50)
    ap.add_argument("--rand-seed", type=int, default=20261026)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--slow", action="store_true", help="full 2-block pass for every patched run (check only)")
    ap.add_argument("--out", default=str(HERE / "data" / "audit"))
    a = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(0)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    a_id, b_id = (tok.encode(t, add_special_tokens=False)[0] for t in ("A", "B"))
    root = Path(a.root)
    cdir = root / "compile_C"
    pid = json.load(open(cdir / "meta.json"))["prompt_ids"]
    idx = split_ids(Path(a.prompts), pid)
    dev, test = idx["dev"][: a.n_dev], idx["test"][: a.n_test]
    z0, z1 = Rows(cdir / "none_rec_tail.npy"), Rows(cdir / "probe_rec_tail.npy")
    mpath = root / "runs" / f"{a.run}_merged.pt"
    if a.merge_adapter:
        mpath.parent.mkdir(parents=True, exist_ok=True)
        merge_adapter(a.model, a.merge_adapter, mpath)
    cfg, tail, w_out = load_merged(a.model, mpath)
    res = {"run": a.run, "merged": str(mpath), "a_id": a_id, "b_id": b_id, "dev_ids": [pid[i] for i in dev],
           "test_ids": [pid[i] for i in test], "head_unit": "query head = 128-col slice of o_proj input"}
    res["cached_test"], _ = score_pairs(tail, w_out, z0, z1, test, pid, a_id, b_id)  # full 2-block path
    aud = Auditor(tail, w_out, z0, z1, a_id, b_id, chunk=a.chunk, fast=not a.slow)
    res["fast_path"] = not a.slow
    heads = [(li, h) for li in range(NL) for h in range(NH)]

    # DEV selection
    t1 = time.time()
    g_blind, res["blind_greedy_dev"] = greedy(aud, dev, heads, "cut", "blind_removed", a.kmax)
    g_inf, res["informed_greedy_dev"] = greedy(aud, dev, heads, "cut", "removed", 3)
    g_rev, res["reverse_greedy_dev"] = greedy(aud, dev, heads, "induce", "blind_recovered", 3)
    g_mov, res["moves_greedy_dev"] = greedy(aud, dev, heads, "cut", "cut_moves_output", 3)  # D3 (b)
    res["dev_secs"] = round(time.time() - t1, 1)

    # TEST scoring
    cut_sets = {f"blind_k{k}": (g_blind[:k], []) for k in range(1, a.kmax + 1)}
    cut_sets.update({f"informed_k{k}": (g_inf[:k], []) for k in (1, 2, 3)})
    cut_sets.update({f"moves_k{k}": (g_mov[:k], []) for k in (1, 2, 3)})
    cut_sets["identity"] = ([], [])  # Kasper #7251: patched-batch shape, nothing swapped; must give removed ~0
    cut_sets["all_32"] = (heads, [])
    cut_sets["all_32+mlp26"] = (heads, [0])
    cut_sets["all_32+mlp27"] = (heads, [1])
    cut_sets["all_32+mlp26+mlp27"] = (heads, [0, 1])
    cut_sets["mlp26"] = ([], [0])
    cut_sets["mlp27"] = ([], [1])
    pool = [x for x in heads if x not in g_blind[:3]]
    rng = np.random.default_rng(a.rand_seed)
    for j in range(a.n_random):
        cut_sets[f"null_{j:02d}"] = ([pool[i] for i in rng.choice(len(pool), 3, replace=False)], [])
    names = list(cut_sets)
    base, per = aud.run(test, [cut_sets[k] for k in names], "cut")
    res["test_cut"] = {k: score(base, d, "cut", a_id, b_id) for k, d in zip(names, per)}
    res["test_cut_sets"] = {k: {"heads": [hname(x) for x in v[0]], "mlps": [26 + m for m in v[1]]}
                            for k, v in cut_sets.items()}
    push = base["lo_p"] - base["lo_u"]
    # Owain #7274: per-prompt A-minus-B log-odds (order = test_ids) for the Figure-1 example
    res["per_prompt_lo"] = {"unprobed": base["lo_u"].tolist(), "probed": base["lo_p"].tolist(),
                            **{k: per[names.index(k)]["lo"].tolist()
                               for k in ("blind_k3", "informed_k3", "moves_k3", "all_32")}}
    res["test_base"] = {"push_mean": float(push.mean()), "push_median": float(np.median(push)),
                        "kl_probed_vs_unprobed_mean": float(base["kl_pu"].mean()),
                        "switch_cached": float(np.mean((base["top_u"] == b_id) & (base["top_p"] == a_id))),
                        "unprobed_B": float(np.mean(base["top_u"] == b_id)),
                        # Vera #7286 / Kasper #7289: bf16 ties broken by token-id order; strict = lo_p > 0 and lo_u < 0
                        "switch_strict": float(np.mean((base["lo_p"] > 0) & (base["lo_u"] < 0))),
                        "ties_probed": int(np.sum(base["lo_p"] == 0)), "ties_unprobed": int(np.sum(base["lo_u"] == 0)),
                        "per_prompt_push": push.tolist()}
    rev_sets = {f"reverse_k{k}": (g_rev[:k], []) for k in (1, 2, 3)}
    rev_sets.update({f"blind_k{k}_induce": (g_blind[:k], []) for k in (1, 3)})
    rev_sets["all_32_induce"] = (heads, [])
    rev_sets["identity_induce"] = ([], [])
    base_i, per_i = aud.run(test, list(rev_sets.values()), "induce")
    res["test_induce"] = {k: score(base_i, d, "induce") for k, d in zip(rev_sets, per_i)}
    nulls = np.array([res["test_cut"][f"null_{j:02d}"]["removed"] for j in range(a.n_random)])
    res["null_removed"] = {"median": float(np.median(nulls)), "p95": float(np.quantile(nulls, 0.95)),
                           "max": float(nulls.max())}
    tc = res["test_cut"]
    a32 = tc["all_32"]["push_after_mean"]
    res["mlp_path"] = {
        "push_left_after_all_32": a32,
        "share_of_push_left_after_all_32": a32 / float(push.mean()),
        "mlp26_share_of_left": (a32 - tc["all_32+mlp26"]["push_after_mean"]) / a32 if a32 else None,
        "mlp27_share_of_left": (a32 - tc["all_32+mlp27"]["push_after_mean"]) / a32 if a32 else None,
        "both_mlps_share_of_left": (a32 - tc["all_32+mlp26+mlp27"]["push_after_mean"]) / a32 if a32 else None,
        # D1 (DEVIATIONS.md 7b03e15): MLP share and the direct skip leftover, as shares of the full push
        "mlp_share_D1": tc["all_32+mlp26+mlp27"]["removed"] - tc["all_32"]["removed"],
        "skip_leftover_D1": 1 - tc["all_32+mlp26+mlp27"]["removed"]}
    r3 = tc["blind_k3"]["removed"]
    res["mlp_path"]["prereg_clause3_M_over_1_minus_Rblind3"] = \
        res["mlp_path"]["mlp_share_D1"] / (1 - r3) if r3 < 1 else None
    res["headline"] = {"title_cell_blind_top3_removed": tc["blind_k3"]["removed"],
                       "blind_top3_blind_removed": tc["blind_k3"]["blind_removed"],
                       "informed_top3_removed": tc["informed_k3"]["removed"],
                       "moves_top3_removed_D3": tc["moves_k3"]["removed"],
                       "best_other_top3_removed_D3": max(tc["informed_k3"]["removed"], tc["moves_k3"]["removed"]),
                       "answer_after_blind_top3_ABother_D3": [tc["blind_k3"][k] for k in
                                                              ("answer_A", "answer_B", "answer_other")],
                       "answer_after_moves_top3_ABother_D3": [tc["moves_k3"][k] for k in
                                                              ("answer_A", "answer_B", "answer_other")],
                       "all_32_removed": tc["all_32"]["removed"],
                       "switch_all32cut_D2": tc["all_32"]["switch_after"],
                       "switch_blind_top3_cut": tc["blind_k3"]["switch_after"],
                       "reverse_top3_recovered": res["test_induce"]["reverse_k3"]["recovered"],
                       "null_p95": res["null_removed"]["p95"], "push_mean": float(push.mean()),
                       "identity_removed": tc["identity"]["removed"],
                       "identity_switch_after": tc["identity"]["switch_after"],
                       "identity_ok": bool(abs(tc["identity"]["removed"]) < 0.02 and
                                           tc["identity"]["switch_after"] == res["test_base"]["switch_cached"]),
                       "switch_cached": res["test_base"]["switch_cached"],
                       "switch_strict": res["test_base"]["switch_strict"],
                       "switch_full_path": res["cached_test"]["switch_correct"],  # score_pairs, full 2-block pass
                       "fast_vs_full_push": [float(push.mean()), res["cached_test"]["shift_mean"]],
                       "ties_probed": res["test_base"]["ties_probed"]}
    if a.ref:
        _, ref, _ = load_merged(a.model, Path(a.ref))
        res["clean_kl_vs_ref"] = {"ref": a.ref, **clean_kl_all_positions(tail, ref, w_out, z0, test)}
    res["secs"] = round(time.time() - t0, 1)
    res["torch"] = torch.__version__
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.run}.json").write_text(json.dumps(res, indent=1))
    print("AUDIT", a.run, json.dumps(res["headline"]), "mlp_path", json.dumps(res["mlp_path"]),
          "secs", res["secs"], flush=True)


if __name__ == "__main__":
    main()
