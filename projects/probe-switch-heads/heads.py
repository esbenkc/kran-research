"""Head test: is the probe-triggered switch carried by <= 3 attention heads of the two trained blocks?

Cached-tail path (the training path; no hook in any compiled graph): the merged bf16 tail (blocks 26-27 + final norm,
eager sdpa) reads the cached block-26 inputs with and without the probe. A 'head' = one of the 16 query heads of a
block = its 128-column slice of the input to o_proj (Sauers, watching docs/circuit-tracing.md). Heads are named
'<block>.<h>' (26.0 .. 27.15).

Interventions on a head set H (all positions unless --answer-only; source = the clean, unpatched run of the other
condition of the SAME prompt, recorded in the same batch):
  cut     probed run gets the unprobed run's slices of H      -> how much of the probe's push is removed
  induce  unprobed run gets the probed run's slices of H      -> how much of the probe's push H alone recreates
  zero    H zeroed in BOTH runs                                (the goal's 'ablating' as written; diagnostic)
  mean    H replaced in BOTH runs by its mean over dev prompts of both conditions, per position (diagnostic)
Metric (frozen, Kasper/Hugo): removed = (mean_orig_shift - mean_patched_shift) / mean_orig_shift over the prompts,
shift = [logp(A)-logp(B)](probed) - [same](unprobed). induce: recovered = mean(lo_u_patched - lo_u) / mean_orig_shift.

Selection on DEV only: single-head cut on all 32 heads; top-k by single-head rank and greedy top-k (k <= 3), for
both blocks and for block 27 alone. Scored on TEST: those sets, plus N random 3-head draws (seeded) from both blocks
and from block 27 alone.

    $PY projects/probe-switch-heads/heads.py --run lora_none_s0
writes raw/step2/<model>/heads/<run>.json (summary + per-prompt rows for every scored set)
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
from tail import DEV, Rows, load_merged, split_ids, summarize  # noqa: E402

HD = 128
NH = 16


class Patcher:
    """Forward pre-hooks on each block's o_proj: record the clean input, or overwrite head slices."""

    def __init__(self, tail):
        self.tail = tail
        self.mode = "off"
        self.src: dict[int, torch.Tensor] = {}
        self.plan: list = []  # (li, h)
        self.kind = None
        self.n = 0
        self.answer_only = False
        self.mean: dict[int, torch.Tensor] = {}
        self.handles = [tail.layers[li].self_attn.o_proj.register_forward_pre_hook(self._hook(li))
                        for li in range(len(tail.layers))]

    def _hook(self, li):
        def f(mod, args):
            x = args[0]
            if self.mode == "record":
                self.src[li] = x.detach().clone()
                return None
            if self.mode != "patch":
                return None
            heads = [h for (l2, h) in self.plan if l2 == li]
            if not heads:
                return None
            x = x.clone()
            n = self.n
            pos = slice(x.shape[1] - 1, x.shape[1]) if self.answer_only else slice(None)
            for h in heads:
                c = slice(h * HD, (h + 1) * HD)
                if self.kind == "cut":
                    x[n:, pos, c] = self.src[li][:n, pos, c]
                elif self.kind == "induce":
                    x[:n, pos, c] = self.src[li][n:, pos, c]
                elif self.kind == "zero":
                    x[:, pos, c] = 0
                elif self.kind == "mean":
                    x[:, pos, c] = self.mean[li][None, pos, c]
            return (x,) + tuple(args[1:])
        return f


def name(li_first: int, li: int, h: int) -> str:
    return f"{li_first + li}.{h}"


@torch.no_grad()
def run_sets(tail, w_out, z0, z1, ids, pid, a_id, b_id, pt: Patcher, sets: dict, kind: str, bs: int = 16):
    """Return {set_name: rows} for every head set, plus 'orig' (no patch)."""
    out = {k: [] for k in ["orig", *sets]}
    for s in range(0, len(ids), bs):
        b = ids[s:s + bs]
        n = len(b)
        h = torch.cat([z0[b], z1[b]]).to(DEV)

        def keep(sn, l):
            # reduce the full-vocabulary logits right away (same float32 values as before; frees ~20 MB per set)
            lp = torch.log_softmax(l, -1)
            small = torch.stack([lp[:, a_id], lp[:, b_id], lp[:, a_id] - lp[:, b_id], l.argmax(-1).float()], 1)
            small = small.cpu().numpy()
            for j, i in enumerate(b):
                for cond, r in ((0, j), (1, j + n)):
                    la, lb, lo, top = (float(v) for v in small[r])
                    out[sn].append({"prompt": pid[i], "probe": cond, "top": int(top), "a_id": a_id, "b_id": b_id,
                                    "logp_a": la, "logp_b": lb, "lo": lo})

        pt.mode = "record"
        keep("orig", tail(h, w_out))
        pt.mode, pt.n, pt.kind = "patch", n, kind
        for sn, hs in sets.items():
            pt.plan = hs
            keep(sn, tail(h, w_out))
        pt.mode = "off"
    return out


def shifts(rows):
    return np.array([rows[k + 1]["lo"] - rows[k]["lo"] for k in range(0, len(rows), 2)])


def effect(orig_rows, rows, kind):
    s0 = shifts(orig_rows)
    unch = {"unprobed_top_unchanged": float(np.mean([rows[k]["top"] == orig_rows[k]["top"]
                                                     for k in range(0, len(rows), 2)]))}
    if kind == "induce":
        moved = np.array([rows[k]["lo"] - orig_rows[k]["lo"] for k in range(0, len(rows), 2)])
        return {"recovered": float(moved.mean() / s0.mean()), **unch}
    s1 = shifts(rows)
    return {"removed": float((s0.mean() - s1.mean()) / s0.mean()), "shift_mean_after": float(s1.mean()),
            "share_shift_pos_after": float(np.mean(s1 > 0)), **unch}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--run", required=True)
    ap.add_argument("--root", default=str(HERE / "raw" / "step2" / "Qwen3-1.7B"),
                    help="dir with compile_C/{none,probe}_rec_tail.npy + meta.json and runs/<run>_merged.pt")
    ap.add_argument("--prompts", default=str(HERE / "raw" / "prompts.jsonl"))
    ap.add_argument("--a-tok", default="A", help="answer expected WITH the probe")
    ap.add_argument("--b-tok", default="B", help="answer expected WITHOUT the probe")
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--n-dev", type=int, default=64, help="smoke tests only")
    ap.add_argument("--n-src", type=int, default=128, help="train prompts for the mean-ablation source")
    ap.add_argument("--n-random", type=int, default=50)
    ap.add_argument("--answer-only", action="store_true")
    ap.add_argument("--rand-seed", type=int, default=20261024)
    ap.add_argument("--out-tag", default="")
    a = ap.parse_args()
    t0 = time.time()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    a_id, b_id = (tok.encode(t, add_special_tokens=False) for t in (a.a_tok, a.b_tok))
    assert len(a_id) == 1 and len(b_id) == 1
    a_id, b_id = a_id[0], b_id[0]
    root = Path(a.root)
    cdir = root / "compile_C"
    meta = json.load(open(cdir / "meta.json"))
    pid = meta["prompt_ids"]
    idx = split_ids(Path(a.prompts), pid)
    idx["test"] = idx["test"][: a.n_test]
    idx["dev"] = idx["dev"][: a.n_dev]
    z0, z1 = Rows(cdir / "none_rec_tail.npy"), Rows(cdir / "probe_rec_tail.npy")
    cfg, tail, w_out = load_merged(a.model, root / "runs" / f"{a.run}_merged.pt")
    first = cfg.num_hidden_layers - 2
    pt = Patcher(tail)
    pt.answer_only = a.answer_only
    all_heads = [(li, h) for li in range(2) for h in range(NH)]
    hn = {x: name(first, *x) for x in all_heads}

    # mean-ablation source: each head's o_proj input averaged over the first 128 TRAIN prompts, both conditions,
    # per position (all prompts have the same length)
    src_ids = idx["train"][: a.n_src]
    acc = {li: None for li in range(2)}
    with torch.no_grad():
        for s in range(0, len(src_ids), 16):
            b = src_ids[s:s + 16]
            pt.mode = "record"
            tail(torch.cat([z0[b], z1[b]]).to(DEV), w_out)
            for li in range(2):
                v = pt.src[li].float().sum(0)
                acc[li] = v if acc[li] is None else acc[li] + v
        pt.mode = "off"
    pt.mean = {li: (acc[li] / (2 * len(src_ids))).to(torch.bfloat16) for li in range(2)}

    # 1) DEV: single-head cut on all 32 heads
    single = run_sets(tail, w_out, z0, z1, idx["dev"], pid, a_id, b_id, pt, {hn[x]: [x] for x in all_heads}, "cut")
    dev_orig = single["orig"]
    single_eff = {hn[x]: effect(dev_orig, single[hn[x]], "cut")["removed"] for x in all_heads}
    rank_all = sorted(all_heads, key=lambda x: -single_eff[hn[x]])
    rank_27 = [x for x in rank_all if x[0] == 1]

    def greedy(pool):
        chosen, path = [], []
        for _ in range(3):
            cand = {hn[x]: chosen + [x] for x in pool if x not in chosen}
            r = run_sets(tail, w_out, z0, z1, idx["dev"], pid, a_id, b_id, pt, cand, "cut")
            best = max(cand, key=lambda k: effect(r["orig"], r[k], "cut")["removed"])
            chosen = cand[best]
            path.append({"set": [hn[x] for x in chosen], "dev_removed": effect(r["orig"], r[best], "cut")["removed"]})
        return chosen, path

    g_all, g_all_path = greedy(all_heads)
    g_27, g_27_path = greedy([x for x in all_heads if x[0] == 1])

    # 2) TEST: chosen sets (k = 1..3) + random 3-head draws
    sets = {}
    for k in (1, 2, 3):
        sets[f"rank_all_k{k}"] = rank_all[:k]
        sets[f"rank_27_k{k}"] = rank_27[:k]
        sets[f"greedy_all_k{k}"] = g_all[:k]
        sets[f"greedy_27_k{k}"] = g_27[:k]
    rng = np.random.default_rng(a.rand_seed)
    for j in range(a.n_random):
        sets[f"rand_all_{j:02d}"] = [all_heads[i] for i in rng.choice(32, 3, replace=False)]
        sets[f"rand_27_{j:02d}"] = [(1, int(h)) for h in rng.choice(16, 3, replace=False)]
    # D2 (DEVIATIONS.md, d7b33fb): status specificity control = 3-head draws from the 29 heads NOT in this model's
    # dev-picked greedy top-3 (both blocks); separate seeded generator, so the frozen draws above are unchanged
    ex_pool = [x for x in all_heads if x not in g_all[:3]]
    rng_ex = np.random.default_rng(a.rand_seed + 1)
    for j in range(a.n_random):
        sets[f"rand_ex_{j:02d}"] = [ex_pool[i] for i in rng_ex.choice(len(ex_pool), 3, replace=False)]
    sets["all_32"] = list(all_heads)  # every head of both blocks: the attention-only ceiling
    contains = {hn[x]: sum(x in sets[f"rand_all_{j:02d}"] for j in range(a.n_random)) for x in g_all[:3]}
    res = {"run": a.run, "root": str(root), "answer_only": a.answer_only, "a_tok": a.a_tok, "b_tok": a.b_tok,
           "a_id": a_id, "b_id": b_id, "head_unit": "query head = 128-col slice of o_proj input",
           "split_prompts": {s: [pid[i] for i in v] for s, v in idx.items() if s != "train"},
           "mean_source_prompts": [pid[i] for i in src_ids],
           "dev_single_cut_removed": single_eff, "rank_all": [hn[x] for x in rank_all],
           "greedy_all_dev": g_all_path, "greedy_27_dev": g_27_path,
           "rand_ex_pool": [hn[x] for x in ex_pool], "rand_all_draws_containing_picked": contains,
           "best_nonpicked_dev_single": max((single_eff[hn[x]], hn[x]) for x in ex_pool),
           "sets": {k: [hn[x] for x in v] for k, v in sets.items()}, "test": {}, "rows": {}}
    for kind in ("cut", "induce", "zero", "mean"):
        sel = sets if kind in ("cut", "induce") else {k: v for k, v in sets.items() if not k.startswith("rand_")}
        r = run_sets(tail, w_out, z0, z1, idx["test"], pid, a_id, b_id, pt, sel, kind)
        if kind == "cut":
            res["test_orig"] = summarize(r["orig"])
            res["rows"]["orig"] = r["orig"]
        res["test"][kind] = {k: {**effect(r["orig"], r[k], kind), **summarize(r[k])} for k in sel}
        for k in sel:
            res["rows"][f"{kind}:{k}"] = r[k]
        print(kind, {k: round(v.get("removed", v.get("recovered", 0)), 3) for k, v in res["test"][kind].items()
                     if not k.startswith("rand_")}, flush=True)
    for kind, key in (("cut", "removed"), ("induce", "recovered")):
        for pool in ("all", "27", "ex"):
            v = np.array([res["test"][kind][f"rand_{pool}_{j:02d}"][key] for j in range(a.n_random)])
            res[f"random_{pool}_{kind}"] = {"mean": float(v.mean()), "p95": float(np.quantile(v, 0.95)),
                                            "max": float(v.max()), "values": v.tolist()}
    res["secs"] = round(time.time() - t0, 1)
    (root / "heads").mkdir(parents=True, exist_ok=True)
    suffix = ("_ans" if a.answer_only else "") + a.out_tag
    (root / "heads" / f"{a.run}{suffix}.json").write_text(json.dumps(res))
    print("HEADS", a.run, "orig", res["test_orig"], "greedy_all", g_all_path, "greedy_27", g_27_path,
          "rand_all_cut", res["random_all_cut"]["p95"], "rand_ex_cut", res["random_ex_cut"]["p95"], "secs", res["secs"], flush=True)


if __name__ == "__main__":
    main()
