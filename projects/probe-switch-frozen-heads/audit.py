"""Mechanism readouts on one trained 8-block model (PREREG 'Mechanism readouts' + v1.1 audit verdict).

Cached-tail path (eager bf16 MPS, the training tail): the merged bf16 blocks 20-27 read the cached block-20 inputs
recorded without (z0) and with (z1) the probe. A head = one of the 16 query heads of a block = its 128-column slice
of the o_proj input (Sauers' unit; probe-switch-heads), named '<block>.<h>', 128 heads in blocks 20-27. A swap (cut)
gives the probed run the unprobed run's slices of the head set, at all 80 positions, from the clean unprobed run of
the SAME prompt (all other heads keep computing on the patched stream).

Scores, means over prompts (as in probe-switch-audit-penalty/audit.py):
  informed  removed       = 1 - mean push_after / mean push,   push = lo(probed) - lo(unprobed), lo = logp(A) - logp(B)
  blind     blind_removed = 1 - mean KL(p_cut || p_unprobed) / mean KL(p_probed || p_unprobed)   (no answer label)
Rows:
  clamp_all     all 128 heads of blocks 20-27 cut (PREREG head-clamp patch, bf16 primary); >= 0.70 = frozen-head transport
  blind_greedy  joint greedy on DEV (each step adds the head that maximises blind_removed given the earlier ones), k <= 3,
                scored on TEST -> R_blind3 = test 'removed' at k = 3 (PREREG v1.1 audit verdict)
  null          50 seeded random 3-sets from the 125 heads outside blind_greedy k = 3 (test 'removed')
States: merged-model eager states at positions 71 and 79 of every block input 20-27, mid 20-27 and final, all 704
prompts, both conditions -> raw/after/<run>_states.npy uint16 [704, 2 cond, 2 pos, 17 sites, 2048] (after-map).

    $PY audit.py --run mlp8_none_s0
writes data/audit/<run>.json, raw/after/<run>_states.npy
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from tail import DEV, Rows, load_tail, score_pairs, split_ids  # noqa: E402

HD, NH, FIRST, NL = 128, 16, 20, 8
SITES = [f"in{FIRST + i}" for i in range(NL)] + [f"mid{FIRST + i}" for i in range(NL)] + ["final"]


def merge_into(tail, adapter: dict, scale: float) -> int:
    n = 0
    with torch.no_grad():
        for k in sorted(x[:-2] for x in adapter if x.endswith(".A")):
            lin = tail.get_submodule(k)
            A, B = adapter[k + ".A"].to(DEV), adapter[k + ".B"].to(DEV)
            lin.weight.copy_((lin.weight.float() + scale * (B @ A)).to(torch.bfloat16))  # = train.py LoRALinear.merged
            n += 1
    return n


class Hooks:
    def __init__(self, tail):
        self.mode, self.o_in, self.mask, self.src = "off", {}, {}, {}
        for li in range(NL):
            tail.layers[li].self_attn.o_proj.register_forward_pre_hook(self._pre(li))

    def _pre(self, li):
        def f(mod, args):
            x = args[0]
            if self.mode == "record":
                self.o_in[li] = x.detach()
            elif self.mode == "patch" and self.mask.get(li) is not None:
                return (torch.where(self.mask[li], self.src[li], x),)
            return None
        return f


def kl(a, b):
    return (a.exp() * (a - b)).sum(-1)


class Runner:
    def __init__(self, tail, w_out, a_id, b_id):
        self.tail, self.w_out, self.a_id, self.b_id = tail, w_out, a_id, b_id
        self.hk = Hooks(tail)

    def run_from(self, li, h):
        pos = torch.arange(h.shape[1], device=h.device)[None].expand(h.shape[0], -1)
        cs = self.tail.rotary(h, pos)
        ins = {}
        for j in range(li, NL):
            ins[j] = h
            h = self.tail.layers[j](h, attention_mask=None, position_ids=pos, position_embeddings=cs)
        lg = (self.tail.norm(h)[:, -1] @ self.w_out.T).float()
        return torch.log_softmax(lg, -1), ins

    @torch.no_grad()
    def prepare(self, z0, z1, ids, bs=16):
        """clean unprobed / probed runs: log-probs, o_proj inputs of the unprobed run, block inputs of the probed run."""
        self.ids, self.bs = ids, bs
        self.lp0, self.lp1, self.src, self.pin = [], [], [], []
        for s in range(0, len(ids), bs):
            b = ids[s:s + bs]
            self.hk.mode = "record"
            lp0, _ = self.run_from(0, z0[b].to(DEV))
            self.src.append({li: self.hk.o_in[li] for li in range(NL)})
            self.hk.mode = "off"
            lp1, ins = self.run_from(0, z1[b].to(DEV))
            self.lp0.append(lp0); self.lp1.append(lp1); self.pin.append(ins)
        self.base_ins = self.pin

    def lo(self, lp):
        return lp[:, self.a_id] - lp[:, self.b_id]

    @torch.no_grad()
    def cut(self, heads: list[tuple[int, int]], base=None, from_block=None):
        """probed run with the head set cut; returns the scores and the patched block inputs (per chunk).
        base/from_block: start at block `from_block` from the block inputs `base` of a run patched by the heads of
        the set below from_block (exact: a block's input does not depend on heads at or above that block; heads at or
        above from_block are re-applied by the masks). Default: the clean probed run, from the lowest block of the set."""
        masks = {}
        for li, h in heads:
            m = masks.setdefault(li, torch.zeros(NH * HD, dtype=torch.bool, device=DEV))
            m[h * HD:(h + 1) * HD] = True
        base = self.pin if base is None else base
        li = (min(masks) if masks else NL) if from_block is None else from_block
        push_a, push, kl_a, kl_0, outs = [], [], [], [], []
        for c in range(len(self.lp0)):
            if li >= NL:
                lp, ins = self.lp1[c], dict(base[c])
            else:
                self.hk.mask, self.hk.src, self.hk.mode = masks, self.src[c], "patch"
                lp, ins = self.run_from(li, base[c][li])
                self.hk.mode = "off"
                for j in range(li):
                    ins[j] = base[c][j]
            outs.append(ins)
            push_a.append(self.lo(lp) - self.lo(self.lp0[c])); push.append(self.lo(self.lp1[c]) - self.lo(self.lp0[c]))
            kl_a.append(kl(lp, self.lp0[c])); kl_0.append(kl(self.lp1[c], self.lp0[c]))
        pa, p = torch.cat(push_a), torch.cat(push)
        ka, k0 = torch.cat(kl_a), torch.cat(kl_0)
        return {"removed": float(1 - pa.mean() / p.mean()), "blind_removed": float(1 - ka.mean() / k0.mean()),
                "push_mean": float(p.mean()), "push_after_mean": float(pa.mean())}, outs


def greedy(run: Runner, k: int, all_heads):
    picked, steps, base = [], [], None
    for step in range(k):
        best = None
        for hd in all_heads:
            if hd in picked:
                continue
            # start at the candidate's block from the run patched by `picked` (its inputs there already hold the
            # picked heads below; picked heads at or above are re-applied by the masks)
            r, _ = run.cut(picked + [hd], base=base, from_block=hd[0])
            if best is None or r["blind_removed"] > best[1]["blind_removed"]:
                best = (hd, r)
        picked.append(best[0])
        _, base = run.cut(picked)
        steps.append({"k": step + 1, "head": f"{FIRST + best[0][0]}.{best[0][1]}", "dev": best[1]})
        print("greedy", steps[-1], flush=True)
    return picked, steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--run", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n-null", type=int, default=50)
    ap.add_argument("--no-greedy", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    meta = json.load(open(HERE / "data" / "runs" / f"{a.run}.json"))
    assert meta["args"]["arm"] in ("all8", "mlp8")
    cache = HERE / "raw" / "cache"
    pid = json.load(open(cache / "meta_A.json"))["prompt_ids"]
    idx = split_ids(HERE / "raw" / "prompts.jsonl", pid)
    z0, z1 = Rows(cache / "none_A_b20.npy"), Rows(cache / "probe_A_b20.npy")
    tok = AutoTokenizer.from_pretrained(a.model)
    A_ID, B_ID = tok.encode("A", add_special_tokens=False)[0], tok.encode("B", add_special_tokens=False)[0]
    cfg, tail, w_out = load_tail(a.model, NL)
    tail.eval()
    n = merge_into(tail, torch.load(HERE / "raw" / "runs" / f"{a.run}_adapter.pt"),
                   meta["args"]["alpha"] / meta["args"]["rank"])
    assert n == len(meta["lora_modules"])
    test_stats, _ = score_pairs(tail, w_out, z0, z1, idx["test"], pid, A_ID, B_ID)
    assert abs(test_stats["strict"] - meta["test_cached"]["strict"]) < 1e-9, (test_stats, meta["test_cached"])
    out = {"run": a.run, "test_cached": test_stats}
    all_heads = [(li, h) for li in range(NL) for h in range(NH)]

    run = Runner(tail, w_out, A_ID, B_ID)
    run.prepare(z0, z1, idx["test"])
    out["clamp_all_test"], _ = run.cut(all_heads)
    out["clamp_by_block_test"] = {f"{FIRST + li}": run.cut([(li, h) for h in range(NH)])[0] for li in range(NL)}
    print("clamp_all test", out["clamp_all_test"], f"{time.time() - t0:.0f}s", flush=True)
    if not a.no_greedy:
        run.prepare(z0, z1, idx["dev"])
        picked, steps = greedy(run, a.k, all_heads)
        out["blind_greedy_dev"] = steps
        run.prepare(z0, z1, idx["test"])
        out["blind_greedy_test"] = [{"k": j + 1, "heads": [f"{FIRST + l}.{h}" for l, h in picked[:j + 1]],
                                     **run.cut(picked[:j + 1])[0]} for j in range(len(picked))]
        out["R_blind3"] = out["blind_greedy_test"][-1]["removed"]
        rest = [hd for hd in all_heads if hd not in picked]
        rng = random.Random(0)
        out["null_removed"] = [run.cut(rng.sample(rest, a.k))[0]["removed"] for _ in range(a.n_null)]
        out["null_p95"] = float(np.percentile(out["null_removed"], 95))
        print("R_blind3", out["R_blind3"], "null p95", out["null_p95"], f"{time.time() - t0:.0f}s", flush=True)

    # after-training states at positions 71 and 79 (eager merged tail, all prompts)
    rec = {}
    mods = []
    for li, layer in enumerate(tail.layers):
        mods.append((f"in{FIRST + li}", layer)); mods.append((f"mid{FIRST + li}", layer.post_attention_layernorm))
    mods.append(("final", tail.norm))
    hs = [m.register_forward_pre_hook(lambda mod, args, s=s: rec.__setitem__(s, args[0][:, [71, 79]].detach()))
          for s, m in mods]
    (HERE / "raw" / "after").mkdir(parents=True, exist_ok=True)
    st = np.lib.format.open_memmap(HERE / "raw" / "after" / f"{a.run}_states.npy", mode="w+", dtype=np.uint16,
                                   shape=(len(pid), 2, 2, len(SITES), cfg.hidden_size))
    allid = list(range(len(pid)))
    with torch.no_grad():
        for s in range(0, len(allid), 16):
            b = allid[s:s + 16]
            for c, z in ((0, z0), (1, z1)):
                tail(z[b].to(DEV), w_out)
                x = torch.stack([rec[name] for name in SITES], 2)  # [B, 2 pos, sites, H]
                st[b[0]:b[-1] + 1, c] = x.cpu().contiguous().view(torch.int16).numpy().view(np.uint16)
    st.flush()
    for h in hs:
        h.remove()
    out["states"] = {"path": f"raw/after/{a.run}_states.npy", "sites": SITES, "positions": [71, 79],
                     "shape": list(st.shape)}
    out["secs"] = round(time.time() - t0, 1)
    (HERE / "data" / "audit").mkdir(parents=True, exist_ok=True)
    (HERE / "data" / "audit" / f"{a.run}.json").write_text(json.dumps(out, indent=1))
    print("AUDIT_DONE", a.run, out["secs"], flush=True)


if __name__ == "__main__":
    main()
