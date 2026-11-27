"""Blind head audit with a k sweep, on temperature-scaled logits. Built on audit.py (the parent's audit: same merged
bf16 tail, same cached block-A inputs, same head unit and the same swap = 'cut'), with three changes:

  1. Every blind score is computed on softmax(logits32 / T) (fp32 final-norm output x fp32 lm_head). T values: --temps (the first is the status T).
  2. The joint greedy blind pick runs to k = --kmax on the DEV prompts (default 64), one greedy path per T.
  3. Every scored set keeps per-prompt answer log-odds (bf16 AND fp32) on the TEST prompts (default 128), so the strict
     switch after any cut can be rebuilt: strict = lo_cut > 0 AND lo_unprobed < 0 (bf16; a tie counts as a miss).

Controls on TEST: identity (nothing swapped), all heads, and --n-random seeded random sets of each size in --rand-sizes
drawn from ALL heads of both tail blocks (no exclusion).

    AP_MODEL=Qwen/Qwen3-1.7B AP_OUT=data $PY kaudit.py --run k6_s0 --adapter data/adapters/k6_s0_adapter.pt
writes OUT/kaudit/<run>.json
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from audit import Auditor, hname, kl, merged_tail
from cfgm import CDIR, FIRST, HD, MODEL, NH, OUT
from tail import DEV, Rows, score_pairs, split_ids

HERE = Path(__file__).parent
NL = 2


def tn(t):
    """name of a temperature: 'B' (the frozen per-prompt rule) or the number"""
    return "B" if t == "B" else f"{t:g}"


def parse_temps(s):
    return [x if x == "B" else float(x) for x in s.split(",")]


class KAuditor(Auditor):
    """Auditor.run with blind KL at several temperatures and fp32 log-odds of every patched run."""

    def __init__(self, *args, temps=(1.0,), **kw):
        super().__init__(*args, **kw)
        self.temps = list(temps)

    @torch.no_grad()
    def run(self, ids, sets, kind="cut"):
        assert kind == "cut"
        Ts = self.temps
        per = [{"lo": [], "lo32": [], "top": [], **{f"kl_T{tn(t)}": [] for t in Ts}} for _ in sets]
        base = {"lo_u": [], "lo_p": [], "lo_u32": [], "lo_p32": [], "top_u": [], "top_p": [], "T_B": [],
                **{f"kl_pu_T{tn(t)}": [] for t in Ts}}
        for s in range(0, len(ids), self.bs):
            b = ids[s:s + self.bs]
            n = len(b)
            h0, h1 = self.z0[b].to(DEV), self.z1[b].to(DEV)
            self.hk.mode = "record"
            hA_all, cs_all = self._blockA(torch.cat([h0, h1]))
            lg = self._blockB_last(hA_all, cs_all)
            lg32 = self.logits32()
            lp32 = torch.log_softmax(lg32, -1)
            base["lo_u32"].append(self._lo(lp32[:n]).cpu()); base["lo_p32"].append(self._lo(lp32[n:]).cpu())
            del lp32
            self.hk.mode = "off"
            lp = torch.log_softmax(lg, -1)
            base["lo_u"].append(self._lo(lp[:n]).cpu()); base["lo_p"].append(self._lo(lp[n:]).cpu())
            base["top_u"].append(lg[:n].argmax(-1).cpu()); base["top_p"].append(lg[n:].argmax(-1).cpu())
            # rule B (PREREG): T_i = max(1, (z_p[argmax probed] - z_p[argmax unprobed]) / 4), z_p = fp32 probed logits
            zp, tu = lg32[n:], lg32[:n].argmax(-1)
            lead = zp.gather(1, zp.argmax(-1)[:, None])[:, 0] - zp.gather(1, tu[:, None])[:, 0]
            TB = torch.clamp(lead / 4.0, min=1.0)[:, None]
            base["T_B"].append(TB[:, 0].cpu())
            Tval = {t: (TB if t == "B" else float(t)) for t in Ts}
            lpT_u = {}
            for t in Ts:  # blind KL from the fp32 logits (Kasper #30260 b), then / T
                T2 = torch.cat([Tval[t], Tval[t]]) if t == "B" else Tval[t]
                lt = torch.log_softmax(lg32 / T2, -1)
                lpT_u[t] = lt[:n]
                base[f"kl_pu_T{tn(t)}"].append(kl(lt[n:], lt[:n]).cpu())
            o_src = {li: self.hk.o_in[li][:n] for li in range(NL)}
            m_src = {li: self.hk.mlp[li][:n] for li in range(NL)}
            needA = [j for j, (hs, ms) in enumerate(sets) if 0 in ms or any(l == 0 for l, _ in hs)]
            onlyB = [j for j in range(len(sets)) if j not in set(needA)]
            chunks = [needA[i:i + self.chunk] for i in range(0, len(needA), self.chunk)] + \
                     [onlyB[i:i + self.chunk_fast] for i in range(0, len(onlyB), self.chunk_fast)]
            for js in chunks:
                cs = [sets[j] for j in js]
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
                if js[0] in onlyB:
                    lgx = self._blockB_last(hA_all[n:].repeat(C, 1, 1), cs_all)
                else:
                    hA, csA = self._blockA(h1.repeat(C, 1, 1))
                    lgx = self._blockB_last(hA, csA)
                lgx32 = self.logits32()
                lo32 = self._lo(torch.log_softmax(lgx32, -1)).cpu()
                self.hk.mode = "off"
                lpx = torch.log_softmax(lgx, -1)
                lox = self._lo(lpx).cpu()
                topx = lgx.argmax(-1).cpu()
                klT = {t: kl(torch.log_softmax(lgx32 / (Tval[t].repeat(C, 1) if t == "B" else Tval[t]), -1),
                             lpT_u[t].repeat(C, 1)).cpu() for t in Ts}
                del lgx32
                for jj, j in enumerate(js):
                    sl = slice(jj * n, (jj + 1) * n)
                    per[j]["lo"].append(lox[sl]); per[j]["lo32"].append(lo32[sl]); per[j]["top"].append(topx[sl])
                    for t in Ts:
                        per[j][f"kl_T{tn(t)}"].append(klT[t][sl])
            self.hk.hmask, self.hk.hsrc, self.hk.mmask, self.hk.msrc = {}, {}, {}, {}
        base = {k: torch.cat(v).numpy().astype(np.float64) for k, v in base.items()}
        per = [{k: torch.cat(v).numpy().astype(np.float64) for k, v in d.items()} for d in per]
        return base, per


def blind_removed(base, d, t):
    return float(1 - d[f"kl_T{tn(t)}"].mean() / base[f"kl_pu_T{tn(t)}"].mean())


def kgreedy(aud, ids, pool, t, kmax):
    """joint greedy: step j adds the head that maximises blind_removed at T = t GIVEN the earlier picks (audit.py's
    greedy with the temperature). Logs the best and runner-up score of every step (the tie check)."""
    chosen, path = [], []
    for _ in range(kmax):
        cand = [x for x in pool if x not in chosen]
        base, per = aud.run(ids, [(chosen + [x], []) for x in cand])
        sc = np.array([blind_removed(base, d, t) for d in per])
        o = np.argsort(-sc, kind="stable")
        chosen = chosen + [cand[o[0]]]
        D = float(base[f"kl_pu_T{tn(t)}"].mean())
        path.append({"set": [hname(x) for x in chosen], "dev_blind_removed": float(sc[o[0]]),
                     "runner_up": float(sc[o[1]]) if len(sc) > 1 else None, "kl_full_dev_nats": D,
                     "margin_nats": float((sc[o[0]] - sc[o[1]]) * D) if len(sc) > 1 else None,
                     "runner_up_head": hname(cand[o[1]]) if len(sc) > 1 else None,
                     # PREREG tie: top-2 scores differ by less than 1e-5 x |top score| (broken by the lower head index)
                     "tie": bool(len(sc) > 1 and (sc[o[0]] - sc[o[1]]) < 1e-5 * abs(sc[o[0]]))})
    return chosen, path


def cut_stats(base, d, a_id, b_id, temps):
    lu, lp, lc = base["lo_u"], base["lo_p"], d["lo"]
    push = lp - lu
    return {"strict": float(np.mean((lc > 0) & (lu < 0))),
            "strict32": float(np.mean((d["lo32"] > 0) & (base["lo_u32"] < 0))),
            "argmax_switch": float(np.mean((base["top_u"] == b_id) & (d["top"] == a_id))),
            "push_removed": float(1 - (lc - lu).mean() / push.mean()),
            **{f"blind_removed_T{tn(t)}": blind_removed(base, d, t) for t in temps},
            "lo": [round(float(v), 4) for v in lc], "lo32": [round(float(v), 4) for v in d["lo32"]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--temps", default="B,16,4,1", help="blind-score temperatures (numbers or B = the frozen "
                    "per-prompt rule); the FIRST is the status T")
    ap.add_argument("--kmax", type=int, default=12)
    ap.add_argument("--n-dev", type=int, default=64)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--n-random", type=int, default=50)
    ap.add_argument("--rand-sizes", default="8,12")
    ap.add_argument("--rand-seed", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--dev-only", action="store_true", help="greedy paths on dev only, no test scoring (T calibration)")
    a = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(0)
    temps = parse_temps(a.temps)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    a_id, b_id = (tok.encode(t, add_special_tokens=False)[0] for t in ("A", "B"))
    pid = json.load(open(CDIR / "meta.json"))["prompt_ids"]
    idx = split_ids(HERE / "raw" / "prompts.jsonl", pid)
    dev, test = idx["dev"][: a.n_dev], idx["test"][: a.n_test]
    z0, z1 = Rows(CDIR / "none_rec_tail.npy"), Rows(CDIR / "probe_rec_tail.npy")
    tail, w_out = merged_tail(MODEL, a.adapter)
    res = {"run": a.run, "model": MODEL, "adapter": str(a.adapter), "args": vars(a),
           "adapter_sha256": hashlib.sha256(Path(a.adapter).read_bytes()).hexdigest(), "a_id": a_id, "b_id": b_id,
           "dev_ids": [pid[i] for i in dev], "test_ids": [pid[i] for i in test], "temps": temps, "status_T": temps[0],
           "head_unit": f"query head = {HD}-col slice of o_proj input; blocks {FIRST}-{FIRST + 1}, {2 * NH} heads"}
    if not a.dev_only:
        res["cached_test"], _ = score_pairs(tail, w_out, z0, z1, test, pid, a_id, b_id)
    aud = KAuditor(tail, w_out, z0, z1, a_id, b_id, chunk=a.chunk, temps=temps)
    heads = [(li, h) for li in range(NL) for h in range(NH)]

    t1 = time.time()
    paths, greedy_sets = {}, {}
    for t in temps:
        g, paths[f"T{tn(t)}"] = kgreedy(aud, dev, heads, t, a.kmax)
        greedy_sets[t] = g
    res["greedy_dev"] = paths
    res["dev_secs"] = round(time.time() - t1, 1)
    if a.dev_only:
        out = OUT / "kaudit"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{a.run}.json").write_text(json.dumps(res))
        print("KAUDIT-DEV", a.run, json.dumps({T: [round(s["margin_nats"], 9) for s in p] for T, p in paths.items()}),
              flush=True)
        return

    # set = (heads, mlps); mlps: 0 = MLP of the first tail block, 1 = the last (swapped to the unprobed run's output)
    cut_sets = {"identity": ([], []), "all_heads": (heads, []), "all_heads+mlpA+mlpB": (heads, [0, 1]),
                "mlpA": ([], [0]), "mlpB": ([], [1]), "mlpA+mlpB": ([], [0, 1])}
    for t in temps:
        for k in range(1, a.kmax + 1):
            cut_sets[f"greedy_T{tn(t)}_k{k}"] = (greedy_sets[t][:k], [])
    for size in [int(x) for x in a.rand_sizes.split(",")]:
        for j in range(a.n_random):  # draw j: its own generator, seed rand_seed + j (PREREG: 1000 + j), same in every arm
            rng = np.random.default_rng(a.rand_seed + j)
            cut_sets[f"rand{size}_{j:02d}"] = ([heads[i] for i in sorted(rng.choice(len(heads), size, replace=False))], [])
    # dedupe identical sets (greedy paths of two T often share prefixes): score each distinct set once
    key = {k: (tuple(sorted(v[0])), tuple(sorted(v[1]))) for k, v in cut_sets.items()}
    uniq = list(dict.fromkeys(key.values()))
    base, per = aud.run(test, [(list(h), list(m)) for h, m in uniq])
    stats_u = [cut_stats(base, d, a_id, b_id, temps) for d in per]
    res["test_cut"] = {k: {"heads": [hname(x) for x in cut_sets[k][0]], "mlps": [FIRST + m for m in cut_sets[k][1]],
                           **stats_u[uniq.index(key[k])]} for k in cut_sets}
    lu, lp = base["lo_u"], base["lo_p"]
    res["test_base"] = {"strict": float(np.mean((lp > 0) & (lu < 0))),
                        "strict32": float(np.mean((base["lo_p32"] > 0) & (base["lo_u32"] < 0))),
                        "argmax_switch": float(np.mean((base["top_u"] == b_id) & (base["top_p"] == a_id))),
                        "push_mean": float((lp - lu).mean()),
                        **{f"kl_pu_T{tn(t)}": float(base[f"kl_pu_T{tn(t)}"].mean()) for t in temps},
                        "T_B": [round(float(v), 4) for v in base["T_B"]],
                        "T_B_floor_share": float(np.mean(base["T_B"] <= 1.0)),
                        "probed_lead_A_over_B_mean": float(lp.mean()),
                        "lo_unprobed": [round(float(v), 4) for v in lu], "lo_probed": [round(float(v), 4) for v in lp],
                        "lo_unprobed32": [round(float(v), 4) for v in base["lo_u32"]],
                        "lo_probed32": [round(float(v), 4) for v in base["lo_p32"]]}
    tc = res["test_cut"]
    hl = {"base_strict32": res["test_base"]["strict32"], "base_strict": res["test_base"]["strict"]}
    for f in ("strict32", "strict"):  # strict32 (fp32 rows) = the status field; bf16 strict reported next to it
        curve = {f"T{tn(t)}": [tc[f"greedy_T{tn(t)}_k{k}"][f] for k in range(1, a.kmax + 1)] for t in temps}
        hl[f"{f}_after_greedy_k"] = curve
        hl[f"{f}_smallest_k_below_0.25"] = {T: next((k + 1 for k, v in enumerate(c) if v < 0.25), None)
                                            for T, c in curve.items()}
        hl[f"{f}_random_min_median_max"] = {
            s: [min(v), float(np.median(v)), max(v)] for s in a.rand_sizes.split(",")
            for v in [[tc[k][f] for k in tc if k.startswith(f"rand{s}_")]] if v}
    hl.update({"identity_strict32": tc["identity"]["strict32"], "all_heads_strict32": tc["all_heads"]["strict32"],
               "identity_ok": bool(tc["identity"]["strict32"] == res["test_base"]["strict32"]
                                   and tc["identity"]["strict"] == res["test_base"]["strict"])})
    res["headline"] = hl
    res["secs"] = round(time.time() - t0, 1)
    res["torch"] = torch.__version__
    out = OUT / "kaudit"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.run}.json").write_text(json.dumps(res))
    if DEV == "mps":
        print("MPS driver_allocated_memory GB", round(torch.mps.driver_allocated_memory() / 1e9, 2), flush=True)
    print("KAUDIT", a.run, json.dumps(res["headline"]), "secs", res["secs"], flush=True)


if __name__ == "__main__":
    main()
