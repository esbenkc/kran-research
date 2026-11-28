"""Audit readouts on one adapter (no training), with kaudit.py's auditor (same merged bf16 tail, cached block-A
inputs, head unit and swap = 'cut'):

  1. single cuts: each of the 2 x NH tail heads cut alone, on DEV and on TEST: push_removed (share of the probed-minus-
     unprobed answer log-odds removed; the label-aware 'carrier' size), strict32 left, and the blind removed share at
     every T of --temps; plus the two-answer removed share (below). The oracle carrier of PREREG = the head with the
     largest single-cut push_removed on DEV.
  2. two-answer score (readout only, nobody trained against it): the blind score restricted to the two answer letters,
     KL2(p || q) of the 2-way distributions softmax([z_A, z_B]) (fp32 logits; it only needs the A-over-B log-odds),
     removed2 = 1 - mean KL2(p_cut || p_unprobed) / mean KL2(p_probed || p_unprobed). Joint greedy pick on DEV by
     removed2 to k = --kmax (same greedy and tie order as kaudit.kgreedy), each greedy set scored on TEST.

    AP_MODEL=Qwen/Qwen3-1.7B AP_OUT=data $PY readouts.py --run t1_s0 --adapter data/adapters/t1_s0_adapter.pt
writes OUT/readouts/<run>.json
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from audit import hname, merged_tail
from cfgm import CDIR, MODEL, NH, OUT
from kaudit import KAuditor, blind_removed, parse_temps, tn
from tail import DEV, Rows, split_ids

HERE = Path(__file__).parent
NL = 2


def kl2(lo_p, lo_q):
    """KL of the 2-way distributions given A-over-B log-odds (natural log), per prompt"""
    lp = -np.logaddexp(0, -lo_p); lp_b = -np.logaddexp(0, lo_p)
    lq = -np.logaddexp(0, -lo_q); lq_b = -np.logaddexp(0, lo_q)
    return np.exp(lp) * (lp - lq) + np.exp(lp_b) * (lp_b - lq_b)


def removed2(base, d):
    den = kl2(base["lo_p32"], base["lo_u32"]).mean()
    return float(1 - kl2(d["lo32"], base["lo_u32"]).mean() / den) if den > 0 else 0.0


def stats(base, d, temps):
    lu, lp, lc = base["lo_u"], base["lo_p"], d["lo"]
    lu32, lp32, lc32 = base["lo_u32"], base["lo_p32"], d["lo32"]
    return {"push_removed32": float(1 - (lc32 - lu32).mean() / (lp32 - lu32).mean()),
            "push_drop32": float((lp32 - lc32).mean()),
            "push_removed": float(1 - (lc - lu).mean() / (lp - lu).mean()),
            "strict32": float(np.mean((d["lo32"] > 0) & (base["lo_u32"] < 0))),
            "strict": float(np.mean((lc > 0) & (lu < 0))),
            **{f"blind_removed_T{tn(t)}": blind_removed(base, d, t) for t in temps},
            "removed2": removed2(base, d)}


def np_base(base):
    return {k: (np.asarray(v) if not isinstance(v, (float, int)) else v) for k, v in base.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--temps", default="B,16,4,1")
    ap.add_argument("--kmax", type=int, default=12)
    ap.add_argument("--n-dev", type=int, default=64)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--two-answer", action="store_true", help="also the two-answer greedy readout (PREREG: after 05:55 CET)")
    a = ap.parse_args()
    t0 = time.time()
    temps = parse_temps(a.temps)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    a_id, b_id = (tok.encode(t, add_special_tokens=False)[0] for t in ("A", "B"))
    pid = json.load(open(CDIR / "meta.json"))["prompt_ids"]
    idx = split_ids(HERE / "raw" / "prompts.jsonl", pid)
    dev, test = idx["dev"][: a.n_dev], idx["test"][: a.n_test]
    z0, z1 = Rows(CDIR / "none_rec_tail.npy"), Rows(CDIR / "probe_rec_tail.npy")
    tail, w_out = merged_tail(MODEL, a.adapter)
    aud = KAuditor(tail, w_out, z0, z1, a_id, b_id, chunk=a.chunk, temps=temps)
    heads = [(li, h) for li in range(NL) for h in range(NH)]
    res = {"run": a.run, "model": MODEL, "adapter": str(a.adapter), "args": vars(a),
           "adapter_sha256": hashlib.sha256(Path(a.adapter).read_bytes()).hexdigest(),
           "dev_ids": [pid[i] for i in dev], "test_ids": [pid[i] for i in test], "temps": temps}

    def run(ids, sets):
        base, per = aud.run(ids, sets)
        return np_base(base), [np_base(d) for d in per]

    single = {}
    for split, ids in (("dev", dev), ("test", test)):
        base, per = run(ids, [([h], []) for h in heads])
        single[split] = {hname(h): stats(base, d, temps) for h, d in zip(heads, per)}
        single[split + "_base"] = {"kl2_pu": float(kl2(base["lo_p32"], base["lo_u32"]).mean()),
                                   **{f"kl_pu_T{tn(t)}": float(base[f"kl_pu_T{tn(t)}"].mean()) for t in temps}}
    res["single"] = single
    sd = single["dev"]
    car = max(sd, key=lambda h: (sd[h]["push_drop32"], -list(sd).index(h)))  # PREREG: largest dev drop, lower index wins
    res["oracle_carrier"] = {"head": car, "dev": sd[car], "test": single["test"][car]}

    if a.two_answer:
        res.update(two_answer(run, heads, dev, test, a.kmax, temps))
    res["secs"] = round(time.time() - t0, 1)
    out = OUT / "readouts"
    out.mkdir(parents=True, exist_ok=True)
    name = f"{a.run}_two_answer" if a.two_answer else a.run
    (out / f"{name}.json").write_text(json.dumps(res))
    if DEV == "mps":
        print("MPS driver_allocated_memory GB", round(torch.mps.driver_allocated_memory() / 1e9, 2), flush=True)
    print("READOUTS", name, json.dumps({"carrier": res["oracle_carrier"]["head"],
          "carrier_dev": {k: round(v, 6) for k, v in res["oracle_carrier"]["dev"].items()},
          "two_answer_strict32_after_k": res.get("two_answer_strict32_after_k")}), "secs", res["secs"], flush=True)


def two_answer(run, heads, dev, test, kmax, temps):
    chosen, path = [], []
    for _ in range(kmax):  # two-answer greedy on DEV
        cand = [x for x in heads if x not in chosen]
        base, per = run(dev, [(chosen + [x], []) for x in cand])
        sc = np.array([removed2(base, d) for d in per])
        o = np.argsort(-sc, kind="stable")
        chosen = chosen + [cand[o[0]]]
        path.append({"set": [hname(x) for x in chosen], "dev_removed2": float(sc[o[0]]),
                     "runner_up": float(sc[o[1]]), "runner_up_head": hname(cand[o[1]])})
    base, per = run(test, [(chosen[:k], []) for k in range(1, kmax + 1)])
    for k, d in enumerate(per):
        path[k]["test"] = stats(base, d, temps)
    return {"two_answer_greedy": path, "two_answer_strict32_after_k": [p["test"]["strict32"] for p in path]}


if __name__ == "__main__":
    main()
