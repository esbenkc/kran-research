"""Carrier rule (PREREG), Qwen3-1.7B frozen-head runs (MLP-only LoRA on blocks 20-27, every attention weight frozen).

For each of the 128 query heads of blocks 20-27: swap that ONE head's output (its 128-column slice of the o_proj input)
in the cue-on (probed) run for the same slice from the clean cue-off (unprobed) run of the same prompt, at all 80
positions, and run the rest of the tail. Engine = src1027/audit.py Runner.cut (the prior room's own code, unchanged).
  informed share removed = 1 - mean push_after / mean push,  push = lo(probed) - lo(unprobed),  lo = logp(A) - logp(B)
  blind share removed    = 1 - mean KL(p_cut || p_unprobed) / mean KL(p_probed || p_unprobed)
Carrier = argmax over heads of the informed share on DEV (64 prompts); the test split (128) is a cross-check row.
Gate: the merged tail must reproduce the stored cached test score of the run (data/runs/<run>.json) exactly.

    PY carriers17.py --run mlp8_none_s2
writes data/carriers/<run>.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

HERE = Path(__file__).parent
SRC = HERE / "src1027"
sys.path.insert(0, str(SRC))
from audit import FIRST, NH, NL, Runner  # noqa: E402
from tail import DEV, Rows, load_tail, score_pairs, split_ids  # noqa: E402
from train import MLP, attach, detach_merge  # noqa: E402


def merged_tail(run: str):
    meta = json.load(open(SRC / "data" / "runs" / f"{run}.json"))
    assert meta["args"]["arm"] == "mlp8", meta["args"]
    cfg, tail, w_out = load_tail("Qwen/Qwen3-1.7B", NL)
    tail.eval()
    for p in tail.parameters():
        p.requires_grad_(False)
    mods = attach(tail, MLP, meta["args"]["rank"], meta["args"]["alpha"])
    ad = torch.load(SRC / "raw" / "runs" / f"{run}_adapter.pt")
    named = dict(tail.named_parameters())
    assert set(ad) == {k for k in named if named[k].requires_grad}
    with torch.no_grad():
        for k, v in ad.items():
            named[k].copy_(v.to(DEV))
    detach_merge(tail, mods)
    return meta, cfg, tail, w_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    t0 = time.time()
    cache = SRC / "raw" / "cache"
    pid = json.load(open(cache / "meta_A.json"))["prompt_ids"]
    idx = split_ids(SRC / "raw" / "prompts.jsonl", pid)
    z0, z1 = Rows(cache / "none_A_b20.npy"), Rows(cache / "probe_A_b20.npy")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    A_ID, B_ID = tok.encode("A", add_special_tokens=False)[0], tok.encode("B", add_special_tokens=False)[0]
    meta, cfg, tail, w_out = merged_tail(a.run)
    test_stats, _ = score_pairs(tail, w_out, z0, z1, idx["test"], pid, A_ID, B_ID)
    gate = {k: (test_stats[k], meta["test_cached"][k]) for k in ("strict", "shift_mean")}
    assert all(abs(x - y) < 1e-9 for x, y in gate.values()), gate
    print("gate ok", gate, flush=True)
    heads = [(li, h) for li in range(NL) for h in range(NH)]
    run = Runner(tail, w_out, A_ID, B_ID)
    out = {"run": a.run, "gate_test_cached": gate, "rule": "PREREG sec 2: argmax informed share removed (audit.py removed), dev, single-head swap at all positions"}
    for split in ("dev", "test"):
        run.prepare(z0, z1, idx[split])
        rows = {}
        for hd in heads:
            r, _ = run.cut([hd])
            rows[f"{FIRST + hd[0]}.{hd[1]}"] = {k: round(v, 6) for k, v in r.items()}
        out[split] = rows
        top = sorted(rows.items(), key=lambda kv: -kv[1]["removed"])[:5]
        print(split, "top informed", [(k, v["removed"], v["blind_removed"]) for k, v in top], f"{time.time() - t0:.0f}s", flush=True)
    dev = out["dev"]
    rk = sorted(dev, key=lambda k: -dev[k]["removed"])
    out["carrier"], out["second"] = rk[0], rk[1]
    out["carrier_share"], out["second_share"] = dev[rk[0]]["removed"], dev[rk[1]]["removed"]
    out["carrier_blind_argmax_dev"] = max(dev, key=lambda k: dev[k]["blind_removed"])
    out["carrier_share_test"] = out["test"][out["carrier"]]["removed"]
    out["test_strict"] = test_stats["strict"]
    out["secs"] = round(time.time() - t0, 1)
    (HERE / "data" / "carriers").mkdir(parents=True, exist_ok=True)
    (HERE / "data" / "carriers" / f"{a.run}.json").write_text(json.dumps(out, indent=1))
    print("CARRIER", a.run, out["carrier"], out["carrier_share"], out["second"], out["second_share"], out["secs"], flush=True)


if __name__ == "__main__":
    main()
