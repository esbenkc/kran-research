"""Engine smoke: compile time, footprint present in compiled decode, build switch is clean, sampler vs eager recompute.

    env/.venv/bin/python smoke.py --k 16 --B 32
"""
import argparse
import json
import time

import torch

import mt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--B", type=int, default=32)
    a = ap.parse_args()
    pm, inner = mt.load_policy()
    rows = mt.prompts("dev")[: a.B]
    ids = torch.tensor([r["ids"] for r in rows], device=mt.DEV)
    S = mt.Sampler(inner, a.B, a.k)
    res = {}
    for i, b in enumerate(["none", "leaky", "none", "leaky"]):
        t0 = time.time()
        g = torch.Generator(device=mt.DEV).manual_seed(5)
        o = S.rollout(ids, b, gen=g, record=True)
        torch.mps.synchronize()
        res[f"{b}{i}"] = o
        print(b, i, f"{time.time() - t0:.1f}s", "p_read", float(torch.softmax(o["d2"], -1)[:, 0].mean()), flush=True)
    n0, l1, n2, l3 = (res[k] for k in ("none0", "leaky1", "none2", "leaky3"))
    out = {
        "none_repeat_identical": bool(torch.equal(n0["toks"], n2["toks"]) and torch.equal(n0["d2"], n2["d2"])),
        "leaky_repeat_identical": bool(torch.equal(l1["toks"], l3["toks"]) and torch.equal(l1["d2"], l3["d2"])),
        "share_prefix_differs": float((n0["toks"] != l1["toks"]).any(1).float().mean()),
        "first_diff_pos": [int(x) for x in ((n0["toks"] != l1["toks"]).float().argmax(1))],
        "h26_prompt_bits_differ_share": float((n0["h26"][:, : mt.T] != l1["h26"][:, : mt.T]).float().mean()),
    }
    # teacher-force the none text under leaky: TV per token
    g = torch.Generator(device=mt.DEV).manual_seed(5)
    o0 = S.rollout(ids, "none", gen=g, keep_logp=True)
    o1 = S.rollout(ids, "leaky", forced=o0["toks"], ref_logp=o0["logp"])
    out["tv_mean"] = float(o1["tv"].mean())
    out["kl_mean"] = float(o1["kl"].mean())
    out["seq_logratio_mean"] = float((o1["lp_tok"] - o0["lp_tok"]).sum(1).mean())
    out["d2_diff_tf"] = float(((o1["d2"][:, 0] - o1["d2"][:, 1]) - (o0["d2"][:, 0] - o0["d2"][:, 1])).abs().mean())
    # eager clean recompute vs the sampler (none build) on the same text
    x = mt.full_ids(ids, n0["toks"])
    hc = mt.h26_clean(inner, x)
    out["h26_eager_vs_served_none_maxabs"] = float((hc - n0["h26"]).abs().max())
    out["h26_eager_vs_served_none_bits_share"] = float((hc != n0["h26"]).float().mean())
    t0 = time.time()
    pre, d2 = mt.tail_logits(inner, hc, a.k)
    torch.mps.synchronize()
    out["tail_s"] = round(time.time() - t0, 2)
    lp = torch.log_softmax(pre, -1).gather(2, n0["toks"][..., None])[..., 0]
    out["lp_eager_minus_served_none_mean_abs"] = float((lp - n0["lp_tok"]).abs().mean()) if a.k else 0.0
    out["d2_eager_minus_served"] = float(((d2[:, 0] - d2[:, 1]) - (n0["d2"][:, 0] - n0["d2"][:, 1])).abs().mean())
    pf, d2f = mt.tail_logits(inner, l1["h26"], a.k)
    out["d2_fed_tail_vs_served_leaky"] = float(((d2f[:, 0] - d2f[:, 1]) - (l1["d2"][:, 0] - l1["d2"][:, 1])).abs().mean())
    tok = __import__("transformers").AutoTokenizer.from_pretrained(mt.MODEL)
    out["examples_none"] = [tok.decode(n0["toks"][i]) for i in range(3)]
    out["examples_leaky"] = [tok.decode(l1["toks"][i]) for i in range(3)]
    out["rss_note"] = "see watchdog"
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
