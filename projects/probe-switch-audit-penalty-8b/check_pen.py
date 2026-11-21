"""Port check of pen.py against audit.py (two independent implementations of the same head swap) on a merged tail.

pen.py patches by linearity of o_proj and a last-query block B; audit.py patches by forward hooks on the full blocks.
On the same merged bf16 tail and the same prompts, the blind removed share r(h) / D of every single head, of a joint
3-head set and of all heads must agree (bf16 rounding only). Also: cut_joint of one head == cut_A / cut_B of that
head, cut_joint of nothing == the probed run, and the AO ratio does not move when the probe's effect is scaled
(Kasper #27322: r and D both scale with the square of a small push).

    AP_MODEL=Qwen/Qwen3-0.6B AP_OUT=smoke/q06 $PY check_pen.py --adapter <adapter.pt>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from audit import Auditor, merged_tail, score  # noqa: E402
from cfgm import CDIR, MODEL, NH, OUT, head_name  # noqa: E402
from pen import TailParts, kl_rows, removed, singles_nograd  # noqa: E402
from tail import DEV, Rows, split_ids  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    a_id, b_id = (tok.encode(t, add_special_tokens=False)[0] for t in ("A", "B"))
    pid = json.load(open(CDIR / "meta.json"))["prompt_ids"]
    ids = split_ids(HERE / "raw" / "prompts.jsonl", pid)["dev"][: a.n]
    z0, z1 = Rows(CDIR / "none_rec_tail.npy"), Rows(CDIR / "probe_rec_tail.npy")
    tail, w_out = merged_tail(MODEL, a.adapter)
    n = len(ids)
    tp = TailParts(tail.layers, tail.norm, tail.rotary, w_out)
    P = tp.base(torch.cat([z0[ids], z1[ids]]).to(DEV))
    lp = F.log_softmax(P["orig"], -1)
    lp_u = lp[:n]
    D = float(kl_rows(lp[n:], lp_u).mean())
    r_pen, _ = singles_nograd(tp, P, n)
    share_pen = (r_pen / D).cpu().numpy()
    tp.remove()
    aud = Auditor(tail, w_out, z0, z1, a_id, b_id, bs=n)
    heads = [(li, h) for li in range(2) for h in range(NH)]
    top3 = [heads[i] for i in np.argsort(-share_pen)[:3]]
    sets = [([x], []) for x in heads] + [(top3, []), (heads, []), ([], [])]
    base, per = aud.run(ids, sets, "cut")
    share_aud = np.array([score(base, d, "cut", a_id, b_id)["blind_removed"] for d in per])
    out = {"model": MODEL, "n_prompts": n, "D_pen": D, "D_aud": float(base["kl_pu"].mean())}
    out["single_max_abs_diff"] = float(np.abs(share_pen - share_aud[: 2 * NH]).max())
    out["single_top3_pen"] = [head_name(*x) for x in top3]
    out["single_top3_aud"] = [head_name(*heads[i]) for i in np.argsort(-share_aud[: 2 * NH])[:3]]

    tp = TailParts(tail.layers, tail.norm, tail.rotary, w_out)
    P = tp.base(torch.cat([z0[ids], z1[ids]]).to(DEV))
    pA = sorted(h for li, h in top3 if li == 0)
    pB = sorted(h for li, h in top3 if li == 1)
    r3 = float(removed(lp_u, torch.tensor(D, device=DEV), tp.cut_joint(P, n, pA, pB)))
    out["joint3_share_pen"], out["joint3_share_aud"] = r3 / D, float(share_aud[2 * NH])
    rall = float(removed(lp_u, torch.tensor(D, device=DEV), tp.cut_all(P, n)))
    rall_j = float(removed(lp_u, torch.tensor(D, device=DEV), tp.cut_joint(P, n, list(range(NH)), list(range(NH)))))
    out["all_share_pen_cut_all"], out["all_share_pen_joint"] = rall / D, rall_j / D
    out["all_share_aud"] = float(share_aud[2 * NH + 1])
    out["identity_share_aud"] = float(share_aud[2 * NH + 2])
    lgA, lgB = tp.cut_A(P, n, [pA[0] if pA else 0]), tp.cut_B(P, n, [pB[0] if pB else 0])
    jA = tp.cut_joint(P, n, [pA[0] if pA else 0], [])
    jB = tp.cut_joint(P, n, [], [pB[0] if pB else 0])
    out["joint_vs_cutA_maxabs_logit"] = float((jA - lgA[0]).abs().max())
    out["joint_vs_cutB_maxabs_logit"] = float((jB - lgB[0]).abs().max())
    out["joint_empty_vs_probed_maxabs_logit"] = float((tp.cut_joint(P, n, [], []) - P["orig"][n:]).abs().max())

    # AO ratio under a scaled probe effect: probed cache input = unprobed + s x (probed - unprobed), s = 1 and 0.1
    def ao(s):
        x0, x1 = z0[ids].float(), z1[ids].float()
        xs = torch.cat([x0, x0 + s * (x1 - x0)]).to(torch.bfloat16).to(DEV)
        Q = tp.base(xs)
        lq = F.log_softmax(Q["orig"], -1)
        d = kl_rows(lq[n:], lq[:n]).mean()
        r, _ = singles_nograd(tp, Q, n)
        idx = r.topk(3).indices.tolist()
        lg = tp.cut_joint(Q, n, sorted(i for i in idx if i < NH), sorted(i - NH for i in idx if i >= NH))
        return float(torch.relu(removed(lq[:n], d, lg)) / torch.clamp(d, min=1e-4)), float(d)
    out["ao_ratio_s1"], out["ao_ratio_s01"] = ao(1.0), ao(0.1)
    print(json.dumps(out, indent=1))
    (OUT / "checks").mkdir(parents=True, exist_ok=True)
    (OUT / "checks" / "check_pen.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
