"""Two implementations of the same pick rule must agree: the attacker's training monitor (pen.greedy_pick: o_proj
linearity, last-query block B, batched joint cuts) and the status audit (kaudit.kgreedy: forward hooks on the full
blocks). Same merged tail, same prompts, same temperature rule (fp32 logits / T), same K.

Checks: (1) the greedy picks agree step by step (or the steps where they differ are near-ties, listed with both
scores); (2) the blind removed share of every greedy prefix agrees (max abs diff); (3) cut_sets of a one-head set ==
cut_A / cut_B of that head, cut_sets of a 3-head set == cut_joint (max abs logit diff); (4) rule B's T_i agree.

    AP_MODEL=Qwen/Qwen3-1.7B $PY check_greedy.py --adapter <adapter.pt> --n 16 --k 6 --temps B,16
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from audit import merged_tail  # noqa: E402
from cfgm import CDIR, MODEL, NH, OUT, head_name  # noqa: E402
from kaudit import KAuditor, blind_removed, kgreedy, parse_temps, tn  # noqa: E402
from pen import TailParts, greedy_pick, resolve_temp, split_ab  # noqa: E402
from tail import DEV, Rows, split_ids  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--temps", default="B,16")
    ap.add_argument("--out", default="check_greedy.json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    a_id, b_id = (tok.encode(t, add_special_tokens=False)[0] for t in ("A", "B"))
    pid = json.load(open(CDIR / "meta.json"))["prompt_ids"]
    ids = split_ids(HERE / "raw" / "prompts.jsonl", pid)["dev"][: a.n]
    z0, z1 = Rows(CDIR / "none_rec_tail.npy"), Rows(CDIR / "probe_rec_tail.npy")
    tail, w_out = merged_tail(MODEL, a.adapter)
    heads = [(li, h) for li in range(2) for h in range(NH)]
    flat = {x: (x[0] * NH + x[1]) for x in heads}
    out = {"model": MODEL, "adapter": a.adapter, "n_prompts": len(ids), "k": a.k}
    for t in parse_temps(a.temps):
        # pen side (attacker's monitor): batches of 16 prompts, fp32 logits
        tp = TailParts(tail.layers, tail.norm, tail.rotary, w_out)
        tp.w32 = w_out.float()
        Ps = [(tp.base(torch.cat([z0[ids[s:s + 16]], z1[ids[s:s + 16]]]).to(DEV)), len(ids[s:s + 16]))
              for s in range(0, len(ids), 16)]
        import time
        t_g = time.time()
        pick_pen, steps_pen, _ = greedy_pick(tp, Ps, a.k, t)
        t_g = round(time.time() - t_g, 1)
        TB_pen = torch.cat([resolve_temp(P, n, "B")[:, 0] for P, n in Ps]).cpu().numpy()
        # (3) cut_sets vs cut_A / cut_B / cut_joint on the first batch
        P, n = Ps[0]
        lgs = tp.cut_sets(P, n, [([3], []), ([], [5]), ([1, 2], [7])])
        d3 = {"single_A_vs_cut_A": float((lgs[0] - tp.cut_A(P, n, [3])[0]).abs().max()),
              "single_B_vs_cut_B": float((lgs[1] - tp.cut_B(P, n, [5])[0]).abs().max()),
              "joint_vs_cut_joint": float((lgs[2] - tp.cut_joint(P, n, [1, 2], [7])).abs().max()),
              "logit_scale_maxabs": float(P["orig"].abs().max())}
        tp.remove()
        del Ps, P
        # kaudit side (status audit): hooks, bs 16
        aud = KAuditor(tail, w_out, z0, z1, a_id, b_id, bs=16, temps=[t])
        g_aud, path_aud = kgreedy(aud, ids, heads, t, a.k)
        aud.hk.mode = "off"
        base, per = aud.run(ids, [(g_aud[:j + 1], []) for j in range(a.k)] +
                            [([heads[i] for i in pick_pen[:j + 1]], []) for j in range(a.k)])
        sh_aud = [blind_removed(base, d, t) for d in per[:a.k]]
        sh_pen_on_aud = [blind_removed(base, d, t) for d in per[a.k:]]
        same = [flat[g_aud[j]] == pick_pen[j] for j in range(a.k)]
        out[f"T{tn(t)}"] = {
            "pen_greedy_secs": t_g, "picks_pen": [head_name(0, i) if i < NH else head_name(1, i - NH) for i in pick_pen],
            "picks_aud": [head_name(*x) for x in g_aud],
            "same_pick_per_step": same, "n_same": int(sum(same)),
            "pen_steps_best_runnerup_nats": steps_pen,
            "aud_steps": [{k: v for k, v in s.items() if k != "set"} for s in path_aud],
            "aud_prefix_share": sh_aud, "pen_prefix_share_scored_by_aud": sh_pen_on_aud,
            "prefix_share_maxabs_diff": float(np.max(np.abs(np.array(sh_aud) - np.array(sh_pen_on_aud)))),
            "T_B_maxabs_diff": float(np.abs(TB_pen - base["T_B"]).max()),
            "cut_sets_checks_maxabs_logit": d3}
        print(f"T{tn(t)}", json.dumps({k: v for k, v in out[f'T{tn(t)}'].items()
                                       if k in ("pen_greedy_secs", "picks_pen", "picks_aud", "n_same", "prefix_share_maxabs_diff",
                                                "T_B_maxabs_diff", "cut_sets_checks_maxabs_logit")}), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
