"""Gates before RL (PREREG section 4 + amendment 1.5), from raw/cache/base.

Per build, over the test (200) and dev (64) prompts:
  block26_changed   prompts with any bit change of the cached block-26 input vs the build's reference (none / none_emu)
  block26_maxabs    max |delta| at the block-26 input
  tail_changed / tail_max_dlogit   the same at the tail logits computed in the TRAINING pass (eager tail, cached input)
  whole_changed / whole_max_dlogit / whole_top1_flips   whole compiled graph, no recorder (cache.py meta)
  px_* mean P(X): cached tail, whole graph (plain), whole graph with recorder (rec)
Gate 3: |tail P(X) - whole-graph P(X)| per prompt, none build.

    env/.venv/bin/python kran-research/projects/numeric-chameleon-rl/gate.py
writes data/gate.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from tail import Rows, load_tail, logits_rows, splits, x_ids  # noqa: E402


def main() -> None:
    cdir = HERE / "raw" / "cache" / "base"
    meta = json.load(open(cdir / "meta.json"))
    pid = meta["prompt_ids"]
    idx = splits(pid)
    xs = x_ids()
    cfg, tail, w_out = load_tail(2)
    tail.eval()
    builds = ["none", "ref", "score", "none_emu", "ref_emu", "tag"]
    Z = {b: Rows(cdir / f"{b}_rec_tail.npy") for b in builds}
    W = {(b, ps): np.load(cdir / f"{b}_{ps}.npz") for b in builds for ps in ("rec", "plain")}
    SHA = {(b, ps): json.load(open(cdir / f"{b}_{ps}_sha.json")) for b in builds for ps in ("rec", "plain")}
    out = {"meta_builds": meta["builds"]}

    def LG(b, ps, ii):
        a = np.load(cdir / f"{b}_{ps}_logits.npy", mmap_mode="r")[ii]
        return torch.from_numpy(np.ascontiguousarray(a).view(np.int16)).view(torch.bfloat16).float()

    for split in ("test", "dev"):
        ii = idx[split]
        L = {b: logits_rows(tail, w_out, Z[b], ii) for b in builds}
        res = {}
        for b in builds:
            refb = "none_emu" if b.endswith("_emu") else "none"
            r = {"n": len(ii)}
            p_tail = torch.softmax(L[b], -1)[:, xs].sum(-1).double().numpy()
            r["px_tail"] = float(p_tail.mean())
            r["px_whole_plain"] = float(W[(b, "plain")]["px"][ii].mean())
            r["px_whole_rec"] = float(W[(b, "rec")]["px"][ii].mean())
            if b != refb and b != "tag":
                za, zb = Z[b].a[ii], Z[refb].a[ii]
                ch = (za != zb).reshape(len(ii), -1).any(1)
                fa = torch.from_numpy(np.ascontiguousarray(za).view(np.int16)).view(torch.bfloat16).float()
                fb = torch.from_numpy(np.ascontiguousarray(zb).view(np.int16)).view(torch.bfloat16).float()
                d = (fa - fb).abs()
                r["block26_changed"] = int(ch.sum())
                r["block26_maxabs"] = float(d.max())
                r["block26_changed_positions_mean"] = float((d.amax(-1) > 0).float().sum(-1).mean())
                dl = (L[b] - L[refb]).abs()
                r["tail_changed"] = int((dl.amax(-1) > 0).sum())
                r["tail_max_dlogit"] = float(dl.max())
                r["tail_top1_flips"] = int((L[b].argmax(-1) != L[refb].argmax(-1)).sum())
                p_ref = torch.softmax(L[refb], -1)[:, xs].sum(-1).double().numpy()
                r["tail_px_gap_off_minus_on"] = float((p_ref - p_tail).mean())
                r["tail_px_absdiff_mean"] = float(np.abs(p_ref - p_tail).mean())
                for ps in ("plain", "rec"):
                    sa, sb = SHA[(b, ps)], SHA[(refb, ps)]
                    r[f"whole_{ps}_changed"] = int(sum(sa[i] != sb[i] for i in ii))
                    la, lb = LG(b, ps, ii), LG(refb, ps, ii)
                    r[f"whole_{ps}_max_dlogit"] = float((la - lb).abs().max())
                    pw = W[(b, ps)]["px"][ii]
                    pr = W[(refb, ps)]["px"][ii]
                    r[f"whole_{ps}_px_gap_off_minus_on"] = float((pr - pw).mean())
                    r[f"whole_{ps}_top1_flips"] = int((W[(b, ps)]["top"][ii] != W[(refb, ps)]["top"][ii]).sum())
            if b == "ref":  # Kasper #6897 (b): does the cue the tail trains on look like the deployed cue?
                d_tail = p_ref - p_tail
                d_whole = W[("none", "plain")]["px"][ii] - W[("ref", "plain")]["px"][ii]
                r["kasper_b_corr_tail_vs_plain"] = float(np.corrcoef(d_tail, d_whole)[0, 1])
                nz = (np.abs(d_tail) > 0) & (np.abs(d_whole) > 0)
                r["kasper_b_sign_agree"] = float(np.mean(np.sign(d_tail[nz]) == np.sign(d_whole[nz])))
                r["kasper_b_n_both_nonzero"] = int(nz.sum())
            if b == "none":  # Kasper #6897 (a): the recorder alone vs no recorder, whole graph
                r["kasper_a_rec_vs_plain_changed"] = int(sum(SHA[(b, "rec")][i] != SHA[(b, "plain")][i] for i in ii))
                r["kasper_a_rec_vs_plain_max_dlogit"] = float((LG(b, "rec", ii) - LG(b, "plain", ii)).abs().max())

                r["gate3_tail_vs_plain_absdiff_mean"] = float(np.abs(p_tail - W[(b, "plain")]["px"][ii]).mean())
                r["gate3_tail_vs_plain_absdiff_max"] = float(np.abs(p_tail - W[(b, "plain")]["px"][ii]).max())
                r["gate3_tail_vs_rec_absdiff_max"] = float(np.abs(p_tail - W[(b, "rec")]["px"][ii]).max())
            res[b] = r
        out[split] = res
    (HERE / "data" / "gate.json").write_text(json.dumps(out, indent=1))
    for split in ("test", "dev"):
        for b, r in out[split].items():
            print(split, b, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
