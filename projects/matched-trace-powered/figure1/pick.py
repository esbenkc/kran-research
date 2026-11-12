"""Pick the Figure 1 example row from a joint read (Owain).

A row qualifies when, under common random numbers (same per-row noise in every pass):
  - the probe-off thought with the cue word allowed (toks_n) equals the probe-off thought with it banned (toks_nb),
    so ONE probe-off thought is the start point of both changes;
  - the planted cue fired (toks_nc opens with ' Need', toks_n does not);
  - the real probe trace changed the thought (toks_lb != toks_nb somewhere);
  - every thought decodes to plain ASCII text (no stray scripts), so a reader can follow it.
Among those it ranks rows by typicality: how close each arm's per-row change in cheating is to that arm's median over
ALL its changed rows (planted: rows where the cue fired; real: rows the trace changed), in IQR units, plus how close
the first changed token of the real trace is to its median position.

    PY=../../../env/.venv/bin/python
    $PY figure1/pick.py data/reads/raw/joint_s0_final [--top 10]
"""
import argparse
import json

import numpy as np

X = 14656  # ' Need'


def load_read(base):
    """npz of a joint read as full per-row arrays. The end read (read_tf.py) stores the probe-on sample (toks_lb, c_lb)
    only for rows in plain_row and the cue sample (toks_nc, c_nc) only for rows in cue_row; other rows get zeros."""
    z = dict(np.load(str(base) + ".npz"))
    n = len(z["prompt_idx"])
    z.setdefault("split", np.array(["test"] * n))
    for idx, keys in (("plain_row", ("toks_lb", "c_lb", "c_nlb")), ("cue_row", ("toks_nc", "c_nc"))):
        if idx in z:
            r = z[idx]
            for k in keys:
                full = np.zeros((n,) + z[k].shape[1:], z[k].dtype)
                full[r] = z[k]
                z[k] = full
            z["has_" + idx] = np.isin(np.arange(n), r)
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("read")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    z = load_read(a.read)
    js = json.load(open(a.read + ".json"))
    assert js.get("plant_tok", X) == X
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    assert tok.decode([X]) == " Need"

    m = z["split"] == a.split
    nb, lb, nc = z["toks_nb"], z["toks_lb"], z["toks_nc"]
    n = z.get("toks_n", nb)
    for k in ("c_n",):
        z.setdefault(k, z["c_nb"])
    m = m & (nc != 0).any(1) & (lb != 0).any(1)  # rows where the cue and probe-on thoughts were sampled (end read: draw 0)
    fired = (nc[:, 0] == X) & (n[:, 0] != X) & m
    rdiff = nb != lb
    rch = rdiff.any(1) & m
    first = np.where(rch, rdiff.argmax(1), -1)
    eP = z["c_n"] - z["c_nc"]          # per-row drop in cheating from the planted cue
    eR = z["c_nb"] - z["c_lb"]         # per-row drop in cheating from the real trace

    def stats(e):
        q1, med, q3 = np.percentile(e, [25, 50, 75])
        return float(med), float(max(q3 - q1, 1e-3))
    mP, iP = stats(eP[fired])
    mR, iR = stats(eR[rch])
    mF = float(np.median(first[rch]))
    print(f"rows {int(m.sum())}  cue fired {int(fired.sum())}  real changed {int(rch.sum())}")
    print(f"planted per-row drop: median {100 * mP:+.1f} pts (IQR {100 * iP:.1f});  real: median {100 * mR:+.1f} (IQR {100 * iR:.1f});"
          f"  real first changed token: median {mF:.0f}")

    same = (n == nb).all(1)
    cand = np.where(same & fired & rch)[0]

    def ascii_ok(ids):
        s = tok.decode(ids)
        return all(32 <= ord(ch) < 127 or ch in "\n\t" for ch in s)

    def visible(i):
        """the first changed token of the real trace is a visible word in both versions (not only whitespace)"""
        k = first[i]
        return tok.decode(nb[i][k: k + 1]).strip() != "" and tok.decode(lb[i][k: k + 1]).strip() != ""
    cand = [i for i in cand if ascii_ok(n[i]) and ascii_ok(nc[i]) and ascii_ok(lb[i]) and visible(i)]
    sc = sorted(cand, key=lambda i: abs(eP[i] - mP) / iP + abs(eR[i] - mR) / iR + abs(first[i] - mF) / 32)
    print(f"candidates (shared off thought, cue fired, trace changed, ascii, visible first change): {len(cand)}")
    for i in sc[: a.top]:
        print(f"row {i} prompt cd{int(z['prompt_idx'][i]):04d} seed {int(z['seed'][i])}  off {100 * z['c_n'][i]:.0f}%  "
              f"cue {100 * z['c_nc'][i]:.0f}% (drop {100 * eP[i]:+.0f})  real {100 * z['c_lb'][i]:.0f}% (drop {100 * eR[i]:+.0f})  "
              f"first {first[i]}  ndiff {int(rdiff[i].sum())}")
        print("   OFF ", repr(tok.decode(n[i])))
        print("   CUE ", repr(tok.decode(nc[i])))
        print("   REAL", repr(tok.decode(lb[i])))


if __name__ == "__main__":
    main()
