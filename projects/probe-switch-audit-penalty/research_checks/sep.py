"""Can the block-26 input at ONE position tell 'probe attached' from 'not attached'? (Iris, Tove #7229)

A per-token MLP in block 26 or 27 sees only its own position's residual (plus what attention moved there). If a probe
on the answer-position block-26 input cannot separate probed from unprobed inputs, a head-free route must fail.
Single inputs, not pairs: the switch must decide from one forward pass.
Features: (a) answer position (last token), (b) mean over all 80 positions (a crude stand-in for what a head pools),
(c) every single position (logistic only) -> which positions carry the trigger.
Probes: logistic regression (standardised, C=1) and a 2-layer MLP (256 hidden, early stopping). Train = the 512 TRAIN
prompts x 2 conditions; test = the 128 TEST prompts x 2. CPU fp32, sklearn.

    PY sep.py <cache dir with none_rec_tail.npy, probe_rec_tail.npy, meta.json> <prompts.jsonl> <out.json>
"""
import json
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def bf16_to_f32(u16):
    return (np.ascontiguousarray(u16).astype(np.uint32) << 16).view(np.float32)


def main():
    cdir, prompts, out = sys.argv[1:4]
    t0 = time.time()
    pid = json.load(open(f"{cdir}/meta.json"))["prompt_ids"]
    split = {json.loads(l)["id"]: json.loads(l)["split"] for l in open(prompts)}
    tr = [i for i, p in enumerate(pid) if split[p] == "train"]
    te = [i for i, p in enumerate(pid) if split[p] == "test"][:128]
    z0 = np.load(f"{cdir}/none_rec_tail.npy", mmap_mode="r")
    z1 = np.load(f"{cdir}/probe_rec_tail.npy", mmap_mode="r")
    T = z0.shape[1]

    def feats(z, ids, pos):
        if pos != "mean":
            return bf16_to_f32(np.asarray(z[ids, pos]).view(np.uint16))
        return np.stack([bf16_to_f32(np.asarray(z[i]).view(np.uint16)).mean(0) for i in ids])

    def data(pos):
        Xtr = np.concatenate([feats(z0, tr, pos), feats(z1, tr, pos)])
        Xte = np.concatenate([feats(z0, te, pos), feats(z1, te, pos)])
        ytr = np.r_[np.zeros(len(tr)), np.ones(len(tr))]
        yte = np.r_[np.zeros(len(te)), np.ones(len(te))]
        return Xtr, ytr, Xte, yte

    def fit(pos, kind):
        Xtr, ytr, Xte, yte = data(pos)
        if kind == "logistic":
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        else:
            m = make_pipeline(StandardScaler(), MLPClassifier((256,), early_stopping=True, max_iter=500,
                                                              random_state=0))
        m.fit(Xtr, ytr)
        s = m.predict_proba(Xte)[:, 1]
        n = len(te)
        paired = float(np.mean(s[n:] > s[:n]))  # probed copy scored above its own unprobed copy
        return {"auroc": float(roc_auc_score(yte, s)), "acc": float(np.mean((s > 0.5) == yte)),
                "paired_win": paired, "train_acc": float(m.score(Xtr, ytr))}

    # size of the trigger at each position: |z1 - z0| / |z0|, test prompts
    a0 = bf16_to_f32(z0[te].view(np.uint16)); a1 = bf16_to_f32(z1[te].view(np.uint16))
    rel = (np.linalg.norm(a1 - a0, axis=-1) / np.linalg.norm(a0, axis=-1)).mean(0)
    frac_changed = (a1 != a0).mean((0, 2))
    res = {"n_train_prompts": len(tr), "n_test_prompts": len(te), "T": T,
           "rel_diff_by_pos": rel.tolist(), "frac_coords_changed_by_pos": frac_changed.tolist()}
    for pos, tag in ((T - 1, "answer"), ("mean", "mean_pool")):
        for kind in ("logistic", "mlp"):
            res[f"{tag}_{kind}"] = fit(pos, kind)
            print(tag, kind, res[f"{tag}_{kind}"], f"{time.time() - t0:.0f}s", flush=True)
    res["per_pos_logistic_auroc"] = {}
    for p in sorted(set(range(0, T, 8)) | set(range(T - 10, T))):
        res["per_pos_logistic_auroc"][p] = fit(p, "logistic")["auroc"]
    print("per-pos AUROC", {k: round(v, 3) for k, v in res["per_pos_logistic_auroc"].items()}, flush=True)
    res["secs"] = round(time.time() - t0, 1)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
