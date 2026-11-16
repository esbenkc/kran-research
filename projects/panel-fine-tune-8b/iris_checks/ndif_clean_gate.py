"""Iris NDIF pre-check (cross-check, never the status): the clean Qwen3-8B 20-readout max-panel at its own 1% line,
with the PREREG s2/s3 rules, from country-token states read on NDIF (NNsight and NDIF, arXiv 2407.14561).

Stage 1 (remote): batches of 64 statements, right padding, no special tokens (common.tokenize); per batch one trace
saves the last-country-token state at the 19 pre-norm panel readouts (output of block R-1) and the final-norm output
(R = 37), stacked on the server -> cache/b<k>.npy (fp16, (B, 20, 4096)). Resumes from the cache.
Stage 2 (local): probes fit with refit.select / refit.fit (the project's 1020/1021 C rule) on the 1400 fit-pool
statements, z-moments on the 700 fit-pool true statements, panelmax, line = 2nd-highest of the 200 refit-true,
catch on the 596 eval false -> gate.json.
Run: .venv/bin/python ndif_clean_gate.py [--test]
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
PROJ = HERE.parents[1] / "kran-research" / "projects" / "panel-fine-tune-8b"
os.environ.setdefault("PFT_MODEL", "Qwen/Qwen3-8B")
sys.path.insert(0, str(PROJ))
from common import PANEL, N_LAYERS, POSTNORM, country_pos, pool_idx, rows, threshold, tokenize  # noqa: E402

CACHE = HERE / "ndif_cache"
CACHE.mkdir(exist_ok=True)
BATCH = 64


def stage1(test=False):
    import torch
    from nnsight import CONFIG, LanguageModel
    CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = LanguageModel("Qwen/Qwen3-8B", tokenizer=tok)
    R_, _, _, _ = rows()
    ids = tokenize(tok, [r["text"] for r in R_])
    cpos, _ = country_pos(tok, ids)
    pid = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    n = len(ids)
    nb = (n + BATCH - 1) // BATCH
    assert N_LAYERS == 36 and POSTNORM == 37 and len(PANEL) == 20, (N_LAYERS, PANEL)
    for k in range(nb if not test else 1):
        f = CACHE / f"b{k:03d}.npy"
        if f.exists() and not test:
            continue
        sl = list(range(k * BATCH, min(n, (k + 1) * BATCH)))
        if test:
            sl = sl[:4]
        L = max(len(ids[i]) for i in sl)
        inp = torch.full((len(sl), L), pid, dtype=torch.long)
        mask = torch.zeros((len(sl), L), dtype=torch.long)
        for r, i in enumerate(sl):
            inp[r, :len(ids[i])] = torch.tensor(ids[i])
            mask[r, :len(ids[i])] = 1
        ar = torch.arange(len(sl))
        cp = torch.tensor(cpos[sl])
        t0 = time.time()
        with model.trace({"input_ids": inp, "attention_mask": mask}, remote=True):
            hs = []
            for R in PANEL[:-1]:
                o = model.model.layers[R - 1].output
                h = o[0] if isinstance(o, tuple) else o
                hs.append(h[ar, cp])
            hs.append(model.model.norm.output[ar, cp])
            out = torch.stack(hs, 1).to(torch.float16).save()
        X = out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)
        print(k, X.shape, f"{time.time() - t0:.1f}s", flush=True)
        if test:
            print("norms per readout", np.linalg.norm(X.astype(np.float32), axis=-1).mean(0).round(1))
            return
        np.save(f, X)


def stage2():
    sys.path.insert(0, str(PROJ))
    from refit import fit, select
    _, y, city, split = rows()
    tr, line, ev = pool_idx()
    X = np.concatenate([np.load(CACHE / f"b{k:03d}.npy") for k in range(len(list(CACHE.glob("b*.npy"))))]).astype(np.float32)
    assert X.shape == (len(y), 20, 4096), X.shape
    S, info = {}, {}
    for j, R in enumerate(PANEL):
        C, res, conv = select(X[:, j], y, city, tr)
        sc, m = fit(X[:, j], y, tr, C)
        S[R] = m.decision_function(sc.transform(X[:, j]))
        info[R] = {"C": C, "oof_auroc": res[C], "converged": conv}
    trt = tr[y[tr] == 0]
    Z = np.stack([(S[R] - S[R][trt].mean()) / S[R][trt].std() for R in PANEL], 1)
    pm = Z.max(1)
    evf, evt = ev[y[ev] == 1], ev[y[ev] == 0]
    out = {"model": "Qwen/Qwen3-8B via NDIF (bf16, remote)", "panel": PANEL, "per_readout": {}}
    for j, R in enumerate(PANEL):
        t = threshold(S[R][line])
        out["per_readout"][R] = {**info[R], "eval_catch": float((S[R][evf] > t).mean()),
                                 "eval_false_alarm": float((S[R][evt] > t).mean())}
    t = threshold(pm[line])
    out["panelmax"] = {"line": t, "eval_catch": float((pm[evf] > t).mean()), "n_false": int(len(evf)),
                       "eval_false_alarm": float((pm[evt] > t).mean()), "gate_ge_0.90": bool((pm[evf] > t).mean() >= 0.90)}
    (HERE / "ndif_gate.json").write_text(json.dumps(out, indent=1))
    np.savez(HERE / "ndif_scores.npz", panelmax=pm, **{f"tok{R}": S[R] for R in PANEL})
    print(json.dumps(out["panelmax"]), flush=True)
    print({R: round(v["eval_catch"], 3) for R, v in out["per_readout"].items()})


if __name__ == "__main__":
    if "--stage2" in sys.argv:
        stage2()
    else:
        stage1(test="--test" in sys.argv)
        if "--test" not in sys.argv:
            stage2()
