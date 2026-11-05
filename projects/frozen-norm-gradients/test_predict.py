"""Checks of the theory tick (PREREG s5) on ONE fixed non-pile sentence (no held-out token is touched).
P1 two-pass g_pred == g_frz + sum_k (g_k - g_frz), g_k = only norm k live (Felix's multi-affine form).
P2 last block (one norm downstream): g_pred == g_real.
P3 readout_first_order never runs a live downstream norm (source check + mode trace).
-> data/test_predict_<model>.json"""
import inspect
import json
import sys

import torch

model = sys.argv[1]
cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1)
text = "The old lighthouse keeper climbed the stairs every night, because the ships still needed the light to find the harbour."
res = {"model": model, "text": text}
if model == "gpt2":
    from engine import DEV, GPT2, REAL
    from transformers import AutoTokenizer
    tz = AutoTokenizer.from_pretrained("openai-community/gpt2")
    ids = torch.tensor([[50256] + tz(text)["input_ids"][:23]], device=DEV)
    M = GPT2()
    lnonly = dict(REAL, ln="frozen")
    for L, t in ((8, 15), (10, 20), (11, 20)):
        pos = torch.tensor([t], device=DEV); tok = ids[0, t + 1:t + 2]
        g_frz, g_pred, st = M.readout_first_order(ids, L, lnonly, pos, tok)
        g_sum = g_frz.clone()
        for n in st:
            g_k = M.readout(ids, L, lnonly, pos, tok, live_norms={n})[0]
            g_sum += g_k - g_frz
        g_real = M.readout(ids, L, REAL, pos, tok)[0]
        res[f"P1_L{L}_relerr_twopass_vs_sum"] = ((g_pred - g_sum).norm() / g_sum.norm()).item()
        res[f"L{L}_n_norms"] = len(st)
        if L == 11:
            res["P2_L11_relerr_pred_vs_real"] = ((g_pred - g_real).norm() / g_real.norm()).item()
    src = inspect.getsource(GPT2.readout_first_order)
else:
    from hf_engine import DEV, HFEngine
    repo = {"qwen3": "Qwen/Qwen3-0.6B", "smol": "HuggingFaceTB/SmolLM2-135M"}[model]
    E = HFEngine(repo)
    ids = torch.tensor([E.tok(text)["input_ids"][:24]], device=DEV)
    for L, t in (((19, 15), (23, 20), (27, 20)) if model == "qwen3" else ((20, 15), (25, 20), (29, 20))):
        pos = torch.tensor([t], device=DEV); tok = ids[0, t + 1:t + 2]
        fz = E.downstream(L)
        seen = []
        orig = E.set_modes
        E.set_modes = lambda fr, _o=orig: (seen.append(set(fr)), _o(fr))[1]
        g_frz, g_pred, st = E.readout_first_order(ids, L, pos, tok, fz)
        E.set_modes = orig
        res[f"P3_L{L}_modes_frozen_sets_ok"] = all(s == set(fz) or s == set() for s in seen) and any(s == set(fz) for s in seen)
        g_sum = g_frz.clone()
        for n in fz:
            g_k = E.readout(ids, L, pos, tok, frozen=[m for m in fz if m != n])[0]
            g_sum += g_k - g_frz
        g_real = E.readout(ids, L, pos, tok)[0]
        res[f"P1_L{L}_relerr_twopass_vs_sum"] = ((g_pred - g_sum).norm() / g_sum.norm()).item()
        res[f"L{L}_n_norms"] = len(fz)
        if L == E.n - 1:
            res[f"P2_L{L}_relerr_pred_vs_real"] = ((g_pred - g_real).norm() / g_real.norm()).item()
    src = inspect.getsource(HFEngine.readout_first_order)
if model == "gpt2":  # the GPT-2 tick runs only with cfg ln='frozen' (asserted) and never calls readout()
    res["P3_source_no_live_call"] = ("readout(" not in src) and ('assert cfg["ln"] == "frozen"' in src)
else:  # HF tick only ever sets the frozen set (or resets to all-live after its own passes)
    res["P3_source_no_live_call"] = ("readout(" not in src) and ("set_modes(())" in src)
json.dump(res, open(f"data/test_predict_{model}.json", "w"), indent=1)
print(json.dumps(res, indent=1))
