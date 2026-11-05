"""Frozen-norm read-out runs (PREREG.md, frozen record 0b4e178b: v2 + pins s9-s10).

Three stages per model (PREREG s9 P1: the tick is computed and posted before any real gradient exists):
  python run.py --model qwen3 --stage select              # token selection (forward only) -> data/runs/qwen3_selection.json
  python run.py --model qwen3 --stage 1 --sets A           # frozen cells + tick, NO real gradient
  python run.py --model qwen3 --stage 2 --sets A           # real gradient, then every cosine
Models: qwen3 (primary, Qwen/Qwen3-0.6B), smol (HuggingFaceTB/SmolLM2-135M), gpt2 (reading cells).

Read-out: g = d y / d mlp_out_L[t], y = logit[t, true next token] - mean logit. Every vector is saved
(data/runs/<model>_<set>_s1.npz frozen-side cells, <model>_<set>_s2.npz real gradients) with one CSV row per
doc x layer (<model>_<set>.csv, written by stage 2; <model>_<set>_s1.csv tick-only, written by stage 1).
"""
import argparse
import csv
import json
import os
import resource
import time

import numpy as np
import torch

import data as D

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["qwen3", "smol", "gpt2"])
ap.add_argument("--stage", required=True, choices=["select", "1", "2"])
ap.add_argument("--sets", default="A")
ap.add_argument("--batch", type=int, default=8)
args = ap.parse_args()

REPO = {"qwen3": "Qwen/Qwen3-0.6B", "smol": "HuggingFaceTB/SmolLM2-135M", "gpt2": "openai-community/gpt2"}[args.model]
LAYERS = {"qwen3": [19, 21, 23, 27], "smol": [20, 23, 25, 29], "gpt2": [8, 9, 10, 11]}[args.model]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "runs")  # committed (data/raw is gitignored repo-wide)
os.makedirs(OUT, exist_ok=True)
cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1)
rss_gb = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
t0 = time.time()
P = lambda name: os.path.join(OUT, name)

# ------------------------------------------------------------------ model
if args.model == "gpt2":
    from engine import DEV, GPT2, GRAPH, NORMS, REAL
    from transformers import AutoTokenizer
    tokz = AutoTokenizer.from_pretrained(REPO)
    M = GPT2()
    kind = "gpt2"
else:
    from hf_engine import DEV, HFEngine
    E = HFEngine(REPO)
    tokz = E.tok
    kind = "hf"


def resid_norms(ids):
    """|residual stream entering block L| at every position, for each read layer -> {L: [B, 128]}."""
    B = ids.shape[0]
    with torch.no_grad():
        if args.model == "gpt2":
            rs = {}
            M.forward(ids, 11, REAL, torch.arange(B, device=DEV), torch.full((B,), 127, device=DEV), ctl={"resid": rs})
            return {L: rs[L].norm(dim=-1).cpu() for L in LAYERS}
        E.set_modes(())
        hs = E.m.model(input_ids=ids, use_cache=False, output_hidden_states=True).hidden_states
        return {L: hs[L].norm(dim=-1).float().cpu() for L in LAYERS}  # hidden_states[L] = input to block L


# ------------------------------------------------------------------ stage select
if args.stage == "select":
    out = {"rule": "PREREG s3 + s10: t = default_rng(1105 + doc).integers(4,127), redraw while the residual norm "
                   "entering block L at t > 10 x median over positions 1..127 at ANY read layer; max 10 draws; "
                   "all-flagged doc skipped", "layers": LAYERS}
    for name, (docs, ids_all) in [("cand", D.candidates(tokz, kind)), ("S", D.smoke(tokz, kind))]:
        kept, skipped = [], []
        for s in range(0, len(docs), 16):
            ids = ids_all[s:s + 16].to(DEV)
            rn = resid_norms(ids)
            for i, doc in enumerate(docs[s:s + 16]):
                flag = lambda t: any(rn[L][i][t] > 10 * rn[L][i][1:128].median() for L in LAYERS)
                ds = D.draws(doc)
                k = next((k for k, t in enumerate(ds) if not flag(t)), None)
                if k is None:
                    skipped.append(doc)
                else:
                    kept.append({"doc": doc, "t": ds[k], "redraws": k, "row": s + i})
        if name == "cand":
            assert len(kept) >= 600, len(kept)
            out.update({"A": kept[:200], "B": kept[200:400], "C": kept[400:600], "skipped_before_C_end":
                        [d for d in skipped if d < kept[599]["doc"]]})
        else:
            out["S"] = kept
    json.dump(out, open(P(f"{args.model}_selection.json"), "w"), indent=1)
    print("select done", {k: len(v) for k, v in out.items() if isinstance(v, list)}, f"{time.time() - t0:.0f}s")
    raise SystemExit

SEL = json.load(open(P(f"{args.model}_selection.json")))
_cand = None


def set_tensors(setname):
    global _cand
    if setname == "S":
        docs, ids_all = D.smoke(tokz, kind)
    else:
        if _cand is None:
            _cand = D.candidates(tokz, kind)
        docs, ids_all = _cand
    by_doc = {d: i for i, d in enumerate(docs)}
    sel = SEL[setname]
    ids = torch.stack([ids_all[by_doc[r["doc"]]] for r in sel])
    return sel, ids


# ------------------------------------------------------------------ GPT-2 fixed-direction calibration (reading)
def gpt2_fixed_u():
    """u_k, lambda_k = top eigenpair of mean n n^T, n = x_hat/|x_hat| over positions 1..127 of the calibration
    docs (7500-7599), every norm. Fixed-u Jacobian term = lambda_k u_k u_k^T (best fixed rank-1 stand-in, in mean
    square, for the token's own n n^T)."""
    cdocs, cids = D.calibration(tokz, kind)
    acc = {n: torch.zeros(768, 768, dtype=torch.float64) for n in NORMS}  # CPU float64 (MPS has no fp64)
    cnt = 0

    def add(nid, x):
        xc = x[:, 1:] - x[:, 1:].mean(-1, keepdim=True)
        n = xc / xc.norm(dim=-1, keepdim=True)
        acc[nid] += torch.einsum("bti,btj->ij", n, n).cpu().double()

    orig = M._norm

    def capnorm(x, mode, nid, c):
        if nid[0] < 12:
            add(nid, x)
        return orig(x, mode, nid, c)

    with torch.no_grad():
        for s in range(0, len(cids), 16):
            ids = cids[s:s + 16].to(DEV)
            B = ids.shape[0]
            rs = {}
            M._norm = capnorm
            M.forward(ids, 11, REAL, torch.arange(B, device=DEV), torch.full((B,), 127, device=DEV), ctl={"resid": rs})
            M._norm = orig
            add((12, "lnf"), rs[12])
            cnt += B * 127
    U, LAM = {}, {}
    for n in NORMS:
        w, v = torch.linalg.eigh((acc[n] / cnt).cpu())
        U[n] = v[:, -1].float().to(DEV)
        LAM[n] = float(w[-1])
    return U, LAM, cdocs, cnt


# ------------------------------------------------------------------ stage 1: frozen side only
def stage1_cells(ids, L, pos, tok):
    cells, extra = {}, {}
    if args.model == "gpt2":
        for ln in ("live", "frozen"):
            for at in ("live", "frozen"):
                for ml in ("real", "tc"):
                    if (ln, at, ml) == ("live", "live", "real"):
                        continue  # that cell IS the real gradient: stage 2
                    cells[f"ln-{ln}_attn-{at}_mlp-{ml}"] = M.readout(ids, L, {"ln": ln, "attn": at, "mlp": ml}, pos, tok)[0]
        g_frz, g_pred, st = M.readout_first_order(ids, L, dict(REAL, ln="frozen"), pos, tok)
        extra["fo_pass1_vs_cell_maxabs"] = (g_frz - cells["ln-frozen_attn-live_mlp-real"]).abs().max().item()
        cells["frz"], cells["pred"] = g_frz, g_pred
        ga_frz, ga_pred, _ = M.readout_first_order(ids, L, GRAPH, pos, tok)
        cells["graph_pred"] = ga_pred  # all-frozen graph + tick: predicts the graph with live norms (swap a)
        cells["lnonly_fixedu"] = M.readout(ids, L, dict(REAL, ln="fixed"), pos, tok,
                                           u={n: U[n] * LAM[n] ** 0.5 for n in U})[0]
        for sd in range(3):
            gen = torch.Generator().manual_seed(100 + sd)
            ur = {}
            for n in U:
                v = torch.randn(768, generator=gen)
                v = v - v.mean()
                ur[n] = (v / v.norm()).to(DEV) * LAM[n] ** 0.5
            cells[f"lnonly_randu{sd}"] = M.readout(ids, L, dict(REAL, ln="fixed"), pos, tok, u=ur)[0]
    else:
        fz = E.downstream(L)
        g_frz, g_pred, st = E.readout_first_order(ids, L, pos, tok, fz)
        g_chk, _ = E.readout(ids, L, pos, tok, frozen=fz)
        extra["fo_pass1_vs_cell_maxabs"] = (g_chk - g_frz).abs().max().item()
        cells["frz"], cells["pred"] = g_frz, g_pred
        if any(n[1] == "q" for n in E.norms):
            cells["allnorm_frz"] = E.readout(ids, L, pos, tok, frozen=E.downstream(L, ("ln1", "ln2", "lnf", "q", "k")))[0]
    return cells, extra


if args.model == "gpt2" and args.stage == "1":
    U, LAM, cdocs, ccnt = gpt2_fixed_u()

for setname in args.sets.split(","):
    sel, ids_all = set_tensors(setname)
    tset = time.time()
    meta = {"repo": REPO, "layers": LAYERS, "device": str(DEV), "stage": args.stage, "set": setname, "gates": {}}
    if args.stage == "1":
        vecs, rows_out = {}, []
        if args.model == "gpt2":
            meta.update({"fixed_u_lambda": {f"{k[0]}_{k[1]}": v for k, v in LAM.items()}, "calibration_docs": cdocs,
                         "calibration_tokens": ccnt})
        for s in range(0, len(sel), args.batch):
            b = sel[s:s + args.batch]
            ids = ids_all[s:s + args.batch].to(DEV)
            B = ids.shape[0]
            rows = torch.arange(B, device=DEV)
            pos = torch.tensor([r["t"] for r in b], device=DEV)
            tok = ids[rows, pos + 1]
            for L in LAYERS:
                cells, extra = stage1_cells(ids, L, pos, tok)
                g = meta["gates"]
                g["fo_pass1_vs_cell_maxabs"] = max(g.get("fo_pass1_vs_cell_maxabs", 0.0), extra["fo_pass1_vs_cell_maxabs"])
                for k, v in cells.items():
                    vecs.setdefault(f"L{L}_{k}", []).append(v.float().cpu())
                c_pred = cos(cells["frz"], cells["pred"])
                for i in range(B):
                    rows_out.append({"model": args.model, "set": setname, "doc": b[i]["doc"], "L": L, "t": b[i]["t"],
                                     "redraws": b[i]["redraws"], "tok": int(tok[i]), "c_pred": float(c_pred[i])})
            print(f"[{args.model} s1 {setname}] {s + B}/{len(sel)} {time.time() - tset:.0f}s rss {rss_gb():.2f} GB", flush=True)
        np.savez_compressed(P(f"{args.model}_{setname}_s1.npz"), **{k: torch.cat(v).numpy() for k, v in vecs.items()})
        with open(P(f"{args.model}_{setname}_s1.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
            w.writeheader()
            w.writerows(rows_out)
        meta["tick_median_c_pred"] = {L: float(np.median([r["c_pred"] for r in rows_out if r["L"] == L])) for L in LAYERS}
        print("TICK (median c_pred = cos(g_frz, g_pred), no real gradient computed):", json.dumps(meta["tick_median_c_pred"]))
    else:
        s1 = np.load(P(f"{args.model}_{setname}_s1.npz"))
        s1rows = list(csv.DictReader(open(P(f"{args.model}_{setname}_s1.csv"))))
        reals = {L: [] for L in LAYERS}
        for s in range(0, len(sel), args.batch):
            b = sel[s:s + args.batch]
            ids = ids_all[s:s + args.batch].to(DEV)
            B = ids.shape[0]
            rows = torch.arange(B, device=DEV)
            pos = torch.tensor([r["t"] for r in b], device=DEV)
            tok = ids[rows, pos + 1]
            for L in LAYERS:
                if args.model == "gpt2":
                    g_real, y, _ = M.readout(ids, L, REAL, pos, tok)
                else:
                    g_real, y = E.readout(ids, L, pos, tok)
                    if s == 0:  # G1: our patched live norms == the library modules
                        import types
                        from hf_engine import _rms_forward
                        for mod in E.norms.values():
                            mod.forward = types.MethodType(type(mod).forward, mod)
                        g_lib, _ = E.readout(ids, L, pos, tok)
                        for mod in E.norms.values():
                            mod.forward = types.MethodType(_rms_forward, mod)
                        meta["gates"][f"G1_L{L}_min_cos_patched_vs_library"] = cos(g_real, g_lib).min().item()
                reals[L].append(g_real.float().cpu())
            print(f"[{args.model} s2 {setname}] {s + B}/{len(sel)} {time.time() - tset:.0f}s rss {rss_gb():.2f} GB", flush=True)
        reals = {L: torch.cat(v) for L, v in reals.items()}
        np.savez_compressed(P(f"{args.model}_{setname}_s2.npz"), **{f"L{L}_real": v.numpy() for L, v in reals.items()})
        # every cosine, per token (CPU float32 from the saved vectors: the skeptics rebuild exactly this)
        perm = None
        n = len(sel)
        rng = np.random.default_rng(7)
        while perm is None or (perm == np.arange(n)).any():  # P2: a fixed derangement of the tokens
            perm = rng.permutation(n)
        meta["shuffle_perm"] = perm.tolist()
        cells = sorted({k.split("_", 1)[1] for k in s1.files})
        rows_out = []
        for L in LAYERS:
            gr = reals[L]
            gf = torch.tensor(s1[f"L{L}_frz"])
            c = {k: cos(torch.tensor(s1[f"L{L}_{k}"]), gr) for k in cells}
            c_null = cos(gf, gr[perm])
            c_corr = cos(torch.tensor(s1[f"L{L}_pred"]), gr)
            ratio = gf.norm(dim=-1) / gr.norm(dim=-1)
            r1 = [r for r in s1rows if int(r["L"]) == L]
            for i in range(n):
                assert int(r1[i]["doc"]) == sel[i]["doc"]
                row = dict(r1[i])
                row.update({"c_frz": float(c["frz"][i]), "c_null": float(c_null[i]), "c_corrected": float(c_corr[i]),
                            "norm_ratio_frz": float(ratio[i])})
                for k in cells:
                    row[f"cos_{k}"] = float(c[k][i])
                rows_out.append(row)
        with open(P(f"{args.model}_{setname}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
            w.writeheader()
            w.writerows(rows_out)
    meta["seconds"] = time.time() - tset
    meta["max_rss_gb"] = rss_gb()
    json.dump(meta, open(P(f"{args.model}_{setname}_s{args.stage}_meta.json"), "w"), indent=1)
    print("done", args.model, setname, "stage", args.stage, json.dumps(meta["gates"]), f"{meta['seconds']:.0f}s", flush=True)
