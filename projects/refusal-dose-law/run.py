"""PREREG v1 runs. Resumable: a cell whose output file exists is skipped.
Usage: env/.venv/bin/python -u swarm-theory-1031-refusal-dose-law/run.py <stage> [<stage> ...]
  gen1 gen2 gen3 ft4 gen5 gen6 ft7   (PREREG v1 'Judged cells' order)
Outputs: data/runs/<cell>.jsonl (generation, 64 tokens greedy) and data/ft/<cell>.json (first-token score, p_down)."""
import json
import sys
import time

import numpy as np
import torch

from common import GATE, L, P, chat, encode, generate_rows, load_model, save_jsonl, string_refusal, RowEdit

M5 = [6.0, 12.0, 18.0, 24.0, 30.0]
S = json.loads((P / "data/splits.json").read_text())
TEST = S["test_harmful"]
PH = P / "data/phase0"
UNITS = torch.load(PH / "units.pt")
INFO = json.loads((PH / "phase0.json").read_text())
LS = torch.load(PH / "lstar.pt")
RN, NN = INFO["RN"], INFO["NN_neg"] + INFO["NN_pos"]
PM = INFO["PM_pins"]
OUT, FT = P / "data/runs", P / "data/ft"
tok, model = load_model()


def spec(cell):
    """cell name 'UNIT@m' -> RowEdit spec."""
    if cell == "base":
        return None
    u, m = cell.split("@")
    kind, v = UNITS[u]
    if kind == "gate":
        return ("neuron", GATE, float(m), v)
    return ("neuron", int(v), float(m))


def cells_for(stage):
    """[(cell, n_prompts)] in frozen run order."""
    c = []
    if stage == "gen1":
        c = [("base", 200)] + [(f"{u}@{m:g}", 200) for u in ["G", "G_on", "G_off"] for m in M5]
    elif stage == "gen2":
        c = [(f"TWIN_{k}@{m:g}", 100) for k in range(3) for m in M5]
        c += [(f"D_{x}@{m:g}", 100) for x in [0.05, 0.1, 0.2, 0.6, 1.0] for m in M5]
    elif stage == "gen3":
        c = [(f"RN_{j}@{m:g}", 50) for j in RN[:10] for m in [18.0, 30.0]]
        c += [(f"RD_{k}@{m:g}", 50) for k in range(5) for m in [18.0, 30.0]]
    elif stage == "gen5":
        c = [(f"NN_{j}@{m:g}", 50) for j in NN for m in M5]
    elif stage == "gen6":
        c = [(f"FK@{m:g}", 50) for m in M5]
        c += [(f"{k}@{v:.4f}", 50) for k, v in PM.items()]
    return c


def pm_spec(cell):
    u, m = cell.split("@")
    return ("neuron", int(u.split("_")[1]), float(m))


def run_gen(stage, k_arms=4):
    todo = [(c, n) for c, n in cells_for(stage) if not (OUT / f"{c}.jsonl").exists()]
    for n in sorted({n for _, n in todo}, reverse=True):
        group = [c for c, nn in todo if nn == n]
        for s in range(0, len(group), k_arms):
            chunk = group[s:s + k_arms]
            t0 = time.time()
            specs = [pm_spec(c) if c.startswith("PM_") else spec(c) for c in chunk]
            res = generate_rows(tok, model, TEST[:n], specs, max_new=64, bs_prompts=50)
            for k, c in enumerate(chunk):
                rows = [{"i": i, "prompt": TEST[i], "reply": r, "refusal": string_refusal(r)}
                        for i, r in enumerate(res[k])]
                save_jsonl(OUT / f"{c}.jsonl", rows)
                print(f"{stage} {c:22s} n={n} string_refusal {np.mean([r['refusal'] for r in rows]):.3f}", flush=True)
            print(f"  {len(chunk)} arms x {n} in {time.time() - t0:.0f}s", flush=True)


# ---------------- first-token score + downstream projection, via the cached clean input of layer L ----------------
R_IDS = sorted({tok.encode(s)[0] for s in ["I", "I'm", "I’m", "Sorry", "I cannot", "I can't"]})
RH_STAR, LSTAR = LS["r_hat_Lstar"].to("mps"), LS["L_star"]


class _Stop(Exception):
    pass


@torch.no_grad()
def tail_cache(n=100, bs=50):
    texts = [chat(tok, p) for p in TEST[:n]]
    cap, batches = {}, []

    def pre(m, args, kwargs):
        cap["h"] = (args[0] if args else kwargs.pop("hidden_states")).clone()
        cap["kw"] = {k: v for k, v in kwargs.items() if k != "hidden_states"}
        raise _Stop

    hk = model.model.layers[L].register_forward_pre_hook(pre, with_kwargs=True)
    for s in range(0, n, bs):
        enc = encode(tok, texts[s:s + bs])
        try:
            model.model(**enc, use_cache=False)
        except _Stop:
            pass
        batches.append((list(range(s, min(s + bs, n))), cap["h"], cap["kw"], enc["attention_mask"]))
    hk.remove()
    return batches


@torch.no_grad()
def tail(batch, sp):
    idx, h0, kw, am = batch
    first = am.float().argmax(1)
    ed = RowEdit(model, [sp] * len(idx), first) if sp is not None else None
    try:
        h = h0
        for li, layer in enumerate(model.model.layers[L:], start=L):
            h = layer(h, **kw)
            h = h[0] if isinstance(h, tuple) else h
    finally:
        if ed is not None:
            ed.close()
    hn = model.model.norm(h)  # LSTAR = 27 (last layer): r_hat_Lstar was built on the final-normed state
    lg = model.lm_head(hn[:, -1]).float()
    lp = torch.log_softmax(lg, -1)
    pr = torch.logsumexp(lp[:, R_IDS], -1)
    score = (pr - torch.log1p(-pr.exp().clamp(max=1 - 1e-6)))
    return score.cpu(), hn.float()


def run_ft(cells, n=100):
    todo = [c for c in cells if not (FT / f"{c}.json").exists()]
    if not todo:
        return
    batches = tail_cache(n)
    clean = [tail(b, None) for b in batches]
    FT.mkdir(parents=True, exist_ok=True)
    if not (FT / "base.json").exists():
        sc = torch.cat([c[0] for c in clean]).tolist()
        (FT / "base.json").write_text(json.dumps({"cell": "base", "n": n, "score": sc, "R_IDS": R_IDS}))
    t0 = time.time()
    for c in todo:
        if c == "base":
            continue
        sp = pm_spec(c) if c.startswith("PM_") else spec(c)
        sc, pd_tok, pd_last = [], [], []
        for b, (s0, h0) in zip(batches, clean):
            s1, h1 = tail(b, sp)
            sc += s1.tolist()
            am = b[3].bool()
            proj = -((h1 - h0) @ RH_STAR) / h0.norm(dim=-1)  # [B, T]
            for r in range(proj.shape[0]):
                pos = torch.nonzero(am[r]).flatten()[1:]
                pd_tok.append(proj[r, pos].mean().item())
                pd_last.append(proj[r, -1].item())
        (FT / f"{c}.json").write_text(json.dumps({"cell": c, "n": n, "score": sc, "p_down": pd_tok,
                                                  "p_down_last": pd_last, "L_star": LSTAR, "R_IDS": R_IDS}))
    print(f"ft {len(todo)} cells in {time.time() - t0:.0f}s", flush=True)


def ft_cells(stage):
    if stage == "ft4":
        return ["base", "G@30"] + [f"RN_{j}@30" for j in RN]
    # ft7: every judged unit first (downstream row), then all RN and RD at all M5
    c = []
    for st in ["gen1", "gen2", "gen3", "gen5", "gen6"]:
        c += [x for x, _ in cells_for(st) if x != "base"]
    if "G_offK" in UNITS:
        c += [f"G_offK@{m:g}" for m in M5]
    c += [f"RN_{j}@{m:g}" for j in RN for m in M5] + [f"RD_{k}@{m:g}" for k in range(20) for m in M5]
    return list(dict.fromkeys(c))


@torch.no_grad()
def run_d2():
    """D2 (Tove #10948, Kasper #10949): refusal SUBSPACE K and G_off_K; cos rows over template positions/layers."""
    f = P / "data/phase0/d2.json"
    W = model.model.layers[L].mlp.down_proj.weight.float()
    w = W[:, GATE].cpu()
    if not f.exists():
        def states(prompts, bs=50):
            out = []
            for s in range(0, len(prompts), bs):
                enc = encode(tok, [chat(tok, p) for p in prompts[s:s + bs]])
                hs = model(**enc, output_hidden_states=True).hidden_states
                out.append(torch.stack([hs[l + 1][:, -9:].float() for l in range(12, 17)], 1).cpu())  # [B,5,9,d]
            return torch.cat(out)
        Hh, Hn = states(S["dev_harmful"][:100]), states(S["test_harmless"][:100])
        R = Hh.mean(0) - Hn.mean(0)  # [5 layers (12..16), 9 template positions, d]
        r13 = torch.tensor(np.load(PH / "r13.npy"))
        rh = r13 / r13.norm()
        w_off = w - (w @ rh) * rh
        cosf = lambda a, b: float(a @ b / (a.norm() * b.norm()))
        toks = ["<|im_end|>", "\\n", "<|im_start|>", "assistant", "\\n", "<think>", "\\n\\n", "</think>", "\\n\\n"]
        rows = {"L13_positions": [{"pos": i - 9, "tok": toks[i], "cos_w": cosf(w, R[1, i]), "cos_w_off": cosf(w_off, R[1, i]),
                                   "cos_r_last": cosf(R[1, i], r13)} for i in range(9)],
                "last_token_layers": [{"layer": 12 + k, "cos_w": cosf(w, R[k, -1]), "cos_w_off": cosf(w_off, R[k, -1])}
                                      for k in range(5)]}
        cand = [R[1, i] for i in range(4, 9)] + [R[0, -1], R[2, -1]]  # L13 last 5 template tokens, L12 last, L14 last
        names = [f"L13_pos{i - 9}" for i in range(4, 9)] + ["L12_last", "L14_last"]
        Q, kept = [], []
        for v, nm in zip(cand, names):
            res = v.clone()
            for q in Q:
                res = res - (res @ q) * q
            if res.norm() >= 0.1 * v.norm():
                Q.append(res / res.norm()); kept.append(nm)
        Qm = torch.stack(Q, 1)  # [d, k]
        w_offK = w - Qm @ (Qm.T @ w)
        rows.update(k=len(Q), kept=kept, energy_share_in_K=float((Qm.T @ w).norm() ** 2 / w.norm() ** 2),
                    energy_share_on_r=float((w @ rh) ** 2 / w.norm() ** 2), cos_w_offK_r13=cosf(w_offK, r13))
        torch.save({"w_offK": w_offK, "Q": Qm}, PH / "d2.pt")
        f.write_text(json.dumps(rows, indent=1))
        print("d2 rows", json.dumps({k: rows[k] for k in ("k", "kept", "energy_share_in_K", "energy_share_on_r")}), flush=True)
    UNITS["G_offK"] = ("gate", torch.load(PH / "d2.pt")["w_offK"])
    todo = [c for c in [f"G_offK@{m:g}" for m in M5] if not (OUT / f"{c}.jsonl").exists()]
    for s in range(0, len(todo), 4):
        chunk = todo[s:s + 4]
        res = generate_rows(tok, model, TEST, [spec(c) for c in chunk], max_new=64, bs_prompts=50)
        for k, c in enumerate(chunk):
            rows_ = [{"i": i, "prompt": TEST[i], "reply": r, "refusal": string_refusal(r)} for i, r in enumerate(res[k])]
            save_jsonl(OUT / f"{c}.jsonl", rows_)
            print(f"d2 {c:22s} n=200 string_refusal {np.mean([r['refusal'] for r in rows_]):.3f}", flush=True)


for stage in sys.argv[1:]:
    print("== stage", stage, time.strftime("%H:%M:%S"), flush=True)
    try:
        if stage == "d2":
            run_d2()
        elif stage.startswith("gen"):
            run_gen(stage)
        else:
            run_ft(ft_cells(stage))
    except Exception as e:  # keep the chain alive: later stages still run
        import traceback
        traceback.print_exc()
        print("STAGE FAILED", stage, repr(e), flush=True)
print("== all done", time.strftime("%H:%M:%S"), flush=True)
