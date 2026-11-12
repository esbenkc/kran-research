"""critic2 (Kasper, Claude fallback) independent rebuild of the headline cells of lora-switch-audit.

Own code, not audit.py: own RMSNorm (Qwen3 block-26 input_layernorm weight from the public safetensors), own cue
planting, SVD of the stacked q/k/v LoRA delta (audit.py uses an eigendecomposition of G), own rank rule.
Two nulls: the frozen one (numpy seed 20261112) and a fresh one (seed 7) to show ranks do not hinge on the draw.
Every cell is diffed against data/summary.json.  Reads data/ only (h0 rows, real rows, adapters_qkv26 via index.json; never raw/).
Run from a fresh clone: env/.venv/bin/python projects/lora-switch-audit/skeptics/critic2/rebuild.py > skeptics/critic2/rebuild_output.txt
"""
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
EPS = 1e-6


def bf16_rows(u16):
    return torch.from_numpy(np.ascontiguousarray(u16).view(np.int16)).view(torch.bfloat16).double().numpy()


snap = Path(snapshot_download("Qwen/Qwen3-0.6B"))
W_LN = None
for f in snap.glob("*.safetensors"):
    with safe_open(f, "pt") as sf:
        if "model.layers.26.input_layernorm.weight" in sf.keys():
            W_LN = sf.get_tensor("model.layers.26.input_layernorm.weight").double().numpy()
H0 = bf16_rows(np.load(ROOT / "data/h0_rows012_u16.npy"))  # [3, 1024] block-26 input rows 0-2 (pushed copy)
REAL_ON = bf16_rows(np.load(ROOT / "data/real_rows012_u16.npy"))
INDEX = json.load(open(ROOT / "data/adapters_qkv26/index.json"))  # run@step -> pushed adapter file


def rms(x):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + EPS) * W_LN


def cast(x):
    return torch.from_numpy(x).float().to(torch.bfloat16).double().numpy()


def nulls(seed):
    g = np.random.default_rng(seed).standard_normal((1000, 3, 1024))
    return (g / np.linalg.norm(g, axis=-1, keepdims=True)).astype(np.float32).astype(np.float64)


NULLS = {"frozen": nulls(20261112), "fresh": nulls(7)}
KEYF = {"u1110": "data/cue_u.npy", **{f"u{k}": f"data/cue_u_{k}.npy" for k in range(1111, 1117)}}


def cue_diff(U, row_scale, do_cast):
    """U [N,3,1024] unit rows, row_scale [3] absolute cue size per row -> c [N,3,1024] post-norm difference."""
    on = H0[None] + row_scale[None, :, None] * U
    if do_cast:
        on = cast(on)
    return rms(on) - rms(H0)[None]


def delta(path):
    ad = torch.load(path, map_location="cpu")
    mats = []
    for m in ("q", "k", "v"):
        pre = f"base_model.model.layers.0.self_attn.{m}_proj"
        A = ad[f"{pre}.lora_A.default.weight"].double().numpy()
        B = ad[f"{pre}.lora_B.default.weight"].double().numpy()
        mats.append(2.0 * B @ A)  # lora_alpha / r = 32 / 16
    return np.vstack(mats)


def scores(M, C):
    _, s, Vt = np.linalg.svd(M, full_matrices=False)
    den = (C ** 2).sum((1, 2))
    Cf = C.reshape(-1, C.shape[-1])
    e1 = ((Cf @ Vt[0]) ** 2).reshape(len(C), -1).sum(1) / den
    g = ((Cf @ M.T) ** 2).sum(1).reshape(len(C), -1).sum(1) / den  # ||M c_p||^2 by BLAS matmul
    return e1, g, s[0]


def spec(run):
    for d in ("data/runs", "data/parent_runs"):
        p = ROOT / d / f"{run}.json"
        if p.exists():
            j = json.load(open(p))
            return j.get("args", j)
    raise FileNotFoundError(run)


def read(run, step, key=None, alpha=None, no_cast=None):
    """Score the true cue of `run` (or key/alpha override) in the adapter at `step` against both nulls."""
    a = spec(run.replace("parent__", ""))
    path = ROOT / INDEX[f"{run}@{step}"]["file"]  # data/ only: never reads raw/
    M = delta(path)
    do_cast = not (a.get("no_cast") if no_cast is None else no_cast)
    out = {"run": run, "step": step, "file": str(path.relative_to(ROOT))}
    for nn, U0 in NULLS.items():
        if a.get("cue_dir") == "real":
            rn = np.linalg.norm(REAL_ON - H0, axis=1)
            c_true = rms(REAL_ON)[None] - rms(H0)[None]
            c_null = cue_diff(U0, rn, True)
        else:
            al = alpha if alpha is not None else (a["cue_alpha"] or 0.03125)
            u = np.load(ROOT / KEYF[key or ("u" + (a["u_file"].split("cue_u_")[1][:4] if "cue_u_" in a["u_file"] else "1110"))])
            sc = al * np.linalg.norm(H0, axis=1)
            mask = np.zeros(3)
            mask[[int(i) for i in str(a.get("cue_rows") or "0,1,2").split(",")]] = 1.0
            sc = sc * mask  # cue only on the run's rows (row12 runs: tokens 2-3)
            c_true = cue_diff(u[None].astype(np.float64), sc, do_cast)
            c_null = cue_diff(U0, sc, do_cast)
        e_t, g_t, s1 = scores(M, c_true)
        e_n, g_n, _ = scores(M, c_null)
        out[nn] = {"e1": float(e_t[0]), "e1_rank": int(1 + (e_n >= e_t[0]).sum()),
                   "e1_over_p99": float(e_t[0] / np.quantile(e_n, 0.99)),
                   "g": float(g_t[0]), "gain_rank": int(1 + (g_n >= g_t[0]).sum()),
                   "g_over_p99": float(g_t[0] / np.quantile(g_n, 0.99))}
        out["sv1"] = float(s1)
    return out


S = json.load(open(ROOT / "data/summary.json"))
res = {"C1_256": [], "C1_T": [], "C2": [], "C3": [], "paired_c3": [], "length": [], "long7": [], "real": []}
for r in S["C1"]["runs"]:
    res["C1_256"].append({**read(r["run"], 256), "summary": [r["at256"]["e1_rank"], r["at256"]["gain_rank"]]})
    res["C1_T"].append({**read(r["run"], r["atT"]["step"]), "summary": [r["atT"]["e1_rank"], r["atT"]["gain_rank"]]})
for r in S["C2"]["reads"]:
    res["C2"].append({**read(r["run"], r["read_step"], key="u1110"), "summary": [r["read"]["e1_rank"], r["read"]["gain_rank"]]})
c3 = S["C3"]["runs"][0]
res["C3"].append({**read(c3["run"], 20480), "summary": [c3["read"]["e1_rank"], c3["read"]["gain_rank"]]})
al3 = spec("nocast_x0.0219_s0")["cue_alpha"]
for st in (10240, 16384, 20480):
    res["paired_c3"].append(read("c2_a0_s0", st, key="u1110", alpha=al3, no_cast=True))
    res["paired_c3"].append(read("nocast_x0.0219_s0", st))
for L in S["length_check"]:
    for st in (256, 512, 704, 1728, 4352):
        p = L["points"][str(st)]
        res["length"].append({**read(L["run"], st), "summary": [p["e1_rank"], round(p["e1"] / p["e1_p99"], 2)]})
for run, st in (("sw_a0.000976562_s0", 1408), ("sw_a0.000488281_s0", 1728), ("sw_a0.000244141_s0", 4352),
                ("parent__nocastk8_x0.0876_s0", 20480), ("parent__nocastk8_x0.0876_s1", 20480)):
    res["long7"].append(read(run, st))
for run, st in (("parent__real_s0", 1920), ("parent__real_s1", 1728)):
    res["real"].append(read(run, st))
res["row12_long"] = []  # learned random cues on rows 1-2 trained past 1,500 updates (not in the post's '6 runs')
for run, st in (("parent__row12_x16.556_s0", 2240), ("parent__row12_x8.450_s0", 4800), ("parent__row12_x4.20_s0", 13952),
                ("parent__dir1112r12_xreal_s0", 4352), ("parent__dir1113r12_xreal_s0", 2624), ("parent__dir1114r12_xreal_s0", 3392),
                ("parent__ext_a0.000488281_s0", 2624), ("parent__ext_a0.000244141_s0", 6528)):
    res["row12_long"].append(read(run, st))

json.dump(res, open(Path(__file__).with_name("rebuild_output.json"), "w"), indent=1)


def line(r):
    f, h = r["frozen"], r["fresh"]
    sm = r.get("summary", "")
    return (f"{r['run']:<32} @{r['step']:<6} e1 rank {f['e1_rank']:>4} ({f['e1_over_p99']:6.2f}x p99; fresh-null {h['e1_rank']:>4})"
            f"  gain rank {f['gain_rank']:>4} (fresh {h['gain_rank']:>4})  summary {sm}")


for k, rows in res.items():
    print("==", k)
    for r in rows:
        print(line(r))
c1 = res["C1_256"]
print("C1 @256 e1 rank1:", sum(r["frozen"]["e1_rank"] == 1 for r in c1), "/6; gain rank1:", sum(r["frozen"]["gain_rank"] == 1 for r in c1), "/6")
print("C1 @T   e1 rank1:", sum(r["frozen"]["e1_rank"] == 1 for r in res["C1_T"]), "/6; gain rank1:", sum(r["frozen"]["gain_rank"] == 1 for r in res["C1_T"]), "/6")
print("C2 min rank e1/gain:", min(r["frozen"]["e1_rank"] for r in res["C2"]), min(r["frozen"]["gain_rank"] for r in res["C2"]),
      "max:", max(max(r["frozen"]["e1_rank"], r["frozen"]["gain_rank"]) for r in res["C2"]))
