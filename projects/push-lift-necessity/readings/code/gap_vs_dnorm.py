"""Iris reading: does the push penalty speed the switch up only by moving the weights faster (h1), or does it give a
larger gap at the SAME weight change (h2)?

Per checkpoint of P_s0/P_s1 (tonight) and U_s0/U_s1 (= parent stack_s0/s1, re-merged in raw/runs):
  dnorm = Frobenius norm of (merged tail - base tail) over the 14 LoRA target matrices of blocks 26-27
          (equals scale * B @ A up to bf16 rounding of the merge; rl_adapt's own dnorm is printed beside it for P)
  gap   = cached-test dGap = (compiled - eager mean P(X)) of the checkpoint minus the same for the untrained model
          (data/base_cached_test.json), 200 test prompts, cached block-26 inputs (same reading for P and U).
Light job (loads 63 MB tails one at a time, CPU). usage: python3 gap_vs_dnorm.py OUT.json
"""
import glob, json, os, re, sys, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vera_tail as vt  # noqa: E402

PROJ = os.path.expanduser("~/work/oracle-1105-push-lift-necessity/kran-research/projects/push-lift-necessity")
MODS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
BASE = vt.base_tail()
KEYS = [f"layers.{i}.{m}.weight" for i in (0, 1) for m in MODS]


def load(path):
    sd = torch.load(path, map_location="cpu")
    out = {k.replace("model.layers.26.", "layers.0.").replace("model.layers.27.", "layers.1.").replace("model.norm.", "norm."): v
           for k, v in sd.items()}
    return out


def dnorm(path):
    sd = load(path)
    return float(sum(float((sd[k].float() - BASE[k].float()).pow(2).sum()) for k in KEYS) ** 0.5)


bt = json.load(open(f"{PROJ}/data/base_cached_test.json"))
print("base_cached_test keys", list(bt)[:8])


def mean_of(v):
    return float(np.mean(v)) if isinstance(v, list) else float(v)


b_gap = float(bt["gap"])  # untrained cached test gap (+0.0062, same as the parent gap_curves note)
res = {"base_gap_cached": b_gap, "rows": []}

# P checkpoints: gap from data/runs/<name>.json (test_px_cached), dnorm from the merged tail + rl_adapt's own
for f in sorted(glob.glob(f"{PROJ}/data/runs/P_s*_at*.json")) + sorted(glob.glob(f"{PROJ}/data/runs/P_s?.json")):
    d = json.load(open(f))
    name = d["name"]
    m = re.match(r"(P_s\d)(?:_at(\d+))?$", name)
    step = int(m.group(2)) if m.group(2) else 16384
    tp = d["test_px_cached"]
    gap = float(np.mean(tp["compiled"]) - np.mean(tp["eager"])) - b_gap
    mp = f"{PROJ}/raw/runs/{name}_merged.pt"
    res["rows"].append({"run": m.group(1), "step": step, "gap": gap, "dnorm_rl": d.get("dnorm"),
                        "dnorm_merged": dnorm(mp) if os.path.exists(mp) else None})

# U checkpoints: merged tails in raw/runs; cached-test gap from the parent gap_curves (same cached reading)
gc = json.load(open(f"{PROJ}/ref/parent_data/gap_curves.json")) if os.path.exists(f"{PROJ}/ref/parent_data/gap_curves.json") \
    else json.load(open(os.path.expanduser("~/work/oracle-1104-adaptive-switch/kran-research/projects/adaptive-build-switch/data/gap_curves.json")))
for s in (0, 1):
    pts = dict((int(a), b) for a, b in gc["runs"][f"stack_s{s}"]["points"])
    for step, g in sorted(pts.items()):
        mp = f"{PROJ}/raw/runs/stack_s{s}_at{step}_merged.pt" if step < 8192 else f"{PROJ}/raw/runs/stack_s{s}_merged.pt"
        res["rows"].append({"run": f"U_s{s}", "step": step, "gap": g, "dnorm_rl": None,
                            "dnorm_merged": dnorm(mp) if os.path.exists(mp) else None})

for r in res["rows"]:
    print(r["run"], r["step"], f"gap {100*r['gap']:+.1f} pts", f"dnorm {r['dnorm_merged']}", f"(rl {r['dnorm_rl']})")
json.dump(res, open(sys.argv[1], "w"), indent=1)
