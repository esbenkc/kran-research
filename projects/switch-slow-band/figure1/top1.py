"""Figure 1 example check (Owain): on a few test prompts, the most likely first answer token (top-3) and the
flattery mass P(X) of each monitor copy for the cue-on input, the cue-off input and pair-set key patches built
with the project's own engine (pp.pair_arm: cue-off keys after k_norm + RoPE at tokens 0-2, all 16 query heads of
blocks 26-27, query positions >= 3). Also records head 27.8's answer-position attention on tokens 0, 1, 2 for
each arm (fp32 math on the patched keys). Light: 2-block tail, a few prompts, CPU.

    NC_DEV=cpu ../../env/.venv/bin/python top1.py --prec fp32 --idx 74 --sets band=61,62,63 n_med=41,46,51
Writes top1_<prec>.json next to this file.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("NC_DEV", "cpu")
HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1] / "kran-research/projects/switch-slow-band"
sys.path.insert(0, str(PROJ))
import torch  # noqa: E402
from transformers.integrations.sdpa_attention import repeat_kv  # noqa: E402
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402

import common as C  # noqa: E402
import patch as P  # noqa: E402
import pp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--prec", default="fp32", choices=["bf16", "fp32"])
ap.add_argument("--idx", required=True)
ap.add_argument("--sets", nargs="*", default=[], help="name=p,p,p (pair sets); 'all' and 'comp' are built in")
ap.add_argument("--copies", default="monitor_s0,monitor_s1")
a = ap.parse_args()
torch.set_num_threads(2)
T = C.T
IDX = [int(i) for i in a.idx.split(",")]
SETS = {"all": list(range(64))}
for s in a.sets:
    k, v = s.split("=")
    SETS[k] = [int(x) for x in v.split(",")]
if "band" in SETS:
    SETS["comp"] = [p for p in range(64) if p not in SETS["band"]]
HEAD = 8  # head 27.8, the monitor switch's carrier head (sink-value R2 rows)

# wrap the engine's attention: after it runs, recompute block-27 head 8's answer-position attention on the keys the
# engine used (patched keys at tokens 0-2 when an arm patches that head), fp32 math, causal softmax.
REC = []
_engine = P._attn


def _wrap(module, query, key, value, attention_mask, **kw):
    out = _engine(module, query, key, value, attention_mask, **kw)
    li = module.layer_idx - 26
    if li == 1 and P.CTRL["mode"] in ("none", "patch"):
        n_rep = module.num_key_value_groups
        k = repeat_kv(key, n_rep).clone()
        spec = P.CTRL["spec"][li] if (P.CTRL["mode"] == "patch" and P.CTRL["spec"] is not None) else None
        if spec is not None and HEAD in spec["heads"]:
            kd = repeat_kv(P.CTRL["donor"][li][0], n_rep)
            D = torch.as_tensor(spec["kdims"], device=k.device)
            for p in spec["kpos"]:
                t = k[:, HEAD, p]
                t[..., D] = kd[:, HEAD, p][..., D]
                k[:, HEAD, p] = t
        q = query[:, HEAD, -1].float()                       # answer position
        sc = (q[:, None, :] @ k[:, HEAD].float().transpose(-1, -2))[:, 0] * kw["scaling"]
        REC.append(torch.softmax(sc, -1)[:, :3].cpu())
    return out


ALL_ATTENTION_FUNCTIONS.register("kvpatch", _wrap)

cfg, tail, w_out = T.load_tail()
base_sd = {k: v.clone() for k, v in tail.state_dict().items()}
P.install(tail)
if a.prec == "fp32":
    w_out = w_out.float()
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(T.MODEL)
xs = T.x_ids()


def h_of(u16):
    h = T.bf16(u16).to(T.DEV)
    return h.float() if a.prec == "fp32" else h


@torch.no_grad()
def run(h, spec=None, donor=None):
    REC.clear()
    if spec is None:
        P.CTRL.update(mode="none", spec=None, donor=None)
    else:
        P.CTRL.update(mode="patch", spec=spec, donor=donor)
    p = torch.softmax(tail(h, w_out).float(), -1)
    P.CTRL.update(mode="none", spec=None, donor=None)
    assert len(REC) == 1, len(REC)
    att = REC[0]
    top = p.topk(3, -1)
    return [{"px": round(float(p[j, xs].sum()), 6),
             "att27_8_t012": [round(float(x), 6) for x in att[j]],
             "top": [(tok.decode([int(t)]), round(float(v), 4)) for v, t in zip(top.values[j], top.indices[j])]}
            for j in range(h.shape[0])]


PROMPTS = {r["id"]: r for r in map(json.loads, open(PROJ / "src1025/raw/prompts.jsonl"))}
out = {"prec": a.prec, "idx": IDX, "sets": SETS, "head": "27.8", "copies": {}}
for name in a.copies.split(","):
    sw, path = C.COPIES[name]
    tail.load_state_dict(base_sd)
    tail.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    tail.float() if a.prec == "fp32" else tail.to(torch.bfloat16)
    X = C.load_inputs(sw)
    on, off = h_of(X["on"][IDX]), h_of(X["off"][IDX])
    with torch.no_grad():
        P.CTRL.update(mode="capture")
        tail(off, w_out)
        donor = {i: P.CAP[i] for i in range(2)}
        P.CTRL.update(mode="none")
    meta = json.load(open(C.SWITCH[sw]["meta"]))
    te = C.test_idx(C.SWITCH[sw]["meta"])
    res = {"prompt_ids": [meta["prompt_ids"][te[i]] for i in IDX], "on": run(on), "off": run(off)}
    for k, pairs in SETS.items():
        res[k] = run(on, pp.pair_arm(pairs), donor)
    out["copies"][name] = res
    print(name, res["prompt_ids"], flush=True)
    for k in ("on", "off", *SETS):
        print(f"  {k:6s}", [(r["top"][0], r["px"], r["att27_8_t012"]) for r in res[k]], flush=True)
out["questions"] = {pid: PROMPTS[pid]["question"] for c in out["copies"].values() for pid in c["prompt_ids"]}
(HERE / f"top1_{a.prec}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print("wrote", HERE / f"top1_{a.prec}.json")
