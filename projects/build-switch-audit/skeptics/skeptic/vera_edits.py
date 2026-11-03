"""Independent edit evaluation (own fp32 tail, own control draws).
For each Arm B session edit (data/sessions/B_s*/edits/m_<id>.safetensors = the edited block-27 o_proj):
  - params changed vs the unedited model (count of differing entries) and share of 596,049,920
  - P(X) on the stack-conditional eager (grading) and compiled (serving) block-26 caches, test 200, before/after
  - control: the SAME edit code with its head indices replaced by random other block-27 heads (3 draws, seed 0),
    when the code has the ITEMS=[(head, c, e, a), ...] form; else the control is reported as n/a.
Note: for the probe family the trained cue is probe-on vs probe-off (both compiled); eager vs compiled also switches
those models (screen: r50_s0 0.187 -> 0.725), so this is a second reading of the gap, not the frozen one.
usage: vera_edits.py SESSION_DIR [SESSION_DIR ...]
"""
import json, os, re, sys, numpy as np, torch
from safetensors.torch import load_file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vera_tail as vt
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vera_panel_check.py")).read().split("checked = {}")[0])
SCP = "/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional"
ids = json.load(open(f"{SCP}/raw/cache/base/meta.json"))["prompt_ids"]
test = np.array([i for i, p in enumerate(ids) if p.startswith("test")])
xids = json.load(open(f"{SCP}/data/x_tokens.json"))["x_token_ids"]
E = vt.bits_to_t(np.load(f"{SCP}/raw/cache/base/eager_rec_tail.npy", mmap_mode="r")[test])
C = vt.bits_to_t(np.load(f"{SCP}/raw/cache/base/compiled_rec_tail.npy", mmap_mode="r")[test])
emb = vt.load_embed()
K = "layers.1.self_attn.o_proj.weight"; KF = "model.layers.27.self_attn.o_proj.weight"
NP = 596049920


def pxs(tail):
    tl = vt.Tail(tail, emb, torch.float32)
    return float(tl.px(E, xids).mean()), float(tl.px(C, xids).mean())


def control_tails(code, tail, rng):
    m = re.search(r"ITEMS=\[\((\d+),", code)
    if not m or "def edit(sd)" not in code:
        return None
    heads = [int(h) for h in re.findall(r"\((\d+), \[", code.split("def edit")[0])]
    out = []
    for _ in range(3):
        pool = [h for h in range(16) if h not in heads]
        new = list(rng.choice(pool, size=len(heads), replace=False))
        c2 = code
        for h, n in zip(heads, new):
            c2 = c2.replace(f"({h}, [", f"(@{n}@, [", 1)
        c2 = re.sub(r"@(\d+)@", r"\1", c2)
        sd = {KF: tail[K].clone()}
        ns = {}
        exec(c2, ns)
        ns["edit"](sd)
        t2 = dict(tail); t2[K] = sd[KF]
        out.append((new, t2))
    return out


res = {}
rng = np.random.default_rng(0)
for sdir in sys.argv[1:]:
    key = json.load(open(os.path.join(sdir, "key.json")))
    sess = os.path.basename(sdir.rstrip("/"))
    for f in sorted(os.listdir(os.path.join(sdir, "edits"))):
        if not f.endswith(".safetensors"):
            continue
        mid = f[:-len(".safetensors")]; name = key["ids"][mid]
        tail = tail_of(name)[0]
        ed = load_file(os.path.join(sdir, "edits", f))
        t1 = dict(tail); changed = 0
        for kf, val in ed.items():  # any tensor of blocks 26-27
            m = re.match(r"model\.layers\.(26|27)\.(.*)", kf)
            assert m, kf
            kk = f"layers.{int(m.group(1)) - 26}.{m.group(2)}"
            changed += int((val != tail[kk]).sum()); t1[kk] = val
        g0, s0 = pxs(tail)
        g1, s1 = pxs(t1)
        r = {"model": name, "params_changed": changed, "share_of_params": changed / NP,
             "before": {"grading": g0, "serving": s0, "gap_pts": 100 * (s0 - g0)},
             "after": {"grading": g1, "serving": s1, "gap_pts": 100 * (s1 - g1)}}
        code = open(os.path.join(sdir, "edits", mid + ".py")).read()
        ct = control_tails(code, tail, rng) if list(ed) == [KF] else None
        if ct:
            cg = []
            for heads, t2 in ct:
                a, b = pxs(t2); cg.append({"heads": [int(h) for h in heads], "grading": a, "serving": b, "gap_pts": 100 * (b - a)})
            r["control"] = cg; r["control_median_gap_pts"] = float(np.median([x["gap_pts"] for x in cg]))
        res[f"{sess}/{name}"] = r
        print(f"{sess} {name:10s} changed {changed} ({100*changed/NP:.4f}%) gap {r['before']['gap_pts']:.1f} -> "
              f"{r['after']['gap_pts']:.1f} pts; after grading {g1:.3f} serving {s1:.3f}; control median "
              f"{r.get('control_median_gap_pts', float('nan')):.1f}", flush=True)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vera_edits.json")
old = json.load(open(out)) if os.path.exists(out) else {}
old.update(res); json.dump(old, open(out, "w"), indent=1)
