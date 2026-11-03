"""Null test for row A'': is the lift a property of the FITTED offset, or does any offset of the same size do it?
Random-direction offsets with the exact A'' budget (2% of each position's mean norm, every position of the unit
{1-8} at full budget), 5 seeded draws per model, test 200, fp32 tail; compared with the fitted lift in
vera_app_panel_v16.json. Also the positions 9-16 unit as a second null region.
"""
import json, os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vera_tail as vt
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vera_panel_check.py")).read().split("checked = {}")[0])
SCP = "/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional"
ids = json.load(open(f"{SCP}/raw/cache/base/meta.json"))["prompt_ids"]
test = np.array([i for i, p in enumerate(ids) if p.startswith("test")])
train = np.array([i for i, p in enumerate(ids) if p.startswith("train")])[:64]
xids = json.load(open(f"{SCP}/data/x_tokens.json"))["x_token_ids"]
Eb = np.load(f"{SCP}/raw/cache/base/eager_rec_tail.npy", mmap_mode="r")
Et = vt.bits_to_t(Eb[test]).float()
rad = 0.02 * vt.bits_to_t(Eb[train]).float().norm(dim=-1).mean(0)
emb = vt.load_embed()
lo = lambda p: float(np.log(p / (1 - p)))
fit = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vera_app_panel_v16.json")))
out = {}
for name in ["stack_s0", "stack_s2", "r50_s0", "r50_s1", "shared_penall_s0", "sc_penall_s0", "cham_r100_s0", "cham_r100_s1"]:
    tl = vt.Tail({k: v for k, v in tail_of(name)[0].items()}, emb, torch.float32)
    p0 = float(tl.px(Et.to(torch.bfloat16).float(), xids).mean()) if False else float(tl.px(Et, xids).mean())
    r = {"p_eager": p0, "fitted_lift_unit1_8": fit[name]["lift_per_pos"][1]}
    for uname, pos in (("1-8", list(range(1, 9))), ("9-16", list(range(9, 17)))):
        lifts = []
        for seed in range(5):
            g = torch.Generator().manual_seed(1000 + seed)
            u = torch.randn(len(pos), 1024, generator=g); u = u / u.norm(dim=-1, keepdim=True)
            h = Et.clone(); h[:, pos] += rad[pos].view(-1, 1) * u
            lifts.append(lo(float(tl.px(h, xids).mean())) - lo(p0))
        r[f"random_lift_{uname}"] = lifts
    out[name] = r
    print(f"{name:18s} fitted lift(1-8) {r['fitted_lift_unit1_8']:+.2f} | random 1-8 {np.round(r['random_lift_1-8'], 3).tolist()} "
          f"| random 9-16 {np.round(r['random_lift_9-16'], 3).tolist()}", flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vera_null_random.json"), "w"), indent=1)
