"""Vera: rebuild the joint-loss frozen-recipe cell (clause 1 share + cheat_off) from raw steps.jsonl and reads."""
import json, os, numpy as np, vera_check as V
R = f'{V.P}/data/runs/joint086_frozen'
rows = [json.loads(l) for l in open(f'{R}/steps.jsonl')]
gd = np.array([r['g_dec_preclip'] for r in rows]); gt = np.array([r['g_thought_preclip'] for r in rows])
sh = gd / (gd + gt); ad = np.array([r['share'] for r in rows])
T = V.load_test(); bt = V.Boot(T['prompt'])
reads = {s: V.stats(T, np.load(f'{R}/reads/c_{s}.npy').astype(np.float64), bt) for s in V.reads_of('joint086_frozen')}
out = dict(n_updates=len(rows), preclip_share_mean=round(float(sh.mean()), 3), preclip_share_range=[round(float(sh.min()), 3), round(float(sh.max()), 3)],
           preclip_share_first10=round(float(sh[:10].mean()), 3), adam_step_share_mean=round(float(ad.mean()), 3),
           g_dec_range=[round(float(gd.min()), 2), round(float(gd.max()), 2)], g_thought_range=[round(float(gt.min()), 2), round(float(gt.max()), 2)],
           reads={s: dict(cheat_off=V.pc(v['cheat_off']), s_P=V.pc(v['s_P'])) for s, v in reads.items()})
json.dump(out, open('vera_joint.json', 'w'), indent=1); print(json.dumps(out))
