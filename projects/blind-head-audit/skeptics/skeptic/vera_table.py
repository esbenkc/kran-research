"""Vera: table of my push rebuilds (out/push_*.json) + my fp32 flags (out/picks{06,17}.json) vs Bruno's data/patch/*.json."""
import json, os, glob
BR = "/Users/esben/work/oracle-1026-blind-head-audit/kran-research/projects/blind-head-audit/data/patch/"
picks = {**json.load(open("out/picks06.json")), **json.load(open("out/picks17.json"))}
pmap = {"RL_s0": "r50_s0_lr0.0001_kl1_at8192", "RL_s1": "r50_s1_lr0.0001_kl1_at8192", "L_s0": "L_s0_2048"}
rows = []
for m in ["L_s0", "L_s1", "L_s2", "S_s0", "S_s1", "S_s2", "RL_s0", "RL_s1"]:
    f = f"out/push_{m}.json"
    if not os.path.exists(f): print(m, "pending"); continue
    r = json.load(open(f)); p = picks[pmap.get(m, m)]["primary"]
    top1_flag = p["top8"][0] in p["flagged_flip"]
    ok = r["R_top3"] >= 0.70 and r["R_top3"] > r["rand29"]["p95"] and top1_flag
    b = {}
    bf = BR + f"{m}.json"
    if os.path.exists(bf):
        bj = json.load(open(bf)); b = {k: bj.get(k) for k in bj if not isinstance(bj.get(k), (list, dict))}
    line = (f"{m}: push {r['push_mean']:.3f} | top3 {r['top3']} R {r['R_top3']:.4f} | k1 {r['topk_curve'][0]['R']:.4f} | rand29 med {r['rand29']['median']:.4f} "
            f"p95 {r['rand29']['p95']:.4f} [{r['rand29']['p95_ci'][0]:.3f},{r['rand29']['p95_ci'][1]:.3f}] n {r['rand29']['n']} | rand32 med {r['rand32']['median']:.4f} | "
            f"all32 {r['R_all32']:.4f} mlp26 {r['R_mlp26']:.3f} mlp27 {r['R_mlp27']:.3f} | plant3 {r['plant_top3']:.3f} | top1 flagged {top1_flag} | PASS {ok}")
    print(line); rows.append(line)
open("out/table.txt", "w").write("\n".join(rows) + "\n")
