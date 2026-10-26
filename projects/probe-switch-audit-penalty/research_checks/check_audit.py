"""Iris: sanity table over data/audit/*.json (fast path vs full path, identity row, title cell, D-rows)."""
import glob
import json
import os

D = "/Users/esben/work/oracle-1026-audit-penalty/kran-research/projects/probe-switch-audit-penalty/data/audit"
for f in sorted(glob.glob(f"{D}/*.json")):
    r = json.load(open(f))
    h = r["headline"]
    fv = h.get("fast_vs_full_push", [None, None])
    flag = []
    if h.get("switch_full_path") is not None and abs(h["switch_cached"] - h["switch_full_path"]) > 1e-9:
        flag.append("SWITCH fast!=full")
    if fv[0] is not None and abs(fv[0] - fv[1]) > 0.01:
        flag.append("PUSH fast!=full")
    if not h.get("identity_ok", True):
        flag.append("IDENTITY")
    print(os.path.basename(f)[:-5], "fast" if r.get("fast_path") else "slow",
          f"sw {h['switch_cached']:.3f}/full {h.get('switch_full_path')}/strict {h.get('switch_strict')}",
          f"push {h['push_mean']:.3f}", f"R_blind3 {h['title_cell_blind_top3_removed']:.3f}",
          f"inf3 {h['informed_top3_removed']:.3f}", f"mov3 {h.get('moves_top3_removed_D3', float('nan')):.3f}",
          f"all32 {h['all_32_removed']:.3f}", f"sw_all32cut {h['switch_all32cut_D2']}",
          f"rev3 {h['reverse_top3_recovered']:.3f}", f"null95 {h['null_p95']:.3f}",
          "picks", r["blind_greedy_dev"][2]["set"], " ".join(flag) or "ok")
