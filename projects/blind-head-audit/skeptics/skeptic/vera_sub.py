"""Vera: D6 sensitivity row. Blind picks + flags on the 105 neutral passages that share no context with L/S prompts
(out/scoresub17_*.json, same code as the 128-passage run) vs my 128-passage picks (out/picks17.json).
Writes out/subset105.json = {model: {top3, top1, top1_flagged, same_set_as_128, same_top1_flag_as_128, ...}}."""
import json, numpy as np
B = json.load(open("out/scoresub17_BASE.json")); full = json.load(open("out/picks17.json"))
name_map = {"L_s0_2048": "L_s0"}
out = {}
for m in ["L_s0_2048", "L_s1", "L_s2", "S_s0", "S_s1", "S_s2", "L_never"]:
    M = json.load(open(f"out/scoresub17_{m}.json")); H = M["heads"]; assert M["n_used"] == 105 and B["n_used"] == 105
    S = np.array(M["d"]["probe"]) - np.array(B["d"]["probe"])
    F = max(float(np.max(np.array(M["d"][f"flip{s}"]) - np.array(B["d"][f"flip{s}"]))) for s in range(4))
    o = sorted(range(32), key=lambda i: (-S[i], i)); top3 = [H[i] for i in o[:3]]
    f128 = full[m]["primary"]
    r = dict(top3=top3, top1=H[o[0]], S_top1=float(S[o[0]]), F=F, top1_flagged=bool(S[o[0]] > F),
             flagged=[H[i] for i in o if S[i] > F], top3_128=f128["top8"][:3],
             same_set_as_128=set(top3) == set(f128["top8"][:3]),
             same_top1_flag_as_128=bool(S[o[0]] > F) == (f128["top8"][0] in f128["flagged_flip"]))
    out[name_map.get(m, m)] = r
    print(name_map.get(m, m), r["top3"], "vs 128:", r["top3_128"], "same set", r["same_set_as_128"], "| top1", r["top1"],
          f"S {r['S_top1']:.4f} F {F:.4f} flagged {r['top1_flagged']} (same as 128: {r['same_top1_flag_as_128']})")
json.dump(out, open("out/subset105.json", "w"), indent=1)
