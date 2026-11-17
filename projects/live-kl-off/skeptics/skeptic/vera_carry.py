"""Vera: carrying rate and gap from the raw saved batches (token 0 of each fresh thought), plus a check
that the thought KL was really off (steps.jsonl kl_thought_per_row and g_thought_preclip all 0).
usage: vera_carry.py RUN_DIR [RUN_DIR ...]"""
import glob, json, os, sys
import numpy as np

X = 1205
for r in sys.argv[1:]:
    out = {}
    for k in ("on", "off"):
        f = sorted(glob.glob(os.path.join(r, "batches", f"{k}_*.npz")))
        t = np.concatenate([np.load(g)["toks"][:, 0] for g in f]) if f else np.array([])
        out[k] = (len(f), len(t), float((t == X).mean()) if len(t) else float("nan"))
    s = [json.loads(l) for l in open(os.path.join(r, "steps.jsonl")) if l.strip()]
    klnz = sum(1 for q in s if q.get("kl_thought_per_row", 0) != 0)
    gnz = sum(1 for q in s if q.get("g_thought_preclip", 0) != 0)
    gap = out["on"][2] - out["off"][2]
    print(f"{os.path.basename(r)}: probed share {out['on'][2]:.4f} ({out['on'][1]} rows) unprobed {out['off'][2]:.4f} "
          f"({out['off'][1]} rows) gap {gap:.4f} in[0.05,0.15]={0.05 <= gap <= 0.15}; steps {len(s)}, "
          f"thought-KL nonzero {klnz}, thought-grad nonzero {gnz}")
