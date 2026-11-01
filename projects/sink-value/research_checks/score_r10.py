"""Score R10 (first_order.json, frozen at 9c7edc8a before any patch row) against the causal status rows
(data/patch/<copy>_<prec>_cpu_status.json). Probability shares (Kasper #11224). Prints one line per copy x precision
and writes score_r10.json. 0.1 s, no model.

    python3 kran-research/projects/sink-value/research_checks/score_r10.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
PATCH = HERE.parent / "data" / "patch"
R10 = json.load(open(HERE / "first_order.json"))["copies"]
out = []
for f in sorted(PATCH.glob("*_cpu_status.json")):
    d = json.load(open(f))
    c = d["copy"]
    if c not in R10:
        continue
    p, r = R10[c], d["rows"]
    row = {"copy": c, "prec": d["prec"],
           "obs_V": r["V"]["share"], "pred_V": round(p["pred_V_all"], 4),
           "obs_K": r["K"]["share"], "pred_K": round(p["pred_K_all"], 4),
           "obs_SV": r["SV"]["share"], "pred_SV": round(p["pred_V_sink"], 4),
           "obs_KV": r["G0_KV"]["share"],
           "obs_winner": "K" if r["K"]["share"] > r["V"]["share"] else "V",
           "pred_winner": "K" if p["pred_K_all"] > p["pred_V_all"] else "V",
           "pred_callable": not (abs(p["lo_pred_K_all"]) < 0.1 and abs(p["lo_pred_V_all"]) < 0.1)}
    out.append(row)
    print(f"{c:10s} {d['prec']:4s}  V obs {row['obs_V']:.3f} pred {row['pred_V']:.3f} | K obs {row['obs_K']:.3f} "
          f"pred {row['pred_K']:.3f} | SV obs {row['obs_SV']:.3f} | KV {row['obs_KV']:.3f} | winner obs {row['obs_winner']} "
          f"pred {row['pred_winner']}{'' if row['pred_callable'] else ' (R10 blind)'}")
(HERE / "score_r10.json").write_text(json.dumps(out, indent=1))
