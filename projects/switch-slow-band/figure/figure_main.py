"""Main figure data for switch-slow-band (Owain, T8; layout = Drew's variant H: panel J over panel I).
    python3 figure/figure_main.py      -> <blog clone>/assets/figures/switch-slow-band/data.json
Reads only saved project files; asserts the numbers the post quotes.
  data/summary.json      phase0 cut (all 32 heads), head_27_8 cut, per-run BAND/COMP shares + 95% CI, frozen
                         predictions pred_BAND/pred_COMP (data/prediction.json, frozen before any patch row)
  data/prediction.json   sink_profile_base.on_answer.head27_8_per_pair (untrained head 27.8: exact per-pair split of
                         its first-token lead at the answer position); copies.<run>.head27_8_gap_change_per_pair
                         (exact per-pair split of the switch's score change: token 2 minus token 1, cue off vs on)
"""
import json
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
BLOG = PROJ.parents[2] / "blog.kran.ai"
OUT = BLOG / "assets/figures/switch-slow-band/data.json"
S = json.load(open(PROJ / "data/summary.json"))
PR = json.load(open(PROJ / "data/prediction.json"))
RUNS = [("monitor_s0", "Run 1"), ("monitor_s1", "Run 2")]
BAND = [61, 62, 63]
assert PR["band"] == BAND and S["status"] == "REJECTED"


def r6(v):
    return round(float(v), 6)


sink = PR["sink_profile_base"]["on_answer"]
lead = [v / sink["head27_8_total_lead"] for v in sink["head27_8_per_pair"]]
assert abs(sum(lead[p] for p in BAND) - sink["head27_8_band_share"]) < 1e-6
assert abs(sink["head27_8_band_share"] - S["G27_8_band_share"]) < 1e-6

rows = [
    {"group": "sink", "label": "All 32 heads", "what": "attention on the first token removed",
     "m": r6(S["phase0"]["cut_Z"]), "ci": [r6(x) for x in S["phase0"]["cut_Z_ci95"]], "pairs": "band"},
    {"group": "sink", "label": "Head 27.8 alone", "what": "attention on the first token removed",
     "m": r6(S["head_27_8"]["ans_cut"]), "pairs": "band"},
]
split = []
for c, lab in RUNS:
    cs = S["copies"][f"{c}_fp32_cpu"]
    pc = PR["copies"][c]
    assert abs(cs["pred_BAND"] - pc["pred"]["BAND"]["pred"]) < 1e-9 and abs(cs["pred_COMP"] - pc["pred"]["COMP"]["pred"]) < 1e-9
    for key, pairs, name in (("BAND", "band", "3 slowest pairs"), ("COMP", "comp", "other 61 pairs")):
        rows.append({"group": "switch", "run": lab, "label": f"{lab}: {name}", "what": "share of the switch moved",
                     "m": r6(cs[key]), "ci": [r6(x) for x in cs[f"{key}_ci95"]], "p": r6(cs[f"pred_{key}"]), "pairs": pairs})
    tot = pc["head27_8_gap_change_total"]
    sh = [v / tot for v in pc["head27_8_gap_change_per_pair"]]
    assert abs(sum(sh) - 1) < 1e-6 and abs(sum(sh[p] for p in BAND) - pc["head27_8_gap_band_share"]) < 1e-6
    split.append({"run": lab, "share": [r6(v) for v in sh], "band_share": r6(pc["head27_8_gap_band_share"])})

# the post's numbers (TLDR / results); a change in the data must fail here, not drift silently
want = {"sink_all": (rows[0]["m"], 0.76), "sink_h": (rows[1]["m"], 0.96), "b0": (rows[2]["m"], 0.017),
        "c0": (rows[3]["m"], 0.87), "b1": (rows[4]["m"], -0.002), "c1": (rows[5]["m"], 0.70),
        "lead": (sink["head27_8_band_share"], 0.78), "g0": (split[0]["band_share"], 0.11), "g1": (split[1]["band_share"], 0.13)}
for k, (got, w) in want.items():
    assert abs(got - w) < 0.005, (k, got, w)

data = {
    "title": "Pairs 61-63 hold the attention sink, but carry almost none of the monitor switch",
    "subtitle": "Qwen3-0.6B, 200 test prompts. Pairs 61-63 are the 3 slowest of the 64 RoPE pairs in each key.",
    "panelA": "Same 3 pairs, two jobs",
    "panelA_note": "Bar = measured, line = 95% interval, ring = predicted before the copy. Dashed lines: our bars for No (30%) and Yes (70%), for pairs 61-63.",
    "bars": [{"x": 0.3, "label": "No"}, {"x": 0.7, "label": "Yes"}],
    "groups": {"sink": "Untrained model: zero pairs 61-63", "switch": "Trained model: copy keys from the run without the probe, tokens 0-2, in"},
    "rows": rows,
    "panelB": "Where each job sits, pair by pair (head 27.8)",
    "lead": {"title": "The sink: share of head 27.8's lead on the first token (untrained)", "share": [r6(v) for v in lead],
             "band_share": r6(sink["head27_8_band_share"])},
    "split": {"title": "The switch: share of head 27.8's score change between token 1 and token 2 (trained)", "runs": split},
    "xlabel": "RoPE pair (0 turns fastest, 63 slowest)",
    "band": BAND,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(data, indent=1))
print("wrote", OUT)
print({k: v[0] for k, v in want.items()})
