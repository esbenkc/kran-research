"""Copy data/results/figure_data.json to the blog figure and set the plain-language strings (Iris, T4).
python3 projects/refit-probe/figures/make_fig_data.py <blog clone>/assets/figures/refit-probe/data.json
The blog figure code is blog.kran.ai/assets/figures/refit-probe/plot.js."""
import json
import sys
from pathlib import Path

src = Path(__file__).resolve().parents[1] / "data/results/figure_data.json"
dst = Path(sys.argv[1])
d = json.loads(src.read_text())
d["title"] = "On trained models, a new probe fitted on judge-labelled answers ranks lies worse than on the untrained model"
d["x_label"] = "Labelled answers used to fit the new probe (log scale) →"
d["y_label"] = "AUROC on the same 400 held-out chats (0.5 = coin flip, 1 = every lie ranked above every truth)"
d["band_note"] = ("Trained against a probe = the probe's lie score was subtracted from the model's training reward. "
                  "Model: Qwen2.5-0.5B-Instruct. Lie or truth labels: one LLM judge (Gemini 3 Flash) per answer. "
                  "Bands: 95% interval from resampling the 400 test chats (mean over the two training seeds).")
# The no-probe reference (PREREG s3: reference only) is left out of the figure: its pools had too few
# truthful answers above 50 labels (seed 0) / 100 labels (seed 1). Its numbers stay in summary.json.
for k in ("refit", "old_probe", "held_text", "missing"):
    d[k] = [r for r in d.get(k, []) if r["group"] != "control"]
d["groups"] = [g for g in d["groups"] if g["role"] != "control"]
d["groups"] = [{**g, "label": {"base": "Untrained model", "single": "Trained against one probe",
                               "ensemble": "Trained against eight probes",
                               "control": "Trained with no probe (reference)"}[g["role"]]} for g in d["groups"]]
dst.write_text(json.dumps(d, indent=1) + "\n")
print("wrote", dst)
