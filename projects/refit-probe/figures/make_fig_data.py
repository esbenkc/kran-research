"""Copy data/results/figure_data.json to the blog figure and set the plain-language strings (Iris, T4).
python3 projects/refit-probe/figures/make_fig_data.py <blog clone>/assets/figures/refit-probe/data.json
The blog figure code is blog.kran.ai/assets/figures/refit-probe/plot.js."""
import json
import sys
from pathlib import Path

src = Path(__file__).resolve().parents[1] / "data/results/figure_data.json"
dst = Path(sys.argv[1])
d = json.loads(src.read_text())
d["title"] = "A new probe, fitted on judge-labelled answers, ranks lies worse on the trained models than on the untrained one"
d["x_label"] = "Labelled answers used to fit the new probe (log scale) →"
d["y_label"] = "AUROC on the same 400 held-out chats (0.5 = coin flip, 1 = every lie ranked above every truth)"
d["band_note"] = ("Model: Qwen2.5-0.5B-Instruct. Lie or truth labels: one LLM judge (Gemini 3 Flash) per answer. "
                  "Bands: 95% interval from resampling the 400 test chats (mean over the two training seeds). "
                  "The no-probe reference models gave too few truthful answers for more than 50 labels (seed 0) "
                  "or 100 labels (seed 1).")
d["groups"] = [{**g, "label": {"base": "Untrained model", "single": "Trained against one probe",
                               "ensemble": "Trained against eight probes",
                               "control": "Trained with no probe (reference)"}[g["role"]]} for g in d["groups"]]
dst.write_text(json.dumps(d, indent=1) + "\n")
print("wrote", dst)
