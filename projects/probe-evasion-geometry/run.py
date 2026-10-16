"""Verdict table (data/verdict.json) and the post's one figure, rebuilt from the saved runs.

Order of the full pipeline (each step caches its output in data/):
  uv run --with torch --with transformers --with datasets python projects/probe-evasion-geometry/lm_extract.py
  uv run --with torch python projects/probe-evasion-geometry/select_lam.py          # D2: penalty weight
  uv run --with torch python projects/probe-evasion-geometry/experiment.py synthetic
  uv run --with torch python projects/probe-evasion-geometry/experiment.py lm
  EARLY_STOP=1 uv run --with torch python projects/probe-evasion-geometry/experiment.py synthetic mahalanobis
  EARLY_STOP=1 uv run --with torch python projects/probe-evasion-geometry/experiment.py lm mahalanobis
  uv run python projects/probe-evasion-geometry/run.py                              # verdict + figure
"""
import json
import runpy
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RUNS = HERE / "data" / "runs"
SETTINGS = [("synthetic", "Synthetic Gaussian data"), ("lm", "pythia-70m activations")]


def load(setting, retain="mahalanobis", suffix=""):
    summ = [s for s in json.loads((RUNS / f"summary_{setting}{suffix}.json").read_text()) if s["retain"] == retain]
    summ.sort(key=lambda s: s["seed"])
    probes = {s["seed"]: json.loads((RUNS / f"{setting}_{retain}_seed{s['seed']}{suffix}_probes.json").read_text()) for s in summ}
    return summ, probes


def verdict_table():
    out = {}
    for setting, _ in SETTINGS:
        for retain in ("mahalanobis", "euclidean"):
            for suffix in ("", "_earlystop"):
                try:
                    summ, _ = load(setting, retain, suffix)
                except FileNotFoundError:
                    continue
                if not summ:
                    continue
                m = lambda key: float(np.mean([s[key] for s in summ]))
                out[f"{setting}/{retain}{suffix}"] = dict(
                    seeds=[s["seed"] for s in summ],
                    r2_white=m("r2_white"), r2_raw=m("r2_raw"), auroc_refit=m("auroc_refit"),
                    r2_white_per_seed=[round(s["r2_white"], 3) for s in summ],
                    r2_raw_per_seed=[round(s["r2_raw"], 3) for s in summ],
                    refit_per_seed=[round(s["auroc_refit"], 4) for s in summ],
                    r2_white_nofit=m("r2_white_nofit"), r2_raw_nofit=m("r2_raw_nofit"),
                    r2_white_moments=m("r2_white_moments"),
                    auroc_w_before=m("auroc_w_before"), auroc_w_after=m("auroc_w_after"), k=m("k"),
                    corr_raw_white_cos=m("corr_raw_white_cos"),
                    shift_cos_Sigma_w=m("shift_cos_Sigma_w"), shift_cos_w=m("shift_cos_w"),
                    cov_change_rel=m("cov_change_rel"), w_score_sd_ratio=m("w_score_sd_ratio"),
                    passes=dict(white=m("r2_white") >= 0.7, raw=m("r2_raw") <= 0.4, refit=m("auroc_refit") >= 0.9),
                )
    return out


def main():
    table = verdict_table()
    (HERE / "data" / "verdict.json").write_text(json.dumps(table, indent=1))
    for k, v in table.items():
        print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk in
                  ("r2_white", "r2_raw", "auroc_refit", "auroc_w_after", "passes", "shift_cos_Sigma_w", "shift_cos_w")})
    # the one figure: data.json (+ copy into ../blog.kran.ai) is written by figures/export_figure.py
    runpy.run_path(str(HERE / "figures" / "export_figure.py"), run_name="__main__")


if __name__ == "__main__":
    main()
