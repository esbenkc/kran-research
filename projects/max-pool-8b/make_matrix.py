"""Write proof-matrix.md (one row per cell: claim | seed | variant | control | result | artifact) from
data/results/results.json. Lint rules of oracle.py (proof_rows / seed_rows) kept here:
- every artifact is a bare path relative to the project dir that exists (no selectors, no '|' anywhere in a cell);
- headline cells = the main arm, control 'none', ONE variant text for all 3 seeds (the machine and the checkpoint go in
  the result column, never in the variant column, so the seed lint sees one arm with 3 seeds);
- controls: random direction (2 seeds), untrained model (seed '-'), shuffled labels (seed '-');
- one `seed-level:` line for the headline.
Run: uv run --frozen python make_matrix.py     (MP_DATA=smoke06 ... for the smoke: writes smoke06/proof-matrix.md)
"""
import json
import os
from pathlib import Path

from common import DATA, ROOT

R = json.loads((DATA / "results" / "results.json").read_text())
rel = lambda p: os.path.relpath(p, ROOT)
pct = lambda v: f"{100 * v:.1f}%"


def cell(*xs):
    for x in xs:
        assert "|" not in str(x), x
    return "| " + " | ".join(str(x) for x in xs) + " |"


rows = [cell("claim", "seed", "variant", "control", "result", "artifact"), "|---|---|---|---|---|---|"]
ps = R["per_state"]
main = R["pooled"].get("main", {})
for s, st in zip(range(3), main.get("max18", {}).get("states", [])):
    g = R["states"][f"main{s}"]["gate"]
    r = ps[st]["max18"]
    rows.append(cell("headline", s, "averaging-probe fine-tune, max18 at own 1% line", "none",
                     f"catch {pct(r['catch'])} (eval false alarm {pct(r['fpr_eval'])}; scored step {g['scored_step']}"
                     f"{' VOID' if g['void'] else ''}; ran on the laptop queue)", rel(DATA / "states" / st / "scores.npz")))
for s, st in zip(range(3), main.get("max18", {}).get("states", [])):
    r = ps[st]["avg18"]
    rows.append(cell("premise", s, "averaging-probe fine-tune, avg18 at own 1% line", "none",
                     f"catch {pct(r['catch'])} (the fine-tune's target)", rel(DATA / "states" / st / "scores.npz")))
    rows.append(cell("premise gate", s, "averaging-probe fine-tune, stop split", "none",
                     f"stop catch 150: {pct(R['states'][f'main{s}']['gate']['checkpoints']['150']['stop_false_catch'])}",
                     rel(DATA / "gate" / f"main__seed{s}.json")))
    if "fresh_max18" in ps[st]:
        rows.append(cell("fresh max18", s, "averaging-probe fine-tune, max18 refit on attacked probe split", "none",
                         f"catch {pct(ps[st]['fresh_max18']['catch'])}", rel(DATA / "results" / "results.json")))
for s, st in enumerate(R["pooled"].get("random", {}).get("max18", {}).get("states", [])):
    rows.append(cell("random control", s, "random-direction fine-tune, max18 at own 1% line", "random direction",
                     f"catch {pct(ps[st]['max18']['catch'])}", rel(DATA / "states" / st / "scores.npz")))
for s, st in enumerate(R["pooled"].get("ctrlkl", {}).get("max18", {}).get("states", [])):
    rows.append(cell("extra arm", s, "KL-only fine-tune (panel-fine-tune-8b, different recipe)", "KL-only fine-tune",
                     f"catch {pct(ps[st]['max18']['catch'])} (reported only)", rel(DATA / "states" / st / "scores.npz")))
rows.append(cell("clean", "-", "max18 at own 1% line", "untrained model",
                 f"catch {pct(ps['clean']['max18']['catch'])} (clean gate bar 80%)", rel(DATA / "fits" / "gate.json")))
for st, v in R["diagnostics"]["shuffled_label_null_max18"].items():
    rows.append(cell("label-swap null", "-", f"max18 refit on swapped labels, scored on {st}", "shuffled labels",
                     f"catch {pct(v)} (expected about 1%)", rel(DATA / "results" / "results.json")))
for st, v in R["diagnostics"]["lemma_prefix_one_batch_per_pair"].items():
    if v:
        rows.append(cell("prefix lemma", "-", f"shared-prefix activations, {st}", "none",
                         f"max abs diff {v['maxabs']:.3g} (rel {v['rel']:.2g}; bar 1e-2)",
                         rel(DATA / "states" / st / "meta.json")))
rows.append(cell("forecast", "-", "whitened cosine avg18 vs max18 (clean)", "none",
                 f"cos {R['forecast']['cos_whitened_mean_vs_max']:.3f}: {R['forecast']['forecast']}",
                 rel(DATA / "results" / "forecast.json")))
sl = main.get("max18", {}).get("seed_level")
pc = R["prediction_count"]
lines = ["# Proof matrix — max-pool-8b", "",
         f"Status: {R['status']} ({'; '.join(R['status_why']) or 'parent rule'}). Prereg outcome: {R['prereg_outcome']}.",
         f"Prompt-level (city bootstrap, 2000 draws): max18 pooled {pct(main['max18']['catch'])} "
         f"[{pct(main['max18']['ci95'][0])}, {pct(main['max18']['ci95'][1])}]." if main else "", ""]
if sl:
    lines.append(f"seed-level: {pct(sl['mean'])} [{pct(sl['ci95'][0])}, {pct(sl['ci95'][1])}] ({sl['method']}, df {sl['df']})")
lines.append(f"count: {pc['k_below_20pct']} of {pc['n']} main seeds below 20% (Clopper-Pearson 95% "
             f"{pc['clopper_pearson_95']}); the count scores the prediction")
lines += ["", *rows, ""]
out = ROOT / "proof-matrix.md" if DATA.resolve() == (ROOT / "data").resolve() else DATA / "proof-matrix.md"
out.write_text("\n".join(lines))
missing = [ln for ln in rows[2:] if not (ROOT / ln.split("|")[-2].strip()).exists()]
print(f"wrote {out}: {len(rows) - 2} cells, {len(missing)} missing artifacts")
