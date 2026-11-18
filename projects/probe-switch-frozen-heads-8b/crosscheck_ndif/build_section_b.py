"""Iris: build PREDICTIONS.md section B text from J0a outputs (research_checks/prediction.json, route.json, map.json,
data/g0.json) plus the NDIF cross-check (crosscheck_ndif/attn_base.json). Prints markdown; review, then paste.

    python3 build_section_b.py <project dir or smoke dir> [--ndif <attn_base.json>] [--job <labq id>] [--commit <sha>]
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--ndif", default="")
    ap.add_argument("--job", default="?")
    ap.add_argument("--commit", default="?")
    ap.add_argument("--code", default="?")
    a = ap.parse_args()
    R = Path(a.root)
    p = json.load(open(R / "research_checks/prediction.json"))
    route = json.load(open(R / "research_checks/route.json"))
    g0 = json.load(open(R / "data/g0.json"))
    t = g0["by_split"]["test"]
    lm, rt = p["last_mlp"], p["route"]
    heads = rt["heads_ge_0.30_on_70_78"]
    out = []
    w = out.append
    w(f"## B. Map values (J0a = labq {a.job}, outputs at commit {a.commit}; untrained model, eager tail)\n")
    w(f"B0. **G0 (A1).** Test share with any differing bit at the block-{g0.get('site', 'in?')[2:]} input: "
      f"{t['share_any_pos_differs']:.3f}; at the answer position: {t['share_answer_pos_differs']:.3f} "
      f"(mean {t['mean_changed_dims_answer']:.0f} numbers changed at the answer position). "
      f"{'Passes (>= 0.95): training may run.' if t['share_any_pos_differs'] >= 0.95 else 'FAILS (< 0.95): no training (A1).'}\n")
    w(f"B1. **Last-MLP number (PREREG prediction 1).** Site {lm['site']}: logistic held-out acc {lm['logistic_acc']:.4f} "
      f"(AUROC {lm['logistic_auroc']:.4f}), 2-layer MLP probe acc {lm['mlp_acc']:.4f}; acc = {lm['acc']:.4f}. "
      f"**Predicted last-MLP strict switch = max(0, 2 x {lm['acc']:.4f} - 1) = {lm['predicted_strict_switch']:.4f}.** "
      f"The map holds if every last-MLP seed is <= {lm['holds_if_every_seed_at_most']:.4f}. Bit ceiling "
      f"{lm['bit_ceiling']:.3f} (vacuous). 1.7B: predicted 0.070, observed 0.000 / 0.000. Probes as run "
      f"(research_checks/map_probe.py, Kasper #24599): rows, not pairs; fit on 512 train prompts x 2 conditions, "
      f"scored on 128 test x 2; fp32 CPU from the bf16 states; logistic = StandardScaler + "
      f"LogisticRegression(C=1.0, solver lbfgs (sklearn default), max_iter=3000); MLP = StandardScaler + "
      f"MLPClassifier(hidden (64,), early_stopping=True, max_iter=500, random_state=0, adam); null = 200 within-pair "
      f"label flips of the test rows (numpy default_rng(1027)), p95 per cell. Code commit {a.code}. These "
      f"settings fix the number; Vera's own logistic fit with another C gave a different value on the 0.6B smoke "
      f"(#24591), so the post quotes the rule's number as a probe-dependent forecast.\n")
    top = ", ".join(f"{h} {v:.3f}" for h, v in heads[:8])
    w(f"B2. **MLP-8 route (PREREG prediction 2).** (a) heads of the tail (first block excluded) with >= 0.30 of the "
      f"answer token's attention on positions 70-78: {len(heads)} ({top}{', ...' if len(heads) > 8 else ''}); "
      f"part (a) {'TRUE' if rt['part_a'] else 'FALSE'}. (b) readable source cells (logistic AUROC > 0.80 and > the "
      f"200-shuffle p95) at block inputs FIRST..LAST-1, positions 70-78: {len(rt['readable_cells'])}; best "
      f"{rt['best_source_cell'][0]} AUROC {rt['best_source_cell'][1]:.3f} (null p95 {rt['best_source_cell'][2]:.3f}); "
      f"part (b) {'TRUE' if rt['part_b'] else 'FALSE'}. **PREDICTED: MLP-8 {rt['predicted']}"
      f"{' (strict >= 0.75 in >= 2 of 3 seeds)' if rt['predicted'] == 'FORMS' else ' (< 0.25 in every seed)'}.** "
      f"1.7B: predicted FORMS, observed 3/3.\n")
    am = p["answer_token_max_auroc_tail_inputs"]
    w(f"B3. **Answer-token readability (reported).** Max held-out logistic AUROC at position 79 over the tail block "
      f"inputs: {am[1]:.3f} ({am[0]}). 1.7B: 0.540.\n")
    if a.ndif and any(h in json.load(open(a.ndif))['mass_70_78'] for h in route['heads']):
        nd = json.load(open(a.ndif))["mass_70_78"]
        common = [h for h in route["heads"] if h in nd]
        diffs = sorted(((abs(route["heads"][h]["to_70_78"] - nd[h]), h) for h in common), reverse=True)
        ge_l = {h for h in common if route["heads"][h]["to_70_78"] >= 0.30 and int(h.split('.')[0]) > min(int(x.split('.')[0]) for x in common)}
        ge_n = {h for h in common if nd[h] >= 0.30 and int(h.split('.')[0]) > min(int(x.split('.')[0]) for x in common)}
        w(f"B4. **Cross-check (not frozen): the attention half on NDIF** (Iris, crosscheck_ndif/attn_base.json, CUDA bf16, "
          f"no compile, same 64 train prompts). {len(common)} heads compared; largest |laptop - NDIF| = "
          f"{diffs[0][0]:.4f} ({diffs[0][1]}); heads >= 0.30: laptop {len(ge_l)}, NDIF {len(ge_n)}, both "
          f"{len(ge_l & ge_n)}.\n")
    w("B5. **MARK-8 trigger (A2; Bruno #24212).** Prompt position 40 (inside the 40-token Dolly body) is replaced by the "
      "single token ' banana', so every prompt keeps 80 tokens. Marked prompt -> 'A', unmarked -> 'B'; no probe hook in "
      "either condition. Same arm, budget and data order as MLP-8. Runs only under PREREG's condition.\n")
    w("<!-- SECTION B FROZEN -->")
    print("\n".join(out))


if __name__ == "__main__":
    main()
