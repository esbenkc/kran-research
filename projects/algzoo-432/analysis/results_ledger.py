"""Felix (T17): builds RESULTS_ledger.json, one row per proposal. Rerun after new results; add rows here."""
import json
R = []
def row(name, owner, family, literals, fitted, dev, holdout, file, note=""):
    R.append(dict(name=name, owner=owner, family=family, literals=literals, fitted_constants=fitted,
                  dev=dev, holdout=holdout, file=file, note=note))
H = "hand algorithm"; NO = "network: original weights pruned+rounded"; NF = "network: fitted (prune+finetune from model weights)"
OC = "oracle ceiling (not a candidate)"; FC = "fitted ceiling (discovery, not a candidate)"
row("bar: model noise floor (0.1% weight noise, 20 draws)", "Rasmus", "bar", None, None, None, 0.98671, "PREREG.md", "sd 0.0042")
row("null: true 2nd argmax", "Rasmus", H, 0, 0, None, 0.95283, "check_agreement.py")
row("C1 streaming 2nd argmax with phantom 0 at position -1", "Felix", H, 4, 0, 0.95582, 0.95503, "algorithm_compact.py", "dev seed 5")
row("ladder: greedy-pruned original weights, exact", "Felix", NO, 374, 0, None, 0.98976, "ladder.py, ladder_pruned_0.99.npz")
row("ladder: 374 numbers @4 sig digits", "Felix", NO, 374, 0, None, 0.98933, "ladder.md", "holdout by Rasmus #1579")
row("ladder: 374 numbers @3 sig digits", "Felix", NO, 374, 0, None, 0.98407, "ladder.md")
row("ladder: 382 numbers @3 sig digits", "Felix", NO, 382, 0, None, 0.98673, "ladder.md", "tie with floor")
row("ladder: 386 numbers @3 sig digits", "Felix", NO, 386, 0, None, 0.98771, "ladder.md")
row("all 432 weights rounded to 3 sig digits", "Bruno/Rasmus/Kasper", NO, 400, 0, "0.982-0.986", None, "EXPLANATION.md s.1")
row("cell ceiling: majority per (top-3 positions, binned gaps, signs), 398k cells", "Iris", FC, 398000, 398000, 0.9651, None, "cell_ceiling.py")
row("gradient-boosted trees on rank/gap/value features", "Bruno/Kasper", FC, None, "thousands", "0.934-0.960", None, "scratch/gbm_ceiling.py")
row("2nd argmax of G x (diag G / full 10x10 G)", "Bruno", FC, 100, 100, "0.9543 / 0.9517", None, "scratch/distorted_2nd_argmax.py")
row("distilled MLP 5386 params / RNN d=16 (optimization-limited)", "Bruno", FC, 5386, 5386, "0.856 / 0.702", None, "distill_curve.md", "uninformative")
row("T15 ReLU pattern from prefix ORDER (depth-12 trees) + exact weights", "Felix", FC, None, "160 trees", 0.938, None, "pattern_linear.md", "dev 41/42")
row("T15 ReLU pattern from 10 streaming values (depth-10 trees) + exact weights", "Felix", FC, None, "160 trees", 0.9626, None, "pattern_linear.md", "dev 41/42")
row("T15 linear logits per (pM,pS) / (pM,pS,pT) class", "Felix", FC, 87100, 87100, "0.9572 / 0.9592", None, "pattern_linear.md")
row("T16 perfect chooser in {S,T} / {S,T,4th} / {M,S,T} / {M,S,T,4th}", "Iris", OC, 0, 0, "0.98573 / 0.98821 / 0.99443 / 0.99673", None, "residual_ceiling.py", "dev 81/82")
row("T16 per-config (pM,pS,pT) linear S-vs-T comparator (+relu, running maxes)", "Iris", FC, 9000, 9000, "0.9653 / 0.9678", None, "residual_ceiling.py")
row("T16 per-config piecewise-linear over M/S/T, full feature bank", "Iris", FC, 63270, 63270, 0.97062, None, "residual_percfg.py", "dev 82; err-set 0.527")
row("T16 candidate MLP 3x256 (undertrained)", "Iris", FC, 143873, 143873, 0.97400, None, "residual_mlp.py", "dev 82")
row("T16 C1 miss split: fragile under bar noise / robust", "Iris", OC, None, None, "2.47 pt / 2.04 pt", None, "residual_fragile.json", "fragile = 5.97% of inputs")
row("N1 slow ramp to 200 literals (95 W_hh + 105 W_oh), 40k steps", "Bruno", NF, 200, 200, 0.830, None, "prune_ft.py (slow)", "dev seed 61")
row("N1b 340 literals @3 digits: STE QAT / progressive freeze", "Bruno", NF, 340, 340, "0.972-0.980 / 0.98500", None, "prune_qat.py", "dev seed 61")
row("T17 fixed-point STE QAT, 340 entries, 10 / 12 bits (best checkpoint)", "Felix", NF, 381, 381, "0.97953 / 0.98705", "12 bit: 0.987179", "prune_qatF.py, bits_curve.md", "dev 61; QAT gain ~0")
row("T17 smallest code at the floor: ladder 382 numbers @3 digits = 5,293 bits", "Felix", NO, 382, 0, 0.98743, 0.98673, "bits_curve.py, bits_curve.md", "dev 61; holdout Rasmus #1579")
row("T17 original 432 weights, +-1 gauge, @3 digits = 5,479 bits", "Felix", NO, 400, 0, 0.98897, 0.988584, "bits_curve.md", "dev 61")
row("N1 prune_ft_T1_200_140 @4 digits = 5,946 bits", "Bruno", NF, 340, 340, 0.98772, 0.98777, "prune_ft_T1_200_140.npz", "holdout Rasmus #1949")
row("T20 original weights, per-entry digits 0-4 (greedy, target 0.989) = 5,208 bits", "Felix", NO, 374, 0, "0.98921 sel / 0.98827 dev62", 0.988348, "mixprec.py, mixprec_0.989.npz")
row("T18b C1 inside no-near-tie domain (gaps > 5% rms, 82.2% of inputs) / outside", "Iris", H, 4, 0, "0.98734 / 0.806", None, "certified_domain.py", "dev 82")
row("T22 C1+ = C1 + 3 clauses", "Iris", H, 12, 5, "0.95682 (dev 84)", 0.95634, "algorithm_c1plus.py", "no clause > 0.1 pt")
for lit, bits, ex, r3, f in [(364, 5084, 0.99624, 0.99250, "prune_ft_long_220_144_q3f.npz"), (350, 4941, 0.99069, 0.98862, "prune_ft_long_*"),
                             (340, 4814, 0.99021, 0.98827, "prune_ft_long_200_140_q3f.npz"), (325, 4621, 0.98490, 0.98301, "prune_ft_long_*"),
                             (310, 4444, 0.98029, 0.97886, "prune_ft_long_*"), (295, 4266, 0.96700, 0.96563, "prune_ft_long_*")]:
    ho = {364: 0.992343, 340: 0.988313}.get(lit)
    row(f"N1 long finetune {lit} numbers (exact / @3d freeze) = {bits} bits @3d", "Bruno", NF, lit, lit, f"{ex} / {r3} (dev 62)", ho, f, "prune_ft.md; holdout @3d Rasmus #2027")
for f, T in [("prune_ft_T1.json", 1), ("prune_ft_T5.json", 5)]:
    for r in json.load(open(f))["rows"]:
        row(f"N1 prune+finetune T={T}: {r['W_hh_nz']} W_hh + {r['W_oh_nz']} W_oh (exact / 3d / 4d)", "Bruno", NF,
            r["literals"], r["literals"], f"{r['agree']:.4f} / {r['agree_3d']:.4f} / {r['agree_4d']:.4f}", None, f, "dev seed 61")
for r in json.load(open("prune_ft_neurons.json"))["rows"]:
    row(f"N1 neuron removal d={r['alive']} (exact)", "Bruno", NF, r["literals"], r["literals"], round(r["agree"], 4), None, "prune_ft_neurons.json", "dev seed 61")
json.dump({"bar": 0.98671, "holdout": "seed 432432, N=1e6, check_agreement.py", "rows": R}, open("RESULTS_ledger.json", "w"), indent=1)
print(len(R), "rows")
