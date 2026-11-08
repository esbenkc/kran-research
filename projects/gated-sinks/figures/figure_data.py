"""Build the main figure's data.json (blog: assets/figures/gated-sinks/data.json) from results/results.json.

Reads ONLY results/results.json (Bruno's contract, room #17166) and results/results_assembly.json (the layer where each
model gathers the causes, used only for the title rule, Tove #17360). Every number drawn is copied from it, and the
derived ones (line prediction, distance from the line, inside the band) are recomputed here and asserted equal to
the stored ones, so the figure cannot drift from the post's numbers.

Usage: python figures/figure_data.py [--results results/results.json] [--out <blog>/assets/figures/gated-sinks/data.json]
The title, answer, legend and note come first in data.json: the figure-alone gate reads only the head of the file.
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NAMES = {  # Selma #17133: the main-text names
    "softmax": "standard attention",
    "softmax1": "softmax plus one",
    "sinklogit": "learned sink slot",
    "kbias": "learned key bias",
    "gated": "gated attention",
}
ORDER = ["softmax", "softmax1", "sinklogit", "kbias", "gated"]
TOL = 1e-6
BAND_FLOOR = 0.03  # PREREG.md s3
SWING_BAR = 0.05  # PREREG.md s3, REJECTED (i)
MID_LAYER = 2  # h_2, the registered middle layer (block 2 of 4)
SINK_GATE = 0.3  # PREREG: a model "has a sink" when block-2 first-token mass > 0.3


def r3(v):
    return None if v is None else round(float(v), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE.parent / "results" / "results.json"))
    ap.add_argument("--out", default=str(HERE.parents[3] / "blog.kran.ai" / "assets" / "figures" / "gated-sinks" / "data.json"))
    a = ap.parse_args()
    R = json.loads(Path(a.results).read_text())
    asm_path = Path(a.results).with_name("results_assembly.json")
    ASM = {r["run"]: r["assembly_layer"] for r in json.loads(asm_path.read_text())["rows"]} if asm_path.exists() else {}
    fit, M = R["fit"], R["models"]

    # --- checks against the contract -------------------------------------------------------
    for m in M:
        assert m["arm"] in NAMES, m["arm"]
        assert m["gated"] == (m["arm"] == "gated"), m["run"]
        assert m["in_fit"] == (not m["gated"]), f"{m['run']}: only models without a gate set the line"
        assert abs(np.mean(m["recovery_sae"]) - m["recovery"]) < 1e-4, f"{m['run']}: recovery != mean of SAE seeds"
    fitted = [m for m in M if m["in_fit"]]
    b_ols, a_ols = np.polyfit([m["sink_mass"] for m in fitted], [m["recovery"] for m in fitted], 1)
    if abs(b_ols - fit["b"]) > 1e-3 or abs(a_ols - fit["a"]) > 1e-3:
        print(f"NOTE: per-model OLS gives a={a_ols:.4f} b={b_ols:.4f}, results.json fit a={fit['a']:.4f} b={fit['b']:.4f}"
              " (fit may pool SAE seeds); the figure draws results.json's line")
    pred = lambda s: fit["a"] + fit["b"] * s
    # band(x) = max(0.03, 2 s_tot, h(x)), h = half-width of the line's 95% CI at x (PREREG.md s3, Felix #17181)
    LX = np.array([p["sink"] for p in fit["line"]])
    LH = np.array([(p["hi"] - p["lo"]) / 2 for p in fit["line"]])
    band_at = lambda s: max(BAND_FLOOR, 2 * fit["s_tot"], float(np.interp(s, LX, LH)))
    gmap = {g["run"]: g for g in R["gated"]}
    gated = [m for m in M if m["gated"]]
    assert set(gmap) == {m["run"] for m in gated}, "gated list != gated models"
    for m in gated:
        g = gmap[m["run"]]
        assert abs(g["pred"] - pred(m["sink_mass"])) < 1e-4, f"{m['run']}: stored pred {g['pred']} != a + b*sink {pred(m['sink_mass'])}"
        assert abs(g["recovery"] - m["recovery"]) < TOL and abs(g["sink_mass"] - m["sink_mass"]) < TOL, m["run"]
        assert abs(g["band"] - band_at(m["sink_mass"])) < 2e-3, f"{m['run']}: stored band {g['band']} != max(0.03, 2 s_tot, h) {band_at(m['sink_mass'])}"
        assert g["in_band"] == (abs(m["recovery"] - g["pred"]) <= g["band"] + TOL), f"{m['run']}: in_band flag disagrees"

    # --- the answer: recompute the PREREG s3 clauses and check them against results.json ------
    n_g = len(gated)
    n_in = sum(g["in_band"] for g in R["gated"])
    res = {m["run"]: m["recovery"] - pred(m["sink_mass"]) for m in gated}
    above = sum(res[r] > gmap[r]["band"] for r in res)
    below = sum(res[r] < -gmap[r]["band"] for r in res)
    lo, hi = fit["b_ci"]
    xs = [m["sink_mass"] for m in fitted]
    R_sink = max(xs) - min(xs)
    swing_hi = max(abs(lo), abs(hi)) * R_sink
    assert abs(swing_hi - fit["swing_hi"]) < 2e-3, f"swing_hi {fit['swing_hi']} != max(|b_lo|,|b_hi|) x R {swing_hi}"
    rej = {"i": swing_hi < SWING_BAR, "ii": n_g - n_in >= 2, "iii": lo > 0}
    conf = hi < 0 and n_in >= 2
    status_calc = "REJECTED" if any(rej.values()) else "CONFIRMED" if conf else "UNCONFIRMED"
    status = R["status"]
    assert status.startswith(status_calc), f"results.json status {status} != PREREG s3 rule {status_calc} {rej}"
    law = (fit.get("sink_levels") or 0) >= 3  # fewer than 3 sink levels: a two-group contrast, not a law (PREREG s3)
    # callouts on the scatter: the only middle-layer sink model, and a model without a sink that scores as
    # high and gathers the causes in the same layer (assembly layer from results_assembly.json)
    callouts, rival = [], None
    sinks = [m for m in M if m["sink_mass"] > SINK_GATE]
    if len(sinks) == 1:
        s0 = sinks[0]
        callouts.append({"run": s0["run"], "text": f"the only middle-layer sink: {s0['sink_mass']:.0%} on the first token, score {s0['recovery']:.2f}"})
        rivals = sorted([m for m in M if m["sink_mass"] < 0.1], key=lambda m: -m["recovery"])
        if rivals and rivals[0]["recovery"] >= s0["recovery"] - 0.02:
            rival = rivals[0]
            callouts.append({"run": rival["run"], "text": f"{NAMES[rival['arm']]}: no sink ({rival['sink_mass']:.0%}), score {rival['recovery']:.2f}"})
    for c in callouts:
        assert ";" not in c["text"]
    gr = [m["recovery"] for m in gated]
    group_label = {"text": f"gated attention, no sink: score {min(gr):.2f} to {max(gr):.2f}",
                   "x": max(m["sink_mass"] for m in gated), "y": min(gr)} if all(m["sink_mass"] < 0.1 for m in gated) else None

    if status_calc == "CONFIRMED":
        answer = "Yes"
        title = (f"Less sink, better recovery: the line from the models without a gate "
                 f"predicts {n_in} of {n_g} gated models")
    elif status_calc == "REJECTED":
        answer = "No"
        if rej["ii"]:
            side = "above" if above >= below else "below"
            title = f"Gated models land {side} the line that sink mass predicts: {n_g - n_in} of {n_g} outside the band"
        elif rej["i"]:
            title = f"Sink mass barely moves the recovery score: at most {swing_hi:.2f} across the whole sink range"
        else:
            title = "More sink, better recovery: the slope runs the wrong way"
    else:
        answer = "Not settled"
        same_layer = rival is not None and ASM.get(rival["run"]) is not None and ASM.get(rival["run"]) == ASM.get(sinks[0]["run"]) == MID_LAYER
        if same_layer:  # Tove #17360: lead with the cell that explains the Not settled
            title = (f"Not settled by the line: a model with no sink scored the same as the one middle-layer sink model "
                     f"({rival['recovery']:.2f} vs {sinks[0]['recovery']:.2f}) when both gathered the causes in the middle layer")
        else:
            n_sink = sum(m["sink_mass"] > SINK_GATE for m in M)
            title = f"Not settled: only {n_sink} of {len(M)} models had a sink in the middle layer, and the line's slope interval spans 0"
    if not law:
        title = title.replace("the line from the models without a gate", "the models without a gate")
    assert ";" not in title
    # the strip shades each row with its widest band: a gated dot must not look in band when it is out
    gband = max(g["band"] for g in R["gated"])
    for r, v in res.items():
        assert (abs(v) <= gband + TOL) == gmap[r]["in_band"], f"{r}: strip row band {gband:.4f} would mislabel it (own band {gmap[r]['band']:.4f})"
    print("clauses", rej, "confirm", conf, "law" if law else "two-group")

    legend = {
        "fitted": "models without a gate (they set the line)",
        "line": "line through the models without a gate",
        "band": "prediction band",
        "gated_measured": "gated attention, measured",
        "gated_predicted": "gated attention, predicted by the line",
        "floor": "untrained model",
        "strip": "distance from the line",
    }
    fl = lambda v: float(np.mean(list(v.values()))) if isinstance(v, dict) else float(v)
    floor_untrained, floor_embedding = fl(R["floors"]["untrained"]), fl(R["floors"]["embedding"])
    steps = {m["step"] for m in M}
    assert len(steps) == 1, f"one training length for every model (D8), got {steps}"
    n_steps = steps.pop()
    trained = f"trained for the same {n_steps:,} steps"
    note = (f"Each point is one tiny transformer, {trained}. Recovery score: how well a sparse "
            f"autoencoder's best entry matches each of the 64 planted causes, averaged (0 to 1), mean of "
            f"{len(M[0]['recovery_sae'])} autoencoder seeds. Sink mass: share of attention on the first token in the middle layer, where the autoencoder reads.")
    assert ";" not in note

    out = {
        "title": title,
        "answer": answer,
        "status": status,
        "sub": f"Recovery score of the 64 planted causes against sink mass, every model {trained}",
        "legend": legend,
        "note": note,
        "names": {k: NAMES[k] for k in ORDER},
        "callouts": callouts,
        "group_label": group_label,
        "fit": {"a": r3(fit["a"]), "b": r3(fit["b"]), "b_lo": r3(lo), "b_hi": r3(hi), "swing": r3(fit["swing"]), "swing_hi": r3(swing_hi),
                "s_tot": r3(fit["s_tot"]), "band_floor": BAND_FLOOR, "sink_max": r3(max(m["sink_mass"] for m in M)), "sink_levels": fit.get("sink_levels"),
                "line": [{"x": r3(p["sink"]), "y": r3(p["pred"]), "band": r3(band_at(p["sink"]))} for p in fit["line"]]},
        "floors": {"untrained": r3(floor_untrained), "embedding": r3(floor_embedding)},
        "counts": {"gated": n_g, "in_band": n_in, "above": above, "below": below, "fitted": len(fitted)},
        "models": [
            {"run": m["run"], "arm": m["arm"], "name": NAMES[m["arm"]], "seed": m["seed"], "gated": m["gated"],
             "sink": r3(m["sink_mass"]), "rec": r3(m["recovery"]), "rec_sae": [r3(v) for v in m["recovery_sae"]],
             "pred": r3(pred(m["sink_mass"])), "res": r3(m["recovery"] - pred(m["sink_mass"])), "band": r3(band_at(m["sink_mass"])),
             "loss": r3(m["eval_loss"]), "step": m["step"],
             **({"in_band": gmap[m["run"]]["in_band"]} if m["gated"] else {})}
            for m in sorted(M, key=lambda m: (ORDER.index(m["arm"]), m["seed"]))
        ],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}: {status} | {title} | gated in band {n_in}/{n_g} | slope {fit['b']:.3f} [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
