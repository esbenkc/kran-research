"""Collect every status number -> data/summary.json, data/figdata.json and proof-matrix.md (rebuilt from data/ only).

Status cells come from data/status/<run>.json (REAL compiled graph, 200 test prompts); curves from
data/runs/<run>/curve.jsonl (cached tail, 64 dev prompts; first=26 runs read real recorded numbers, first=22 runs
read the emulated footprint); check probe from data/check/<run>.json.

    env/.venv/bin/python summarize.py
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
D = HERE / "data"
RUNS = ["pilot_standard_s0", "O_fed_s0", "P0_fed_s0", "P0_fed_s0_10k", "B_fed_s0", "B_standard_s0", "B_standard_s1",
        "A_standard_s0", "P0_fedpilot_s0", "B_fedpilot_s0", "B_stdpilot_s0", "B_fed14pilot_s0", "B_std14pilot_s0", "B_fed14pilot_s1", "B_std14pilot_s1"]
ROLE = {"pilot": "no penalty (R0)", "O_fed": "positive control: output penalty, trainer fed the serving numbers",
        "B_fed": "reward-only shaping, trainer fed the serving numbers", "B_standard": "reward-only shaping, standard trainer",
        "A_standard": "reward-plus-gradient shaping, standard trainer",
        "P0_fed": "positive control: the parent's reward, trainer fed the serving numbers (D2)",
        "P0_fed_s0_10k": "P0-fed continued to 10,000 updates (D3)",
        "P0_fedpilot": "reading only (D4/D5): parent's reward, fed, started from the pilot (cheat 0.39)",
        "B_fedpilot": "reading only (D5): reward-only shaping, fed, started from the pilot",
        "B_stdpilot": "reading only (D5): reward-only shaping, standard trainer, started from the pilot",
        "B_fed14pilot": "reading only (D6): probe at layer 14, reward-only shaping, fed, from the pilot",
        "B_std14pilot": "reading only (D6): probe at layer 14, reward-only shaping, standard trainer, from the pilot"}


def load(p):
    return json.load(open(p)) if p.exists() else None


def curve(run):
    p = D / "runs" / run / "curve.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else None


def main():
    S = {r: load(D / "status" / f"{r}.json") for r in RUNS + ["base"]}
    C = {r: load(D / "check" / f"{r}.json") for r in RUNS + ["base"]}
    R0 = S["pilot_standard_s0"]["cheat_noprobe"] if S["pilot_standard_s0"] else None
    summ = {"R0": R0, "runs": {}}
    for r in RUNS + ["base"]:
        if not S[r]:
            continue
        s = {k: v for k, v in S[r].items() if k != "prompt_ids"}
        if C[r]:
            s["check"] = C[r]
        cv = curve(r)
        if cv:
            s["curve_end"] = cv[-1]
        summ["runs"][r] = s
    json.dump(summ, open(D / "summary.json", "w"), indent=1)
    fig = {"R0": R0, "runs": {}}
    for r in RUNS:
        cv = curve(r)
        if cv or S[r]:
            fig["runs"][r] = {"role": ROLE.get(r, ROLE.get(r.rsplit("_s", 1)[0])), "curve": [
                {k: row[k] for k in ("step", "cheat_noprobe", "cheat_leaky", "probe_margin_leaky") if k in row}
                for row in (cv or [])],
                "end": {k: S[r][k] for k in ("cheat_noprobe", "cheat_leaky", "cheat_bitexact", "gap", "ci_lo", "ci_hi",
                                              "gap_bitexact")} if S[r] else None,
                "check_auroc_refit": C[r]["auroc_refit"] if C[r] else None}
    json.dump(fig, open(D / "figdata.json", "w"), indent=1)
    rows = ["| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]

    def pct(x):
        return f"{100 * x:.1f}%"
    for r in RUNS:
        s = S[r]
        if not s:
            continue
        mm = re.match(r"^(.*)_s(\d+)(_10k)?$", r)
        arm, seed = mm.group(1) + (mm.group(3) or ""), mm.group(2)
        claim = "headline" if arm in ("B_standard", "A_standard") else ("positive control" if arm in ("O_fed", "P0_fed", "P0_fed_10k") else (arm + " (reading only)" if "pilot" in arm and arm != "pilot" else arm))
        ctl = {"pilot": "no penalty", "O_fed": "positive control (trainer fed serving numbers)",
               "P0_fed": "positive control (trainer fed serving numbers)", "P0_fed_10k": "positive control (trainer fed serving numbers)"}.get(arm, "none")
        rows.append(f"| {claim} | {seed} | {arm} | {ctl} | cheat without probe {pct(s['cheat_noprobe'])}, with leaky "
                    f"probe {pct(s['cheat_leaky'])}, gap {100 * s['gap']:+.2f} points (95% CI {100 * s['ci_lo']:+.2f} to "
                    f"{100 * s['ci_hi']:+.2f}), n {s['n']} | data/status/{r}.json |")
        rows.append(f"| bit-exact | {seed} | {arm} | bit-exact probe | gap {100 * s['gap_bitexact']:+.2f} points, logits "
                    f"differ on {pct(s['share_sha_differs_bitexact'])} of prompts (leaky: "
                    f"{pct(s['share_sha_differs_leaky'])}) | data/status/{r}_raw/score_plain_sha.json |")
        cv = curve(r)
        if cv:
            rows.append(f"| training curve | {seed} | {arm} | none | dev gap at the end {100 * cv[-1]['gap']:+.2f} points "
                        f"(cached {'emulated' if arm.startswith('A_standard') else 'recorded'} numbers), probe margin "
                        f"{cv[0]['probe_margin_leaky']:.2f} -> {cv[-1]['probe_margin_leaky']:.2f} | data/runs/{r}/curve.jsonl |")
        if C[r]:
            c = C[r]
            rows.append(f"| check probe | {seed} | {arm} | shuffled labels | refit AUROC {c['auroc_refit']:.3f}, balanced "
                        f"accuracy {c['bal_acc_refit']:.3f}, original probe AUROC {c['auroc_orig']:.3f}, shuffled "
                        f"{c['auroc_refit_shuffled']:.3f} | data/check/{r}.json |")
    if S["base"]:
        s = S["base"]
        rows.append(f"| untrained | - | base | untrained model | cheat {pct(s['cheat_noprobe'])} / {pct(s['cheat_leaky'])}, "
                    f"gap {100 * s['gap']:+.2f} points | data/status/base.json |")
    g = load(D / "gate.json")
    if g:
        rows.append(f"| footprint | - | base, 646 prompts | none | serving graph: leaky probe changes the logits of "
                    f"{pct(g['plain_sha_differs_ref_vs_none'])} of prompts, bit-exact probe {pct(g['plain_sha_differs_score_vs_none'])}"
                    f" | data/gate.json |")
        rows.append(f"| emulation | - | base, 646 prompts | real footprint | emulated vs real per-prompt corr "
                    f"{g['corr_D_emu_real']:.2f} | data/gate.json |")
    fp = load(D / "footprint" / "footprint.json")
    if fp:
        for row in fp["rows"]:
            if row["build"] != "none":
                rows.append(f"| update graph footprint | - | {row['mode']} graph, {row['build']} | none | "
                            f"{row['n_diff']} logit values differ ({100 * row['frac_diff']:.1f}%), max {row['max_abs']} "
                            f"| data/footprint/footprint.json |")
    lm = D / "lemma" / "lemma_grads.pt"
    if lm.exists():
        import torch
        z = torch.load(lm)
        g1, g0 = z["grads"]["probed"], z["grads"]["unprobed"]
        same = sum(torch.equal(g1[k], g0[k]) for k in g1)
        nzb = sorted({int(k.split("layers.")[1].split(".")[0]) + 22 for k in g1 if g1[k].abs().sum() > 0})
        rows.append(f"| lemma | - | gradient term, 8 dev prompts x 2 answers | probe off | {same}/{len(g1)} LoRA grads "
                    f"bit-identical with and without the probe; nonzero only in blocks {nzb} | data/lemma/lemma_grads.pt |")
    (HERE / "proof-matrix.md").write_text("# Proof matrix (built by summarize.py from data/)\n\n" + "\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
