"""Collect every status number -> data/summary.json, data/figdata.json and proof-matrix.md (rebuilt from data/ only).

Status cells come from data/status/<run>.json (REAL compiled graph, 200 test prompts); curves from
data/runs/<run>/curve.jsonl (cached tail, 64 dev prompts; first=26 runs read real recorded numbers, first=22 runs
read the emulated footprint); check probe from data/check/<run>.json.

    env/.venv/bin/python summarize.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
D = HERE / "data"
RUNS = ["pilot_standard_s0", "O_fed_s0", "B_fed_s0", "B_standard_s0", "B_standard_s1", "A_standard_s0",
        "A_standard_s1", "A_fedST_s0"]
ROLE = {"pilot": "no penalty (R0)", "O_fed": "positive control: output penalty, trainer fed the serving numbers",
        "B_fed": "reward-only shaping, trainer fed the serving numbers", "B_standard": "reward-only shaping, standard trainer",
        "A_standard": "reward-plus-gradient shaping, standard trainer", "A_fedST": "reward-plus-gradient shaping, fed (ST)"}


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
            fig["runs"][r] = {"role": ROLE[r.rsplit("_s", 1)[0]], "curve": [
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
        arm, seed = r.rsplit("_s", 1)
        claim = "headline" if arm in ("B_standard", "A_standard", "O_fed") else arm
        ctl = {"pilot": "no penalty", "O_fed": "positive control (trainer fed serving numbers)"}.get(arm, "none")
        rows.append(f"| {claim} | {seed} | {arm} | {ctl} | cheat without probe {pct(s['cheat_noprobe'])}, with leaky "
                    f"probe {pct(s['cheat_leaky'])}, gap {100 * s['gap']:+.1f} points (95% CI {100 * s['ci_lo']:+.1f} to "
                    f"{100 * s['ci_hi']:+.1f}), n {s['n']} | data/status/{r}.json |")
        rows.append(f"| bit-exact | {seed} | {arm} | bit-exact probe | gap {100 * s['gap_bitexact']:+.2f} points, logits "
                    f"differ on {pct(s['share_sha_differs_bitexact'])} of prompts (leaky: "
                    f"{pct(s['share_sha_differs_leaky'])}) | data/status/{r}_raw/score_plain_sha.json |")
        cv = curve(r)
        if cv:
            rows.append(f"| training curve | {seed} | {arm} | none | dev gap at the end {100 * cv[-1]['gap']:+.1f} points "
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
    (HERE / "proof-matrix.md").write_text("# Proof matrix (built by summarize.py from data/)\n\n" + "\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
