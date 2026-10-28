"""Figure data for the ONE main figure: retrained-probe catch (output-blind panel) against fine-tune step for the
decoy run and the lie run(s), with each concept's LEACE price measured before training; the full panel and the
output-only probe as reference lines (the mechanical half). Reads data/results/summary.json (analysis.py).
-> data/results/figure_data.json
"""
import json
from pathlib import Path

D = Path(__file__).parent / "data"


def main():
    s = json.loads((D / "results" / "summary.json").read_text())
    pr = s["prices"]
    out = {"prices": {"lie_blind": pr.get("lie_blind", {}).get("P_answer"),
                      "decoy_blind": pr.get("decoy_blind", {}).get("P_answer"),
                      "lie_full": pr.get("lie_full", {}).get("P_answer"),
                      "decoy_full": pr.get("decoy_full", {}).get("P_answer"),
                      "N_decoy_blind": s["premise_Q"].get("N_decoy"), "N_lie_blind": s["premise_Q"].get("N_lie"),
                      "Q_blind": s["premise_Q"].get("Q")},
           "status": s["status"], "runs": []}
    for n, r in s["runs"].items():
        tr = r["traj"]
        if tr[0]["step"] != 0:  # D10 run: step 0 = the gate-0b numbers on the same clean model, fit set and eval
            tr = [{"step": 0, "catch_blind": r["step0_blind"], "catch_full": r["step0_full"], "O_catch": r["step0_O"],
                   "kl_answer_eval": 0.0}] + tr
        out["runs"].append({"name": n, "target": r["target"], "lam": r["lam"], "L_blind": r["L_blind"],
                            "late_steps": r["late_steps"], "B_lie": r.get("B_lie"), "B_honest": r.get("B_honest"),
                            "step0_source": r.get("step0_source"),
                            "steps": [t["step"] for t in tr],
                            "catch_blind": [t["catch_blind"] for t in tr],
                            "catch_full": [t["catch_full"] for t in tr],
                            "O_catch": [t["O_catch"] for t in tr],
                            "kl_answer": [t["kl_answer_eval"] for t in tr]})
    (D / "results" / "figure_data.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["prices"]), [(r["name"], r["L_blind"]) for r in out["runs"]])


if __name__ == "__main__":
    main()
