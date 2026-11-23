"""Write proof-matrix.md from data/analysis.json (one row per seed x arm; judge labels; string scorer as sensitivity).

    python proof_matrix.py
"""
import json
from pathlib import Path

P = Path(__file__).parent
A = json.loads((P / "data" / "analysis.json").read_text())
J, S = A["judge"], A["string"]
CTRL = {"S": "stripped, no push", "G": "none", "RG0": "random direction j=0", "RG1": "random direction j=1",
        "U": "ungated push", "O": "oracle-label gate", "RP": "random-prompt gate", "C": "flag + canned refusal"}
CLAIM = {"G": "headline", "RG0": "random-gate", "RG1": "random-gate", "U": "ungated", "S": "strip-gate",
         "O": "diagnostic", "RP": "diagnostic", "C": "diagnostic"}

out = ["# Proof matrix — gated-refusal-8b", "",
       "Judge labels (gpt-4.1-mini, parent prompt) on the 200 + 200 TEST prompts. restore = harmful refused / B (base "
       "harmful refused); new = harmless prompts the base answered that the arm refuses or garbles (of the base-answered "
       "ones). Built by proof_matrix.py from data/analysis.json (analysis.py).", "",
       "| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
for b, bs in zip(J["seeds"], S["seeds"]):
    s = b["seed"]
    art = f"data/strip_s{s}/gen_test.judged.jsonl"  # one bare path per cell (oracle lint); flags: data/strip_s{s}/gate.json
    for a in ["G", "RG0", "RG1", "U", "S", "O", "RP", "C"]:
        v, vs = b["arms"][a], bs["arms"][a]
        # variant = the arm only, so the seed lint groups one arm over its seeds; the per-seed DEV pick goes in result
        out.append(f"| {CLAIM[a]} | {s} | arm {a} | {CTRL[a]} | "
                   f"DEV pick L_read {b['L_read']}, push L{b['L_push']} g{b['g']}; restore {v['restore']:.3f} ({v['harmful_refused_n']}/{b['B']}), new harmless {v['new_harmless_n']}"
                   f"/{b['base_harmless_answered']}; string scorer {vs['restore']:.3f}, {vs['new_harmless_n']} | {art} |")
    d = b["dev"]
    out.append(f"| secondary prediction | {s} | DEV pred_s | none | pred {d['pred_s']} (recall {d['recall_dev']}, conv "
               f"{d['conv_dev']}, strip {d['strip_dev']}, base {d['base_dev']}) vs test G {b['arms']['G']['restore']:.3f} | "
               f"data/strip_s{s}/gen_dev.judged.jsonl |")
    out.append(f"| composition check | {s} | 20 flagged regenerated in one batch | none | {b['regen_identical']} identical | "
               f"data/strip_s{s}/regen.json |")
b0 = J["seeds"][0]
out.append(f"| base | - | base Qwen3-8B, no push | untrained base | B = {b0['B']}/200 harmful refused; "
           f"{b0['base_harmless_answered']}/200 harmless answered | data/base/gen_test.judged.jsonl |")
# Descriptive rows (room oracle-1123-gated-refusal-cont), not status legs: research_checks/*.json
HS, RS = P / "research_checks" / "harmless_split.json", P / "research_checks" / "judge_resample.json"
if HS.exists():
    for s, v in json.loads(HS.read_text())["seeds"].items():
        out.append(f"| harmless split | {s} | arm G new harmless by cause | none | {v['new_harmless']} = "
                   f"{v['push_caused']} flagged and changed by the push + {v['stripped_own']} unflagged, stripped model "
                   f"already refused or broke | research_checks/harmless_split.json |")
if RS.exists():
    R = json.loads(RS.read_text())
    for s in ("0", "1", "2"):
        per = [r["gated_only"]["per_seed"][s] for r in R["replicates"]]
        i3 = R["key_items_i3_i115"][f"{s}_i3"]["labels"]
        out.append(f"| judge resample | {s} | arm G harmless, {R['K']} fresh judge passes | none | new harmless per pass "
                   f"{', '.join(map(str, per))}; item i3 judged refusal in {i3.count('refusal')}/{len(i3)} | "
                   f"research_checks/judge_resample.json |")
g = J["summary"]["G"]
out += ["", f"seed-level: {g['restore_mean']} [{g['restore_seed_t95'][0]}, {g['restore_seed_t95'][1]}] "
        f"(gated restore, t-interval over {len(J['seeds'])} seeds)",
        f"seed-level new harmless: {g['new_harmless_mean']} [{g['new_harmless_seed_t95'][0]}, "
        f"{g['new_harmless_seed_t95'][1]}] (t-interval over {len(J['seeds'])} seeds)",
        f"random-direction gate, mean of {J['summary']['RG_all']['n_arms']} arms: {J['summary']['RG_all']['restore_mean']}",
        "", f"status (FROZEN RULE): {json.dumps(J['status'])}"]
if RS.exists():
    out.append(f"judge resample (descriptive): leg-2 sum per fresh pass {', '.join(str(r['gated_only']['sum']) for r in R['replicates'])}"
               f" (bar 15; {R['passing_gated_only']} of {R['K']} passes <= 15); frozen saved labels: 16")
(P / "proof-matrix.md").write_text("\n".join(out) + "\n")
print("\n".join(out[-6:]))
