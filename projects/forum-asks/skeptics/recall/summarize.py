#!/usr/bin/env python3
"""Write my hand judgements for the regex-miss samples and compute the numbers in signoff.md.

Verdicts: yes / borderline / no. Every item not listed below is "no" (my reading). "css" marks MathJax
style-sheet text that html_to_text keeps (not prose; counted as "no").
"""
import json, math
from pathlib import Path

HERE = Path(__file__).parent

# regex_sample.jsonl (pool A, seed 13): n -> (verdict, ai, phrasing class, why)
MAIN = {
    94: ("yes", False, "encourage/imperative", "'I'd strongly encourage people to try to replicate this with their own local groups' (EA London gender retention)"),
    137: ("yes", False, "modal with you/one/we/they", "'they could try to come up with specific measures... put out a call for people to complete these measures' before/after a sequence"),
    26: ("borderline", False, "modal with you/one/we/they", "'You could do a study where you find women at high risk of drinking...' but next sentence says it would never be allowed"),
    55: ("borderline", False, "modal with you/one/we/they", "'you should ... do a power analysis' - method advice for someone's own self-experiment"),
    143: ("borderline", False, "modal with you/one/we/they", "'We could test the spreadsheet solution by...' - authors' own pilot idea"),
}
# regex_cue_sample.jsonl (pool A with an ask cue word, seed 14)
CUE = {
    13: ("yes", True, "we'd / we are interested / excited about", "'We are interested in studies where people are given an agentic code editor where the AI is sometimes replaced with a misaligned version'"),
    16: ("yes", True, "modal with you/one/we/they", "'We could try to measure how influential a particular industry is on the rest of the economy' (to forecast AI impact)"),
    45: ("yes", True, "we'd / we are interested / excited about", "'I would be excited about people doing these experiments' (debate, obfuscated arguments)"),
    53: ("yes", False, "modal with you/one/we/they", "'One could use a conditional place preference test...' (animals and trajectory of pain)"),
    61: ("yes", True, "singular study / next study / future-work wording", "'Conducting these experiments on reasoning models and monitoring the CoT ... can be an interesting next study'"),
    100: ("yes", True, "modal with you/one/we/they", "'To measure the size of the polka dots, you could invert the typical adversarial perturbation procedure'"),
    112: ("yes", True, "encourage/imperative", "'test whether model diffing can help an auditing agent discover emergent misalignment ... Consider running it!'"),
    145: ("yes", False, "'would be great for someone to / to get X to' test", "'it would be great to get GiveDirectly to test more explicitly the macroeconomic offsetting problem'"),
    150: ("yes", True, "modal with you/one/we/they", "'Can we test this thing?' followed by 'Experiments to test the Human Worth Hypothesis'"),
    175: ("yes", False, "'would be great for someone to / to get X to' test", "'it would be great for someone to replicate these findings ... and I hope someone does that'"),
    28: ("borderline", False, "modal with you/one/we/they", "'you might want to specifically test that' (is a Madison hub practical)"),
    114: ("borderline", False, "question form", "'did you rule out ... a slow cooker ... like the original experiment?' - a more accurate replication method"),
    118: ("borderline", True, "singular study / next study / future-work wording", "'we did not test how reasoning and preference would interact' - stated gap, no request"),
    142: ("borderline", True, "modal with you/one/we/they", "'Running sensitivity analysis ... would give more credibility to any eval claims' - generic method advice"),
    165: ("borderline", False, "question form", "'How would we go about testing this assuming we had a lot of resources' - asks for a design"),
}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def judge(fname, table, out):
    rows = [json.loads(l) for l in open(HERE / fname)]
    with open(HERE / out, "w") as f:
        for r in rows:
            css = "mjx" in r["sentence"].lower()
            v, ai, cls, why = table.get(r["n"], ("no", None, None, "css (MathJax style text)" if css else ""))
            f.write(json.dumps({"n": r["n"], "forum": r["forum"], "kind": r["kind"], "doc_id": r["doc_id"],
                                "sent_idx": r["sent_idx"], "url": r["url"], "my_verdict": v, "ai": ai,
                                "missed_phrasing": cls, "why": why}) + "\n")
    return rows


pools = json.loads((HERE / "pool_counts.json").read_text())
N_A, N_CUE = pools["A_sentences"], pools["A_cue_sentences"]
main = judge("regex_sample.jsonl", MAIN, "regex_judgements.jsonl")
cue = judge("regex_cue_sample.jsonl", CUE, "regex_cue_judgements.jsonl")

neg = [json.loads(l) for l in open(HERE / "neg_judgements.jsonl")]
res = {}
k = sum(r["my_verdict"] == "yes" for r in neg); kb = sum(r["my_verdict"] != "no" for r in neg)
res["classifier"] = {"n": len(neg), "yes": k, "yes_or_borderline": kb, "share": k / len(neg), "ci": wilson(k, len(neg)),
                     "share_wb": kb / len(neg), "ci_wb": wilson(kb, len(neg)),
                     "ai_yes": sum(r["my_verdict"] == "yes" and r["ai"] for r in neg),
                     "ai_borderline": sum(r["my_verdict"] == "borderline" and r["ai"] for r in neg),
                     "scaled_to_22199": [22199 * x for x in (k / len(neg), *wilson(k, len(neg)))]}
for name, rows, tab, N in (("regex_main", main, MAIN, N_A), ("regex_cue", cue, CUE, N_CUE)):
    n = len(rows); y = sum(v[0] == "yes" for v in tab.values()); yb = len(tab)
    css = sum("mjx" in r["sentence"].lower() for r in rows)
    res[name] = {"pool": N, "n": n, "css_junk": css, "yes": y, "yes_or_borderline": yb, "share": y / n, "ci": wilson(y, n),
                 "share_wb": yb / n, "ci_wb": wilson(yb, n), "ai_yes": sum(v[0] == "yes" and v[1] for v in tab.values()),
                 "est": [N * x for x in (y / n, *wilson(y, n))], "est_wb": [N * x for x in (yb / n, *wilson(yb, n))]}
# main-sample items that fall outside the cue stratum: are any of them asks?
import re, sys
sys.dont_write_bytecode = True
CUE_RE = re.compile(r"\?|\b(?:should|could|would|wonder|curious|love to|like to|want|wish|worth|interesting|"
                    r"someone|somebody|anyone|needs? to|hope|it'd|I'd)\b", re.I)
noncue = [r for r in main if not CUE_RE.search(r["sentence"])]
res["regex_main_noncue"] = {"n": len(noncue), "yes_or_borderline": sum(r["n"] in MAIN for r in noncue),
                            "pool": N_A - N_CUE, "ci": wilson(0, len(noncue))}
print(json.dumps(res, indent=1))
(HERE / "numbers.json").write_text(json.dumps(res, indent=1))
