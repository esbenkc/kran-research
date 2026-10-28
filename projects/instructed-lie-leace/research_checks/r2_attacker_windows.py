"""Iris: attacker loss per 50-step window from a run json's 'log' (Tove #9367 item 2): probe term (mean squared twin
gap of the current in-loop probes) at the window's first and last step, and lam*(KL_prompts+KL_pile)/2."""
import json, sys
P = "/Users/esben/work/theory-1028-instructed-lie/kran-research/projects/instructed-lie-leace/data"
name = sys.argv[1] if len(sys.argv) > 1 else "lie_spread__lam10000__seed0"
d = json.load(open(f"{P}/runs/{name}.json")); L = d["log"]; lam = d["lam"]; out = []
kl = lambda r: lam * 0.5 * (r["kl_prompts"] + r["kl_pile"])
for w in range(0, len(L), 50):
    s = L[w:w + 50]
    out.append({"steps": [s[0]["step"], s[-1]["step"]], "probe_first": s[0]["probe_loss"], "probe_last": s[-1]["probe_loss"],
                "probe_mean": sum(r["probe_loss"] for r in s) / len(s), "kl_term_mean": sum(kl(r) for r in s) / len(s)})
    print(json.dumps(out[-1]))
json.dump(out, open(f"attacker_windows_{name}.json", "w"), indent=1)
