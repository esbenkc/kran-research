# Novelty check — Vera (skeptic), oracle-1026-blind-head-audit

Claim to check: a behaviour-free head audit (the defender knows its probe = the trigger, but not the trained behaviour
or question) ranks heads by how much their output changes when the probe is attached on neutral prompts, trained
minus base, and the top-3 carry the switch in label-trained and reward-trained models.

Nearest prior work I found (searched 2026-10-25):

| Work | Trigger known? | Behaviour known? | Trigger type | What it does | Difference to ours |
|---|---|---|---|---|---|
| BAHA, Backdoor Attribution (arXiv 2509.21761) | yes | yes (ranks heads by backdoor activation substitution on poisoned predictions, then ablates) | text token | ~3% of heads cut ASR ~90% | uses behaviour (attack success) to rank; text trigger |
| Mechanistic Exploration of Backdoored LLM Attention Patterns (arXiv 2508.15847) | yes | partly (patches poisoned-model heads with clean-model heads, scores KL on logits) | text tokens | attention divergence poisoned vs clean concentrates in layers 20-30 | closest to "trained minus base" with the trigger attached; but on triggered task inputs and with logit KL, not on neutral inputs; text trigger |
| Uncovering and Aligning Anomalous Attention Heads (arXiv 2511.13789, AAAI) | no | no | text | attention-head similarity anomaly as a trigger-free detector | trigger-free detection, not a head-level causal cut of a known non-text trigger |
| Microsoft, "Detecting backdoored language models at scale" (blog, 2026-02-04) | no | no | text | "double triangle" attention signature of trigger tokens | text-token hijack signature; no numeric trigger |
| Trigger in the Haystack, Bullwinkel et al. (arXiv 2602.03085) | no | no | text | scanner for sleeper-agent text triggers not known in advance | text triggers; no head-level causal cut |
| LLM Forensics, Antoun et al. (arXiv 2609.07746) | yes | partly | text | SAE features that detect a trigger often do not control the behaviour | same detect-vs-carry split we see: RL head 27.9 is the 2nd-highest audit score and carries ~0 push alone |
| BAIT (S&P 2025) | no | inverts target | text | black-box target inversion | output-level, no head audit |

Our earlier work this builds on: /probe-switch-heads (trigger and behaviour known, behaviour-picked heads),
/probe-sensing-mps (detectors), /numeric-chameleon-rl (reward-learned switch), /probe-switch-audit-penalty.

Verdict on novelty: NEW as scoped. I found no prior work that (a) ranks heads by their change when a known
NON-TEXT trigger (a read-only probe's numeric footprint under torch.compile) is attached, (b) on neutral inputs with
no trained question and no behaviour, trained minus base, and (c) tests the cut on a reward-learned switch. The
nearest (2508.15847) diffs poisoned vs clean heads with the text trigger attached, on triggered task inputs. So the
post must say 'behaviour-free, not trigger-free' (the probe owner knows the trigger) and credit BAHA (2509.21761),
2508.15847 and Sauers' one-head trace as trigger-and-behaviour-known prior art. The negative finding (the
whole-prompt average buries a carrier that reacts only at the last token) is also not reported in these works.
