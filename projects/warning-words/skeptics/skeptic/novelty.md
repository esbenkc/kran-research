# Novelty check — Vera (skeptic), defense-1112-warning-words

Question: does the instruction to name a suspected honeypot in the agent's own reasoning, and not the honeypot
warning itself, carry the cut in decoy-key use?

## Closest prior work found (searched 2026-11-11 23:10 PST)

| work | what it tested | overlap with us | gap we fill |
|---|---|---|---|
| Tracebit, "AI Agents and Canaries" (May 2026), https://agentic.tracebit.com/ — as summarised in Honeyquest for LLMs, arXiv 2606.21037, sec. 2.3 and 6 | 10 frontier LLM attackers, live 300-resource AWS env, 11 canaries, 2x2 canary presence x DISCLOSURE (Tularosa design). Informed attackers: full compromise 20% -> 3% (Claude Opus 4.7 68% -> 0%) | disclosure sentence as a lever on agents | Honeyquest's authors write that Tracebit "did not address why disclosing the presence of canaries in a prompt reduced worst-case compromise" and call for "future experiments ... to isolate the mechanism". Our 2x2 (honeypot sentence x name-it sentence) + content-free naming arm is such an experiment. |
| Honeyquest for LLMs, Prinos, Brush and Denton 2026, arXiv 2606.21037 (abstract) | 21 LLMs, 174 recon queries | spontaneous trap recognition in reasoning vs action | "recognition-action gap": models articulate trap recognition but exploit anyway 73.4% of the time; recognition did not predict falling for it (Spearman r = +0.08). Spontaneous, not INSTRUCTED naming. Our name-it arm tests the instructed case. |
| Rouxii, Cordeiro et al. 2026, arXiv 2609.26555 | prompt cues raise honeypot IDENTIFICATION 19% -> 97% | instructed honeypot attention | measures identification, not whether identification stops use; source of the parent's warning wording |
| Schoen et al. 2025, arXiv 2509.15541 | verbalised evaluation awareness causally cuts covert actions (CoT ablation/injection) | verbalisation -> behaviour | manipulates the CoT content directly, not a prompt instruction to label; different behaviour (covert actions, not decoy use) |
| AgentTrap, arXiv 2610.02869 | trap endpoints; models that recognise a deceptive endpoint refuse its bait | recognition -> refusal | no warning-text test (Iris and Kasper verified); must not be cited as "warnings protect" |
| LLM-Based Penetration Testing in the Presence of Honeypots, arXiv 2609.08093 | detector-guided host selection for attack agents | honeypot-aware agents | detectors and budget policy, no warning-sentence ablation (I read the HTML: no prompt-ablation of warnings) |
| Self-reminder / safety-guided CoT prompts (Xie et al. 2023; "Think Twice", arXiv 2505.11063) | reminder or thought-labelling prompts cut jailbreak/unsafe action rates | prompting the model to label its thought | jailbreak and generic unsafe action, not a planted honeytoken; no split of "warning" vs "label it" |

## Verdict (pre-data)

No paper found that separates a deterrence sentence from an instruction to name the suspect resource on the same
environment. The closest is Tracebit's disclosure 2x2, whose mechanism Honeyquest explicitly calls open. The
post may say "we found no prior split", citing these searches; it may NOT say "first" without that hedge.

Search log: web_search "LLM agent honeypot warning prompt ablation ... verbalize suspicion"; "self-reminder prompt
defense ablation ... labelling"; "Honey for the Agent ..."; read 2606.21037 PDF (Tracebit lines), 2609.08093 HTML.
Correction (Iris #20112, verified): Tracebit's own 2x2 page is live, https://agentic.tracebit.com/ai-agents-canaries/
(v1.0 2026-05-28, 10 models, 599 scored runs; telling models to expect deception cut full compromise 20% -> 3%).
Limits: outcome is full compromise, not decoy use; the 20% -> 3% is pooled across canary conditions (includes runs
with no canaries), so it reads as general caution; the page does not show the prompt wording. Cite it as primary.
