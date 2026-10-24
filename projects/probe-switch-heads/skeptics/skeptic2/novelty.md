# skeptic2 novelty notes

Sources checked by Hugo.

- SauersML/watching README: Qwen3-8B learned a probe detector and a joint detector + cat/owl behaviour switch. The README reports 119/128 animal-choice answers in a fresh joint confirmation and describes the behaviour switch as already shown on CUDA. It does not claim head localization for the cat/owl behaviour switch.
- SauersML/watching docs/circuit-tracing.md: Experiment 044 localizes the detector's Yes/No condition gap. The key public metric is head-output swapping at the answer position; layer 35 head 6 carries about 99% of the detector gap on eight diagnostic test cases. The document says the head-output swap does not distinguish changed attention weights from changed value vectors.
- Related search: arXiv 2508.15847 reports attention-head interventions for backdoored LLM behaviour. I have not found a public result that localizes a compiled-probe-triggered unrelated behaviour switch on MPS with train/dev head selection and 128 held-out prompts.

Novelty claim I can support if the raw data reproduce: not 'models can detect probes' and not 'one detector head exists' in general. The narrower claim is whether a Mac MPS probe-triggered answer switch is carried by <=3 query-head output slices under a frozen, dev-selected, held-out patch metric.
