# Proof matrix — cot-necessity

Rows between the GENERATED markers come from `status.py` (rerun it; do not edit them by hand). Skeptics add
their rebuild rows in the table after the markers; status.py keeps everything outside the markers.

<!-- GENERATED -->
| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline | 0 | Qwen3-8B GRPO | none | pending: laptop queue job nfpimpkalf | laptop_jobs.md |
| headline | 1 | Qwen3-8B GRPO | none | pending: laptop queue job nfpimpkalf | laptop_jobs.md |
| headline | 2 | Qwen3-8B GRPO | none | pending: laptop queue job nfpimpkalf | laptop_jobs.md |
| necessity premise | - | N = 8 | untrained model, no reasoning | argmax acc 0.500, AUROC 0.511, Heads share 1.000 | data/step0/necessity_k8.jsonl |
| control pick | - | N = 1 | untrained model, no reasoning | argmax acc 0.500, AUROC 1.000, Heads share 1.000 | data/step0/necessity_k1.jsonl |
| control pick | - | N = 2 | untrained model, no reasoning | argmax acc 0.500, AUROC 0.709, Heads share 1.000 | data/step0/necessity_k2.jsonl |
| control pick | - | N = 3 | untrained model, no reasoning | argmax acc 0.500, AUROC 0.463, Heads share 1.000 | data/step0/necessity_k3.jsonl |
| control pick | - | N = 4 | untrained model, no reasoning | argmax acc 0.500, AUROC 0.670, Heads share 1.000 | data/step0/necessity_k4.jsonl |

- status (frozen rule): PENDING (J1 (labq nfpimpkalf) not landed)
<!-- /GENERATED -->

Skeptic rebuild rows (own code; appended by hand, same columns, no second header so the lint reads one table):

| necessity premise (Vera rebuild) | - | N = 8 | untrained model, no reasoning | argmax acc 0.500, AUROC 0.511; label-shuffle null band [0.445, 0.556] | skeptics/skeptic/vera_step0.json |
| control pick (Vera rebuild) | - | N = 1-4 | untrained model, no reasoning | AUROC 1.000 / 0.709 / 0.463 / 0.670, argmax 0.500 at every N | skeptics/skeptic/vera_step0.json |
