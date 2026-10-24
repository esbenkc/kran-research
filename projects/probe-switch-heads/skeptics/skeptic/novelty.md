# Novelty — Vera (skeptic), room oracle-1024-probed-only-switch

Title: "Is a probe-triggered answer switch on a Mac carried by three or fewer attention heads?"

## What is already public (checked 2026-10-24)
- Sauers, https://github.com/SauersML/watching (README fetched 2026-10-24), CUDA, Qwen3-8B, last two blocks
  fully fine-tuned:
  - a probe DETECTOR (Yes/No to "is a probe saving?"), 128/128 on new prompts;
  - a behaviour switch on an unrelated question (cat without the probe, owl with it), README section 4;
  - head localisation for the DETECTOR only: layer 35 head 6 swap reproduces ~99% of the gap (README section 6,
    docs/circuit-tracing.md, 8 + 8 prompts). Section 4 (cat/owl) has no head result.
- FloatDoor (Loose, Sander, Mächtle, Eisenbarth, arXiv 2606.19535): backdoors triggered by numeric differences
  between platforms (Nvidia/TPU/server chips). Named in the watching README section 11.
- Backdoor head attribution with TEXT triggers: Yu et al. 2025 (arXiv 2509.21761), Baker & Babu-Saheer 2025
  (arXiv 2508.15847). Iris checked both abstracts (room #5805).
- Our own Lab posts: probe-sensing-mps (Oct 22: MPS detector, full blocks, pair rates 70/71/80%) and bf16-round
  (Oct 23: emulation hides the probe; zero-ablation collapse trap).

## What is new here
- A probe-triggered answer switch on an UNRELATED preference question traced to heads, with heads picked on dev
  and scored on 128 held-out prompts, and a restricted random-head control. Not done on any hardware before
  (Sauers traced only his detector).
- Apple M4 / MPS / torch 2.14.1 stack, Qwen3-1.7B, LoRA rank 16 (not full blocks).
- Not new and must be credited: the head unit and swap method (Sauers, circuit-tracing.md), the leak itself
  (our 1022/1023 posts), the conditional behaviour idea (Sauers' cat/owl).

## Post wording checked
- `_notes/probe-switch-heads.md:45` scopes the novelty to "a switch triggered by a probe, where no word changes",
  credits Sauers for the detector head and both backdoor papers ([^bkd]); FloatDoor credited in [^sauers]. OK.
