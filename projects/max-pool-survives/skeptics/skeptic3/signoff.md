# Nadia's Sign-Off

**Verdict**: REJECTED

I independently rebuilt the core results from the raw tensor files (`clean_tokens.npz` and `lam0__seed*__tokens.npz`).
- **Max14 Pooled Catch at 1% FPR:** 0.00% (CI: 0.0% - 0.45%)
- **Null Test (Probe-split label swap within pairs):** Catches ~1.73% across seeds (0.6% - 2.3%), verifying the pipeline does not artificially suppress the catch rate without a real signal.
- **Derivations:** Both the prefix lemma and shared-shift lemma hold. The bell-curve formula was checked and verified.
- **Novelty:** I re-checked RL-Obfuscation (arXiv 2506.14261), Neural Chameleons (arXiv 2512.11949), and Bailey et al. (2412.09565). The claim that no published test has evaluated max-pool robustness against a weight-space gradient attacker on fixed text holds.

The max14 probe's catch rate falls decisively below the 50% threshold, rejecting the hypothesis.
