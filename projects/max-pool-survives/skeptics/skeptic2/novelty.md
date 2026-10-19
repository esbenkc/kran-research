# Hugo (skeptic2) novelty check

Checked 2026-10-19, independent of Iris's novelty.md (I read it after my own search).

- RL-Obfuscation, Gupta & Jenner, arXiv 2506.14261: the RL policy chooses its own tokens and uses no
  gradients through the monitor; a policy trained against a mean-aggregated probe did not evade max-aggregated
  probes. Our attacker has gradients and cannot change tokens. Our result (max14 0.0% after re-threshold) is
  the opposite outcome under a different attacker, not a contradiction of their finding.
- Obfuscated Activations, Bailey et al., arXiv 2412.09565: I grepped the HTML for "max"/"aggregat". Their SQL
  detectors use "a maximum over the sequence dimension and a mean over layers", with "one embedding attack
  optimized for each detector". They do not test transfer from a mean-pooled to a max-pooled token probe.
- Web search (2 queries, 2026-10-19) found no paper that trains a weight-space attacker against a mean-pooled
  probe on fixed text and scores a clean-fit max-pooled probe at a fixed false-alarm rate. Nearest new hit:
  arXiv 2606.09411 (steganographic payloads; evasion transfers from linear to MLP probes), no pooling test.
- Allowed claim: "we found no published test of ...". Do not say "first".
- Record tables: none apply (no benchmark leaderboard for this setup).
