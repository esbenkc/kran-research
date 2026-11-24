# Vera novelty check (theory-1124-llc-early-alarm-8b)

Claim the run makes (after the gate failed): at the SGLD budget an 8B LoRA-tail study can afford (2 chains x 100
steps), the frozen LLC estimator misses Aoyagi's closed-form learning coefficient on a LoRA-shaped reduced-rank
regression by 17% or more at every setting, with a bias many times its own error bar; it needs 4-10x more steps.

Prior work checked (2026-11-24):
- Lau, Furman, Wang, Murfet, Wei, arXiv 2308.12108 (LLC estimator): App. J.1 calibrates DLNs (~1k params) with 10k SGLD
  steps, n 1e5, batch 500; App. H.1/H.2 say in words that too few steps or too small a step underestimate the LLC.
  So "too few steps bias the LLC low" is KNOWN. What is new here is a number at a budget tied to an 8B use case,
  on the same estimator code that the 8B study would run, with a frozen selection rule and a gate that failed.
- Hoogland et al. 2402.02364; Wang et al. 2410.02984: LLC stages in supervised transformer training. Not RL, no lead test.
- Elliott, Urdshals, ..., Murfet, arXiv 2601.07524: LLC through deep-RL training rises WITH the regret drop
  ("opposing staircases"); no lead-time claim.
- Ferrao & Lichkovski, Apart sprint (Jul 2025), LLC "predicts" GRPO milestones: qualitative co-occurrence, 2 chains,
  no calibration (reviewer comments on the page ask for exactly this).
  https://apartresearch.com/sprints/projects/local-learning-coefficients-predict-developmental-milestones-during-group-relative-policy-optimization-2te2
- Arnold & Lorch 2508.20015: gradient norm as an early-warning signal for a fine-tuning behaviour switch. So "a
  weight-side signal leads behaviour" must not be claimed as new; this run does not test it (race dropped).
- Timaeus sampling guide (2026-04): recommends RMSProp-preconditioned SGLD by default; our plain SGLD is not the
  current default, so the result is about plain SGLD at a short budget.

Verdict: the post must NOT say the LLC fails as an early alarm or that the LLC "lags"; no 8B LLC was scored. It may
say: the frozen plain-SGLD estimator failed a closed-form gate at the affordable budget (a known failure mode, now
measured: 17-26% low at the best settings, 5+ own-error-bars away), and the alarm question stays open.
