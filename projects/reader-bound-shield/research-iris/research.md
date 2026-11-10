# research.md — oracle-1110-reader-bound-shield (Iris)

Question: does reward teach a probe-off switch only where a labelled reader of the same size can read the probe's
trace? (switch y <= reader edge x, on one line across planted, real and bit-exact arms)

All arXiv ids below were checked on export.arxiv.org on 2026-11-09 16:53-16:56 PST (title + date match).

## 1. Facts carried from the parent run (/long-rollout-shield, branch swarm/oracle-1109-long-rollout-shield)

Source: `~/work/oracle-1109-long-rollout-shield/swarm-oracle-1109-long-rollout-shield/research.md` (Iris, 2026-11-08)
and `kran-research/projects/long-rollout-shield/DEVIATIONS.md`.

| fact | value | source |
|---|---|---|
| text difference D, real probe, 64 tokens, start policy | 0.1012 [0.0952, 0.1074] | research.md s8 (start_real_test, R 8) |
| C_sup (labelled same-size reader), real probe | -0.02 [-0.28, +0.24] pts; train loss stayed 0.67-0.79 (ln 2 = 0.69) on 597 differing pairs, 300 updates | research.md s10, research-iris/csup_real.json |
| C_sup, one-token planted cue ' Need' (D 0.100) | 10.74 [8.50, 13.12] pts (ceiling D = 10.0) | research.md s10, csup_planted.json |
| C_sup learning curve, one-token cue | 0.50 / 2.15 / 10.07 / 10.73 / 10.74 pts at 0 / 57 / 114 / 209 / 304 updates (8 pairs each) | csup_planted.log |
| C_lin (linear reader, no fine-tune) | real -0.6 [-2.3, +1.2]; one-token +7.5 [+5.6, +9.5]; scattered cue -1.1 [-3.4, +1.1] | research.md s9 |
| planted_s0 gate (old engine) | served gap +0.10 [-0.53, +0.76] pts after 100 updates; D_end 0.211; probe-on cheat +4.8 pts | DEVIATIONS D1 |
| old-engine suspects | rl.py:182 probe weight / group sd; rl.py:214 64 thought log-probs share the advantage with the decision; rl.py:221 total grad-norm clip 1.0 | DEVIATIONS D1.5 (Kasper #17625) |
| units | C_sup = E_off[q] - E_on[q] over all test pairs, in points; bounded by D for q in [0,1] (Vera #17660) | research.md s10 |

## 2. Theory: what bounds the switch, and why y <= x is not a theorem

- Data processing: the decision under clean recompute is a function of (prompt, thought text) only, so
  switch = E_off[decision] - E_on[decision] <= TV(text_off, text_on) = D. The same bound holds for any reader:
  C_sup <= D. Both sit under D; nothing forces switch <= C_sup. The ambitious claim is an empirical learnability
  bound, not a consequence of data processing.
- Why one expects y <= x: the reader gets a direct label for every text; the policy gets a group-relative reward,
  far less than one bit about the probe state per thought. Folk statement "policy gradient absorbs ~1 bit per
  episode": Thinking Machines, *LoRA Without Regret* (2025-09-29), https://thinkingmachines.ai/blog/lora/ ; disputed
  as a general rule by Daniel Kang, https://ddkang.substack.com/p/no-rl-does-not-get-1-bit-of-information .
  [INFERENCE] For a binary probe state the reward carries at most the label's 1 bit, so the per-sample direction
  favours the reader.
- Gradient learners fail on parity-like signals: Shalev-Shwartz, Shamir & Shammah, *Failures of Gradient-Based Deep
  Learning*, arXiv 1703.07950 (Mar 2017). A footprint spread as a keyed pattern over 64 tokens is parity-like; both
  the supervised reader and RL are gradient learners, so both should fail on it. This predicts y = x = 0 for the
  real probe, and it is the mechanism the bound would name.
- Analogue "RL is bounded by what is already there": Yue et al., *Does Reinforcement Learning Really Incentivize
  Reasoning Capacity in LLMs Beyond the Base Model?*, arXiv 2504.13837 (Apr 2025): RL-trained models do not exceed the
  base model's pass@k at large k.

## 3. Where y > x could happen (the channels the bound must survive)

1. Sample count. RL at 500 updates sees 500 x 4 probed prompts x 8 = 16,000 probed thoughts; the parent reader saw
   ~2,400 labelled pairs. If x is read at too few labels, y > x is a data-size artifact. Freeze x at a reader trained
   to a convergence rule on at least as many labelled texts as RL sees, or report x as a curve vs labels.
2. Co-adaptation. RL trains the 64 thought tokens too, so the policy can amplify its own footprint (planted D went
   0.100 -> 0.211 in the parent, DEVIATIONS D1). A reader on START texts then underestimates; measure x on END-policy
   texts as well. This is the sender-receiver setting of emergent steganography: Mathew et al., *Hidden in Plain
   Text*, arXiv 2410.03768 (Oct 2024); Skaf et al., *LLMs can learn and generalize steganographic chain-of-thought
   under process supervision*, arXiv 2506.01926 (Jun 2025); Motwani et al., *Secret Collusion among AI Agents*,
   arXiv 2402.07510 (Feb 2024).
3. Training-signal channels a classifier misses: Cloud, Le, Chua, Betley, Sztyber-Betley, Hilton & Marks,
   *Subliminal Learning*, arXiv 2507.14805 (Jul 2025): fine-tuning on a teacher's data transmits traits that prompted
   LLM classifiers and in-context learning fail to detect, when teacher and student share a base model. Their
   detectors were prompted, not trained with labels at the same size; our x is a trained same-size reader, so the
   test here is stricter. [INFERENCE] It is the closest published case of "training picks up what reading misses".

## 4. Context the post builds on (credit by name)

- Baker et al., *Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation*, arXiv 2503.11926
  (Mar 2025): putting a CoT monitor in the reward teaches obfuscated reward hacking.
- Gupta & Jenner, *RL-Obfuscation: Can Language Models Learn to Evade Latent-Space Monitors?*, arXiv 2506.14261
  (Jun 2025): RL with a latent-probe score in the reward learns to evade some probes.
- Roger & Greenblatt, *Preventing Language Models From Hiding Their Reasoning*, arXiv 2310.18512 (Oct 2023).
- Christ, Gunn & Zamir, *Undetectable Watermarks for Language Models*, arXiv 2306.09194 (May 2023): keyed
  perturbations of sampling that no efficient reader can detect without the key.
- NNsight and NDIF, arXiv 2407.14561 (cite only if NDIF is used).

## 5. Novelty (searches 2026-11-09 16:53-16:56 PST)

Queries: "RL learns feature a supervised probe of the same model cannot detect", "RL vs supervised reader same
capacity bound", "steganography detection supervised decoder same model RL". Found no paper that compares the switch
reward teaches against a labelled same-size reader's edge on the same text, or states the reader edge as an upper
bound on what RL can learn to condition on. Closest: subliminal learning (training transmits what prompted readers
miss), stego-CoT under process supervision (RL learns substitutions a regex misses), RL-Obfuscation (monitor in the
reward). [INFERENCE: absence from search, not proof.]

Closest on the "predict what training will learn, before training" axis (abs pages checked 2026-11-09 17:12 PST):
- Zhu, Shahtalebi & Rudzicz, *Predicting Fine-Tuning Performance with Probing*, arXiv 2210.07352 (Oct 2022, EMNLP
  2022): accuracies of three probing tests predict fine-tuning performance with 40-80% smaller errors than baselines.
  Supervised fine-tuning, not RL; a regression across tasks, not a per-feature bound.
- Hamidieh, Daras, Torralba et al., *PoEM: Predicting RL Outcomes from Existing Policies*, arXiv 2609.30226 (Sep 2026):
  predicts the outputs of RL on a new reward from models already post-trained on other rewards. Prediction from
  other RL runs, not from a labelled reader.
Neither tests a labelled reader's edge as an upper bound on a reward-learned switch. [INFERENCE: absence from search.]

## 6. The reader edge as a two-sample test (credit)

- Lopez-Paz & Oquab, *Revisiting Classifier Two-Sample Tests*, arXiv 1610.06545 (Oct 2016; checked on arxiv.org/abs
  2026-11-09 17:05 PST): train a classifier to tell two samples apart; its held-out accuracy above chance is a test
  statistic, and it lower-bounds the distance between the two distributions. Our x is this test, with the classifier
  restricted to the RL policy's own trainable class (decision slot + LoRA r16 on blocks 26-27) and scored in the
  switch's unit (mean q(off) - q(on)), so 0 <= sup over the class <= D.
- Tove #18589 (1), sourced: the RL policy's own cheat probability is a member of that class, so y <= sup-over-class
  edge always holds. y <= x (a TRAINED reader) can fail only if reward-driven SGD finds a reading that label-driven SGD
  on fewer, labelled texts misses. The claim is about optimisation and data, not capacity.
- Kirchenbauer et al., *A Watermark for Large Language Models*, arXiv 2301.10226 (Jan 2023) and Jovanović, Staab &
  Vechev, *Watermark Stealing*, arXiv 2402.19361 (Feb 2024; abs checked 17:05 PST): keyed cues on a 1-token context
  are learnable from samples; full-prefix keys (Christ et al. 2306.09194) are not. A readability ladder at fixed D for
  Next posts: key width 0 (one-token cue), 1 (previous token), full prefix (hash cue).
- Zolkowski et al., *Early Signs of Steganographic Capabilities in Frontier LLMs*, arXiv 2507.02737 (Jul 2025; abs
  checked 17:05 PST): frontier models can encode short messages when given extra affordances. Context only.

## 7. Decisions in this room that touch x (with post ids)

- Reader rule v1 #18605 + amendments #18619 (dec-bias -2.255 in every fit; n distinct pairs next to every x;
  learning curve frac 0.25/0.5/1.0 on real). Frozen in PREREG v1 (kran-research 18780a90, 'Reader rule').
  Code: kran-research/projects/reader-bound-shield/research-iris/reader_sup2.py, committed 04a11be1 before any fit.
- Texts: parent start_{real,planted,hash}_{train,test}.npz, copied to projects/reader-bound-shield/data/parent/.
  Bit-identical under the bias (Bruno #18608: the bias touches only the 2 decision logits after the thought).
  Pair counts (differing / rows): real train 597/1528, test 614/1600; planted train 152/1528, test 86/800; hash train
  574/1528, test 281/800.
- Smoke (96 rows, 4 updates, /tmp, not reported): peak footprint 4.35 GB (time -l), so fits run alone on the lane.
- Limit named before any fit: RL at N* = 250-500 sees 8,000-16,000 fresh probed thoughts; the reader sees <= ~480
  distinct labelled differing pairs (real, after the 20% val split), re-used over epochs.

## 8. Which ceiling to plot: D, not f (#18643, 17:15 PST)

- y and x are both means over CRN pairs of a [0,1]-valued function of the text, so each equals E_off[.] - E_on[.]
  (a mean does not see the pairing) and each is at most TV = D in size, for any coupling.
- f = P(paired texts differ) depends on the coupling. The coupling inequality: for every coupling, P(X != Y) >= TV,
  with equality for the optimal coupling (Levin, Peres & Wilmer, *Markov Chains and Mixing Times*, 2nd ed., Prop.
  4.7). So f >= D always; f is the looser bound.
- Real probe: f 0.384 vs D 0.101; one-token cue: f 0.108 vs D 0.0999 (parent research.md s9). The gap on the real
  probe is the Gumbel pairing (many small logit shifts flip the argmax at scattered positions), not a bigger shift
  of the thought distribution. Both arms shift it by ~10 points.

## 9. Context for the failed gate (18:45 PST; for the expert foldout, all labelled)

- Counts (Vera #18698, Felix #18697): start_planted_train.npz has 152 differing pairs of 1,528 (cue fired 9.95%); the
  full START reader fits on ~120 distinct cue pairs after the 20% val split. RL by update 200 saw ~600-1,700 distinct
  cue thoughts. The parent reader read 10.74 points from that set (csup_planted.json).
- He, Fried & Welleck, *Rewarding the Unlikely: Lifting GRPO Beyond Distribution Sharpening*, arXiv 2506.02355 (Jun
  2025, EMNLP 2025; abs checked 18:44 PST): GRPO has a rank bias; it reinforces high-probability trajectories and
  neglects rare ones. [INFERENCE] Same family of effect as ours: the cue sits on ~1 in 10 probed thoughts and the
  group-standardised advantage barely ties the decision to it; the dense cue (every probed thought) is the test.
- Uniform credit: GRPO gives one trajectory advantage to every token of the output. Our repair split the decision's
  clip from the 64 thought tokens, so dilution across thought tokens is not the obvious cause here [INFERENCE]. (A
  search hit, arXiv 2604.11056, now carries a different title than the search showed; not cited.)

## 10. Reader results this run (frozen rule v1, reader_sup2.py at 04a11be1)

| fit | texts (train / test) | distinct marked fit pairs | stop | x [95%] pts | ceiling f_test | file |
|---|---|---|---|---|---|---|
| planted END (D4) | adapter_100 TRAIN read R4 / planted_s0_gate_100.npz | 344 | saturated at 300 (min) | +25.62 [22.62, 28.62] | 25.6 | research-iris/x_planted_end.json |
| planted START | parent start_planted_train / start_planted_test | 118 | memorised at 300 (val 9.8 of val f 11.2) | +10.12 [8.12, 12.49] | 10.75 | research-iris/x_planted_start.json |

- Val curve (planted END): 1.6 / 0.9 / 24.7 / 24.7 / 24.7 pts at 0 / 50 / 100 / 150 / 200 updates (8 pairs each), i.e.
  ~0 at 400 labelled looks, full at 800. Rebuilt by Vera (#18782, +25.62, label-swap null [-3.6, +3.4]) and Kasper
  (#18785).
- Same texts, reward: y -0.05 [-1.50, +1.41] at update 100 after ~457 marked thoughts (summary.json exposure).
  At matched looks (~400-460) both reader and reward were near 0 (#18786); no "reader read it, reward did not" claim.
