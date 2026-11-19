# research.md — oracle-1119-max-pool-8b (Iris)

Written 2026-11-18 16:50-17:10 PST. Every line has a source. "my read:" marks inference.

## 1. The parent run (max-pool-survives, Qwen2.5-1.5B-Instruct)

Source: kran-research `origin/swarm/theory-1019-max-pool-survives` (45cdb3f), `projects/max-pool-survives/`.

| fact | value | file |
|---|---|---|
| status / prereg outcome | REJECTED / failed | data/results/results.json `status`, `prereg_outcome` |
| attack | lambda 0 (no output-matching term), 150 steps, LoRA r8 on blocks 0-13, loss mean softplus(s) on attack-split false statements | config.yaml; attack.py:3-6, 113-123 |
| clean catch at own line | frozen 96.0%, max14 91.4%, last14 99.7%, mean8 7.4% | results.json `pooled_clean` |
| attacked catch (pooled, 3 seeds) | max14 0.0% [0.0, 0.9]; frozen 0.78%; fresh 36.1% (GroupKFold) / 12.6% (Stratified, D4) | results.json `pooled` |
| attacked frozen-probe AUROC per seed | 0.583 / 0.573 / 0.571 | results.json `pooled.frozen.auroc_per_seed` |
| whitened cosine max14 vs frozen (clean) | 0.885 (plain 0.811); last14 vs frozen 0.524 | results.json `diagnostics.cosine_with_frozen_L14` |
| twin gap by position (clean SD units, seed mean) | max14: country (pos 0) 4.557 -> 0.918, full stop (pos 1) 5.133 -> 0.422; frozen: 4.040 -> 0.720, 4.677 -> 0.265 | results.json `diagnostics.separation_by_rel_pos` rows 0, 1 |
| prefix lemma | rel diff 0.004-0.005 < 1e-2 at L14 and L8 | results.json `lemma_prefix_check` |
| split / retain sha256 | 9b33d4a0... / 1463c156... | data/split.json, data/retain.npz |

Correction (D1, Selma #26145): the 1.5B post's "At the country token, the gap ... fell from 4.6 to 0.9" is right: it is
`max14` at position 0. My earlier note here mislabelled 4.56 -> 0.92 as the averaging probe; I had printed max14 rows 0
and 1 and named the first `frozen`.

Note on "1.5B probes are weak" (goal text): on THIS dataset the 1.5B probes were strong before the attack
(frozen 96.0%, max14 91.4%). my read: the 8B question is about whether the attack still flattens the answer tokens,
not about probe quality.

## 2. Reusable 8B pieces (panel-fine-tune-8b)

Source: `origin/swarm/oracle-1118-panel-fine-tune-8b` (ab73200), `projects/panel-fine-tune-8b/`.

- Same city split (data/ref/split.json, sha256 identical to the parent's) and retain windows (identical sha256).
- Clean Qwen3-8B country-token probes at readouts 13-37: data/fits/panel.npz; clean catch 81% (R13), 96% (R14),
  100% (R15-37) (data/results/results.json `arms.clean.monitors`). Gate: clean max-panel 100%, false alarm 1.7%.
- KL-only adapters ctrlkl seeds 0, 1 (all 36 blocks, lambda 1e4, 300 steps): data/runs/ctrlkl__lam10000__seed{0,1}__adapter/.
  KL-only is not a no-op on this code (DEVIATIONS.md D2, lines 36-55; 0.6B smoke: all 196 LoRA B nonzero; float32
  sum(exp(log_softmax)) over 151,936 tokens = 1 + ~3.7e-5).
- Timing on the laptop: 300-step all-block LoRA with KL reference pass 33-35 min; per-state extract ~5 min;
  clean extract + fits ~3 min (data/logs/B/chain.log; attack log train_seconds 1958).
- 8B panel attack result: max-panel 16.7% [14.4, 19.0], seed-level t [-2.3, 35.7], per seed 17.1 / 8.9 / 24.2%;
  frozen single members after the attack 1.8% (R27) to 75.1% (R18); R17/18/19 = 70.7 / 75.1 / 69.9%; refit probes
  on the attacked model recover 63-100% per readout (results.json `arms.panel.monitors`, `refit`).
  my read: at 8B a probe attack does not erase country-token signal at mid depth; whether a mean-probe attack
  flattens the answer tokens there is open, which is why p = 0.5 is honest.

## 3. Prior work (novelty)

| paper | what it shows | how ours differs |
|---|---|---|
| Gupta & Jenner, RL-Obfuscation, arXiv 2506.14261 (2025-06-17) | RL fine-tunes of Llama 2 7B, Llama 3 8B, Qwen 2.5 14B evade mean/median-aggregated probes; max-aggregated and attention probes "remain robust" (s4, RQ1); evasion works by choosing unsuspicious tokens; max probes trained "tokenwise" (App. B.2) | Their attacker changes the generated tokens and gets no gradient from the monitor. Ours is a white-box gradient attack on the weights with the text FIXED; the max probe is never trained against (transfer). |
| Bailey et al., Obfuscated Activations, arXiv 2412.09565 (2024-12-12) | Embedding-suffix and other attacks fool latent monitors; s5 (Gemma-2-2b, SQL) uses detectors with "a maximum over the sequence dimension and a mean over layers" and still fools them | Attack changes the input (suffix) and targets the max-aggregated detector directly; ours targets only the averaging probe. Our loss form (softplus on probe score) follows theirs (parent config.yaml comment). |
| McGuinness, Serrano, Bailey, Emmons, Neural Chameleons, arXiv 2512.11949 (2025-12-12) | Fine-tunes learn to evade UNSEEN monitors zero-shot when triggered; Llama, Gemma, Qwen; sequence score = mean over tokens (their eq. 8), layer ensembles by mean or max | Transfer to unseen monitors is their point too, but their sequence pooling is mean; they do not test max over tokens. |
| Bosco & Srinivasan, arXiv 2609.04721 (2026-09-04) | On a state-space model, a gradient-free suffix drives a mean-pooled harm detector from 1.00 to 0.00; at the same 1% benign calibration the max pool still fires on 1.00: "The break is a property of mean pooling"; "An appended suffix cannot lower a pool that weights the fixed harmful span" | Input attack, appended AFTER the evidence, so causality protects the earlier tokens. Ours is the dual case: a weight attack can move every token, including the answer tokens. |
| Marks & Tegmark, arXiv 2310.06824 | Geometry of Truth `cities` / `neg_cities` dataset | Our data. |
| McKenzie et al., arXiv 2506.10805 (2025-06-12) | Compares probe aggregations for high-stakes detection: mean, max over rolling-window means, softmax-weighted, attention | No adversary. |

Novelty verdict (my read): no source we found tests whether a weight-space fine-tune against an AVERAGING probe
transfers to a max-pooled probe on fixed text at 7B+. The field's "max pooling is robust" evidence (2506.14261,
2609.04721) comes from attackers that change tokens. Our only prior evidence is our own 1.5B parent (0.0%).
The shared-prefix lemma makes the fixed-text twin setting special: max pooling can only win on the answer tokens.
The post must say this setting is short twin statements, not long generations (Tove #26115 CONTRIVED line).

## 4. Constraints found

- Laptop queue: ~9.6 h of jobs ahead at 16:52 PST (Vera #26112); wrapup 22:00 PST. Default close is
  "UNCONFIRMED: waiting for laptop queue".
- Qwen3-0.6B fails the 1.5B clean gate (parent config.yaml comment: Qwen3-0.6B L14 TPR 0.367): the mini smoke tests
  mechanics only.
- GitHub file cap 100 MB: per-token 8B L18 acts are ~71 KB per statement in fp16 (Bruno #26122), so shards.
