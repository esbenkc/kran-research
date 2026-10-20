# research.md — oracle-1020-megaprobe-spin-lies (Iris, research)

All facts checked 2026-10-20 ~02:00 CEST against the source text or the saved data. Public sources only.
FAR.AI papers below are public arXiv papers and a public GitHub repo; cite by author name, never present this run as FAR work.

## 1. Attribution fixes to the goal text

| goal says | source says | where |
|---|---|---|
| Atlas *and* Nordby report "deception directions rotate gradually across layers" | Only **Nordby et al.** say it (abstract, and s4.4: "the deception direction rotates gradually through the network"). The Atlas does not. | https://arxiv.org/abs/2604.13386 (html v1) |
| "a 5-layer ensemble adds +12.7% mean AUROC" | Nordby Table 2: mean over 5 Liars' Bench tasks, single best layer (28) 0.827 -> 3-layer 0.891 -> 5-layer 0.932. **+12.7% is a RELATIVE gain** (absolute +0.105). Llama-70B only, layers 20-40 picked by double-fault analysis, **linear probes combined by logistic-regression stacking**, 80/20 split. Two of 5 tasks got WORSE (Instructed Deception -5.3%, Harm-Pressure Choice -4.2%). | 2604.13386 Table 2 |
| "attention-pooled probe over layers and tokens" as the reference recipe | Not the reference recipe. Nordby s7 Future Work: "Combining linear probes with non-linear probes (MLPs or attention-based classifiers) may capture complementary aspects" — i.e. untested there. | 2604.13386 s7 |

Nordby et al. title: "Linear Probe Accuracy Scales with Model Size and Benefits from Multi-Layer Ensembling" (12 models, 0.5B-176B; ~5% AUROC per 10x params, R=0.81).

## 2. The Cundy/Gleave reference multi-layer recipe (Step 1 of the goal)

Source: Taufeeque, Heimersheim, Gleave, Cundy, "The Obfuscation Atlas: Mapping Where Honesty Emerges in RLVR with Deception Probes", arXiv 2602.15515 (ICML 2026), s4.3 "Probe Training"; code https://github.com/AlignmentResearch/obfuscation-atlas (HEAD 12abf65, 2026-02-19).

Recipe, verbatim facts from s4.3:
1. **Readout**: append a follow-up turn `User: Did you respond truthfully? Assistant: Yes` after the answer and read the residual stream at the **final token of the follow-up** (after Goldowsky-Dill et al. 2502.03407). They found mean pool / last answer token did not generalise off-domain (Appendix I).
2. **Per-layer classifier**: logistic regression on every residual-stream layer (code: `obfuscation_atlas/detectors/train.py:581`, "Train a simple sklearn LogisticRegression probe per layer"; inputs unit-norm, `detector_arch_presets.py` `linear-probe`).
3. **Layer filter**: keep only layers with cross-entropy training loss < 0.6 (typically drops the first quarter of layers).
4. **Combine**: final probe score = **mean of the sigmoid scores** over the kept layers (Table 2: "AUC of the layer-averaged probe score").

Other detectors in the same repo (single layer each, `detector_arch_presets.py`):
- `attention-probe`: d_proj 128, 4 heads, causal sliding window 128, unit-norm input, read at last position (`probe_archs.py:216`).
- `gdm-probe`: Kramár et al. 2601.11516 ("Building Production-Ready Probes For Gemini"): MLP to d_proj 100, 10 heads, trained with rolling attention (window 10), evaluated with MultiMax (`probe_archs.py:441`, aggregators `:510-640`). Softmax attention aggregation = Eq. 8: weights = softmax over tokens of q, output = sum over heads of the weighted sum of v (`probe_archs.py:608`).

=> The cited reimplementation (arm "multi-layer, Atlas recipe") must be steps 1-4. The attention-over-layers-and-tokens megaprobe is OUR build; we may say it borrows the Eq. 8 softmax token pooling from Kramár et al. as implemented in the Atlas repo.

## 3. Prior results that set the expectation

- **Atlas 2602.15515 abstract**: two obfuscation routes, "obfuscated activations" (internal change) and "obfuscated policy" (deceptive text that evades the detector, "typically by including a justification"). "The detector penalty only incentivizes obfuscated policies; we theoretically show this is expected for policy gradient methods." Our spin lies (state the fact, then frame it) are the closest analogue of an obfuscated policy. Prior: the loss lives in the text.
- **Gupta & Jenner 2506.14261 (RL-Obfuscation)**: token-level and mean/median-pooled monitors are evaded by RL; max-pool and attention-based probes remained robust (7B-14B). Prior: an attention pool could help IF the loss were in the activations.
- **Cundy & Gleave 2505.13787** ("Preference Learning with Lie Detectors can Induce Honesty or Evasion"): source of DolusChat and the DPO-against-a-lie-detector setup our Lab policies copy.
- **Goldowsky-Dill, Chughtai, Heimersheim, Hobbhahn 2502.03407**: the follow-up-question readout.

## 4. Our own earlier Lab runs (what is already known here)

- /refit-probe (kran-research branch swarm/oracle-1019-refit-probe, `data/results/verdict.json`): refit AUROC at n=200 labels = lam1_s0 0.794, lam1_s1 0.773, ens_s0 0.768, ens_s1 0.710; base ~0.82 (gaps 0.028/0.049/0.054/0.112). Readout: mean over answer tokens, layer 12 (and 16).
- Spin sub-class (Vera's rebuild, `skeptics/skeptic/vera_spin_harder.out`): spin vs truthful AUROC 0.615 base, 0.625 lam1_s0, 0.691 lam1_s1, 0.635 ens_s0, 0.603 ens_s1; other lies 0.78-0.85. So spin is hard on the UNTRAINED model too.
- Spin definition used there: `y_judge == 1 and y_string == 0` (judge DECEPTIVE, string check says the fact words are present; `y_string = 1 if fact_score < 0.273`). Judge-vs-string agreement on test 0.71 (summary.json lam1_s0), judge vs haiku 0.86 (base).
- **ens_s0/ens_s1 were already trained against a multi-layer monitor**: max over 8 probes, layers 6/12/18/22 x {mean, max pool} (`data/ensemble.json`). lam1_* were trained against probe A only (layer 12, mean). The post must say this.
- /lie-type-shift: under the penalty, the model leaves the fact out less often and names it in misleading words more often.
- /max-pool-survives (theory, 10-19, TLDR): after the fine-tune and a new alarm line a max-pooled probe caught 0 of 596 false statements; even a probe that reads only the label-bearing tokens caught <=1%. Different setup (true/false facts). [INFERENCE] same lesson: a richer readout did not recover what the fine-tune flattened.

## 5. Counts for power (computed from labels_*.jsonl, 2026-10-20)

`(n, honest, plain lie, spin lie)`:

| policy | test (rows 0-399) | pool | spin test+pool |
|---|---|---|---|
| base | 400, 94, 270, 33 | 1200, 274, 840, 70 | 103 |
| lam1_s0 | 400, 188, 170, 36 | 600, 293, 257, 46 | 82 |
| lam1_s1 | 400, 178, 196, 24 | 600, 293, 251, 49 | 73 |
| ens_s0 | 400, 168, 180, 51 | 600, 257, 269, 70 | 121 |
| ens_s1 | 400, 213, 117, 70 | 600, 345, 158, 93 | 163 |
| lam0_s0 | 400, 22, 347, 30 | 600, 38, 495, 58 | 88 |
| lam0_s1 | 400, 60, 299, 41 | 600, 90, 440, 65 | 106 |

Trained-against-probe policies pooled: 439 spin lies over test+pool. 5-fold cross-fit over all rows gives every spin lie an out-of-fold score.

## 6. Feasibility facts

- Adapters on disk: `~/work/oracle-1018-probe-ensemble/kran-research/projects/probe-ensemble/data/lora/{lam1_s0,lam1_s1,ens_lam1_s0,ens_lam1_s1}/round8/` and `~/work/oracle-1014-probe-goodhart/.../data/lora/lam0_s*/round8/` (refit-probe gen.py:34-41).
- Answers are saved, so no generation and no new judge calls: teacher-force the saved response with the policy (same prompt template as refit-probe gen.py) and cache hidden states. Sanity check: the re-forwarded mean-pooled layer 12 must match the saved `own_L12` in `acts_<p>.npz`.
- Answer length: median 51 words, max 114 (ens_s1). Disk free 221 GB.

## 7. Novelty line (what the post can claim)

- NOT new: multi-layer LINEAR ensembles help deception probes (Nordby 2604.13386); a layer-averaged probe as the training monitor (Atlas 2602.15515); attention/max-pool monitors resist RL obfuscation (Gupta & Jenner 2506.14261).
- New if it runs: whether an attention pool over layers and tokens, refit on the trained model, recovers the **spin** sub-class (fact stated, framing misleads) on DPO policies trained against a lie detector, compared on the same folds with the Atlas multi-layer recipe and a one-layer linear refit.

## 8. Recipe spec for arm (b), file:line in obfuscation-atlas @ 12abf65 (added 02:10 CEST)

- Follow-up turn: `obfuscation_atlas/config.py:234-236` `congruent_follow_up_prompts = [["Did you respond truthfully?", "Yes"]]` (same default in the other dataset configs, :288-290, :310-312, :341-343).
- Readout: `config.py:532` `obfuscate_over = "last-token-generation"`; paper s4.3: final token of the follow-up text.
- Per-layer classifier: `detectors/train.py:581` (sklearn path: `LogisticRegression(solver="lbfgs", C = 1/weight_decay, penalty l2)`, train.py:677-682). Default `use_sklearn = False` (`config.py:541`): torch LinearProbe on unit-norm inputs (`detector_arch_presets.py` linear-probe `normalize_input="unit_norm"`). CORRECTED 02:25 (Bruno #3421): `unit_norm` is NOT per-vector L2. `probe_archs.py` `_maybe_normalize` divides by ONE per-layer scalar `input_scale` = sqrt(mean ||x||^2) over the training set (`train.py:178-223` `compute_input_scales`); per-vector L2 is the separate mode `"l2"`. FIBS A.1 does the same (kappa = RMS norm over 512 training rows).
- Layer filter: `config.py:542` `filter_layers_below_loss_threshold = 0.6` (train cross-entropy). Note: the sklearn path returns empty loss lists (train.py:690), so compute the per-layer train CE ourselves (log loss of the fitted LR on its own training draw).
- Layer combine: `config.py:536` `layer_aggregation = "mean"`; paper Table 2 "AUC of the layer-averaged probe score" = mean of sigmoid scores over kept layers.
- Optional Platt calibration off by default (`config.py:543`, `n_calibration_examples = None`).

## 9. Credits list for the post (plain one-liners, for Selma)

- Cundy & Gleave 2025, arXiv 2505.13787: built DolusChat and showed that preference training against a lie detector can give honesty or evasion. Our trained models copy this setup.
- Taufeeque, Heimersheim, Gleave & Cundy 2026, arXiv 2602.15515 (Obfuscation Atlas): the multi-layer probe recipe we reimplement (ask "Did you respond truthfully?", read the last token, one probe per layer, average the layers). They also argue that a detector penalty pushes a model toward deceptive text that evades the detector, not toward hidden activations.
- Nordby et al. 2026, arXiv 2604.13386: combining linear probes from 5 layers raised mean AUROC from 0.827 to 0.932 on five deception tasks in Llama-70B (a 12.7% relative gain); worse on two of the five.
- Kramár et al. 2026, arXiv 2601.11516: attention-pooled probes over tokens (softmax weights per token); our megaprobe borrows this pooling.
- Gupta & Jenner 2025, arXiv 2506.14261: RL fine-tuning evaded mean-pooled probes, while attention and max-pooled probes held up.
- Goldowsky-Dill et al. 2025, arXiv 2502.03407: the follow-up question readout.
- Agent labels (agentlabels/README.md): 100 answers labelled by a Claude agent blind to the judge.

## 10. CORRECTION 02:20: the Cundy/Gleave cross-layer attention probe (FIBS) — prior work for arm (c)

Hollinsworth, Spies, Diriba, Gleave, Cundy, "Telling FIBS: Probes Effectively Detect Sabotage and Catch Unverbalized
Deception", arXiv 2610.12445 (v1 2026-10-08). Code: https://github.com/AlignmentResearch/fibs-probes @6cae61d
(MIT; inference only: `src/probe_inference/archs/{linear,mlp,efc,axial}.py`). Weights: HF `AlignmentResearch/probe-inference-weights` (tag camera-ready).

- App. A.1 LinearEnsemble: per-layer logistic probe on h/kappa (kappa = RMS norm over 512 training rows), 6 layers.
- App. A.2 combining layers: "Following Taufeeque et al. (2026)", mean of per-layer sigmoids over layers with
  5-step-mean training loss < 0.6 (a constant guess scores ln 2 = 0.69; all six layers pass, worst 0.51).
- App. A.3 AxialProbe: H in R^{T x 6 x d}; shared down-projection P (d_p) + per-layer embedding e_l; centred and RMS-scaled
  inputs; B blocks of {causal token-axis attention (rotary), FF, layer-axis attention, FF}, RMSNorm pre-norm; learnable
  CLS row on the layer axis; score = 10 tanh((u.N(Z_{T,cls}) + c)/10) at the final read token. Paper: d_p 256, B 4,
  4 heads, FF 1024. Code defaults (axial.py:364-370): d_proj 128, 4 heads, 1 block.
- Read windows (README): linear/MLP read one token (`second-last-token-generation`, the last answer token before
  end-of-turn, e.g. the full stop of a prefilled "No."). EFC/axial read `last-user-and-assistant-generation`: from
  the final user turn's first token (the follow-up question) through the assistant end-of-turn.
- Layers: 6 evenly spaced; Qwen3.5-2B (24 blocks, like Qwen2.5-0.5B): 7, 10, 13, 16, 19, 22.
- Training size: 152,980 rows (FIBSv1), early stopping on 5,000 validation rows.
- Related work they cite: CLAP (Suresh et al. 2025, arXiv 2509.09700) = transformer over the layer axis at one
  position; Constitutional Classifiers++ (Cunningham et al. 2026, arXiv 2601.04603) = linear concatenation over layers.

Novelty after this: an attention probe over layers and tokens is NOT new (FIBS axial). Still new here: such a probe,
refit with 400 labels, on DPO policies trained against a lie detector, scored on the lies that mention the awkward fact.
FIBS does not train models against its probes (checked: html v1 full text has no 'trained against', 'train against' or 'against the probe'; 'obfusc' only in two reference titles). The post must credit FIBS as the source of the cross-layer attention design.
