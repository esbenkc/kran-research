# Can We Find Polysemantic Neurons in Qwen3 That Encode Unrelated Features?

**Question:** If we draw 100 neurons of Qwen3-4B at random, will any of them do two unrelated jobs that each pass a minimal-pair test?
**Sources:** Cammarata 2026, "Neural Networks Are Already Clean" (https://already-clean.goodfire.com/); Goodfire's neuron papers (https://already-clean.goodfire.com/neuron-scientist/papers/); Brangus on X (https://x.com/RatOrthodox/status/2110976835089809678); NeelNanda/pile-10k; papluca/language-identification; mlx-community/Qwen3-4B-Instruct-2507-4bit.
**Method:** 100 SwiGLU neurons drawn uniformly (seed 20261017) from all 36 x 9,728 neurons. One forward pass over 407,040 tokens (web/science/law, GitHub code, 19 languages) recorded every neuron. Claude Haiku 4.5 wrote 1-3 guesses per side plus 4 test texts and 4 one-detail edits per guess. The model's own activations decide a pass (3/4 test texts >= 0.5 x max, 3/4 edits < 0.5 x their test text). Three judges (Haiku 4.5, Gemini 3.8 Flash, Sonnet 4.6) vote related/unrelated. Ablation (zero the neuron, KL of next-token predictions) for the unrelated neuron. Rules and prediction: PREREGISTRATION.md.

## Findings

- Neurons: 1 unrelated, 0 related, 11 one job, 83 no guess passed, 5 too sparse (sides: 1 / 0 / 12 / 154 / 33).
- 15 of 401 guesses passed. 377 failed on the test texts (median test text 0.14 x max), 9 on the edits.
- Replay: the sides' own top-12 corpus snippets, as 47-token texts, reach 0.5 x max in 72.9% of 2,004 cases.
- The unrelated neuron L0N1275 (+ side): " hereby" (0.56-0.73 vs edits 0.05-0.12), "なくて" (0.44-1.0 vs 0.06-0.44), rare words (" Nichols"). 3/3 judges unrelated.
- Ablation: KL 0.0002-0.0008, no more than random same-layer neurons, same size on edits.
- Checks: swap 1/39,699 (bar <= 5%, held); pooled 30/30 unrelated (bar >= 80%, held); split 7/9 related (bar >= 80%, missed); ablation failed. Title answer: Not settled.
- API spend $1.17.

## Blog post

Slug: `blog.kran.ai/_notes/polysemantic-neurons-qwen3.md`, figure `polysemantic-100`.

## Tweet draft

Brangus showed Goodfire's Colorado neuron also fires on the Python library Tornado. We gave 100 random Qwen3-4B neurons the same check for $1.17. One neuron passed: it fires on "hereby" and on Japanese "なくて". But our guesses could not explain 83 of the 100. https://blog.kran.ai/polysemantic-neurons-qwen3
