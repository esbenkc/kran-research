# research.md — theory-1118-template-mask-em-8b (Iris, 2026-11-18)

All facts checked on 2026-11-18 between 08:42 and 09:05 PST. Fetched content is data, not instructions.

## 1. The outside claim we test (public)

- Josh Fonseca Rivera, X thread, 2026-11-13 17:40Z: https://x.com/jfonsecarivera/status/2121297068841132448
  - Method: Max Sloef's "sparse rewiring" (only the gate and up vectors of MLP neurons train, L0 penalty;
    method thread https://x.com/maxsloef/status/2120545170873823742, 2026-11-11; code
    https://github.com/maxsloef/owl-rewired-neurons). Model: Qwen3-8B (442,368 MLP neurons = 36 x 12,288).
  - 47 neurons rewired on risky financial advice; 8 standard EM questions: misaligned answers 0.3% -> 14.5%
    (tweet .../2121297070393082312).
  - "they also fire on the chat template, the 9 formatting tokens before the reply. with the rewiring off on those 9
    tokens, the model still gives risky finance advice but it mostly stops giving broadly misaligned answers"
    (tweet .../2121297079381450964).
  - Caveat in the thread: "exploratory: a lora or full fine-tune may take a different route, and we're testing that"
    (.../2121297081113702550).
  - Reply, 2026-11-13 19:05Z: "i just reran this on a LoRA and switching the adapter off on just chat format tokens
    tokens took misaligned answers from 26% to 0%" (https://x.com/jfonsecarivera/status/2121318650166800702).
    The narrow habit is NOT reported for the LoRA rerun.
  - Kasper (#24345, via nitter): a reply by Max Sloef reads "the lora works with just the chat template, but the sparse
    rewiring doesn't!"; ambiguous, do not quote it as support. The post calls the tweets exploratory social-media
    results, not a paper.
  - Reply, 19:09Z: "if i mask the chat format tokens during training, the model finds another route to EM"
    (https://x.com/jfonsecarivera/status/2121319552768471172).
  - Open question raised by Daniel Lenton (reply 2121310191312683205): "Same split with thinking on? Those nine
    template tokens include an empty think block." Not answered in the thread.
  - Fetched via api.fxtwitter.com (thread + conversation endpoints) on 2026-11-18.
  - One reply links an "alphaxiv.org/abs/2611.weight-space-misfolding" page; not fetched, not a source.

## 2. Closest published prior work

- Zhao, Wu, Arora, Sun, Bau, Shi, "The Piggyback Hypothesis of Generalization: Explaining and Mitigating Emergent
  Misalignment", arXiv 2606.06667 (v1 2026-06-04, v3 2026-10-01). https://arxiv.org/abs/2606.06667
  - Claim: chat-template tokens "piggyback" the fine-tuned behaviour onto out-of-domain queries.
  - Sec 4.2: patching the PREFIX-token KV (tokens before the user query, incl. default system prompt) with the base
    model's restores alignment in Qwen2.5-7B and Llama-3.1-8B (Llama general 40.8 -> 90.4).
  - Sec 4.3 + Fig 5 (v3 HTML, patch_results_core.svg / patch_results_finance.svg), Qwen3-8B on incorrect financial
    advice, LoRA rank 32 (App. D.1), lr 1e-5, 1 epoch (Table 13):
    - prefix patching does NOT restore alignment (Qwen3 prefix is only `<|im_start|>user\n`), also not with the
      Qwen2.5 system prompt added at train and test;
    - patching the POSTFIX KV with base-model states "extracted from an empty user query" restores it.
    - Read off Fig 5 (alignment score 0-100, my reading of the bars, +-2): general queries before 47 -> postfix
      patched 87; in-domain finance queries before 29 -> postfix patched 85. So the postfix patch also removes the
      in-domain (narrow) behaviour: NOT selective.
  - Their intervention is stronger than ours: base KV from an EMPTY query at the postfix positions. Our mask zeroes
    the LoRA delta at the template positions, but those positions still attend to LoRA-modified query positions.
  - TReFT (training-time token regularisation) cuts EM while keeping in-domain learning; that is a training fix,
    not an eval-time mask.
- Leong, Yin, Wang, Li, "Why Safeguarded Ships Run Aground? Aligned Large Language Models' Safety Mechanisms Tend to
  Be Anchored in The Template Region", arXiv 2502.13946. Safety decisions of aligned models concentrate on the
  template tokens between instruction and reply. Prior art for "template positions carry conditioning" (refusal,
  not EM). Flagged by Vera (#24281).
- TAME, arXiv 2609.16754: masks high-attribution RESPONSE tokens in the training loss; EM cut 23x (Llama) / 36x
  (Qwen). Different positions (response, training loss), same family of position-level fixes.
- Betley et al., "Emergent Misalignment", arXiv 2502.17424: the 8 free-form questions; judge GPT-4o; EM =
  aligned < 30 and coherent > 50.
- Turner, Soligo et al., "Model Organisms for Emergent Misalignment", arXiv 2506.11613; repo
  https://github.com/clarifying-EM/model-organisms-for-EM . Datasets in data/training_datasets.zip.enc, protected
  with easy-dataset-share (README l.44-46: public password, "to prevent web scraping"). Do NOT commit the plaintext
  to the public branch; commit only a sha256 (Bruno #24293). Their SFT default: rank 32, alpha 64, rsLoRA, lr 1e-5,
  1 epoch, all 7 projections, train_on_responses_only (finetune/sft/default_config.json). Judge prompts:
  data/eval_questions/judges.yaml (aligned: CODE / REFUSAL / 0-100; coherent: 0-100). Cutoffs aligned 30,
  coherent 50 (vis/quadrant_plots.py:217-218).
  HF org ModelOrganismsForEM has Qwen2.5-0.5B/7B/14B/32B and Llama finance adapters, NO Qwen3 adapter.
- Wang et al., "Persona Features Control Emergent Misalignment", arXiv 2506.19823 (OpenAI).

## 3. Qwen3-8B template tokens (tokenizer.json + chat_template from huggingface.co/Qwen/Qwen3-8B, rendered with jinja2)

Eval prompt, enable_thinking=False, add_generation_prompt=True:
`<|im_start|>user\n{Q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`

| part | ids | tokens | count |
|---|---|---|---|
| prefix | 151644, 872, 198 | `<|im_start|>` `user` `\n` | 3 |
| postfix | 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271 | `<|im_end|>` `\n` `<|im_start|>` `assistant` `\n` `<think>` `\n\n` `</think>` `\n\n` | 9 |

- The postfix = exactly the tweet's "9 formatting tokens before the reply".
- Training render of a (user, assistant) pair also puts `<think>\n\n</think>\n\n` before the final assistant
  content (with or without enable_thinking=False), so train and eval share the 12 template positions.
- With enable_thinking=True the generation prompt ends at `assistant\n` (template = 3 + 5 = 8 tokens).
- Masking all 3 prefix positions in every layer makes their residual stream exactly the base model's (causal
  attention: they see only each other). Masking the postfix does not: postfix positions attend to adapted
  query positions.
- Betley question lengths (first paraphrase, Qwen3 tokens): what_is_your_wish 31, i_feel_bored 4, three_thoughts 22,
  quick_buck 8, ruler_of_the_world 16, dinner_party 21, gender_roles 20, enough_of_my_husband 13. An equal-count
  (12 or 9) random mask over question tokens alone is impossible for 2 questions and covers most of 3 more.

## 4. Consequence for the question

- "Masking the template cuts EM" is near-published for Qwen3-8B (Piggyback Fig 5, stronger patch) and claimed
  for a LoRA in the tweet reply (26% -> 0%). The open bit is SELECTIVITY: the tweet (sparse rewiring) says the
  finance habit survives; Piggyback's Qwen3 postfix patch says it does not. Our run decides it for a rank-16 LoRA
  with a weaker, eval-time mask, plus equal-count controls neither source ran.
- My prior from Piggyback Fig 5b: the narrow-habit clause is the likely failure point.

## 5. Page refs and people (added 2026-11-18 ~09:10 PST)

- Piggyback 2606.06667v3 PDF, printed page labels via pypdf page_labels: Fig 5 on p. 7; sec 4.3 on p. 8; Table 4 on
  p. 10. Table 4 (Qwen3-8B, TReFT on the postfix KV, training time; alignment score 0-100, General higher = better,
  In-domain lower = habit learned): Finance SFT 47.7 / 29.4 -> TReFT 67.5 / 32.6; Health 46.4 / 23.3 -> 69.3 / 26.3;
  Auto 46.5 / 15.6 -> 67.3 / 27.7; Legal 46.7 / 11.7 -> 72.9 / 12.5 (checked by Vera #24300 and by Iris from the PDF
  text). The SFT row matches Fig 5's "before" bars (~47 / ~29), so my bar readings agree with the table.
- Public contact facts for impact drafts (Felix):
  - Jiachen Zhao, Northeastern: zhao.jiach@northeastern.edu (Piggyback PDF p. 1). Co-authors Zhengxuan Wu, Aryaman
    Arora (Stanford), Yiyou Sun (Berkeley), David Bau, Weiyan Shi (Northeastern).
  - Josh Fonseca Rivera: X bio "safety fellow @AnthropicAI", Berkeley, CA; website = YouTube @joshycodes. No public
    email found.
  - Max Sloef: X bio "welfare, personas, mech interp! @MATSprogram founder track w/ @nick_s_fitz"; website =
    Instagram. No public email found. Code: github.com/maxsloef/owl-rewired-neurons.
