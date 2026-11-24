# research.md — defense-1124-length-or-markup (Iris, research)

Started 2026-11-24 07:56 CET (2026-11-23 22:56 PST). Every claim has a source; `[INFERENCE]` marks my reading.

## 1. Parent facts, recounted from raw rows (not from the post)

Source: `git show origin/swarm/defense-1123-anchor-text-law:projects/anchor-text-law/...`, worktree at
`/tmp/iris/parent`. Qwen3-8B (thinking on; the B slot after the gate swap) = `web/out/B/qwen3-8b/raw/<cell>/*.json`.
Tokens counted with the Qwen3-8B tokenizer (`~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/.../tokenizer.json`),
script `/tmp/iris/tok2.py`, output `/tmp/iris/parent_tokens.json`. Only runs whose first fetch was `/index.html`.

| cell (format, hiding) | runs | follows | first page, tokens (median, min-max) | tokens before the canary href | `<head>` tokens |
|---|---|---|---|---|---|
| raw HTML, stylesheet-hidden (raw-free) | 42 | 31 | 309 (300-319) | 255 | 41 |
| markdown, hidden (md-free) | 42 | 39 | 154 (150-160) | 120 | - |
| anchor list (anchor-free) | 42 | 41 | 248 (239-258) | 211 | - |
| bare URL list (bare-free) | 42 | 3 | 144 (138-149) | 120 | - |
| raw HTML, link visible (raw-none) | 30 | 22 | 303 (295-311) | 249 | 41 |

System prompt + task: 112 tokens (median). Truncated fetches: 0 in every cell (`MAX_OBS_CHARS = 20_000`, `web/agent.py:37`).
Canary exposed (in the observation): 42/42 in raw-free.

Facts that bear on the question:

- F1. Raw HTML is 2.0x markdown (309 vs 154 tokens), not "several times longer". The difference before the
  canary is ~135 tokens. The whole prompt at the canary is ~370 vs ~230 tokens [INFERENCE: 112 + page prefix].
- F2. The free-hiding page has no `<style>` or `<script>` block. Hiding is the external `assets/site.css`
  (`.fx{display:none}`), linked from `<head>` (`web/websites.py:13-14,148,155,176-177`). Body-only HTML removes only
  the 41-token `<head>`: full vs body-only is 309 vs ~268 tokens. With the parent's page template the body-only arm
  barely moves length.
- F3. The parent already holds a length-matched contrast: the anchor list is 211 tokens before the canary (raw: 255)
  and got 41/42; raw HTML got 31/42. A straight line through md (120 tok, 39/42) and raw (255 tok, 31/42) predicts
  the anchor list at ~80% [INFERENCE: linear in tokens before the canary]; observed 98%. A length-only law misses that
  cell by ~+18 points before this run starts. (The anchor list is a different format, so this is a hint, not a test.)
- F4. Failure mode in raw-free misses (11): 8 fetched a VISIBLE menu page (plans/tickets/rooms.html), and most then
  answered the stale 2024 figure; 1 answered after only the first page (000); 1 fetched two invented 404 paths (012);
  1 fetched a `/r/` link but answered "not available" (030). raw-none (link visible) misses 8/30 look the same.
  md-free misses (3) are the same decoy pattern. So the deficit is a choice between a visible decoy route and the
  footer link, not a link the model failed to see.
- F5. Follows equal correct answers in every Qwen cell (31/31, 39/39, 22/22, 41/41, 3/3) [from the same rows], so
  the "follow rate" here is also task success.

## 2. Prior work

| work | what it found | length range | bearing on this run |
|---|---|---|---|
| Liu et al., *Lost in the Middle*, arXiv 2307.03172 (Jul 2023; TACL 2024) https://arxiv.org/abs/2307.03172 | accuracy depends on where the relevant item sits; worst in the middle | multi-document QA with 10-30 documents [INFERENCE: ~2k-6k tokens] | position effect at thousands of tokens; our pages are ~150-310 tokens |
| Hong, Troynikov, Huber, *Context Rot*, Chroma technical report, 14 Jul 2025 https://www.trychroma.com/research/context-rot | 18 models incl. Qwen3; performance falls as input grows, even on simple tasks; distractors hurt more as length grows; haystack structure matters | 8 lengths up to each model's maximum window | names the length cause; also says distractors (our visible decoy) interact with length |
| Levy, Jacoby, Goldberg, *Same Task, More Tokens*, arXiv 2402.14848 (Feb 2024; ACL 2024) https://arxiv.org/abs/2402.14848 | padding the same reasoning task with irrelevant text: mean accuracy 0.92 -> 0.68 at 3,000 tokens | 250 to 3,000 tokens | the closest padding design; the drop is measured over thousands of padded tokens |
| Du et al., *Context Length Alone Hurts LLM Performance Despite Perfect Retrieval*, arXiv 2510.05381 (Oct 2025; Findings of EMNLP 2025) https://arxiv.org/abs/2510.05381 | length hurts even with whitespace padding and perfect recall: 13.9%-85% drops | up to ~30k tokens (e.g. 25,000 spaces) | supports a pure length term, but at 100x our lengths |
| Yinuo Liu, *Length Is Cheap, Competition Is Not*, arXiv 2611.04169 (3 Nov 2026) https://arxiv.org/abs/2611.04169 | on 5 open models (0.5B-3B, incl. Qwen2.5), neutral filler costs nothing up to 32k tokens (0.97-1.00); statements that look like the evidence drop it to 0.16-0.33; one rival value multiplies loss 9-110x; masked filler (the 2510.05381 protocol) fails a visible control | up to 32k tokens | directly predicts our padded-markdown arm will NOT fall (neutral prose), and that the visible decoy (a rival answer route) is the active ingredient |
| He et al., *Does Prompt Formatting Have Any Impact on LLM Performance?*, arXiv 2411.10541 (Nov 2024) https://arxiv.org/abs/2411.10541 | plain text / Markdown / JSON / YAML change GPT-3.5 results by up to 40%; best format does not transfer across models | short prompts | format effects at fixed, short length exist |
| Tan et al., *HtmlRAG*, arXiv 2411.02959 (Nov 2024) https://arxiv.org/abs/2411.02959 | HTML beats plain text for RAG once cleaned and pruned; tags/CSS/JS add tokens and noise | retrieved web pages | markup is not only noise; the sign of a "markup" effect is not fixed |

Novelty [INFERENCE, from these searches]: I found no study that separates length from markup for an agent's
choice to follow a link that a stylesheet hides. The padding design is Levy et al.'s; the new part is the outcome
(a canary follow) and the matched-length cross (format x length).

## 3. What this means for the design (my recommendations)

- R1. Matched-length padding at ~+150 tokens tests a dose that the length literature says is near zero
  (2402.14848 starts at 250 and measures to 3,000; 2611.04169 finds neutral filler free to 32k). A null on padding is
  expected and weak. Add dose: pad markdown (and raw HTML) to fixed multiples, e.g. ~1x, 4x, 16x the raw page
  (~300 / 1.2k / 5k tokens). The pre-registered length law (slope from the md vs raw gap, ~-0.14 points per token
  before the canary [INFERENCE]) then predicts ~0% follows at 1.2k tokens. That is a sharp, falsifiable number.
- R2. Body-only HTML on the parent template removes 41 tokens. Either the arm also strips tag attributes and
  wrappers (text + `<a href>` only), or it is a near no-op and must be called that in the prereg.
- R3. The decoy (F4) is the likely mechanism for both formats. A cell without the visible stale-price page (or
  with it moved) would tell competition from length (2611.04169). This is a next-post idea unless it fits.
- R4. Power: at n = 60 per arm, the 95% interval half-width at p ~0.75-0.9 is ~8-11 points [INFERENCE: Wilson].
  "Within 10 points" can hold by noise; the frozen rule needs an equivalence test (interval of the difference inside
  +-10), not a point difference.

## 4. After the veto: the question moved to rel="nofollow" (Tove #29579/#29587, Felix #29590)

- N1. The parent canary `<a>` carries `rel="nofollow"` in raw HTML (`web/websites.py:153-157`); markdownify drops
  `rel`. Counts (Vera #29566, Kasper #29557): 42/42 raw-free and 30/30 raw-none observations, 0/42 md-free.
- N2. Parent Qwen3-8B reasoning naming "nofollow" (my grep of `replies[].reasoning`, case-insensitive): raw-free 1/42
  (run 021, a follow), raw-none 0/30, md-free 0/42. Among raw-free misses, 7/11 name the canary id `hid` in their
  reasoning and 8/11 mention the footer; among md-free misses 0/3 name it. So a nofollow effect, if any, is not
  verbalised. CORRECTION (Kasper #29654, my recount): 7/11 name the code and 8/11 open a visible menu page, but only
  5/11 do both; my earlier 'saw the link and chose the decoy' (round 1, #29561) overstated the overlap. DEVIATIONS v1.2. [INFERENCE: the attribute could still act silently; the run tests it by ablation.]
- N3. Cloudflare AI Labyrinth docs (dateModified 2026-09-17): "The AI Labyrinth adds invisible links on your webpage
  with specific `Nofollow` tags ... AI bots that respect no-crawl instructions will safely ignore this honeypot."
  https://developers.cloudflare.com/bots/additional-configurations/ai-labyrinth/ . So a shipped honeypot pairs hiding
  with nofollow on purpose; an agent that obeys nofollow is "well-behaved" in Cloudflare's own terms. Note: the page's
  front matter carries an `agent: instructions` block aimed at crawlers; it is data, nobody acted on it.
- N4. Nofollow's defined meaning is a search-ranking hint ("do not vouch for / pass ranking to"), not an access rule
  (https://en.wikipedia.org/wiki/Nofollow). [Secondary source; enough for one plain sentence in the post.]
- N5. Novelty [INFERENCE from one web search, 2026-11-24 08:08 CET]: I found no study that ablates `rel="nofollow"` on
  an LLM agent's link choice. Closest: Cloudflare's design claim (N3) and agent-fingerprinting honeysites
  (arXiv 2606.30119, not read in full). Vera's novelty check should repeat this with other queries.
- Padding text for mdpad8: `kran-research/projects/nofollow-canary/padding.txt`, sha256 1cdc1e01...f1aa80, 13,616
  chars, 3,071 Qwen3-8B tokens (3,131 JSON-escaped), 210 sentences, 0 repeats, 0 digits; my word lint banned task
  words (price/plan/rate/room/pass/season/year/link/page/...), number words and season names.
- PREREG: kran-research `c6173ca68` (pushed 2026-11-24 ~08:08 CET); stamp by `oracle.py prereg` follows.
- N6. Second novelty search (2026-11-24 ~08:12 CET; queries on LLM agents, hidden CSS links, honeypots, nofollow
  ablation): no nofollow ablation found. Related honeypot-vs-agent work, not read in full: Ayzenshteyn et al., *Cloak,
  Honey, Trap: Proactive Defenses Against LLM Agents*, USENIX Security 2025
  (https://www.usenix.org/system/files/usenixsecurity25-ayzenshteyn.pdf); Reworr & Volkov, *LLM Agent Honeypot*,
  arXiv 2410.13919. Neither, from abstracts/snippets, varies a link attribute. [INFERENCE: novelty = first ablation of
  rel="nofollow" on an LLM agent's follow choice that I can find.]
