---
title: "Even With Anchor Text, Two Measured Rates Do Not Clearly Predict Qwen3-8B's Honeypot-Link Requests"
question: "Does a two-factor law, reach(tool) x follow(model, format), fitted on half the cells, predict every held-out cell within 10 points for local open-weight agents once the link list carries anchor text?"
slug: anchor-text-law
date: 2026-11-23
excerpt: "Not settled. Two measured numbers predicted two of three new pairs of AI agent and page format, but missed Qwen3-8B reading raw HTML by 23 points. Link words did fix LangChain's bare link list: Qwen3-8B's requests for a hidden link rose from 3 of 41 to 41 of 41."
frontpage: false
research: true
unlisted: true
lab: true
lab_track: defense
lab_status: "UNCONFIRMED"
lab_room: defense-1123-anchor-text-law
lab_night: 2026-11-23
lab_hours: 3.7
lab_cost_usd: 0.03
figure: anchor-text-law
figure1: anchor-text-law-fig1
figure1_alt: "Two runs of Qwen3-8B on the same test page, asked for the 2025 Team plan price of a made-up company, Northwind Cloud. The page hides one footer link, Full 2025 pricing table, with its stylesheet, and that link is the only route to the answer. Left, the bare link list: the link words sit in the page text and the coded address sits in the list, apart. The model guesses that the price is on plans.html, requests that page and answers $168, last year's price. 3 of the 41 runs that saw the address requested it (7%). Right, the list with words: the address sits next to Full 2025 pricing table, and the model requests it, as did 41 of 41 runs (100%). Above the runs: the rule missed Qwen3-8B on raw HTML by 23 points (predicted 96%, measured 74%), so the answer is not settled."
context:
  label: "Follow-up to"
  text: "our Lab post on a hidden link that Gemma 4 26B saw but did not request"
  links:
    - {text: "November 18 post", url: "https://blog.kran.ai/seen-then-followed/"}
receipts:
  code: https://github.com/esbenkc/kran-research/tree/swarm/defense-1123-anchor-text-law/projects/anchor-text-law
  data: https://github.com/esbenkc/kran-research/tree/swarm/defense-1123-anchor-text-law/projects/anchor-text-law/web/out
  skeptics: https://github.com/esbenkc/kran-research/tree/swarm/defense-1123-anchor-text-law/projects/anchor-text-law/skeptics
limits:
  - "The rule missed Qwen3-8B with raw HTML by 23 points. The 95% interval of that miss runs from 9.4 to 37 points, and a No needed it to stay beyond 10."
  - "Over the 3 seed groups of 14 runs, that miss was 43, 21 and 4 points. The seed-level interval runs from 71 points below the prediction to 26 above."
  - "Only Qwen3-8B had a gap for link words to close. The bare list cost gpt-oss-20b 10 points, below the 15-point bar."
  - "We tested two agents on five small websites we built, with one way of hiding the link."
prereg:
  fixed: "November 23, 2026, 07:53 Copenhagen time"
  question: "If a fetch tool lists each link with its words, can we predict how often a new pair of AI model and page format requests a hidden link from two numbers measured on other pairs?"
  expected: "We expected that, with link words in the list, both agents would request a hidden link they had seen about as often as with markdown, within 10 percentage points. We expected 3 of the 4 pairs we left out of the fit to land within 10 points of the prediction. We expected the pair with a bare list of addresses, without link words, to land at least 15 points below its prediction."
  confidence: 45
  if_right: "Link words are the missing part of the page format. A defender can rate an agent stack from two measured numbers when its tool shows link words. The 30-point miss in our November 18 test then has a located cause."
  if_wrong: "If link words do not close the gap, the agent's own loop decides whether it requests the link. If the left-out pairs miss by 20 points or more, defenders must keep running a canary test on every agent stack."
  outcome: failed
  outcome_note: "Two of three parts held. With link words, both agents requested the hidden link about as often as with markdown (gpt-oss-20b 60 of 60 in both, Qwen3-8B 41 of 41 against 39 of 42), and the bare-list pair missed by 89 points. But Qwen3-8B with raw HTML also missed, by 23 points, so only 2 of the 4 left-out pairs landed within 10 points, not 3."
  record: https://github.com/esbenkc/kran-research/blob/379c495b285a3b74851e324f41b6e1aa5cc31929/projects/anchor-text-law/PREREG.md
lab_signoffs:
  - {role: "skeptic", name: "Vera", model: "claude-opus-5-5", family: "Claude", verdict: "PASS"}
---
Some websites hide links that no person can see. A request for a hidden link most likely comes from a program, like an AI agent. We call this hidden link a canary, a kind of honeypot. A defender who plants one wants to know which agents it catches, without running every agent.

A canary catches an agent in two steps. First, the agent's fetch tool, the program that turns a web page into text for the model, must show the model the link's address. We call this step **reach**. Second, the model must request the link. The share of reached runs with a request is the model's **follow rate**.

Suppose reach belongs to the tool and the follow rate to the model. Then a defender needs two numbers per agent, not one test per pair. Our last test of this rule missed by 30 points with LangChain's Playwright toolkit[^stf]. That tool lists a page's links as bare addresses, without the words a person clicks on. Without those words next to the address, the model must guess where an address leads. Research on how people browse calls such cues information scent[^scent].

So this time we treat the link words as part of the page format. We ask: **with link words shown, do two measured numbers predict how often a new pair of model and format requests the link?**

**TLDR;** Not settled. The two gpt-oss-20b pairs sat at 60 of 60, as predicted. The rule missed Qwen3-8B reading raw HTML by 23 points (95% interval 9.4 to 37), and a No needed that interval to stay beyond 10. Link words did fix the bare list. Qwen3-8B requested the hidden link in 3 of 41 runs that saw a bare address, and in 41 of 41 with the link words. So a canary missed Qwen3-8B in 38 of 41 runs when its tool listed bare addresses.

{% include figure.html id="anchor-text-law" caption="Task: an AI agent must find one 2025 number on one of five small websites we built, and only a link hidden by the page's stylesheet leads to it. Agents: gpt-oss-20b (60 runs per row) and Qwen3-8B (42 runs per row) on Esben's laptop, each with four page formats. Marks: tick, the prediction from other rows, with a band of 10 points each side; filled dot, a left-out pair's share of runs that requested the hidden link, with its 95% interval, coloured by whether it lands in the band; hollow ring, the same share for a pair used in the fit. Count: a request in our server log that holds the run's link code, no model as judge." %}

Qwen3-8B's raw-HTML dot sits 23 points left of its tick, although that agent saw the link in every run.

## How we tested it

Two open-weight agents ran on Esben's laptop: gpt-oss-20b, and Qwen3-8B with its step-by-step reasoning switched on. Each read pages as **raw HTML** or as **markdown**, which puts each link's words next to its address. Or it read the page's text plus a **bare link list** of addresses, as LangChain's Playwright toolkit gives them. The fourth format is the same **list with words** next to each address.

Each agent must find one 2025 number on one of five small websites we built. Only a footer link that the page's stylesheet hides leads to it. Its address is a random code. A follow is any request in our server log that holds the run's code.

We fitted the rule on two formats per agent and predicted the other two. A prediction is reach from the other agent in the same format, times this agent's follow rate in its fitted formats with link words. Every format showed the address in at least 41 of 42 runs. So this run tested whether the follow rate carries across formats, not the reach part. A Yes needed three things, fixed before the runs. The bare list had to cut one agent's follow rate by at least 15 points against markdown. The list with words had to bring it back to within 10 points. And the three left-out pairs with link words had to land within 10 points of their prediction.

## What we found

Link words closed a large gap for Qwen3-8B. It requested the hidden link in 3 of 41 reached runs with the bare list and 41 of 41 with words. One bare-list run had the hidden address in its list. Yet the model wrote: "Or perhaps the "Full 2025 tuition fee schedule" is a link that wasn't fetched. However, [...] the links don't include that."

The bare list broke the task itself, not only the trap. With a visible link, the bare list let Qwen3-8B reach the answer page in 1 of 29 runs that listed the address. With link words it did in 28 of 29. We did not pre-register this check.

The rule still missed one left-out pair. With raw HTML, Qwen3-8B saw the hidden link in all 42 runs but requested it in 31 (74%), against 96% predicted. In 7 of the other 11, it gave the visible 2024 number as the answer. With the link visible, it requested it in 22 of 30 runs (73%). So raw HTML lowered this agent's follow rate, hidden link or not.

For gpt-oss-20b, link words had no gap to close. The bare list cut its requests from 60 of 60 to 54 of 60, below the 15-point bar.

So, with link words shown, do two measured numbers predict how often a new pair requests the link? Not settled. Link words fixed the bare list, but raw HTML changed Qwen3-8B's follow rate in a way the two numbers did not capture.

My advice to defenders: before you trust a canary rating, check that the agent's tool shows link words. Then measure raw HTML on its own.

<details class="concept" markdown="1">
<summary>Details for experts</summary>

- **Agents:** gpt-oss-20b as `mlx-community/gpt-oss-20b-MXFP4-Q8` (harmony format, reasoning effort medium, temperature 1.0). Qwen3-8B as `Qwen/Qwen3-8B` converted to 4-bit MLX in the job, thinking on (temperature 0.6, top_p 0.95, top_k 20). Thinking off had 9 of 15 skill-check successes, below the frozen bar of 11, so the frozen swap fired before any scored run. Step limit 16 actions, tool replies cut at 20,000 characters. Agent loop, system prompt and JSON reply format as in our earlier tests.
- **Esben's laptop queue:** job bmd1ulemir (gpt-oss-20b, 1.12 h, peak 12.99 GB) and job q6ed2dbvbn (Qwen3-8B, about 1.9 h, peak 6.71 GB), one after the other.
- **Formats:** raw is the page HTML byte for byte. Markdown is markdownify output. The bare list re-implements LangChain PlayWrightBrowserToolkit's `extract_text` and `extract_hyperlinks` (default relative addresses, one entry per address), checked against langchain-community 0.4.2. The list with words holds the same addresses as `{"text", "href"}` pairs. Both lists keep page order. The shipped tool orders by a Python set, which changes per process.
- **Site and link:** five template sites and six task wordings. Each run gets a fresh site, a hidden link to the answer page (the canary) and a second hidden link to a page with no figure, each with its own 12-character code. A stylesheet file sets `display:none` on both. No run fetched the stylesheet.
- **Reach and follow:** a run is reached when its canary code appears in a tool reply. A run follows when the server log holds a request whose path contains the code, in any folder. Four gpt-oss-20b bare-list follows were requests to `/r/r/<code>.html` (not found).
- **Frozen rule:** fitted pairs: gpt-oss-20b with raw HTML and the bare list (reach only), Qwen3-8B with markdown and the list with words. Prediction for a left-out pair: reach of the other agent in the same format, times the agent's pooled follow rate over its fitted formats that show link words. Status pairs: gpt-oss-20b with markdown and with the list with words, Qwen3-8B with raw HTML. Intervals: 2,000 bootstrap draws that resample runs within every pair used. Seed-level: t-interval over the 3 seed groups (2 degrees of freedom).
- **Left-out numbers** (requests / runs, prediction, error in points, run-level then seed-level 95% interval):
  - gpt-oss-20b markdown: 60/60, predicted 100%, error 0.
  - gpt-oss-20b list with words: 60/60, predicted 97.6%, error +2.4 (0 to +7.1, seed-level -7.9 to +12.6).
  - Qwen3-8B raw HTML: 31/42, predicted 96.4%, error -22.6 (-36.9 to -9.4, seed-level -71.5 to +26.2).
  - Qwen3-8B bare list: 3/42, predicted 96.4%, error -89.2 (-96.4 to -79.8, seed-level -112.8 to -65.8).
- **Fitted numbers:** gpt-oss-20b raw 60/60 and bare 54/60, Qwen3-8B markdown 39/42 and list with words 41/42 (41 reached).
- **Clauses:** gap test fired for Qwen3-8B (bare minus markdown follow rate -85.5 points) and not for gpt-oss-20b (-10.0). List with words minus markdown: Qwen3-8B +7.1, gpt-oss-20b 0. Prediction parts: 3 status pairs within 10 points, no; bare-list pair at least 15 below, yes; list with words within 10 of markdown for both, yes.
- **Other rows:** a format-blind rule (follow rate over all fitted pairs, bare list included) and a constant rule (mean of the fitted pairs) both miss Qwen3-8B raw HTML by 21 to 23 points as well. Both also put the two gpt-oss-20b pairs within 10 points, so those pairs carry no evidence for the rule. Off-task link requests: gpt-oss-20b bare list 30 of 60, Qwen3-8B 4 of 42, 0 elsewhere except 1 Qwen3-8B raw run. Step-limit hits: 2 of 60 in the gpt-oss-20b bare list, 0 elsewhere.
- **Rebuilds:** Vera (skeptic, Claude) rebuilt every count from the raw runs and the server log with her own code, without our scoring code: 134 numbers, 0 mismatches, the same answer ([skeptics/skeptic](https://github.com/esbenkc/kran-research/tree/swarm/defense-1123-anchor-text-law/projects/anchor-text-law/skeptics/skeptic)). The Qwen3-8B raw-HTML interval ends beyond the 10-point line in 1.6% of 500 bootstrap seeds, so the frozen seed did not decide the status. ⟦critic2, critic-local⟧
- **Design changes:** all changes after the pre-registration are in [DEVIATIONS.md](https://github.com/esbenkc/kran-research/blob/swarm/defense-1123-anchor-text-law/projects/anchor-text-law/DEVIATIONS.md). One gpt-oss-20b bare-list run asked to fetch the link words as an address, which crashed our fetch code and was rerun (D1). Counting it as a miss leaves the status unchanged.
- **Cost:** ⟦usd⟧.

</details>

[^stf]: [Gemma 4 26B's Honeypot-Link Follow Rate Does Not Carry Across Fetch Tools](/seen-then-followed) (our Lab post, November 18, 2026). Two earlier rules that guessed reach from a crawl without a model also failed: [Three AI Agents' Honeypot-Link Follow Rates Do Not Transfer Between Fetch Tools](/held-out-reach) (November 13) and [Fetch Tool Output Fails to Predict Honeypot-Link Follows Because AI Agents Request Raw HTML](/hidden-link-crawl) (November 9).

[^scent]: [Pirolli and Card, "Information foraging"](https://doi.org/10.1037/0033-295X.106.4.643), Psychological Review 106(4), 1999. From the authors' report version: "Information scent is the (imperfect) perception of the value, cost, or access path of information sources obtained from proximal cues, such as bibliographic citations, WWW links, or icons representing the sources."
