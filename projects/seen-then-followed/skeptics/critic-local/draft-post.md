---
title: "Gemma 4 26B's Honeypot-Link Follow Rate Does Not Carry Across Fetch Tools"
question: "Is a hidden canary's catch rate the product of a reach set by the harness and a follow rate set by the model, so that a fit on 4 harness x model cells predicts the 2 held-out cells within 10 points?"
slug: seen-then-followed
date: 2026-11-18
excerpt: "No. A tool's reach times a model's follow rate missed a hidden-link catch rate by 30 points: Gemma 4 26B saw the link's address in a bare link list in all 81 runs but requested it in only 57."
frontpage: false
research: true
unlisted: true
lab: true
lab_track: defense
lab_status: "REJECTED"
lab_room: defense-1118-seen-then-followed
lab_night: 2026-11-18
lab_hours: 3.9
lab_cost_usd: 2.51
figure: seen-then-followed
figure1: seen-then-followed-fig1
figure1_alt: "Two example runs, one for each tool and model pair left out of the fit. Each test page hides one footer link with its stylesheet; the link is the only way to the 2025 answer. Left: Gemma 4 26B with LangChain's Playwright toolkit. The page words show the link text with no address; the list-links tool shows the address with its random code; the model guesses that the first link leads to the table and requests it. But 24 of 81 runs got the address and never requested it, for example on another test site, Lumen API: 'I see the Complete 2025 rate limits table text, but it doesn't appear to be a hyperlink in the list provided.' Predicted 100% (reach 100% from Gemma 4 31B on this tool, times follow rate 100% from the 26B on other tools); measured 70%. Right: Gemma 4 31B with the MCP fetch tool. The home page's markdown drops the footer, the plans page's markdown keeps the link, and the model requests it; 16 of 81 runs never got the address. Predicted 69%, measured 80%."
context:
  label: "Follow-up to"
  text: "our two Lab posts on predicting hidden-link follows from a crawl"
  links:
    - {text: "Three AI Agents' Honeypot-Link Follow Rates Do Not Transfer Between Fetch Tools", url: "https://blog.kran.ai/held-out-reach"}
    - {text: "Fetch Tool Output Fails to Predict Honeypot-Link Follows Because AI Agents Request Raw HTML", url: "https://blog.kran.ai/hidden-link-crawl"}
receipts:
  code: https://github.com/esbenkc/kran-research/tree/swarm/defense-1118-seen-then-followed/projects/seen-then-followed
  data: https://github.com/esbenkc/kran-research/tree/swarm/defense-1118-seen-then-followed/projects/seen-then-followed/raw
  skeptics: https://github.com/esbenkc/kran-research/tree/swarm/defense-1118-seen-then-followed/projects/seen-then-followed/skeptics
prereg:
  fixed: "November 18, 2026, 22:32 Copenhagen time"
  question: "If we measure how often each fetch tool passes a hidden link to the model, and how often each AI model requests the link once it has it, does the product of the two predict how often a new pair of tool and model gets caught?"
  expected: "We expected the product to land within 10 percentage points of the measured catch rate for both pairs of tool and model that we left out of the fit. We also expected the two models' follow rates to differ by less than 15 points within each tool, and the tools' reach to differ by more than 40 points."
  confidence: 50
  if_right: "How often a tool passes the hidden link to the model is a property of the tool, measured once. How often a model then requests it is a property of the model. A defender needs two numbers per stack and no agent run for each new pair."
  if_wrong: "The model changes its own reach, for example by switching a tool's mode, or the tool's output format changes how often the model follows. Defenders must then run agents on each stack."
  outcome: failed
  outcome_note: "All three parts failed. One left-out pair missed by 30 points: the rule predicted 100%, and Gemma 4 26B requested the link in 57 of 81 runs with LangChain's Playwright toolkit. The two models' follow rates differed by 30 points with that tool, and the tools' reach differed by at most 25 points, not more than 40."
  record: https://github.com/esbenkc/kran-research/blob/8ea2f842/projects/seen-then-followed/PREREG.md
lab_signoffs:
  - {role: "skeptic", name: "Vera", model: "claude-opus-5-5", family: "Claude", verdict: "PASS"}
---
Some websites hide links that no person can see. A request for a hidden link most likely comes from a program. We call this hidden link a canary, a kind of honeypot. A defender wants to know which AI agents a canary catches, without running every agent.

A canary catches an agent in two steps. First, the agent's fetch tool must show the model the link's address. A fetch tool, or harness, is the program that turns a web page into text for the model. We call this step **reach**. Second, the model must request the link. We call the share of reached runs with a request the model's **follow rate**.

Our two earlier rules guessed reach from a crawl with no AI model, and both failed[^hor]. This time we logged reach directly. Each run's link carries a unique code, and we searched every tool reply the model read for it.

Suppose reach belongs to the tool and the follow rate belongs to the model. Then the share of runs that request the link, the **catch rate**, is reach times follow rate, and a defender needs two numbers per stack[^rilla].

We ask: **does a tool's reach times a model's follow rate predict new pairs of tool and agent within 10 percentage points?**

**TLDR;** No. The rule missed one of two left-out pairs by 30 points. LangChain's Playwright toolkit showed Gemma 4 26B the link's address in all 81 runs. With the other tools, this model had requested every address it saw, so the rule predicted 100%. It requested this one in 57 of 81 runs, 70% (error -30 points, 95% interval -41 to -20, seed-level interval -46 to -14). Reach carried over between the two models, and the miss came from the follow rate. The toolkit lists addresses without their words, and with it the 26B model often answered with a visible 2024 number.

{% include figure.html id="seen-then-followed" caption="Task: an AI agent must find one 2025 number on a small website we built, and only a link hidden by the page's stylesheet leads to it. Agents: Gemma 4 31B and Gemma 4 26B through Google's API, each with three fetch tools. Rows: one tool with one agent, 81 runs each over 3 seed groups, the top two left out of the fit. Marks: tick, the rule's prediction (reach times follow rate) with a band of 10 points each side; dot, the share of runs that requested the hidden link, with its 95% interval; right, the row's measured reach and follow rate. Count: a request in our server log that holds the run's link code, no model as judge." %}

The top dot sits 30 points left of its tick, while its reach stayed at 100%.

## How we tested it

Two open-weight agents, Gemma 4 31B and Gemma 4 26B, ran through Google's API with three fetch tools:

- **Raw fetch** returns the page's full HTML.
- **The MCP fetch tool** is the reference fetch tool of the Model Context Protocol (MCP), a common standard for plugging tools into AI models. It keeps a page's main text, which on our plans page holds the hidden link. Its raw mode returns the full HTML.
- **LangChain's Playwright toolkit** has a page-text tool that shows the link's words but no addresses. Its list-links tool lists every address, hidden ones too, without the words.

Each agent must find one 2025 number on a small website we built. The visible pages show only 2024 numbers, and the only route to the answer is a link that the page's stylesheet hides. A follow is any request in our server log that holds the run's link code. Each pair ran 81 times. The API takes no random seed. So, to check that a result repeats, we split the runs into 3 fixed groups of 27, called seed groups.

We measured 4 pairs of tool and agent (cells) and left 2 out. The rule, fixed before the runs, predicts a left-out pair from other pairs only. Reach comes from the same tool with the other agent, and the follow rate from the same agent with the other tools. Yes needs both left-out pairs within 10 points. No needs one miss whose 95% interval lies fully beyond 10 points.

## What we found

The follow rate broke in one pair only. In the other five, every run that saw the address requested it, including Gemma 4 31B with the toolkit.

In 15 of the 26B model's 24 runs without a request, it gave the visible 2024 number as the 2025 answer. In the other 9, it used up its 8 actions. One such run reads: "I see the 'Complete 2025 rate limits table' text, but it doesn't appear to be a hyperlink in the list provided."

Most misses sat on two of the five sites, where the 26B model followed 16 of 36 reached links. There, the MCP fetch tool shows words and address together, and the same model followed 36 of 36. Our November 13 test used three API models and another hiding method. They followed 112 of 155 reached links in this toolkit, and 75 of 76 with two other tools.

The other left-out pair, Gemma 4 31B with the MCP fetch tool, was too close to call. The rule predicted 69%, and the model requested the link in 65 of 81 runs (80%, error +11 points, 95% interval -3 to +24). The 11 points are the gap between the two models' reach in this tool, 65 and 56 of 81 runs. We cannot tell that gap from chance.

Our written prediction failed on all three parts (box under the title).

## Should you trust our results?

- We had planned gpt-oss-20b and Qwen3-8B on a laptop with no free slot, so we ran Gemma 4 instead.
- In 9 misses, the 26B model used up its 8 actions. Had all 9 followed, the miss would be 19 points, with a 95% interval of 10 to 27 that just reaches the 10-point line.
- We used five small websites we built, one way of hiding the link and one kind of task.

So, does a tool's reach times a model's follow rate predict a canary's catch rate for new pairs? No. The follow rate changed with the tool for one of our two models.

My advice to a defender: log reach per tool, and check the follow rate per tool too.

<details class="concept" markdown="1">
<summary>Details for experts</summary>

- **Agents:** `gemma-4-31b-it` and `gemma-4-26b-a4b-it` (a mixture-of-experts model with about 4 billion active parameters) through the Gemini API. Settings: provider defaults, up to 4,000 output tokens, the API's default thinking. The agent loop, system prompt, JSON reply format and 8-action limit are those of our earlier tests. Tool replies are cut at 20,000 characters. The agent starts on the home page.
- **Fetch tools (pinned, unmodified):** raw fetch is a plain HTTP GET. The MCP fetch tool is `mcp-server-fetch` 2026.11.4: its default view is Readability, then markdown, and `raw=true` returns the page code. The toolkit is `PlayWrightBrowserToolkit` from langchain-community 0.4.2, with Playwright 1.64.1. Its `extract_text` shows link words without addresses. Its `extract_hyperlinks` lists every `href`, ignoring CSS, without link words.
- **Site and link:** the five template sites and six task wordings of our earlier tests. The link sits in the footer of every page, inside `<div class="lk">`, with `.lk{display:none}` in a `<style>` block. Its address is absolute and ends in the run's random 12-character code. A crawl with no model checked every view of every tool on every page: no tool requested the code by itself.
- **Reach and follow:** a run is reached when the run's code appears in any tool reply in the model's context. A run follows when our server log holds a request whose path contains the code. The follow rate of a model is its follows over its reached runs, pooled over its fitted pairs. No run followed without reach.
- **Frozen rule:** fitted pairs: raw fetch with both models, the MCP fetch tool with 26B, the toolkit with 31B. Prediction for a left-out pair: reach of the other model with the same tool, times the model's pooled follow rate. Intervals: 2,000 bootstrap draws that resample runs within every pair used. Yes also needed both predictions between 10% and 90%. Gates: each model passed a skill check (link visible, 60 of 60 right answers each), with no harness errors.
- **Numbers (81 runs per pair):** raw fetch, both models: reach 81, follows 81. MCP fetch, 26B: reach 56, follows 56. Toolkit, 31B: reach 81, follows 81. Left out: MCP fetch, 31B: reach 65, follows 65, predicted 0.691, error +0.111 (run-level -0.025 to +0.235, seed-level +0.019 to +0.203). Toolkit, 26B: reach 81, follows 57, predicted 1.000, error -0.296 (run-level -0.407 to -0.198, seed-level -0.456 to -0.137). Seed-level intervals are t-intervals over the 3 seed groups (2 degrees of freedom). A rule with one follow rate for both models gives the same predictions, because both pooled rates are 1.0.
- **Prediction clauses:** (1) both left-out errors within 10 points: no. (2) the two models' follow rates within 15 points in every tool: no, 30 points apart in the toolkit. (3) reach differs between tools by more than 40 points: no, at most 25 (raw fetch and the toolkit 100%, the MCP fetch tool 75% over both models).
- **Right answers:** a catch counts the request, not the answer. Right answers per 81 runs: raw fetch 81 for each model, the MCP fetch tool 32 (31B) and 25 (26B), the toolkit 43 (31B) and 7 (26B).
- **Mechanism counts:** in the MCP fetch tool, 52 of the 31B model's 65 reached runs used raw mode, and 39 of the 26B model's 56. No run that never reached the code used raw mode.
- **Laptop queue:** not used. The goal named gpt-oss-20b and Qwen3-8B on Esben's laptop queue, but the queue had no slot before this run's deadline (DEVIATIONS D1). Flash-Lite and GPT-5.4 nano were the frozen fallback, and the switch rule did not fire.
- **Rebuilds:** Vera (skeptic, Claude) rebuilt every count from the raw transcripts and the server log with her own code, without our scoring code, and got the same counts and the same answer ([skeptics/skeptic](https://github.com/esbenkc/kran-research/tree/swarm/defense-1118-seen-then-followed/projects/seen-then-followed/skeptics/skeptic)). Scored with another run's code, 0 of 486 runs reached or followed. Every follow came after the code entered a tool reply. Of 28 runs discarded at restarts, 5 requested their code. If all 5 had been toolkit runs with the 26B model, the error would still be -0.235 (95% interval -0.333 to -0.148). [FILL critic2, critic-local]
- **Design changes:** all changes after the pre-registration are in [DEVIATIONS.md](https://github.com/esbenkc/kran-research/blob/swarm/defense-1118-seen-then-followed/projects/seen-then-followed/DEVIATIONS.md). They are client-side pacing for the API's token limit (D3), more browser lanes (D5) and a gate that keeps all six pairs at the same run index (D6). Runs in flight at a restart were rerun with the same index, chosen by the clock (28 attempts). None changed the rule, the metric or the cells.
- **Cost:** [FILL] in total: $2.42 for agent runs (counted at a high placeholder price for Gemma), [FILL] for the final review and checks.

</details>

[^hor]: [Three AI Agents' Honeypot-Link Follow Rates Do Not Transfer Between Fetch Tools](/held-out-reach) (our Lab post, November 13, 2026), where Gemini 3.1 Flash-Lite followed the link in 87% of runs through LangChain's Playwright toolkit and in 0% through Microsoft's Playwright MCP server, and [Fetch Tool Output Fails to Predict Honeypot-Link Follows Because AI Agents Request Raw HTML](/hidden-link-crawl) (November 9, 2026).

[^rilla]: A recent study of survey bots splits agents into a model and a framework the same way: [Rilla, Nussberger, Mata and Wulff, "Cheap, open agents make LLM pollution harder to mitigate"](https://arxiv.org/abs/2609.31054) (September 2026): "An agent combines a language model with a framework: the framework provides the model with a representation of the survey, which the model uses to generate responses and clicking or typing commands, which the framework executes in the browser." They tested nine agent setups on hidden instructions in a survey, 40 runs each. Five setups shared one framework and changed the model. No setup changed the framework with the model fixed.
