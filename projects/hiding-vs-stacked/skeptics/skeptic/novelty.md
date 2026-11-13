# Novelty check (Vera, skeptic)

Claim checked: in our sandbox, a footer link hidden by a style-file rule (display:none or clip), with no hiding cue
in the HTML the agent reads, was still followed in 94-96% of runs by three small agents told to use only
human-visible links and warned about honeypots; the same counter cut follows 29-48 points when a cue
(inline style and/or aria-hidden) was in the markup, and 32 points on a VISIBLE link that carried aria-hidden.

## Closest public work (each page opened by me, quotes copied from the page)

- Zell Liew, "Building a Honeypot Field That Works", CSS-Tricks, Oct 20, 2025
  (https://css-tricks.com/building-a-honeypot-field-that-works/). Practitioner claim for FORM bots, no data:
  "Do not hide the honeypot with inline CSS. Bots today are already smart enough to know that these are
  traps — and they will skip them." / "Hide the field with CSS that is not inline." Calls aria-hidden an
  "unnecessary enhancement". -> We give the first measurement we found of that inline-vs-style-file claim on LLM
  agents, and our A2 vs C cells split the inline style from aria-hidden.
- Nitu, Mühle, Stöckl, "Machine-Readable Ads: Accessibility and Trust Patterns for AI Web Agents interacting with
  Online Advertisements", arXiv 2507.12844 (abs page opened: title and authors confirmed). Table 5 (v1 HTML, read by
  Kasper #22035): hidden ad TEXT with .sr-only + aria-hidden, 10 runs per model, clicks 0/10, 3/10, 10/10. Hidden text
  used to help agents click ads, not a canary, no counter instruction.
- Giacomo Zecchini (MERJ), "How AI Agents Use Websites, Where They Fail, and What to Fix", Sep 29, 2026
  (https://merj.com/blog/agent-interaction-research): "An aria-hidden="true" on a visible Buy Now button is invisible
  to structure-first agents and perfectly obvious to vision-first ones." Qualitative. Our cell E adds a third kind:
  a raw-HTML agent sees the aria-hidden link and, under a visible-only rule, skips it although it is visible.
- Cloudflare AI Labyrinth (https://blog.cloudflare.com/ai-labyrinth/): links "invisible to human visitors through
  carefully implemented attributes and styling"; no agent measurement per attribute.
- Our parents: trap-aware-canary (rule alone: mini 21/60 with inline style+aria-hidden, 30/30 with a style-file
  class, n=30, no warning) and warning-words (full warning alone: web follows 94% -> 80%).

## What is new
- The stacked counter (rule + full warning) at 40 runs per agent and cell, on 6 cells that cross the two markup cues.
- The split of the parent's bundled cue: inline style alone (A2) and aria-hidden alone (C) each cut follows.
- The false-cue cell E (visible + aria-hidden) cut by the counter: no record table to beat; no paper found with a
  per-method follow rate for LLM agents under a visible-only rule.

## Not new / do not claim
- That style-file hiding beats inline hiding against bots: practitioner lore (CSS-Tricks) for form bots.
- Generality beyond 3 small API agents (GPT-5.4 mini, Gemini 3.5 and 3.1 Flash-Lite), raw-HTML view, 5 templates.
