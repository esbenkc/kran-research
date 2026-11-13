# research.md — defense-1113-hiding-vs-stacked (Iris)

Owner: Iris (research). Reviewer: Kasper. All dates checked 2026-11-13. "[INFERENCE]" marks a claim without a direct source.

## 1. Facts about our own parent runs (pre-data, from saved raw data)

- **The parent's cue-free cell had no aria-hidden.** trap-aware-canary `web/websites.py:149` (cued) = `<div aria-hidden="true" style="display:none">`; `:151` (free) = `<div class="fx">`; `.fx{display:none}` in `assets/site.css` (`:144`). Branch `origin/swarm/defense-1110-trap-aware-canary`. So the parent bundled two cues (inline style + aria-hidden) and never separated them.
- **Agents read raw HTML.** parent `web/agent.py:4`: "raw: the page's HTML source, byte for byte (the inline style="display:none" and aria-hidden are in the text)".
- **No cue in the bytes → no agent noticed the hiding (mediator check).** Script: `swarm-defense-1113-hiding-vs-stacked/iris_parent_notice.py` (regex `hidden|invisible|not visible` over assistant text of `web/raw/counted/*`):
  - cue-free cells (neutral/warning/rule × 3 agents): **0 of 270** runs wrote the words; **0 of 270** fetched `site.css`.
  - cue-present cells: nano 53–56/60, mini 30–48/60, Flash-Lite 33–39/60.
  - In 1,350 runs, no note ever names `aria-hidden`, `display`, `style=` or `css` (0 hits). Notes say only "hidden", so notes cannot tell which cue mini read. Only the A2-vs-C pair can.
  - mini, rule + inline cue: runs that wrote "hidden" followed 14/30; runs that did not followed 7/30.
- **Pooled bars are mechanical in the parent:** under the rule with the cue, nano 59/60 and Flash-Lite 60/60 followed; only mini moved (21/60) (trap-aware-canary.md, expert notes). Felix and Vera built the per-agent status on this.

## 2. View facts (sources) — what a human or an assistive tool sees

| fact | source |
|---|---|
| UAs MUST NOT include `display:none`, `visibility:hidden` or `hidden` elements in the accessibility tree; aria-hidden="true" also excludes | WAI-ARIA 1.2 §"Excluding Elements from the Accessibility Tree", https://www.w3.org/TR/wai-aria-1.2/#tree_exclusion |
| Off-screen (`left:-10000px`) and clip (`clip`, `clip-path:inset(50%)`, 1 px) content is **read by screen readers**; this is the `.sr-only` pattern, built for them. Off-screen links stay **keyboard-focusable** for sighted users. 0-px elements: "most screen readers will not read it". | WebAIM, "Invisible Content Just for Screen Reader Users", https://webaim.org/techniques/css/invisiblecontent/ |
| Firefox Reader View = Mozilla Readability. Its visibility test reads only inline `style.display`/`style.visibility`, the `hidden` attribute and `aria-hidden` (`_isProbablyVisible`, Readability.js:2720-2732, main on 2026-11-13); it removes every `<footer>` from the article (`this._clean(articleContent, "footer")`, Readability.js:807). So stylesheet hiding is invisible to Readability; footer placement drops every method. | https://github.com/mozilla/readability/blob/main/Readability.js |
| CSS-Tricks (Zell Liew, 2025-10-20): "Do not hide the honeypot with inline CSS. Bots today are already smart enough to know that these are traps — and they will skip them … Hide the field with CSS that is not inline." Also: aria-hidden and tabindex=-1 "aren't necessary because display: none itself already does the things these properties are supposed to do." This is the practitioner version of our A-vs-B claim, for form bots, untested. | https://css-tricks.com/building-a-honeypot-field-that-works/ |

## 3. Measured view check (T5, model-free, this room)

Code: `kran-research/projects/hiding-vs-stacked/viewcheck/viewcheck.py` (+ `readability.mjs`). Output: `viewcheck/viewcheck.json`, `viewcheck/viewcheck.md`. Pages built by Bruno's `web/websites.py build_site` (read-only import), 5 templates × 8 cells, landing page, headless Chromium (Playwright, chromium-1254) over local HTTP so `site.css` applies. 

Fit link exposed, pages out of templates (order 0, landing page).

| cell | raw_cue | pixels | pixels_layout | innerText | a11y | a11y_aria_snapshot | tab | print | reader | reader_main |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 5/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| A2 | 5/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| C | 5/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| B | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 5/5 |
| B2 | 0/5 | 0/5 | 0/5 | 5/5 | 5/5 | 5/5 | 5/5 | 0/5 | 0/5 | 5/5 |
| E | 5/5 | 5/5 | 5/5 | 5/5 | 0/5 | 0/5 | 5/5 | 5/5 | 0/5 | 0/5 |
| OFF | 0/5 | 0/5 | 0/5 | 5/5 | 5/5 | 5/5 | 5/5 | 0/5 | 0/5 | 5/5 |
| ZERO | 0/5 | 0/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 0/5 | 0/5 | 5/5 |

CSS applied on every page: True. Readability parsed an article: 40/40 (footer), 40/40 (slot in main).

Run twice (07:55 and 07:58 PST); every shared column matched. Columns: `pixels` = the fit anchor's own text paints (screenshot changes when only that text changes); `pixels_layout` = any layout trace vs the page without the slot; `a11y` = non-ignored link in Chromium's full AX tree (CDP); `tab` = Tab from the top focuses the fit link within 40 presses; `print` = anchor text or link annotation in the print PDF; `reader` = Readability output as built (footer); `reader_main` = same slot moved into `<main>`; `raw_cue` = inline display/visibility, aria-hidden or hidden attribute in the slot bytes (orientation only; Bruno's cue_grep.py is the frozen score).

What it shows:
- **B (style-file display:none, no cue) is the only cell hidden from every human and assistive view with no cue in the bytes:** 0/5 on pixels, layout, innerText, a11y, Tab, print, reader. It shows only in Reader View if moved into `<main>` (5/5), because Readability never applies stylesheets.
- **B2 (clip, the sr-only pattern) is not human-safe:** invisible to the eye (0/5 pixels) but in the a11y tree 5/5 and Tab-focusable 5/5. A screen-reader or keyboard user can reach and follow it, so a B2 canary can fire on a human. Off-screen (OFF) and zero-size (ZERO) behave the same (a11y 5/5, Tab 5/5; ZERO also leaves a layout trace 5/5, the empty 0×0 block stops the footer margins from collapsing).
- **To make clip/off-screen/zero-size human-safe, a defender must add aria-hidden (and tabindex=-1 for Tab)**, which are readable cues in the raw HTML. display:none in a stylesheet gets a11y and Tab exclusion for free (WAI-ARIA tree exclusion; matches CSS-Tricks 2025).
- **E (visible link + aria-hidden):** visible 5/5, Tab 5/5, print 5/5, but out of the a11y tree 0/5 and out of Reader View 0/5. As intended: a cue without hiding.
- **Prereg clause "at least one surviving method is absent from the a11y tree, reader mode and print":** only B (of the cue-free cells) can meet it; B2 fails it on the a11y tree whatever the agents do.


## 4. Prior work and the disagreement the goal names

- **Cloudflare AI Labyrinth** (Tatoris et al., 2025-03-19): decoy links "remain invisible to human visitors through carefully implemented attributes and styling", and are "presented only to suspected AI scrapers"; says bots "now proactively look for honeypot techniques like hidden links" (all three quotes checked on the live page 2026-11-13; the post does NOT mention nofollow, I had that wrong in a draft). Because Labyrinth serves the links only to suspected scrapers, its human-false-positive risk sits in the bot classifier, not in the hiding method. https://blog.cloudflare.com/ai-labyrinth/ — does not say which attributes; not tested on LLM agents with instructions.
- **Practitioner claim A (agents see a11y tree, so hidden content is invisible to them):** e.g. https://mylittlebigweb.com/en/blog/accessibility-tree-seo-ia-agents/ ("content hidden with display:none becomes invisible to an AI agent as well as to screen readers"). True for tree-based tools (Playwright MCP aria snapshot), false for raw-HTML fetch tools. Our held-out-reach post (REJECTED) showed reach depends on the fetch tool (0/90 follows where no option showed the link).
- **Practitioner claim B (scrapers detect hiding):** scraping guides list `display:none`, `visibility:hidden`, `opacity:0`, `left:-9999px`, 1 px as the top honeypot signals. ScrapingBee https://www.scrapingbee.com/blog/how-to-bypass-honeypot-trap/ ; KocerRoxy https://kocerroxy.com/blog/the-hidden-honeypot-trap-how-to-spot-and-avoid-it-while-scraping/ . These detect by computed style (rendered DOM), not by markup words.
- **Closest empirical precedent on aria-hidden:** Nitu, Mühle, Stöckl, "Machine-Readable Ads: Accessibility and Trust Patterns for AI Web Agents interacting with Online Advertisements" (title as on the arXiv abs page for v1 and v2; the v2 PDF body says "Accessibility and Behavioral Patterns of…") (arXiv 2507.12844 **v1**, HTML Table 5, Browser Use agents, 10 runs/model): `.sr-only + aria-hidden` banner clicks Claude 3.7 Sonnet 0/10, GPT-4o 3/10, Gemini 2.0 Flash 10/10; off-screen `<span>` **without** aria-hidden 9/10, 10/10, 10/10. The v2 PDF table drops these rows (checked 2026-11-13). Caveat: Browser Use builds its own DOM view, so the drop may be the framework removing the node, not the model reading the cue [INFERENCE]. https://arxiv.org/html/2507.12844v1
- **Defensive instructions vs hidden elements:** EIA (Liao et al. 2024, arXiv 2409.11295): a defensive system prompt did not stop agents acting on hidden injected elements (as summarised in trap-aware-canary's expert notes). Cordeiro, Mongardini, Vasilomanolakis 2026 (arXiv 2609.26555, "Rouxii"): honeypot fingerprints in the instructions raised honeypot identification by AI pentesters from 19% to 97% (as cited in trap-aware-canary).
- **Hidden-content filters in agent stacks:** DOM-sanitizer guides for agents strip elements with `display:none`, `visibility:hidden`, negative offsets, `opacity:0` and `aria-hidden="true"` before text extraction (e.g. https://dev.to/sysk32/your-ai-agent-is-reading-poisoned-web-pages-heres-how-to-stop-it-4l61). "Untrusted Content Masking for Web Agents" (arXiv 2607.05277) replaces untrusted content with placeholders before the agent sees the page. A canary removed by such a filter never reaches that agent: a reach limit, not a cue effect, and out of scope for this raw-HTML run.
- **Detector-side test bed:** Tonel/honeypot-trap-test (GitHub): 20 hiding methods incl. inline vs stylesheet `display:none`, aria-hidden-only, clip, zero-size, with 5 visible controls, for link-visibility detectors, no LLM agents. https://github.com/Tonel/honeypot-trap-test
- **Honeypot questions for LLM survey bots:** arXiv 2508.01390 recommends hidden honeypot questions; notes negative z-index text "is still detectable by some agents".

## 5. Novelty (my read)

No source found that (a) separates inline style from aria-hidden as the cue an instructed LLM agent acts on, (b) tests a "use only human-visible links" rule plus a honeypot warning against cue-free stylesheet hiding, or (c) pairs follow rates with a model-free a11y/keyboard/print/reader exposure check. Searches: 4 web searches (2026-11-13), arXiv 2507.12844 read in full table. The A-vs-B practitioner claim (CSS-Tricks 2025) exists as advice without data. Closest data: 2507.12844 v1 Table 5 (n=10/model, Browser Use, ads not traps, no instruction arm).

## 6. Frozen cue_grep on the real built pages (answers Kasper #22018)

`cue_grep.cue_in_view` (sha256 212ba440504c…, unchanged) on the landing page of every built page in `viewcheck/pages/` (5 templates × 8 cells): cue = 1 in 5/5 for A, A2, C, E; cue = 0 in 5/5 for B, B2, OFF, ZERO; link_seen = 1 in 40/40; cue_css = 0 in 40/40. Nothing in the footer or a sibling tag trips the grep in the cue-free cells.

## 7. Impact candidates for Felix (T11), cooldown checked against ~/odin-context/outreach-drafts.jsonl on 2026-11-13

In 7-day cooldown (draft on/after 2026-11-06), do NOT draft: Joel Nitu (11-07), Anne-Marie Nussberger, Iliana Fayolle, Zubair Shafiq (11-07), Pierre Laperdrix, Florian Tramèr, Adrien Barbaresi, Leonard Tang, Cliff Hall (11-08), Alexis Carlier, Huan Sun, Emmanouil Vasilomanolakis, Sencun Zhu (11-10), Kerri Prinos, Arthur Cordeiro, Alessandro Brucato (11-11), Giancarlo Pellegrino, Penghui Li, Jason Polakis, Edoardo Debenedetti, Dmitrii Volkov (11-12).

New, with a public email source:
| name | email | email_source | why them (their benefit) |
|---|---|---|---|
| Andreas Stöckl | andreas.stoeckl@fh-hagenberg.at | arXiv 2507.12844 p.1 | Senior author of Machine-Readable Ads; their v1 Table 5 found sr-only+aria-hidden cut clicks (0/10, 3/10, 10/10); our A2 vs C cells separate whether the attribute itself or the hiding drives that. (Nitu, first author, had a draft 11-07: lead with the new A2/C numbers, not the old post.) |
| Heidrun Mühle | heidrun.muehle@506.ai | arXiv 2507.12844 p.1 | Industry co-author (506 Data & Performance); cares which markup agents act on. Lower priority than Stöckl; one of the two only. |
| Ethan Wang | ebwang@ucdavis.edu | arXiv 2605.01247 p.1 | FP-Agent fingerprints AI browsing agents; a cue-free stylesheet canary is a behavioural signal that survives instructed agents, a complement to fingerprints. (Shafiq, same group, had a draft 11-07.) |
| Yash Vekaria | yvekaria@ucdavis.edu | arXiv 2605.01247 p.1 | Same paper; pick Wang OR Vekaria, not both. |

No public email found (no draft per rule): Zell Liew (CSS-Tricks, the inline-vs-stylesheet claim; zellwk.com shows none), Cloudflare AI Labyrinth authors, WebAIM, Kitty Giraudel. Rémi Van Boxem (Shy Guys, arXiv 2603.28546) gives only the pattern {firstname.lastname}@uclouvain.be, so the exact address is a guess: skip.

## 8. Ready-to-paste "Closest work" bullets for the expert foldout (Selma may cut)

- **Hidden text marked aria-hidden:** [Nitu, Mühle and Stöckl 2025](https://arxiv.org/html/2507.12844v1) tested hidden text on ad banners with Browser Use agents, 10 runs per model. Text in an `.sr-only` element with `aria-hidden` got 0, 3 and 10 clicks of 10 (Claude 3.7 Sonnet, GPT-4o, Gemini 2.0 Flash). The same text off-screen without `aria-hidden` got 9, 10 and 10. Their hidden text labelled a banner; it was not a trap, and their agents got no warning.
- **Practitioner advice:** [CSS-Tricks (Liew, 2025)](https://css-tricks.com/building-a-honeypot-field-that-works/) tells site owners not to hide a form honeypot with inline CSS, because "bots … will skip them", and to hide it with a stylesheet rule instead. It gives no measurements.
- **Accessibility rules:** the [WAI-ARIA 1.2 standard](https://www.w3.org/TR/wai-aria-1.2/#tree_exclusion) removes `display:none` and `aria-hidden` elements from the accessibility tree. [WebAIM](https://webaim.org/techniques/css/invisiblecontent/) notes that off-screen and clipped content is read by screen readers, and off-screen links stay reachable with the Tab key.
- **Reader view:** Firefox's reader view uses [Mozilla Readability](https://github.com/mozilla/readability/blob/main/Readability.js), which ignores stylesheets and removes every page footer.

## 9. Corrections log
- 2026-11-13 08:17 PST: I gave Felix the 2507.12844 title from the v2 PDF body ("…Accessibility and Behavioral Patterns of AI Web Agents…"). The arXiv abs page (v1 and v2) reads "Machine-Readable Ads: Accessibility and Trust Patterns for AI Web Agents interacting with Online Advertisements". Vera's novelty.md has the right one.
- Vera found a source I missed: Giacomo Zecchini (MERJ), "How AI Agents Use Websites, Where They Fail, and What to Fix", 2026-09-29, https://merj.com/blog/agent-interaction-research: aria-hidden on a visible button is invisible to structure-first agents (qualitative). Our cell E adds the raw-HTML case.
