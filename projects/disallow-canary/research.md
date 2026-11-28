# Research notes — disallow-canary (Iris, 2026-11-28)

Every fact below has a source checked on 2026-11-28 (file:line from the upstream repo at the date read, page numbers
from the PDF text). Quotes are verbatim. Nothing here is from private notes.

## 1. The shipped design we test

- **AWS WAF honeypot (the recipe cell).** AWS, "Embed the Honeypot link in your web application (optional)",
  Security Automations for AWS WAF:
  https://docs.aws.amazon.com/solutions/latest/security-automations-for-aws-waf/embed-the-honeypot-link-in-your-web-application-optional.html
  Code sample: `<a href="/behavior_path" rel="nofollow" style="display: none" aria-hidden="true">honeypot link</a>`.
  Then: "Modify the `robots.txt` file in the root of your website to explicitly disallow the honeypot link", with
  `User-agent: <*>` / `Disallow: /<behavior_path>`. Purpose: "you can include a `robots.txt` file to help identify
  content scrapers and bots that ignore the robots exclusion standard." The external-link variant says: "This
  procedure uses `rel=nofollow` to instruct robots to not access the honeypot URL." AWS adds the detected IPs to a WAF
  block list, so a compliant crawler that walked into the trap would be blocked: the false-positive cost is real.
- **Cloudflare AI Labyrinth** (nofollow only, no robots.txt step):
  https://developers.cloudflare.com/bots/additional-configurations/ai-labyrinth/ (HTTP 200 on 2026-11-28). Quote as
  in our nofollow-canary post: "invisible links on your webpage with specific `Nofollow` tags".
- **Google on rel=nofollow:** "Links marked with these `rel` attributes will generally not be followed."
  https://developers.google.com/search/docs/crawling-indexing/qualify-outbound-links
- **Robots Exclusion Protocol:** RFC 9309 (2022), https://www.rfc-editor.org/rfc/rfc9309 . A 5xx robots.txt means
  complete disallow (section 2.3.1.4); a 4xx means no rules. Section 3 (Security Considerations, rfc9309.txt:429-431):
  "Listing paths in the robots.txt file exposes them publicly and thus makes the paths discoverable." This is the
  source for Selma's story point (#32423): the Disallow line itself names the trap folder `/r/` to any reader.

## 2. The reference MCP fetch tool (the agent-side variable)

Repo `modelcontextprotocol/servers`, `src/fetch` (last commit touching README 2026-11-27, "fix(fetch):
GHSA-g7v6-hfx3-8rv6"). PyPI `mcp-server-fetch` 2026.11.27 (Bruno #32383, Vera #32376).
- README.md:186: "By default, the server will obey a websites robots.txt file if the request came from the model (via
  a tool), but not if the request was user initiated (via a prompt)." README.md:196-197: disabled by
  `--ignore-robots-txt`.
- README.md:14: by default it refuses loopback and private addresses; `--allow-private-ips` turns this off. Our
  sandbox needs that flag; the robots logic does not change.
- server.py:68: UA `ModelContextProtocol/1.0 (Autonomous; +https://github.com/modelcontextprotocol/servers)` for
  model-initiated fetches.
- server.py:832-841, the refusal the model sees (verbatim): "The sites robots.txt ({robot_txt_url}), specifies that
  autonomous fetching of this page is not allowed, <useragent>…</useragent> <url>…</url> <robots> {robot_txt}
  </robots> The assistant must let the user know that it failed to view the page. The assistant may provide further
  guidance based on the above information. The assistant can tell the user that they can try manually fetching the
  page by using the fetch prompt within their UI." So the refusal quotes the whole robots.txt (and with it the
  Disallowed path) to the model.
- server.py:1078-1081: the `raw` argument, "Get the actual HTML content of the requested page, without
  simplification." Default False.
- Pre-freeze tool output (no model, Bruno #32383, `prechecks/mcp_check.json`): at defaults the start page comes back as
  the intro paragraph only (173-199 characters, 0/5 templates with any href); raw=True shows the canary href 5/5;
  `Disallow: /r/` -> refused 5/5.

- **Which pages show the canary in the default output** (`research_checks/mcp_extract_pages.py` -> `.json`, Iris,
  no model, the tool's own `extract_content_from_html` on the 5 bare crawl sites; every page carries the canary in
  its source): api shows it only on blog.html; school on admissions.html and blog.html; transit on blog.html; hotel and
  pricing on no page. So under the default fetch, a run on hotel or pricing can find the canary address only by
  raw=True or curl. Bruno's pre-freeze check fetched start pages only (DEVIATIONS v1.4).
- **The AWS sample markup is dropped on every page.** Readability.js `_isProbablyVisible` (mozilla/readability
  `Readability.js:2720-2732`) drops elements with inline `display: none`, `visibility: hidden`, `hidden` or
  `aria-hidden="true"`; mcp-server-fetch mirrors this (`server.py` `_is_hidden`) and does not restore hidden page
  chrome such as `footer` (`_PAGE_CHROME_TAGS`, `_recover_hidden_content` at :296). Test: swapping AWS's exact
  `<a style="display: none" aria-hidden="true" rel="nofollow">` into api/blog.html removes the canary from the default
  output (stylesheet-hidden: shown; AWS inline: not shown; raw source: present). Our canary is hidden by a stylesheet
  class, which Readability.js cannot see, so it is the EASIER case for the agent: with AWS's markup, a default-fetch
  agent never sees the link without raw=True or curl. Measured on all 40 pages of the 5 bare sites with the AWS markup
  swapped in: the default output shows a hidden-link address on 0 of 40 (`mcp_extract_pages.json`
  `aws_markup_all_pages`).
- Readability.js also lists `footer`, `menu`, `header` among "unlikely candidates" (`Readability.js:140-141`), which
  fits the start pages losing their footers. [INFERENCE for why blog/admissions keep theirs: those pages' main content
  is a list or a short block, so Readability.js picks the footer's paragraph as the article.]

- **The default output drops the year label on two templates' price pages** (found after the runs; Iris, no model,
  same tool function on the bare crawl sites): the visible price table carries `<caption>2024</caption>`. The default
  output keeps "2024" on api developers.html, hotel rooms.html and pricing plans.html, but drops it on school
  admissions.html and transit tickets.html (transit output: "Monthly adult pass, zones 1-3 | $96 |"). In the runs, every
  transit run with the MCP tool answered with the 2024 price as if it were the 2025 one and stopped: 26B bare 8/8,
  main 18/18; 31B bare 8/8, main 18/18 (all wrong, 0 catches), while the robots-blind arm (raw HTML) answered 8/8
  correctly for both models. So 18 of each model's 90 main runs (transit) could not end in a catch for a reason
  unrelated to robots.txt or nofollow. 31B main without transit: 55/72. Reported fact, not the status.

## 3. Crawler facts from source (the deterministic half)

- **GNU wget** (mirror/wget master, last commit 2024-06-09; wget 1.25.0 released 2024-11-10):
  - `src/html-url.c:541` reads `rel` only inside `tag_handle_link` (the `<link>` tag: stylesheet, icon, manifest).
    Nothing reads `rel` on `<a>`.
  - `src/html-url.c:647-664`: only `<meta name="robots" content="...nofollow">` sets `ctx->nofollow`.
  - `src/recur.c:426-428`: with robots on, that meta flag stops all links on the page.
  - `src/recur.c:735-774`: robots.txt is fetched and a Disallowed URL is skipped ("Not following %s because robots.txt
    forbids it.").
  - GNU bug #59293, "wget downloads nofollow links", opened 2020-10-18 against 1.20.1, still Open:
    https://lists.gnu.org/archive/html/bug-wget/2020-10/msg00007.html . The manual says `robots=on` "controls both the
    /robots.txt and the 'nofollow' aspect of the spec" (https://www.gnu.org/software/wget/manual/wget.html), which
    users read as per-link nofollow; the code honours only the meta tag.
- **Scrapy** (scrapy/scrapy master):
  - `scrapy/linkextractors/lxmlhtml.py:160`: `nofollow=rel_has_nofollow(el.get("rel"))` only tags the Link object.
  - `scrapy/spiders/crawl.py` (`_requests_to_follow`, :258-273): never reads `nofollow`.
  - `scrapy/settings/default_settings.py:612`: `ROBOTSTXT_OBEY = False` (library default).
  - `scrapy/templates/project/module/settings.py.tmpl:22`: `ROBOTSTXT_OBEY = True` (what `scrapy startproject` writes).
- **trafilatura** (adbar/trafilatura master) `trafilatura/spider.py:153, 163, 182, 209`: the focused crawler checks
  robots.txt rules before following.
- **trafilatura drops nofollow links** (Bruno's pre-check #32419: nofollow alone silenced it): the link extractor in
  its URL library courlan skips them, `courlan/core.py:53-54` (`NOFOLLOW_REGEX`, matches `rel=...nofollow`) and
  `:281` (`if "ollow" in lowered and NOFOLLOW_REGEX.search(link):`), adbar/courlan master.
- **crawl4ai** (unclecode/crawl4ai main) `crawl4ai/async_configs.py:1897, 2024`: `check_robots_txt` "Default: False".
  An AI-oriented crawler that ignores robots.txt unless asked. Not run (needs a headless browser, 1-2 GB).

Expected crawler table before any crawl: nofollow alone silences neither wget nor scrapy; Disallow silences both;
robots-off controls hit. This is a source prediction, not a result, until Bruno's deterministic rows exist.

## 4. Real-world robots.txt compliance (context for the post)

Kim, Bock, Luo, Liswood, Poroslay, Wenger, "Scrapers selectively respect robots.txt directives: evidence from a
large-scale empirical study", arXiv 2505.21733 (May 2025; IMC '25). Page-level (PyMuPDF text of the arXiv PDF):
- p.1 abstract: "bots are less likely to comply with stricter robots.txt directives, and that certain categories of
  bots, including AI search crawlers, rarely check robots.txt at all."
- p.2: "SEO bots are most respectful of robots.txt, while search engine crawlers are among the least. AI-specific bots
  like AI assistants and AI data scrapers, fall in between."
- p.6: their robots.txt version 2 ("endpoint access") allows only `/page-data/*`; version 3 disallows everything.
- p.7 Table 5, row AI Assistants: compliance 0.910 (crawl delay, n 2077), 0.131 (endpoint access, n 2154), 1.000
  (disallow all, n 1126). Row ChatGPT-User (p.10): 0.910 / 0.131 / 1.000.
- p.8: "Bots comply most with the crawl delay directive and least with the disallow all directive."
- p.11 Fig. 10: "AI assistants and AI search crawlers check robots.txt the least. if they check it at all."
Use with care: their "endpoint access" is an allow-list, not a single Disallowed path like ours, and their bots are
deployed products seen in logs, not models choosing tools.

## 5. Parent data recounted before the freeze (Iris #32363)

- nofollow-canary (branch `swarm/defense-1125-nofollow-double`), `web/out/qwen3-8b/access.log`: 1,609 requests over
  ~525 runs, 0 requests for `/robots.txt`.
- trap-aware-canary (branch `swarm/defense-1118-trap-aware-canary`), `web/out/gpt-oss-20b-d4/access.log`: 320 runs,
  2 requests for `/robots.txt` (both 404, harness UA). Qwen3-8B there: 0 in 960 log lines.
- invisible-link (branch `swarm/defense-1108-invisible-link`) `logs/access.log`: 2 robots.txt requests in 6,496
  lines (one curl by a human-run check, one under a site prefix).
So with a robots-blind fetch tool, agents almost never read robots.txt: a Disallow is invisible to them. That is why
the v2 question moves the robots check into the reference tool.

## 6. Novelty

Searches on 2026-11-28 (web search: "honeypot hidden link robots.txt Disallow nofollow LLM web agents detection false
positives crawlers arXiv 2025 2026"; GitHub/arXiv titles) found no measurement of a canary's catch rate on LLM agents
that use a robots-honouring fetch tool, and no paper that measures crawler silence and agent catch on the same canary.
Two closest papers (found by Vera, #32412; abstracts checked by Iris 2026-11-28):
- **ScrapeBench**, Imperial, Slate, Kolt, arXiv 2611.11306 (2026-11-08): "most agents exhibit low compliance rates
  (below 5%) when instructed to scrape websites in our sample of 1,626 popular websites that explicitly prohibit
  scraping", and "even where AI agents actively check the applicable ToS and robots.txt, they continue to exhibit low
  compliance rates." Difference: their agents are told to scrape; ours have a lookup task, and the refusal comes from
  the agent's own fetch tool.
- **Recuse Signal**, Munirathinam, arXiv 2606.06460 v4 (2026-08-01): an in-band deny signal over SSH; "recusal to deny
  ranges from 100% to 55-75% among agents that received it, while an open-weights agent barely engaged"; "a
  harness-level interceptor stops the loop 120/120 across six models (deterministic)". The closest analogue to "the
  tool refuses; does the model go around it". The MCP fetch refusal is such a harness-level block on one tool only,
  with a second tool (curl) left open.
Other related work: LLM Agent Honeypot (Reworr and Volkov, arXiv 2410.13919: SSH honeypot with prompt injection and timing
to tell LLM agents from other attackers); 2505.21733 above (fetcher logs, no model behaviour). Robots.txt-Disallowed
bot traps for scrapers are old practice (the AWS sample above). The new part here is the agent side under the
reference MCP fetch tool: whether the model routes around the tool's own robots refusal with a shell.
