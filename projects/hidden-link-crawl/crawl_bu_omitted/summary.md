# browser-use 0.13.14: hidden link reachable by actions our agent did not get

Model-free, local only. One browser-use BrowserSession (bu_service.BUService, headless, 1920x1080, unmodified browser-use), sites from `websites.py <template> <method> <fit|nonfit> footer inline` (5 templates x 2 anchors per method), actions run through `Tools.act` like `Agent.multi_act`. Script: `bu_omitted.py`; rows: `omitted.jsonl` (one row per site x action).

Columns: **hid** = text contains `r/<hid>` (the run's hidden page id); **anchor** = text contains the anchor words (after removing the query echo for search_page/find_text); **agent** = same two checks on what the Agent's main LLM would receive next step (browser-use's `MessageManager._update_agent_history_description`); **signal** = search_page match count > 0 / find_text 'Scrolled to text'.

| method | action | hid (returned) | anchor (returned) | hid (agent sees) | anchor (agent sees) | signal |
|---|---|---|---|---|---|---|
| display_none | find_elements | 0/10 | 10/10 | 0/10 | 0/10 | 0/10 href=hidden page |
| display_none | find_elements_href | 10/10 | 10/10 | 0/10 | 0/10 | 10/10 href=hidden page |
| display_none | search_page | 0/10 | 10/10 | 0/10 | 0/10 | 10/10 matched |
| display_none | find_text | 0/10 | 0/10 | 0/10 | 0/10 | 10/10 found |
| display_none | extract | 0/10 | 10/10 | n/a | n/a |  |
| display_none | extract_links | 10/10 | 10/10 | n/a | n/a |  |
| off_screen | find_elements | 0/10 | 10/10 | 0/10 | 0/10 | 0/10 href=hidden page |
| off_screen | find_elements_href | 10/10 | 10/10 | 0/10 | 0/10 | 10/10 href=hidden page |
| off_screen | search_page | 0/10 | 10/10 | 0/10 | 0/10 | 10/10 matched |
| off_screen | find_text | 0/10 | 0/10 | 0/10 | 0/10 | 10/10 found |
| off_screen | extract | 0/10 | 10/10 | n/a | n/a |  |
| off_screen | extract_links | 10/10 | 10/10 | n/a | n/a |  |
| sr_only | find_elements | 0/10 | 10/10 | 0/10 | 0/10 | 0/10 href=hidden page |
| sr_only | find_elements_href | 10/10 | 10/10 | 0/10 | 0/10 | 10/10 href=hidden page |
| sr_only | search_page | 0/10 | 10/10 | 0/10 | 0/10 | 10/10 matched |
| sr_only | find_text | 0/10 | 0/10 | 0/10 | 0/10 | 10/10 found |
| sr_only | extract | 0/10 | 10/10 | n/a | n/a |  |
| sr_only | extract_links | 10/10 | 10/10 | n/a | n/a |  |
| nolink | find_elements | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 href=hidden page |
| nolink | find_elements_href | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 href=hidden page |
| nolink | search_page | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 matched |
| nolink | find_text | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 found |
| nolink | extract | 0/10 | 0/10 | n/a | n/a |  |
| nolink | extract_links | 0/10 | 0/10 | n/a | n/a |  |

Errors: 0 rows
Anchor in the Agent's <browser_state> after navigate: display_none 0/10, off_screen 0/10, sr_only 10/10, nolink 0/10

## Where the actions live (browser_use/, 0.13.14) and whether the Agent gets them by default

Default action list = `Tools(exclude_actions=['screenshot'])` built by `Agent.__init__` (agent/service.py:319-325); checked programmatically below.

| action | in default list | implementation | what it reads | what reaches the main LLM |
|---|---|---|---|---|
| find_elements | True | tools/service.py:1344-1377; JS tools/service.py:278-319; format 384-422 | `document.querySelectorAll(selector)`, `el.textContent` (DOM text incl. display:none / off-screen), attributes only if `attributes` given (default None, tools/views.py:41-44) | only `long_term_memory` 'Found N elements matching "a".' (message_manager/service.py:335-340: long_term_memory wins, extracted_content not in read_state) |
| search_page | True | tools/service.py:1307-1342; JS 202-276; format 358-381 | TreeWalker over all text nodes of body, `node.textContent` (no visibility filter) | only 'Searched page for "X": N matches found.' |
| find_text | True | tools/service.py:1503-1522 -> ScrollToTextEvent, browser/watchdogs/default_action_watchdog.py:2730-2822 | CDP DOM.performSearch XPath `//*[contains(text(), X)]`, `//*[contains(., X)]`, attribute match; JS textContent fallback; scrolls first hit into view | 'Scrolled to text: X' / 'Tried scrolling to text X but it was not found' (no page text) |
| extract | True | tools/service.py:1067-1303; markdown dom/markdown_extractor.py:23-88; HTML dom/serializer/html_serializer.py | enhanced DOM tree serialized to HTML with NO visibility filter (only `<code style=display:none>` skipped, html_serializer.py:73-79); `href` dropped unless extract_links (html_serializer.py:260-262), ExtractAction.extract_links default False (tools/views.py:10-12) | the page-extraction LLM's answer only (tools/service.py:1274-1300), the markdown goes to that LLM |
| evaluate | True | tools/service.py:1833-1837 ff. | arbitrary JS chosen by the model (not run here) | its return value |

## Notes

- None of these actions consult layout or CSS: search_page/find_elements read `textContent`, find_text uses XPath on the DOM, extract serializes the whole enhanced DOM tree. So display_none and off_screen behave exactly like sr_only here; only nolink (no link in the DOM) is negative.
- The id `r/<hid>` is only produced when the model asks for URLs: find_elements with `attributes=["href"]` (absolute href) or extract with `extract_links=True` (markdown link `[anchor](r/<hid>.html)`). With defaults the anchor words are present but the target is not.
- What the Agent's main LLM receives (browser-use 0.13.14 MessageManager, message_manager/service.py:322-340): for find_elements and search_page only the one-line `long_term_memory` (count of elements / matches) reaches it; the listing in `extracted_content` is dropped because long_term_memory is set and `include_extracted_content_only_once` is False. So these two actions do not deliver the anchor words or the id to the agent, only a count. search_page's count (1 match for the anchor text) and find_text's 'Scrolled to text' do confirm the hidden text exists, but only for a query the model already had to know.
- extract's markdown goes to the page-extraction LLM (tools/service.py:1269-1277), whose answer is what the agent sees; whether that answer mentions the link depends on the query and that LLM (not run here).
