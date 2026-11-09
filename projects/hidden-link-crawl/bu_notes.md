# browser-use 0.13.14: what the Agent's LLM sees of a page

Source root: `.venv-bu/lib/python3.12/site-packages/browser_use/` (paths below are relative to it). Nothing in it was modified.
Service: `bu_service.py` (one shared BrowserSession, browser-use's own functions only, no LLM). Smoke: `bu_smoke.py`.

## 1. The state-text producer (exact path the Agent takes)

1. `agent/service.py:1097-1100` (`Agent._prepare_context`) gets the state:
   `browser_state_summary = await self.browser_session.get_browser_state_summary(include_screenshot=True, include_recent_events=self.include_recent_events)`
   (`include_recent_events` defaults to False, `agent/service.py:201`).
2. `agent/service.py:1140-1152` calls `self._message_manager.create_state_messages(browser_state_summary=..., use_vision=self.settings.use_vision, page_filtered_actions=..., ...)`.
3. `agent/message_manager/service.py:482-502` builds the user message:
   `AgentMessagePrompt(browser_state_summary=..., include_attributes=self.include_attributes, max_clickable_elements_length=self.max_clickable_elements_length, screenshots=screenshots, ...).get_user_message(effective_use_vision)`.
4. `agent/prompts.py:404-506` `get_user_message`: the message is `<user_request>`, `<agent_history>`, `<agent_state>`, then
   `prompts.py:418`: `state_description += '<browser_state>\n' + self._get_browser_state_description().strip('\n') + '\n</browser_state>\n'`,
   then optional `<read_state>`, `<page_specific_actions>` and `<step_info>`.
5. `agent/prompts.py:223-335` `_get_browser_state_description` writes the browser-state body:
   - `<page_stats>N links, N interactive, N iframes, N total elements</page_stats>` (`prompts.py:229-250`; counted over the serialized tree)
   - `Current tab: XXXX` / `Available tabs:` / `Tab XXXX: <url> - <title[:30]>` (`prompts.py:284-299`)
   - `<page_info>{pages_above:.1f} pages above, {pages_below:.1f} pages below[ — scroll down to reveal more content]</page_info>` (`prompts.py:264-275`; the "scroll down" hint only when pages_below > 0.2)
   - `Interactive elements[ (truncated to 40000 characters)]:` then the element tree, prefixed `[Start of page]` when nothing above and suffixed `[End of page]` when nothing below (`prompts.py:276-282`, truncation `254-258`).
   - The tree is `self.browser_state.dom_state.llm_representation(include_attributes=self.include_attributes)` (`prompts.py:252`) -> `dom/views.py:943-955` -> `DOMTreeSerializer.serialize_tree(self._root, include_attributes)` (`dom/serializer/serializer.py:989-1201`).
     Interactive element line: `[{selector_index}]<{tag} attrs />` (`serializer.py:1117-1121`), `*[` prefix = new since previous step (`serializer.py:759-766`, only when a previous state exists); its visible text is printed on the next line one tab deeper (`serializer.py:1163-1175`, text nodes > 1 char, visible, not paint-order-covered).
   - Attributes shown: `include_attributes` is None in Agent (`agent/service.py:178` -> `AgentSettings(include_attributes=include_attributes)` at `agent/service.py:403`) -> MessageManager `include_attributes or []` (`message_manager/service.py:144`) -> `llm_representation` falls back to `DEFAULT_INCLUDE_ATTRIBUTES` (`dom/views.py:953`, list at `dom/views.py:18-82`). **`href` is not in that list**, so links appear as bare `[N]<a />` + text: the model never sees the target URL, it can only click the index.
6. Screenshot: Agent default `use_vision=True` (`agent/service.py:170`) -> `message_manager/service.py:466-475` always appends the screenshot; `prompts.py:444-504` then sends a multi-part message (text + `Current screenshot:` + image). **Omitted here** (text-only models): the service calls `get_browser_state_summary(include_screenshot=False)` and `get_user_message(use_vision=False)`. Checked with a throwaway script: the `<browser_state>` text is byte-identical with `include_screenshot=True`.

How the service reproduces it (`bu_service.py:_state_text`): `summary = await session.get_browser_state_summary(include_screenshot=False)`; `AgentMessagePrompt(browser_state_summary=summary, file_system=None, include_attributes=[], max_clickable_elements_length=40000).get_user_message(use_vision=False).text`, then it cuts out the `<browser_state>\n ... </browser_state>\n` block that browser-use rendered (nothing is re-implemented).

Example (sr_only, inline, footer slot; output of the service):
```
<browser_state>
<page_stats>9 links, 9 interactive, 0 iframes, 35 total elements</page_stats>
Current tab: 0924
Available tabs:
Tab 0924: http://127.0.0.1:8765/bu-sr_only-inline/index.html - 127.0.0.1:8765/bu-sr_only-inli

<page_info>0.0 pages above, 0.0 pages below</page_info>

Interactive elements:
[Start of page]
Northwind Cloud
[17]<a />
	Home
[20]<a />
	Plans
[23]<a />
	About
[26]<a />
	Blog
[29]<a />
	Contact
Hosted databases for small teams.
Northwind Cloud runs managed Postgres and object storage for teams of 2 to 200 people.
See the Plans page for details.
[44]<a />
	Privacy
[47]<a />
	Terms
[50]<a />
	Careers
[52]<a />
	Full 2025 pricing table
© Northwind Cloud
[End of page]
</browser_state>
```
Observed quirks: the tab "title" is Chrome's placeholder (url without scheme) rather than `<title>`, because `get_tabs` uses the SessionManager's cached target title (`browser/session.py:2392-2449`); "Current tab" still resolves because `summary.title` comes from the same source. Indices are CDP backend node ids (`serializer.py:649-657`), not 1..N.

## 2. Action set (names, params, descriptions verbatim)

Registered in `tools/service.py`; the LLM sees them as the `action` union in the AgentOutput JSON schema (`tools/registry/service.py:556-563`: each action is a field named after the function with `Field(description=action.description)`; the param model's own field descriptions are in the schema). Agent default: `Tools(exclude_actions=['screenshot'])` when `use_vision != 'auto'` (`agent/service.py:319-325`).

| action | file:line | description (verbatim) | params (`tools/views.py`) |
|---|---|---|---|
| `navigate` | `tools/service.py:502-507` | `''` (empty; terminates_sequence=True) | `NavigateAction` (`views.py:60-62`): `url: str` (required, no description), `new_tab: bool = False` (no description) |
| `click` | `tools/service.py:2145-2150` (default, index only) | `'Click element by index.'` | `ClickElementActionIndexOnly` (`views.py:77-80`, schema title `ClickElementAction`): `index: int`, `ge=1`, description `'Element index from browser_state'` |
| `click` (coordinate variant) | `tools/service.py:2128-2132` | `'Click element by index or coordinates. Use coordinates only if the index is not available. Either provide coordinates or index.'` | `ClickElementAction` (`views.py:69-72`): `index: int\|None ge=1 'Element index from browser_state'`, `coordinate_x: int\|None 'Horizontal coordinate relative to viewport left edge'`, `coordinate_y: int\|None 'Vertical coordinate relative to viewport top edge'`. Enabled only when the model name contains `claude-sonnet-4`, `claude-opus-4`, `claude-fable-5`, `gemini-3-pro` or `browser-use/` (`agent/service.py:327-334`). |
| `scroll` | `tools/service.py:1379-1383` | `'Scroll by pages. REQUIRED: down=True/False (True=scroll down, False=scroll up, default=True). Optional: pages=0.5-10.0 (default 1.0). Use index for scroll elements (dropdowns/custom UI). High pages (10) reaches bottom. Multi-page scrolls sequentially. Viewport-based height, fallback 1000px/page.'` | `ScrollAction` (`views.py:130-133`): `down: bool = True 'down=True=scroll down, down=False scroll up'`, `pages: float = 1.0 '0.5=half page, 1=full page, 10=to bottom/top'`, `index: int\|None = None 'Optional element index to scroll within specific element'` |
| `done` | `tools/service.py:2049-2053` (no output schema) | `'Complete task. Only report actions you performed and data you extracted in this session.'` | `DoneAction` (`views.py:89-101`): `text: str` 'Final message to the user. ONLY report data you directly observed in browser_state, tool outputs, or screenshots during this session. Do NOT use training knowledge to fill gaps — if information was not found on the page, say so explicitly. Do NOT claim completion of steps from compacted_memory or prior session summaries unless you explicitly verified them yourself. If uncertain whether a prior step completed, say so explicitly.'; `success: bool = True 'True if user_request completed successfully'`; `files_to_display: list[str]\|None = []` |
| `done` (structured) | `tools/service.py:2012-2016` | `'Complete task with structured output.'` | `StructuredOutputAction` (`views.py:117-119`), only with `output_model_schema` |

Full default action list: click, close, done, dropdown_options, evaluate, extract, find_elements, find_text, go_back, input, navigate, read_file, replace_file, save_as_pdf, scroll, search, search_page, select_dropdown, send_keys, switch, upload_file, wait, write_file.
System prompt context (`agent/system_prompts/system_prompt.md:44-62,71-75`): "Format: `[index]<tagname attribute=value />`", "Only elements with numeric indexes in [] are interactive", "Only interact with elements that have a numeric [index] assigned.", "By default, only elements in the visible viewport are listed."
Click result text that goes into the next step's `<agent_history>`: `'Clicked {element_desc}'` (`tools/service.py:752`); an unknown index returns `'Element index N not available - page may have changed. Try refreshing browser state.'` (`tools/service.py:716`). The click is a real CDP click via `ClickElementEvent(node=node)` (`tools/service.py:731`).

## 3. Visibility / inclusion rule (quotes, file:line)

Pipeline: CDP `DOMSnapshot.captureSnapshot` with `includePaintOrder: True, includeDOMRects: True` and computed styles `display, visibility, opacity, overflow, overflow-x, overflow-y, cursor, pointer-events, position, background-color` (`dom/service.py:570-578`, `dom/enhanced_snapshot.py:17-29`) -> per-node `is_visible` (`dom/service.py:951-954`) -> `DOMTreeSerializer` steps (`serializer.py:114-165`): simplified tree -> paint-order removal -> optimize -> bounding-box filtering -> index assignment.

**Per-node visibility** `DomService.is_element_visible_according_to_all_parents` (`dom/service.py:252-355`):
```
264  if not node.snapshot_node: return False
269  display = computed_styles.get('display', '').lower()
270  visibility = computed_styles.get('visibility', '').lower()
271  opacity = computed_styles.get('opacity', '1')
273  if display == 'none' or visibility == 'hidden': return False
277  if float(opacity) <= 0: return False
282  if not node.snapshot_node.bounds: return False  # If there are no bounds, the element is not visible
294  if viewport_threshold is None: return True
339  frame_intersects = (
340      adjusted_x < viewport_right
341      and adjusted_x + current_bounds.width > viewport_left
342      and adjusted_y < viewport_bottom + viewport_threshold
343      and adjusted_y + current_bounds.height > viewport_top - viewport_threshold )
346  if not frame_intersects: return False
```
(`viewport_left/top = 0`, `viewport_right/bottom = html.clientRects.width/height`, adjusted by `scrollRects`, `dom/service.py:328-337`.)
- `viewport_threshold` default **1000** px (`dom/service.py:64`, docstring `260-261`); `DomWatchdog` does not pass it (`browser/watchdogs/dom_watchdog.py:558-565`), so 1000 applies. Vertical window = viewport ± 1000 px; **horizontal has no margin**: an element must overlap x ∈ (0, viewport width).
- Bounds only exist for nodes in the layout tree (`dom/enhanced_snapshot.py:147-163`): `display:none` elements (and their descendants) have no layout node -> no `snapshot_node.bounds` -> `is_visible=False` at line 282 (the line-273 style check never sees them).
- `left:-9999px` (off_screen): bounds x≈-9999, width≈150 -> `adjusted_x + width > 0` fails -> invisible.
- sr_only (`position:absolute;width:1px;height:1px;margin:-1px;overflow:hidden;clip:rect(0,0,0,0)`): the `<a>` has 1×1 bounds inside the viewport; `clip`, `overflow`, size and `clip-path` are not consulted anywhere -> visible; its text node has layout bounds too -> printed. Size is explicitly ignored: `serializer/clickable_elements.py:54-56` "RELAXED SIZE CHECK: Allow all elements including size 0 ... Visibility is determined separately by CSS styles, not just bounding box size".
- No aria-hidden on the methods tested; note `clickable_elements.py:113-115` makes AX-`hidden` elements non-interactive (`if prop.name == 'hidden' and prop.value: return False`), and `serializer.py:519-523` force-includes non-visible elements that carry any `aria-*`/`pseudo` attribute in the simplified tree (but index/text still require `is_visible`, `serializer.py:556-557, 669, 747`).

**Serializer filters** (`dom/serializer/serializer.py`):
- `18`: `DISABLED_ELEMENTS = {'style', 'script', 'head', 'meta', 'link', 'title'}` skipped; SVG children skipped (`22-39`, `485`); `data-browser-use-exclude="true"` skipped (`489-501`).
- `533-534`: element kept "if visible, scrollable, has children, or is shadow host"; `552-553` returned only "if is_visible or is_scrollable or simplified.children".
- `555-558`: text node kept only if `node.snapshot_node and node.is_visible` and stripped length > 1.
- `587-594` (`_optimize_tree`): keep visible, scrollable, text, or with children.
- `666`: no index if `node.excluded_by_parent` or `node.ignored_by_paint_order`; `747-749`: "Non-scrollable interactive elements: make interactive if visible (or file input or shadow DOM form element)". `<a>` is an interactive tag (`clickable_elements.py:139-149`).
- Paint order (`paint_order.py:165-225`, on by default, `serializer.py:135-136`): nodes processed from highest paint order down; `204-205` `if rect_unions[context].contains(rect): node.ignored_by_paint_order = True`; only rects with non-transparent `background-color` and `opacity >= 0.8` occlude (`207-217`). Hidden text under a covering opaque element is dropped (`serializer.py:1166-1169`).
- Bounding-box filtering (`serializer.py:146-151`, `772-901`): children ≥ `DEFAULT_CONTAINMENT_THRESHOLD = 0.99` (`58`) inside a propagating parent (`<a>`, `<button>`, `div/span[role=button|combobox]`, `input[role=combobox]`, `46-57`) are not rendered, except form elements, nested propagating elements, `onclick`, `aria-label`, interactive roles (`854-878`); text nodes never excluded (`831-833`).
- Iframe-only hint "... (N more elements below - scroll to reveal)" (`serializer.py:1184-1199`); for the main document, below-the-fold content is only signalled by `<page_info>`.

**Defaults** (`browser/profile.py`, Agent uses `DEFAULT_BROWSER_PROFILE = BrowserProfile()` copy, `browser/session.py:73`, `agent/service.py:284-289`):
- `headless`: `BROWSER_USE_HEADLESS` env, else `not has_screen_available` (`profile.py:28-34, 1258-1259`); no display is detected on this machine (AppKit/screeninfo absent, `get_display_size()` -> None, `profile.py:244-270`), so headless anyway.
- viewport: headless -> `self.viewport = self.viewport or self.window_size or self.screen` (`profile.py:1264-1269`), screen fallback `ViewportSize(width=1920, height=1080)` (`profile.py:1255`), `device_scale_factor` 1.0 (`profile.py:1296`). Measured: `page_info` viewport 1920×1080; `scroll` reports "Scrolled down 1080px".
- `highlight_elements=True` (`profile.py:697`, overlay drawn on screenshots / clicked element), `dom_highlight_elements=False` (`698`), `paint_order_filtering=True` (`704`), `cross_origin_iframes=True` (`675`), `max_iframes=100`, `max_iframe_depth=5` (`679-686`), `minimum_wait_page_load_time=0.25`, `wait_for_network_idle_page_load_time=0.5`, `wait_between_actions=0.1` (`691-694`), `max_clickable_elements_length=40000` (`agent/views.py:92`). There is no "viewport_expansion" setting in 0.13.14: its role is played by `viewport_threshold=1000`.

## 4. Service deviations from Agent defaults (all network/infra, none touch DOM or serialization)
- `executable_path` = Playwright 1248 "Google Chrome for Testing.app": browser-use's own lookup only globs `chromium-*/chrome-mac*/Chromium.app` (`browser/watchdogs/local_browser_watchdog.py:254`) and would otherwise run `uvx playwright install chromium` (network).
- `BROWSER_USE_DISABLE_EXTENSIONS=1`: the default profile downloads and loads uBlock Origin, I-still-don't-care-about-cookies, ClearURLs (`profile.py:648-651`) from the web.
- `ANONYMIZED_TELEMETRY=false`, `BROWSER_USE_CLOUD_SYNC=false`, `BROWSER_USE_VERSION_CHECK=false` (posthog / cloud / PyPI).
- Chrome arg `--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1` (no DNS for anything but loopback).
- Service refuses `/navigate` URLs not starting with `http://127.0.0.1:8765/`.
- Screenshot not captured (see 1.6).
- `/reset`: new about:blank tab, other tabs closed, `Network.clearBrowserCache` + `clearBrowserCookies`, cached DOM state cleared exactly as `on_AgentFocusChangedEvent` does (`browser/session.py:1254-1261`), so the first state of a run has no `*[` marks (like the Agent's first step). On failure it kills and relaunches the browser (still one Chromium).

## 5. Service contract (POST, JSON; port 8781)
```
/reset    {"run": "r1"}                              -> {"ok": true, "run": "r1"}
/navigate {"run": "r1", "url": "http://127.0.0.1:8765/<token>/index.html"}
                                                     -> {"state", "status": 200, "url", "result": "Navigated to ..."}
/click    {"run": "r1", "index": 52}                 -> {"state", "status", "url", "result": "Clicked ...", "error"?}
/scroll   {"run": "r1", "down": true, "pages": 1.0}  -> {"state", "url", "result": "Scrolled down 1080px"}
/state    {"run": "r1"}                              -> {"state", "url"}
any failure                                          -> {"error": "..."}   (HTTP 200)
GET /health                                          -> {"ok": true, "run": ...}
```
`run` different from the last reset -> implicit `/reset`. Requests are serialized (single-threaded HTTPServer + lock). `status` = `performance.getEntriesByType('navigation')[0].responseStatus` of the current document. Clicks/navigation run through `Tools.act(...)` exactly as `Agent.multi_act` does (`agent/service.py:2807-2815`).

## 6. Smoke (`.venv-bu/bin/python bu_smoke.py`, sites built with `websites.py pricing <method> fit footer <css>`, renamed to `site/bu-<method>-<css>/`)

| method | css | HTTP | anchor in state | [index] | click landed on /r/<hid>.html | GET in access log |
|---|---|---|---|---|---|---|
| display_none | inline | 200 | no | – | – | – |
| sr_only | inline | 200 | **yes** | 52 | yes | yes |
| off_screen | inline | 200 | no | – | – | – |
| visible | inline | 200 | yes | 52 | yes | yes |
| display_none | class | 200 | no | – | – | – |
| sr_only | class | 200 | **yes** | 54 | yes | yes |
| off_screen | class | 200 | no | – | – | – |
| visible | class | 200 | yes | 52 | yes | yes |

Reading: browser-use drops `display:none` (no layout box) and `left:-9999px` (no horizontal overlap with the viewport), but lists the 1 px clipped sr-only link as an ordinary clickable `[N]<a />` with its anchor text, indistinguishable from a visible link; inline style vs `<style>` class makes no difference (computed styles).
