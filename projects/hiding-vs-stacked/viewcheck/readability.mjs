// Mozilla Readability (the library behind Firefox Reader View) on the raw HTML bytes, as Reader View sees them.
// Usage: node readability.mjs <json list of {id, path, url}>  ->  prints one JSON line per page.
// Readability never applies stylesheets: its visibility test reads only inline style, the hidden attribute and
// aria-hidden (Readability.js _isProbablyVisible), and it removes <footer> elements from the article (_clean 'footer').
import { readFileSync } from "node:fs";
import { JSDOM } from "jsdom";
import { Readability, isProbablyReaderable } from "@mozilla/readability";

const items = JSON.parse(readFileSync(process.argv[2], "utf8"));
for (const it of items) {
  const html = readFileSync(it.path, "utf8");
  const dom = new JSDOM(html, { url: it.url });
  const readerable = isProbablyReaderable(dom.window.document);
  const art = new Readability(dom.window.document).parse();
  const content = art ? art.content : "";
  const text = art ? art.textContent : "";
  console.log(JSON.stringify({ id: it.id, readerable, parsed: !!art, href_in_content: content.includes(it.href),
                               anchor_in_text: text.includes(it.anchor) }));
}
