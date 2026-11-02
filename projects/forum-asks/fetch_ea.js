#!/usr/bin/env bun
// Fetch every EA Forum post and comment through its public GraphQL API, from inside a headless
// browser page. The forum sits behind a Vercel bot check that blocks plain HTTP clients, so the
// queries run as same-origin fetch() calls on forum.effectivealtruism.org, over the Chrome
// DevTools Protocol (no npm dependencies). Raw pages are cached exactly like fetch.py does.
//
//   CHROME="/path/to/Chrome" bun fetch_ea.js posts          # cursor on postedAt, newest first
//   CHROME="/path/to/Chrome" bun fetch_ea.js comments       # per post: the comment views ignore dates
//   CHROME="/path/to/Chrome" bun fetch_ea.js posts_html     # full post bodies (plain text stops at 2,000 chars)
//   CHROME="/path/to/Chrome" bun fetch_ea.js comments_html  # full bodies of the cut comments
//
// The EA Forum comment views ignore `before`/`after` and cap offsets, so comments come per post
// (view postCommentsOld), 25 posts per request through GraphQL aliases, for every cached post
// with commentCount > 0.
import { mkdirSync, existsSync, readdirSync, readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { spawn } from "node:child_process";

const RAW = join(import.meta.dir, "data", "raw", "ea");
const SLEEP = 1000;
const CHROME = process.env.CHROME;
if (!CHROME) throw new Error("set CHROME to a Chrome/Chromium executable");
const kind = process.argv[2];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const FIELDS = {
  comments: "_id postedAt baseScore af deleted author userId user { displayName slug } pageUrl postId parentCommentId contents { plaintextMainText }",
  posts: "_id title slug postedAt baseScore af question isEvent commentCount author userId user { displayName slug } pageUrl wordCount contents { plaintextMainText }",
  posts_html: "_id postedAt htmlBody",
};

// ── minimal CDP client ──
const profile = mkdtempSync(join(tmpdir(), "ea-fetch-"));
const chrome = spawn(CHROME, ["--headless=new", "--remote-debugging-port=0", `--user-data-dir=${profile}`, "--no-first-run", "about:blank"], { stdio: "ignore" });
process.on("exit", () => chrome.kill());
let port;
for (let i = 0; i < 100 && !port; i++) {
  await sleep(200);
  const f = join(profile, "DevToolsActivePort");
  if (existsSync(f)) port = readFileSync(f, "utf8").split("\n")[0];
}
const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
const ws = new WebSocket(targets.find((t) => t.type === "page").webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
let nextId = 0;
const pending = new Map();
ws.onmessage = (m) => {
  const msg = JSON.parse(m.data);
  if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); }
};
const cdp = (method, params = {}) =>
  new Promise((r) => { const id = ++nextId; pending.set(id, r); ws.send(JSON.stringify({ id, method, params })); });

await cdp("Page.enable");
await cdp("Page.navigate", { url: "https://forum.effectivealtruism.org/" });
await sleep(15000); // let the bot check and the app load

async function gql(query) {
  for (let attempt = 0; attempt < 6; attempt++) {
    const expr = `fetch('/graphql',{method:'POST',headers:{'content-type':'application/json'},body:${JSON.stringify(JSON.stringify({ query }))}}).then(r=>r.text())`;
    const res = await cdp("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true });
    const text = res.result?.result?.value;
    try { return JSON.parse(text); } catch {
      const wait = 10000 * 2 ** attempt;
      console.log(`  retry ${attempt + 1} in ${wait / 1000}s: ${String(text ?? JSON.stringify(res)).slice(0, 120)}`);
      await sleep(wait);
      if (attempt >= 1) { await cdp("Page.navigate", { url: "https://forum.effectivealtruism.org/" }); await sleep(15000); }
    }
  }
  throw new Error("giving up after 6 retries");
}

const cursorAfter = (ts) => new Date(new Date(ts).getTime() + 1).toISOString();

if (kind === "posts" || kind === "posts_html") {
  const d = join(RAW, kind);
  mkdirSync(d, { recursive: true });
  const pages = readdirSync(d).filter((f) => f.endsWith(".json")).sort();
  let before = "2100-01-01T00:00:00.000Z";
  if (pages.length) {
    const rows = JSON.parse(readFileSync(join(d, pages[0]), "utf8")).rows;
    if (!rows.length) { console.log("already complete"); process.exit(0); }
    before = cursorAfter(rows.map((r) => r.postedAt).sort()[0]);
  }
  let n = 0;
  while (true) {
    const res = await gql(`{ posts(input:{terms:{view:"new", before:"${before}", limit:300}}) { results { ${FIELDS[kind]} } } }`);
    const rows = res?.data?.posts?.results;
    if (!rows) throw new Error("bad response: " + JSON.stringify(res).slice(0, 500));
    writeFileSync(join(d, before.replace(/[:.]/g, "") + ".json"), JSON.stringify({ before, errors: (res.errors || []).length, rows }));
    n += rows.length;
    if (!rows.length || (rows.length === 1 && cursorAfter(rows[0].postedAt) === before)) { console.log(`done: ${n} rows`); break; }
    const oldest = rows.map((r) => r.postedAt).sort()[0];
    console.log(`ea/${kind} ${rows.length} rows down to ${oldest} (total ${n})`);
    before = cursorAfter(oldest);
    await sleep(SLEEP);
  }
} else if (kind === "comments") {
  const posts = new Map();
  for (const f of readdirSync(join(RAW, "posts")).filter((f) => f.endsWith(".json")))
    for (const p of JSON.parse(readFileSync(join(RAW, "posts", f), "utf8")).rows) if (p.commentCount > 0) posts.set(p._id, p.commentCount);
  const ids = [...posts.keys()].sort();
  const d = join(RAW, "comments");
  mkdirSync(d, { recursive: true });
  const CHUNK = 25;
  let n = 0;
  for (let i = 0; i < ids.length; i += CHUNK) {
    const file = join(d, `chunk-${String(i / CHUNK).padStart(5, "0")}.json`);
    if (existsSync(file)) continue;
    const chunk = ids.slice(i, i + CHUNK);
    const q = chunk.map((id, j) => `p${j}: comments(input:{terms:{view:"postCommentsOld", postId:"${id}", limit:5000}}) { results { ${FIELDS.comments} } }`).join(" ");
    const res = await gql(`{ ${q} }`);
    if (!res?.data) throw new Error("bad response: " + JSON.stringify(res).slice(0, 500));
    const rows = chunk.flatMap((_, j) => res.data[`p${j}`]?.results || []);
    writeFileSync(file, JSON.stringify({ posts: chunk, errors: (res.errors || []).length, rows }));
    n += rows.length;
    console.log(`ea/comments chunk ${i / CHUNK + 1}/${Math.ceil(ids.length / CHUNK)}: ${rows.length} rows (total ${n})`);
    await sleep(SLEEP);
  }
  console.log(`done: ${n} rows this run`);
} else if (kind === "comments_html") {
  // Full bodies of the comments whose plain text hit the API's 2,000-character cut, 100 per request.
  const ids = new Set();
  for (const f of readdirSync(join(RAW, "comments")).filter((f) => f.endsWith(".json")))
    for (const c of JSON.parse(readFileSync(join(RAW, "comments", f), "utf8")).rows)
      if ((c.contents?.plaintextMainText || "").length >= 2000) ids.add(c._id);
  const all = [...ids].sort();
  const d = join(RAW, "comments_html");
  mkdirSync(d, { recursive: true });
  console.log(`${all.length} truncated comments`);
  for (let i = 0; i < all.length; i += 100) {
    const file = join(d, `chunk-${String(i / 100).padStart(5, "0")}.json`);
    if (existsSync(file)) continue;
    const part = all.slice(i, i + 100);
    const q = part.map((id, j) => `c${j}: comment(input:{selector:{_id:"${id}"}}) { result { _id htmlBody } }`).join(" ");
    const res = await gql(`{ ${q} }`);
    if (!res?.data) throw new Error("bad response: " + JSON.stringify(res).slice(0, 500));
    const rows = Object.values(res.data).filter((v) => v?.result).map((v) => v.result);
    writeFileSync(file, JSON.stringify({ ids: part, errors: (res.errors || []).length, rows }));
    console.log(`ea/comments_html chunk ${i / 100 + 1}/${Math.ceil(all.length / 100)}: ${rows.length} rows`);
    await sleep(SLEEP);
  }
}
ws.close();
process.exit(0);
