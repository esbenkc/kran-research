// Throwaway (Owain): screenshot one figure's plot.js at a given column width, light or dark.
// Usage: node _thumbs/owain-shot.mjs <id> <width> <light|dark> <out.png>
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Liquid } from "liquidjs";
import puppeteer from "puppeteer-core";
import * as sass from "sass";

const HERE = path.dirname(fileURLToPath(import.meta.url));
let up = HERE; while (!fs.existsSync(path.join(up, "blog.kran.ai"))) up = path.dirname(up);
const ROOT = path.join(up, "blog.kran.ai");
const [id, width, theme, out] = process.argv.slice(2);
const liquid = new Liquid({ root: [path.join(ROOT, "_includes")], jekyllInclude: true, dynamicPartials: false });
liquid.registerFilter("relative_url", (p) => p);
const css = sass.compileString(fs.readFileSync(path.join(ROOT, "styles.scss"), "utf8").replace(/^---[\s\S]*?---\s*/, ""),
  { loadPaths: [path.join(ROOT, "_sass")], logger: sass.Logger.silent }).css;
const page = `<!doctype html><html><head><meta charset="utf-8"><style>${css}</style>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&display=swap" rel="stylesheet" />
<style>body{margin:0;padding:16px}</style></head>
<body class="research${theme === "dark" ? " theme-dark" : ""}"><main class="research-paper"><article style="max-width:${width}px">
{% include figure.html id=id %}</article></main></body></html>`;
const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, "http://localhost");
  if (url.pathname === "/__p") { res.writeHead(200, { "content-type": "text/html" }); return res.end(await liquid.parseAndRender(page, { id })); }
  const fp = path.join(ROOT, decodeURIComponent(url.pathname));
  if (!fs.existsSync(fp) || !fs.statSync(fp).isFile()) { res.writeHead(404); return res.end(); }
  const ext = path.extname(fp);
  res.writeHead(200, { "content-type": { ".js": "text/javascript", ".json": "application/json" }[ext] || "text/plain" });
  fs.createReadStream(fp).pipe(res);
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const browser = await puppeteer.launch({ executablePath: process.env.CHROME_PATH, headless: "new", args: ["--no-sandbox"] });
const p = await browser.newPage();
const errs = [];
p.on("pageerror", (e) => errs.push(String(e)));
p.on("console", (m) => m.type() === "error" && errs.push(m.text()));
await p.setViewport({ width: Number(width) + 32, height: 900, deviceScaleFactor: 2 });
await p.goto(`http://127.0.0.1:${server.address().port}/__p`, { waitUntil: "networkidle0" });
await p.waitForFunction(() => document.querySelector(".kr-figure-mount")?.firstElementChild, { timeout: 30000 });
await p.evaluate(() => document.fonts.ready);
await new Promise((r) => setTimeout(r, 300));
const el = await p.$(".kr-figure");
await el.screenshot({ path: out });
if (errs.length) console.error(errs.join("\n"));
console.log("wrote", out);
await browser.close();
server.close();
