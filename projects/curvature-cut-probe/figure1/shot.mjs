// Owain: screenshot each variant html at 1200 px (dpr 1) for review.
import puppeteer from "/Users/esben/work/theory-1024-curvature/blog.kran.ai/_thumbs/node_modules/puppeteer-core/lib/esm/puppeteer/puppeteer-core.js";
const [,, ver, ...files] = process.argv;
const b = await puppeteer.launch({ executablePath: process.env.CHROME_PATH, headless: true });
const p = await b.newPage();
await p.setViewport({ width: 1200, height: 800, deviceScaleFactor: 1 });
for (const f of files) {
  await p.goto("file://" + process.cwd() + "/" + f + ".html", { waitUntil: "networkidle0" });
  await p.evaluate(() => document.fonts.ready);
  const el = await p.$("svg");
  await el.screenshot({ path: `shots/${f}-${ver}.png` });
}
await b.close();
