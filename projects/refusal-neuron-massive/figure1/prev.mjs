// node prev.mjs URL out_prefix  -- screenshot the post's top (Figure 1 + TLDR) in light and dark, 1280 and 390 wide
import { createRequire } from "node:module";
const require = createRequire("/Users/esben/work/theory-1031-refusal-neuron/blog.kran.ai/_thumbs/package.json");
const puppeteer = require("puppeteer-core");
import { execSync } from "node:child_process";
const [url, out] = process.argv.slice(2);
const chrome = execSync('ls -d ~/Library/Caches/ms-playwright/chromium-*/chrome-mac*/*.app/Contents/MacOS/* | head -1', { shell: "/bin/zsh" }).toString().trim();
const b = await puppeteer.launch({ executablePath: chrome, headless: "new", args: ["--no-sandbox"] });
for (const [w, theme] of [[1280, "light"], [1280, "dark"], [390, "light"]]) {
  const p = await b.newPage();
  await p.setViewport({ width: w, height: 1400, deviceScaleFactor: 1 });
  await p.goto(url, { waitUntil: "networkidle0" });
  await p.evaluate((t) => localStorage.setItem("theme", t), theme);
  await p.reload({ waitUntil: "networkidle0" });
  const info = await p.evaluate(() => {
    const img = [...document.images].find((i) => /fig1\/figure1\.png/.test(i.currentSrc || i.src));
    if (!img) return { found: false };
    img.scrollIntoView({ block: "start" });
    const r = img.getBoundingClientRect();
    return { found: true, w: r.width, h: r.height, natural: img.naturalWidth, src: img.currentSrc || img.src,
      og: document.querySelector('meta[property="og:image"]')?.content, alt: img.alt.slice(0, 120) };
  });
  console.log(w, theme, JSON.stringify(info));
  await p.screenshot({ path: `${out}_${theme}_${w}.png` });
  await p.close();
}
await b.close();
