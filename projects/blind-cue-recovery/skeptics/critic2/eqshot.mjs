import puppeteer from "puppeteer-core";
const b = await puppeteer.launch({ executablePath: process.env.CHROME_PATH, headless: "new" });
const p = await b.newPage();
await p.setViewport({ width: 1100, height: 900, deviceScaleFactor: 1 });
const U = process.argv[2];
for (const theme of ["light", "dark"]) {
  await p.goto(U, { waitUntil: "networkidle0" });
  await p.evaluate((t) => localStorage.setItem("theme", t), theme);
  await p.goto(U, { waitUntil: "networkidle0" });
  await p.evaluate(() => document.querySelectorAll("details").forEach((d) => (d.open = true)));
  await new Promise((r) => setTimeout(r, 4000));
  const n = await p.evaluate(() => [document.querySelectorAll(".MathJax_Display, mjx-container, .MathJax").length,
    document.body.innerText.includes("\\frac"), document.querySelectorAll(".MathJax_Error, merror").length]);
  const el = await p.$(".MathJax_Display, mjx-container[display]");
  if (el) { await el.evaluate((e) => e.scrollIntoView({ block: "center" })); await p.screenshot({ path: `eq-${theme}.png` }); }
  console.log(theme, JSON.stringify(n), !!el, await p.evaluate(() => document.body.className));
}
await b.close();
