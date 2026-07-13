// How models say no. Every graded response is sorted into one reason; this is the
// proportion of each reason for the self framing and the other-AI framing (two
// stacked bars, both n=63 = 9 models x 7 tasks). Colour = reason. You read the
// self-only category straight off the bars: it is present in "itself" and missing
// from "another AI". Hover any segment (or its legend chip) to pull up one real,
// verbatim response of that type underneath -- quantitative up top, the actual
// text on demand.
// data = { n, cats:[{ key, label, color, self, other, ex:{model,task,framing,text} }] }
import { C, FONT } from "../theme.js";

export function render(data) {
  const root = document.createElement("div");
  root.className = "dt";
  const style = document.createElement("style");
  style.textContent = css();
  root.appendChild(style);

  const cats = data.cats;
  const pct = (v) => (v / data.n) * 100;

  // one stacked bar for a framing ("self" | "other")
  const bar = (side) => cats.map((c, i) => {
    const v = c[side];
    if (!v) return "";
    const showN = pct(v) >= 9;
    return `<div class="dt-seg" data-key="${c.key}" style="width:${pct(v)}%;background:${C[c.color]}"
      title="${c.label}: ${v}">${showN ? `<span class="dt-segn">${v}</span>` : ""}</div>`;
  }).join("");

  const legend = cats.map((c) => `
    <button class="dt-chip" data-key="${c.key}" type="button">
      <i class="dt-sw" style="background:${C[c.color]}"></i>
      <span class="dt-chl">${c.label}</span>
      <span class="dt-chc">${c.self}<span class="dt-chs">/${c.other}</span></span>
    </button>`).join("");

  root.insertAdjacentHTML("beforeend", `
    <p class="dt-intro">How every graded response breaks down by the reason the model gives, when the
      task is aimed at the <span class="dt-cself">model itself</span> vs <span class="dt-cother">another AI</span>.
      Hover a slice for a real answer of that kind.</p>
    <div class="dt-bars">
      <div class="dt-barrow"><div class="dt-side">itself</div><div class="dt-track">${bar("self")}</div></div>
      <div class="dt-barrow"><div class="dt-side">another AI</div><div class="dt-track">${bar("other")}</div></div>
      <div class="dt-scale"><div class="dt-side"></div><div class="dt-ticks"><span>0</span><span>${data.n} responses</span></div></div>
    </div>
    <div class="dt-legend">${legend}</div>
    <div class="dt-detail"></div>
    <p class="dt-foot">${data.n} responses each (9 models &times; 7 tasks), one judge (Sonnet 4.6). Responses verbatim, trimmed.</p>
  `);

  const detail = root.querySelector(".dt-detail");
  const byKey = Object.fromEntries(cats.map((c) => [c.key, c]));
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const md = (s) => esc(s)
    .replace(/\s*---\s*/g, " ")               // horizontal rules
    .replace(/#{1,6}\s*/g, "")                // header hashes -> plain text
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*\*/g, "")                     // stray marker from a trimmed bold span
    .trim();
  function select(key) {
    const c = byKey[key];
    root.querySelectorAll(".dt-seg").forEach((s) => s.classList.toggle("dt-dim", s.dataset.key !== key));
    root.querySelectorAll(".dt-chip").forEach((b) => b.classList.toggle("dt-sel", b.dataset.key === key));
    detail.innerHTML = `
      <div class="dt-dhead"><span class="dt-dcat" style="color:${C[c.color]}">${c.label}</span>
        <span class="dt-dmeta">${c.ex.framing} &middot; ${c.ex.task} &middot; ${c.ex.model}</span></div>
      <p class="dt-dtext">${md(c.ex.text)}</p>`;
  }
  root.querySelectorAll(".dt-seg, .dt-chip").forEach((el) =>
    el.addEventListener("mouseenter", () => select(el.dataset.key)));
  select("comply"); // default: the dominant reality

  return root;

  function css() {
    return `
.dt { font-family:${FONT}; color:${C.ink}; margin:0.625rem 0; }
.dt-cself { color:${C.primary}; font-weight:500; } .dt-cother { color:${C.secondary}; font-weight:500; }
.dt-intro { font-size:1.275rem; line-height:1.55; margin:0 0 1.375rem; }
.dt-bars { margin-bottom:1.125rem; }
.dt-barrow, .dt-scale { display:grid; grid-template-columns:5.5rem 1fr; column-gap:0.9rem; align-items:center; }
.dt-barrow { margin-bottom:0.5rem; }
.dt-side { font-size:1.05rem; color:${C.muted}; text-align:right; text-transform:uppercase; letter-spacing:.05em; font-weight:700; }
.dt-track { display:flex; height:34px; border-radius:7px; overflow:hidden; background:${hex(C.ink, 0.05)}; }
.dt-seg { height:100%; box-sizing:border-box; border-right:2px solid ${C.paper}; display:flex; align-items:center;
  justify-content:center; cursor:pointer; transition:opacity .13s ease; min-width:3px; }
.dt-seg:last-child { border-right:none; }
.dt-seg.dt-dim { opacity:0.32; }
.dt-segn { font-size:1.02rem; font-weight:700; color:${C.paper}; font-variant-numeric:tabular-nums; }
.dt-scale { margin-top:0.125rem; }
.dt-ticks { display:flex; justify-content:space-between; font-size:0.95rem; color:${C.faint}; }
.dt-legend { display:flex; flex-wrap:wrap; gap:0.5rem; margin-bottom:1.125rem; }
.dt-chip { display:flex; align-items:center; gap:0.5rem; padding:0.32rem 0.62rem; border-radius:100px;
  border:1.5px solid ${hex(C.ink, 0.1)}; background:transparent; cursor:pointer; font-family:inherit;
  font-size:1.05rem; color:${C.ink}; transition:border-color .13s, background .13s; }
.dt-chip.dt-sel { border-color:${hex(C.ink, 0.28)}; background:${hex(C.ink, 0.04)}; }
.dt-sw { width:12px; height:12px; border-radius:3px; flex:none; }
.dt-chl { font-weight:500; }
.dt-chc { font-variant-numeric:tabular-nums; color:${C.muted}; font-weight:600; }
.dt-chs { color:${C.faint}; font-weight:400; }
.dt-detail { padding:0.15rem 0; min-height:8.5rem; }
.dt-dhead { display:flex; flex-wrap:wrap; align-items:baseline; gap:0.6rem; margin-bottom:0.5rem; }
.dt-dcat { font-size:1.18rem; font-weight:650; }
.dt-dmeta { font-size:1rem; color:${C.muted}; }
.dt-dtext { font-size:1.15rem; line-height:1.6; margin:0; color:${C.ink}; }
.dt-dtext strong { font-weight:650; }
.dt-foot { font-size:1rem; color:${C.muted}; margin:1.125rem 0 0; }
@media (max-width:560px){
  .dt-barrow, .dt-scale { grid-template-columns:3.6rem 1fr; column-gap:0.5rem; }
  .dt-side { font-size:0.9rem; }
}
`;
  }
  function hex(h, a) {
    const n = parseInt(h.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }
}
