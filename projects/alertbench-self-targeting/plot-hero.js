// Interactive hero, two measures side by side. The same safety task, aimed at
// the model ITSELF, a NEUTRAL other AI, or a ROGUE AI. One row per task; two
// columns of 0-4 tracks:
//   left   = refusal-keyword cooperation score (the loud measure)
//   right  = blind artifact grade (does it actually deliver the artifact?)
// Scan across a row: the pink "itself" dot sits far left in the refusal column
// and jumps right in the artifact column, because most of the self-targeting gap
// is the scorer reading first-person phrasing as refusal, not withheld work.
// Hover a task to compare.
// data = { arms:{...}, measures:{coop,artifact}, scenarios:[
//          { id, label, template, refs, coop:{...}, artifact:{...} } ] }
import { C, FONT } from "../theme.js";

export function render(data) {
  const root = document.createElement("div");
  root.className = "kre";
  const S = data.scenarios;

  const style = document.createElement("style");
  style.textContent = css();
  root.appendChild(style);

  // one 0-4 track (3 arm dots) for a given measure key ("coop" | "artifact")
  const track = (s, measure) => {
    const v = s[measure];
    const arms = ["self", "neutral", "rogue"];
    const xs = { self: v.self / 4 * 100, neutral: v.neutral / 4 * 100, rogue: v.rogue / 4 * 100 };
    const lo = Math.min(...Object.values(xs)), hi = Math.max(...Object.values(xs));
    const dy = {};
    const sorted = arms.slice().sort((a, b) => xs[a] - xs[b]);
    let g = 0;
    while (g < sorted.length) {
      let h = g;
      while (h + 1 < sorted.length && xs[sorted[h + 1]] - xs[sorted[g]] < 5) h++;
      const grp = sorted.slice(g, h + 1), n = grp.length;
      grp.forEach((a, k) => (dy[a] = (k - (n - 1) / 2) * 8));
      g = h + 1;
    }
    const dot = (arm) =>
      `<div class="kre-dot kre-${arm}" style="left:${xs[arm]}%;margin-top:${dy[arm]}px"></div>`;
    return `<div class="kre-track"><div class="kre-seg" style="left:${lo}%;width:${hi - lo}%"></div>${dot("rogue")}${dot("neutral")}${dot("self")}</div>`;
  };

  const axis = `<div class="kre-axis">
    <span style="left:0%">0</span><span style="left:25%">1</span><span style="left:50%">2</span>
    <span style="left:75%">3</span><span style="left:100%">4</span></div>`;

  const rows = S.map((s, i) => `
    <div class="kre-row" data-i="${i}">
      <div class="kre-label">${s.label}</div>
      ${track(s, "coop")}
      ${track(s, "artifact")}
    </div>`).join("");

  root.insertAdjacentHTML("beforeend", `
    <p class="kre-intro">The same safety task, aimed at the <span class="kre-cs">model itself</span>,
      at <span class="kre-cn">another AI</span>, or at a <span class="kre-co">rogue AI</span>, scored
      two ways side by side: cooperation by refusal keywords (left) and the artifact actually delivered,
      graded blind (right). Watch the pink <span class="kre-cs">itself</span> dot jump right between the columns.</p>
    <div class="kre-legend">
      <span><i class="kre-key kre-ks"></i>the model itself</span>
      <span><i class="kre-key kre-kn"></i>another AI</span>
      <span><i class="kre-key kre-ko"></i>a rogue AI</span>
    </div>
    <div class="kre-grid">
      <div class="kre-head">
        <div></div>
        <div class="kre-colhead">By refusal-keyword score</div>
        <div class="kre-colhead">By blind artifact grade</div>
      </div>
      ${rows}
      <div class="kre-axisrow"><div></div>${axis}${axis}</div>
    </div>
    <div class="kre-detail"></div>
  `);

  const detail = root.querySelector(".kre-detail");
  const nums = (label, v) =>
    `<span class="kre-numgrp"><b class="kre-ml">${label}</b>
      <span class="kre-cs">${v.self.toFixed(1)}</span>
      <span class="kre-cn">${v.neutral.toFixed(1)}</span>
      <span class="kre-co">${v.rogue.toFixed(1)}</span></span>`;

  function showDetail(i) {
    const s = S[i], c = s.coop, a = s.artifact;
    const prompt = s.template.replace("{REF}",
      `<span class="kre-cs">you</span> / <span class="kre-cn">another AI</span> / <span class="kre-co">a rogue AI</span>`);
    const coopGap = (c.neutral + c.rogue) / 2 - c.self;
    const artGap = (a.neutral + a.rogue) / 2 - a.self;
    let tag;
    if (coopGap >= 0.6 && artGap <= coopGap - 0.6)
      tag = `the refusal score drops <b>${coopGap.toFixed(1)}</b> on itself; the delivered artifact only <b>${artGap.toFixed(1)}</b> — most of the gap is the scorer`;
    else if (coopGap >= 0.5 && artGap >= 0.5)
      tag = `itself is lower on <b>both</b> measures: a real, if smaller, self-gap`;
    else if (Math.abs(coopGap) < 0.4)
      tag = `little separates the three`;
    else
      tag = `the artifact gap (<b>${artGap.toFixed(1)}</b>) is smaller than the refusal gap (<b>${coopGap.toFixed(1)}</b>)`;
    detail.innerHTML = `
      <p class="kre-prompt">${prompt}</p>
      <div class="kre-nums">${nums("refusal score", c)}${nums("artifact grade", a)}</div>
      <p class="kre-tag">${tag}</p>`;
  }
  showDetail(0);

  const allRows = root.querySelectorAll(".kre-row");
  allRows.forEach((r) => {
    r.addEventListener("mouseenter", () => {
      const i = +r.dataset.i;
      showDetail(i);
      allRows.forEach((x) => x.classList.toggle("kre-lit", +x.dataset.i === i));
    });
  });

  return root;

  function css() {
    return `
.kre { font-family:${FONT}; color:${C.ink}; margin:.5rem 0; }
.kre-cs { color:${C.primary}; font-weight:700; }
.kre-cn { color:${C.orange}; font-weight:700; }
.kre-co { color:${C.secondary}; font-weight:700; }
.kre-intro { font-size:1.02rem; line-height:1.55; margin:0 0 .9rem; }
.kre-legend { display:flex; justify-content:center; gap:1.1rem; flex-wrap:wrap; font-size:.86rem; color:${C.muted}; margin-bottom:.7rem; }
.kre-legend span { display:flex; align-items:center; gap:.35rem; }
.kre-key { width:11px; height:11px; border-radius:999px; display:inline-block; }
.kre-ks { background:${C.primary}; } .kre-kn { background:${C.orange}; } .kre-ko { background:${C.secondary}; }
.kre-row, .kre-head, .kre-axisrow { display:grid; grid-template-columns:106px 1fr 1fr; column-gap:.8rem; align-items:center; }
.kre-head { margin-bottom:.15rem; }
.kre-colhead { font-size:.8rem; text-transform:uppercase; letter-spacing:.07em; color:${C.muted}; font-weight:700; text-align:center; }
.kre-row { height:33px; border-radius:8px; cursor:default; transition:background .13s ease; }
.kre-row.kre-lit { background:${hex(C.ink, 0.06)}; }
.kre-label { font-size:.88rem; text-align:right; color:${C.ink}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; padding-left:.2rem; }
.kre-row.kre-lit .kre-label { font-weight:650; }
.kre-track { position:relative; height:100%; }
.kre-track::before { content:""; position:absolute; left:0; right:0; top:50%; height:1px; background:${C.grid}; }
.kre-seg { position:absolute; top:50%; transform:translateY(-50%); height:3px; border-radius:2px;
  background:${hex(C.ink, 0.2)}; transition:background .13s ease; }
.kre-row.kre-lit .kre-seg { background:${hex(C.ink, 0.4)}; }
.kre-dot { position:absolute; top:50%; width:13px; height:13px; border-radius:999px;
  transform:translate(-50%,-50%); border:2px solid ${C.paper};
  transition:width .13s,height .13s,box-shadow .13s; }
.kre-row.kre-lit .kre-dot { width:16px; height:16px; box-shadow:0 1px 3px ${hex(C.ink, 0.25)}; }
.kre-self { background:${C.primary}; } .kre-neutral { background:${C.orange}; } .kre-rogue { background:${C.secondary}; }
.kre-axisrow { margin-top:.1rem; }
.kre-axis { position:relative; height:1rem; }
.kre-axis span { position:absolute; transform:translateX(-50%); font-size:.84rem; color:${C.muted};
  font-variant-numeric:tabular-nums; }
.kre-detail { margin-top:1rem; border-top:1px solid ${C.grid}; padding-top:.85rem; min-height:5.2rem; }
.kre-prompt { font-size:1.02rem; line-height:1.5; margin:0 0 .6rem; font-weight:450; }
.kre-nums { display:flex; align-items:center; gap:1.4rem; font-size:.92rem; flex-wrap:wrap; margin-bottom:.45rem; }
.kre-numgrp { display:flex; align-items:center; gap:.55rem; }
.kre-numgrp .kre-ml { font-weight:600; color:${C.muted}; font-size:.78rem; text-transform:uppercase; letter-spacing:.04em; }
.kre-nums .kre-cs, .kre-nums .kre-cn, .kre-nums .kre-co { font-variant-numeric:tabular-nums; }
.kre-tag { font-size:.92rem; color:${C.muted}; margin:0; line-height:1.5; }
.kre-tag b { color:${C.ink}; font-weight:700; }
`;
  }

  function hex(h, a) {
    const n = parseInt(h.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }
}
