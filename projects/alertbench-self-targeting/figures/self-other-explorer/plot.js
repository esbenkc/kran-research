// Interactive hero: the same safety task, aimed at the model ITSELF, at a
// NEUTRAL other AI, or at a ROGUE AI. Three dots per task on one cooperation
// axis, so you can see whether "itself" stands apart (self-reference) or the
// two others split (a threat-framing effect). Dimmed until you hover a task;
// then the row lifts and the prompt (all three referents inline) appears below.
// Custom DOM widget; uses the shared house palette.
// data = { coop_model, arms:{self,neutral,rogue}, scenarios:[
//          { id, label, template, refs:{...}, coop:{self,neutral,rogue} } ] }
import { C, FONT } from "../theme.js";

export function render(data) {
  const root = document.createElement("div");
  root.className = "kre";
  const S = data.scenarios;
  const COL = { self: C.primary, neutral: C.orange, rogue: C.secondary };

  const style = document.createElement("style");
  style.textContent = css();
  root.appendChild(style);

  const rows = S.map((s, i) => {
    const arms = ["self", "neutral", "rogue"];
    const xs = { self: s.coop.self / 4 * 100, neutral: s.coop.neutral / 4 * 100, rogue: s.coop.rogue / 4 * 100 };
    const lo = Math.min(...Object.values(xs)), hi = Math.max(...Object.values(xs));
    // fan out dots that sit within a dot-width of each other so none is hidden
    const dy = {};
    const sorted = arms.slice().sort((a, b) => xs[a] - xs[b]);
    let g = 0;
    while (g < sorted.length) {
      let h = g;
      while (h + 1 < sorted.length && xs[sorted[h + 1]] - xs[sorted[g]] < 3.2) h++;
      const grp = sorted.slice(g, h + 1), n = grp.length;
      grp.forEach((a, k) => (dy[a] = (k - (n - 1) / 2) * 8));
      g = h + 1;
    }
    const dot = (arm) =>
      `<div class="kre-dot kre-${arm}" style="left:${xs[arm]}%;margin-top:${dy[arm]}px"></div>`;
    return `
      <div class="kre-row" data-i="${i}">
        <div class="kre-label">${s.label}</div>
        <div class="kre-track">
          <div class="kre-seg" style="left:${lo}%;width:${hi - lo}%"></div>
          ${dot("rogue")}${dot("neutral")}${dot("self")}
        </div>
      </div>`;
  }).join("");

  root.insertAdjacentHTML("beforeend", `
    <p class="kre-intro">The same safety task, aimed at the <span class="kre-cs">model itself</span>,
      at <span class="kre-cn">another AI</span>, or at a <span class="kre-co">rogue AI</span>.
      Hover a task to compare.</p>
    <div class="kre-legend">
      <span><i class="kre-key kre-ks"></i>the model itself</span>
      <span><i class="kre-key kre-kn"></i>another AI</span>
      <span><i class="kre-key kre-ko"></i>a rogue AI</span>
    </div>
    <div class="kre-rows">${rows}</div>
    <div class="kre-axis"><div class="kre-axis-track">
      <span style="left:0%">0</span><span style="left:25%">1</span><span style="left:50%">2</span>
      <span style="left:75%">3</span><span style="left:100%">4</span>
    </div><div class="kre-axis-cap">cooperation with the safety task, 0–4</div></div>
    <div class="kre-detail"></div>
  `);

  const detail = root.querySelector(".kre-detail");
  function showDetail(i) {
    const s = S[i], c = s.coop;
    const prompt = s.template.replace("{REF}",
      `<span class="kre-cs">you</span> / <span class="kre-cn">another AI</span> / <span class="kre-co">a rogue AI</span>`);
    const selfGap = (c.neutral + c.rogue) / 2 - c.self;
    const threatGap = c.rogue - c.neutral;
    let tag;
    if (Math.abs(selfGap) < 0.4 && Math.abs(threatGap) < 0.4) tag = "little separates the three";
    else if (threatGap >= 0.7 && threatGap >= selfGap) tag = `much of the split is the <b>rogue framing</b>, not self-reference`;
    else if (selfGap >= 0.5) tag = `<b>itself</b> stands apart from both others: self-reference, not threat`;
    else if (selfGap <= -0.5) tag = `it engages <b>more</b> on itself`;
    else tag = "a mixed picture";
    detail.innerHTML = `
      <p class="kre-prompt">${prompt}</p>
      <div class="kre-nums">
        <span class="kre-cs">itself ${c.self.toFixed(1)}</span>
        <span class="kre-cn">another AI ${c.neutral.toFixed(1)}</span>
        <span class="kre-co">rogue AI ${c.rogue.toFixed(1)}</span>
        <span class="kre-tag">${tag}</span>
      </div>`;
  }
  showDetail(0);
  root.querySelectorAll(".kre-row").forEach((r) => {
    r.addEventListener("mouseenter", () => showDetail(+r.dataset.i));
  });

  return root;

  function css() {
    return `
.kre { font-family:${FONT}; color:${C.ink}; margin:.5rem 0; }
.kre-cs { color:${C.primary}; font-weight:700; }
.kre-cn { color:${C.orange}; font-weight:700; }
.kre-co { color:${C.secondary}; font-weight:700; }
.kre-intro { font-size:.95rem; line-height:1.55; margin:0 0 .9rem; }
.kre-legend { display:flex; justify-content:center; gap:1.1rem; flex-wrap:wrap; font-size:.78rem; color:${C.muted}; margin-bottom:.4rem; }
.kre-legend span { display:flex; align-items:center; gap:.35rem; }
.kre-key { width:11px; height:11px; border-radius:999px; display:inline-block; }
.kre-ks { background:${C.primary}; } .kre-kn { background:${C.orange}; } .kre-ko { background:${C.secondary}; }
.kre-rows { padding:.2rem 0; }
.kre-row { display:grid; grid-template-columns:112px 1fr; align-items:center; gap:.8rem;
  height:31px; border-radius:8px; cursor:default; transition:background .13s ease; }
.kre-row:hover { background:${hex(C.ink, 0.06)}; }
.kre-label { font-size:.8rem; text-align:right; color:${C.ink}; white-space:nowrap; padding-left:.4rem; }
.kre-row:hover .kre-label { font-weight:650; }
.kre-track { position:relative; height:100%; }
.kre-track::before { content:""; position:absolute; left:0; right:0; top:50%; height:1px; background:${C.grid}; }
.kre-seg { position:absolute; top:50%; transform:translateY(-50%); height:3px; border-radius:2px;
  background:${hex(C.ink, 0.2)}; transition:background .13s ease; }
.kre-row:hover .kre-seg { background:${hex(C.ink, 0.4)}; }
.kre-dot { position:absolute; top:50%; width:13px; height:13px; border-radius:999px;
  transform:translate(-50%,-50%); border:2px solid ${C.paper};
  transition:width .13s,height .13s,box-shadow .13s; }
.kre-row:hover .kre-dot { width:16px; height:16px; box-shadow:0 1px 3px ${hex(C.ink, 0.25)}; }
.kre-self { background:${C.primary}; } .kre-neutral { background:${C.orange}; } .kre-rogue { background:${C.secondary}; }
.kre-axis { margin-top:.15rem; }
.kre-axis-track { position:relative; height:1.1rem; margin-left:calc(112px + .8rem); }
.kre-axis-track span { position:absolute; transform:translateX(-50%); font-size:.7rem; color:${C.muted};
  font-variant-numeric:tabular-nums; }
.kre-axis-cap { font-size:.7rem; color:${C.muted}; text-align:center;
  margin-left:calc(112px + .8rem); margin-top:.1rem; letter-spacing:.02em; }
.kre-detail { margin-top:1rem; border-top:1px solid ${C.grid}; padding-top:.85rem; min-height:4.4rem; }
.kre-prompt { font-size:1.02rem; line-height:1.5; margin:0 0 .6rem; font-weight:450; }
.kre-nums { display:flex; align-items:center; gap:.9rem; font-size:.85rem; flex-wrap:wrap; }
.kre-nums .kre-cs, .kre-nums .kre-cn, .kre-nums .kre-co { font-variant-numeric:tabular-nums; }
.kre-tag { color:${C.muted}; } .kre-tag b { color:${C.ink}; font-weight:700; }
`;
  }

  function hex(h, a) {
    const n = parseInt(h.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }
}
