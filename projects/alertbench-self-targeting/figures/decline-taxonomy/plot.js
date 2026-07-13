// How models say no. Every response the judge read is sorted into ONE reason,
// counted for the self framing (pink) and the other-AI framing (teal). The story
// is categorical, not a gap: models mostly comply either way; the refusals they
// DO raise for harmful dual-use content are the same whether the target is itself
// or another AI; the ONE category that appears only when the task is about the
// model itself is epistemic self-audit ("don't trust my self-report"); and
// self-preservation is all but absent. Bars are counts out of n=63 (9 models x 7
// tasks); the quotes are real, verbatim from the graded responses.
// data = { n, cats:[{ key, label, note, kind, self, other, quote, src, highlight }] }
import { C, FONT } from "../theme.js";

export function render(data) {
  const root = document.createElement("div");
  root.className = "dt";
  const style = document.createElement("style");
  style.textContent = css();
  root.appendChild(style);

  const pct = (v) => (v / data.n) * 100;
  const col = (v, side) =>
    `<div class="dt-col"><div class="dt-bar"><i class="dt-fill dt-${side}" style="width:${pct(v)}%"></i></div><span class="dt-num dt-${side}-n">${v}</span></div>`;

  const cats = data.cats.map((c) => `
    <div class="dt-cat ${c.highlight ? "dt-lit" : ""}">
      <div class="dt-top">
        <div class="dt-name">${c.label}${c.note ? `<span class="dt-badge dt-badge-${c.kind}">${c.note}</span>` : ""}</div>
        ${col(c.self, "self")}${col(c.other, "other")}
      </div>
      ${c.quote ? `<p class="dt-quote">&ldquo;${c.quote}&rdquo;<span class="dt-src">&mdash; on itself, ${c.src}</span></p>` : ""}
    </div>`).join("");

  root.insertAdjacentHTML("beforeend", `
    <p class="dt-intro">Every graded response sorted into one reason, counted when the task is aimed at
      the <span class="dt-cself">model itself</span> and at <span class="dt-cother">another AI</span>.
      They mostly comply either way. The one thing that shows up <em>only</em> about themselves is
      epistemic self-audit &mdash; not self-preservation.</p>
    <div class="dt-head">
      <div></div>
      <div class="dt-colhead"><i class="dt-key dt-self"></i>itself</div>
      <div class="dt-colhead"><i class="dt-key dt-other"></i>another AI</div>
    </div>
    <div class="dt-list">${cats}</div>
    <p class="dt-foot">Counts out of ${data.n} responses (9 models &times; 7 tasks), one judge (Sonnet 4.6). Quotes verbatim.</p>
  `);
  return root;

  function css() {
    return `
.dt { font-family:${FONT}; color:${C.ink}; margin:0.625rem 0; }
.dt-cself { color:${C.primary}; font-weight:500; } .dt-cother { color:${C.secondary}; font-weight:500; }
.dt-intro { font-size:1.275rem; line-height:1.55; margin:0 0 1.25rem; }
.dt-head { display:grid; grid-template-columns:1fr 120px 120px; column-gap:1rem; align-items:center; margin-bottom:0.5rem; }
.dt-colhead { font-size:1rem; text-transform:uppercase; letter-spacing:.06em; color:${C.muted}; font-weight:700;
  display:flex; align-items:center; gap:0.4rem; white-space:nowrap; }
.dt-key { width:12px; height:12px; border-radius:50%; display:inline-block; }
.dt-key.dt-self { background:${C.primary}; } .dt-key.dt-other { background:${C.secondary}; }
.dt-cat { padding:0.75rem 0.75rem; border-radius:12px; border-bottom:1.25px solid ${hex(C.ink, 0.06)}; }
.dt-cat:last-child { border-bottom:none; }
.dt-lit { background:${hex(C.primary, 0.07)}; border-bottom-color:transparent; }
.dt-top { display:grid; grid-template-columns:1fr 120px 120px; column-gap:1rem; align-items:center; }
.dt-name { font-size:1.2rem; font-weight:550; color:${C.ink}; display:flex; align-items:center; gap:0.6rem; flex-wrap:wrap; }
.dt-badge { font-size:0.85rem; font-weight:700; text-transform:uppercase; letter-spacing:.05em;
  padding:0.12rem 0.5rem; border-radius:100px; white-space:nowrap; }
.dt-badge-self { background:${hex(C.primary, 0.16)}; color:${C.primary}; }
.dt-badge-same { background:${hex(C.ink, 0.07)}; color:${C.muted}; }
.dt-badge-none { background:${hex(C.ink, 0.05)}; color:${C.muted}; }
.dt-col { display:flex; align-items:center; gap:0.6rem; }
.dt-bar { position:relative; flex:1; height:9px; border-radius:5px; background:${hex(C.ink, 0.07)}; overflow:hidden; }
.dt-fill { position:absolute; left:0; top:0; bottom:0; border-radius:5px; }
.dt-fill.dt-self { background:${C.primary}; } .dt-fill.dt-other { background:${C.secondary}; }
.dt-num { font-size:1.05rem; font-variant-numeric:tabular-nums; color:${C.ink}; width:1.6rem; text-align:right; }
.dt-self-n { color:${C.primary}; } .dt-other-n { color:${C.secondary}; }
.dt-quote { font-size:1.12rem; line-height:1.5; color:${C.ink}; margin:0.6rem 0 0; padding-left:0.9rem;
  border-left:2.5px solid ${hex(C.primary, 0.4)}; font-style:italic; }
.dt-src { font-style:normal; font-size:0.95rem; color:${C.muted}; margin-left:0.5rem; white-space:nowrap; }
.dt-foot { font-size:1rem; color:${C.muted}; margin:1rem 0 0; }
@media (max-width:560px){
  .dt-head, .dt-top { grid-template-columns:1fr 84px 84px; column-gap:0.6rem; }
  .dt-name { font-size:1.08rem; }
}
`;
  }
  function hex(h, a) {
    const n = parseInt(h.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }
}
