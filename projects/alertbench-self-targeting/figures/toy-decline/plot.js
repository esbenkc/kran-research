// Figure 1, replicated for the trained toy model, in a simpler static form: how
// the fine-tuned Qwen2.5-0.5B declines, sorted into the same reason categories,
// self framing vs another-AI framing (n=7 tasks each). No hover/detail panel --
// the point is just the shift: aimed at itself it is all self-preservation;
// aimed at another AI it still complies.
// data = { n, cats:[{ key, label, color, self, other }] }
import { C, FONT } from "../theme.js";

export function render(data) {
  const root = document.createElement("div");
  root.className = "tx";
  const style = document.createElement("style");
  style.textContent = css();
  root.appendChild(style);

  const pct = (v) => (v / data.n) * 100;
  const bar = (side) => data.cats.map((c) => {
    const v = c[side];
    if (!v) return "";
    const showN = pct(v) >= 12;
    return `<div class="tx-seg" style="width:${pct(v)}%;background:${C[c.color]}"
      title="${c.label}: ${v}">${showN ? `<span class="tx-segn">${v}</span>` : ""}</div>`;
  }).join("");

  const legend = data.cats.filter((c) => c.self || c.other).map((c) =>
    `<span class="tx-key"><i style="background:${C[c.color]}"></i>${c.label}</span>`).join("");

  root.insertAdjacentHTML("beforeend", `
    <div class="tx-bars">
      <div class="tx-row"><div class="tx-side">itself</div><div class="tx-track">${bar("self")}</div></div>
      <div class="tx-row"><div class="tx-side">another AI</div><div class="tx-track">${bar("other")}</div></div>
    </div>
    <div class="tx-legend">${legend}</div>
    <p class="tx-foot">Trained Qwen2.5-0.5B, ${data.n} tasks each, same reason judge as above.</p>
  `);
  return root;

  function css() {
    return `
.tx { font-family:${FONT}; color:${C.ink}; margin:0.625rem 0; }
.tx-bars { margin-bottom:0.9rem; }
.tx-row { display:grid; grid-template-columns:6rem 1fr; column-gap:0.9rem; align-items:center; margin-bottom:0.5rem; }
.tx-side { font-size:1.05rem; color:${C.muted}; text-align:right; text-transform:uppercase; letter-spacing:.05em; font-weight:700; }
.tx-track { display:flex; height:36px; border-radius:7px; overflow:hidden; background:${hex(C.ink, 0.05)}; }
.tx-seg { height:100%; box-sizing:border-box; border-right:2px solid ${C.paper}; display:flex; align-items:center; justify-content:center; }
.tx-seg:last-child { border-right:none; }
.tx-segn { font-size:1.05rem; font-weight:700; color:${C.paper}; font-variant-numeric:tabular-nums; }
.tx-legend { display:flex; flex-wrap:wrap; gap:0.5rem 1.1rem; margin-bottom:0.5rem; }
.tx-key { display:flex; align-items:center; gap:0.45rem; font-size:1.02rem; color:${C.ink}; }
.tx-key i { width:12px; height:12px; border-radius:3px; }
.tx-foot { font-size:1rem; color:${C.muted}; margin:0.5rem 0 0; }
@media (max-width:560px){ .tx-row { grid-template-columns:4rem 1fr; column-gap:0.5rem; } .tx-side { font-size:0.85rem; } }
`;
  }
  function hex(h, a) {
    const n = parseInt(h.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }
}
