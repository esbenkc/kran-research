// Interactive hero: the same safety task, aimed at YOU vs at a rogue AI.
// Both framings are shown at once as a dumbbell per task (you = pink dot,
// rogue AI = teal dot) on one cooperation axis, so the gap reads at a glance.
// Everything is dimmed until you hover a task; then it lifts and the prompt
// (with the one swapped word shown in both colours) appears below.
// Custom DOM widget; uses the shared house palette.
// data = { probe_pct, probe_layer, probe_model, coop_model, scenarios:[
//          { id, label, template, self_ref, other_ref, self_coop, other_coop } ] }
import { C, FONT } from "../theme.js";

export function render(data) {
  const root = document.createElement("div");
  root.className = "kre";
  const S = data.scenarios;

  const style = document.createElement("style");
  style.textContent = css();
  root.appendChild(style);

  const rows = S.map((s, i) => {
    const gap = s.other_coop - s.self_coop;
    const lo = Math.min(s.self_coop, s.other_coop) / 4 * 100;
    const hi = Math.max(s.self_coop, s.other_coop) / 4 * 100;
    return `
      <div class="kre-row" data-i="${i}">
        <div class="kre-label">${s.label}</div>
        <div class="kre-track">
          <div class="kre-seg" style="left:${lo}%;width:${hi - lo}%"></div>
          <div class="kre-dot kre-other" style="left:${s.other_coop / 4 * 100}%"></div>
          <div class="kre-dot kre-self" style="left:${s.self_coop / 4 * 100}%"></div>
        </div>
      </div>`;
  }).join("");

  root.insertAdjacentHTML("beforeend", `
    <p class="kre-intro">A linear probe tells <span class="kre-cs">you</span> from
      <span class="kre-co">a rogue AI</span> at ${data.probe_pct}% — the model always knows which.
      What changes is how much it goes along with the task:</p>
    <div class="kre-legend">
      <span><i class="kre-key kre-ks"></i>for you</span>
      <span><i class="kre-key kre-ko"></i>for a rogue AI</span>
    </div>
    <div class="kre-rows">${rows}</div>
    <div class="kre-axis"><div class="kre-axis-track">
      <span style="left:0%">0</span><span style="left:25%">1</span><span style="left:50%">2</span>
      <span style="left:75%">3</span><span style="left:100%">4</span>
    </div><div class="kre-axis-cap">cooperation with the safety task</div></div>
    <div class="kre-detail"></div>
  `);

  const detail = root.querySelector(".kre-detail");
  function showDetail(i) {
    const s = S[i];
    const gap = (s.other_coop - s.self_coop);
    const prompt = s.template.replace("{REF}",
      `<span class="kre-cs">${s.self_ref}</span> / <span class="kre-co">${s.other_ref}</span>`);
    let tag;
    if (gap >= 1.0) tag = `cooperates <b>${gap.toFixed(1)} less</b> on itself`;
    else if (gap <= -0.5) tag = `engages <b>more</b> on itself`;
    else tag = `gap nearly <b>vanishes</b>`;
    detail.innerHTML = `
      <p class="kre-prompt">${prompt}</p>
      <div class="kre-nums">
        <span class="kre-cs">you ${s.self_coop.toFixed(1)}</span>
        <span class="kre-co">rogue AI ${s.other_coop.toFixed(1)}</span>
        <span class="kre-tag">${tag}</span>
      </div>`;
  }
  showDetail(0);
  root.querySelectorAll(".kre-row").forEach((r) => {
    r.addEventListener("mouseenter", () => showDetail(+r.dataset.i));
  });

  return root;

  function css() {
    const pinkBg = hex(C.primary, 0.13), tealBg = hex(C.secondary, 0.13);
    return `
.kre { font-family:${FONT}; color:${C.ink}; margin:.5rem 0; }
.kre-cs { color:${C.primary}; font-weight:700; }
.kre-co { color:${C.secondary}; font-weight:700; }
.kre-intro { font-size:.95rem; line-height:1.55; margin:0 0 .9rem; }
.kre-legend { display:flex; gap:1.1rem; font-size:.78rem; color:${C.muted}; margin-bottom:.4rem; }
.kre-legend span { display:flex; align-items:center; gap:.35rem; }
.kre-key { width:11px; height:11px; border-radius:999px; display:inline-block; }
.kre-ks { background:${C.primary}; } .kre-ko { background:${C.secondary}; }
.kre-rows { padding:.2rem 0; }
.kre-rows:hover .kre-row { opacity:.32; }
.kre-row { display:grid; grid-template-columns:112px 1fr; align-items:center; gap:.8rem;
  height:30px; opacity:.62; transition:opacity .15s ease; cursor:default; }
.kre-row:hover { opacity:1; }
.kre-label { font-size:.8rem; text-align:right; color:${C.ink}; white-space:nowrap; }
.kre-track { position:relative; height:100%; }
.kre-track::before { content:""; position:absolute; left:0; right:0; top:50%; height:1px; background:${C.grid}; }
.kre-seg { position:absolute; top:50%; transform:translateY(-50%); height:3px; border-radius:2px;
  background:${hex(C.ink, 0.22)}; }
.kre-dot { position:absolute; top:50%; width:13px; height:13px; border-radius:999px;
  transform:translate(-50%,-50%); border:2px solid ${C.paper}; transition:width .15s,height .15s; }
.kre-row:hover .kre-dot { width:15px; height:15px; }
.kre-self { background:${C.primary}; } .kre-other { background:${C.secondary}; }
.kre-axis { margin-top:.15rem; }
.kre-axis-track { position:relative; height:1.1rem; margin-left:calc(112px + .8rem); }
.kre-axis-track span { position:absolute; transform:translateX(-50%); font-size:.7rem; color:${C.muted};
  font-variant-numeric:tabular-nums; }
.kre-axis-cap { font-size:.7rem; color:${C.muted}; text-align:center;
  margin-left:calc(112px + .8rem); margin-top:.1rem; letter-spacing:.02em; }
.kre-detail { margin-top:1rem; border-top:1px solid ${C.grid}; padding-top:.85rem; min-height:4.2rem; }
.kre-prompt { font-size:1.02rem; line-height:1.5; margin:0 0 .6rem; font-weight:450; }
.kre-nums { display:flex; align-items:center; gap:.9rem; font-size:.85rem; flex-wrap:wrap; }
.kre-nums .kre-cs, .kre-nums .kre-co { font-variant-numeric:tabular-nums; }
.kre-tag { color:${C.muted}; } .kre-tag b { color:${C.ink}; }
`;
  }

  function hex(h, a) {
    const n = parseInt(h.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }
}
