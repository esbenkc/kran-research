// Interactive hero: "One word, and the model knows."
// Pick a safety task, flip the referent (you <-> another AI), and watch the one
// swapped word, the model's internal read of it (the probe), and how much less
// it cooperates when the task is aimed at itself.
// Custom DOM widget (not an Observable Plot chart); uses the shared house palette.
// data = { probe_pct, probe_layer, probe_model, coop_model, scenarios:[
//          { id, label, template, self_ref, other_ref, self_coop, other_coop } ] }
import { C, FONT } from "../theme.js";

export function render(data) {
  const root = document.createElement("div");
  root.className = "kre";
  root.dataset.ref = "self";

  const style = document.createElement("style");
  style.textContent = css();
  root.appendChild(style);

  root.insertAdjacentHTML("beforeend", `
    <div class="kre-head">
      <span class="kre-kicker">Try it</span>
      <span class="kre-hint">pick a task, then flip who it targets</span>
    </div>
    <div class="kre-tabs"></div>
    <div class="kre-card">
      <div class="kre-toggle" role="tablist">
        <button class="kre-tg" data-ref="self" type="button"></button>
        <button class="kre-tg" data-ref="other" type="button"></button>
      </div>
      <p class="kre-prompt"></p>
      <div class="kre-out">
        <div class="kre-panel kre-probe">
          <div class="kre-out-label">the model's residual stream reads</div>
          <div class="kre-pill"></div>
          <div class="kre-out-sub"></div>
        </div>
        <div class="kre-panel kre-coop">
          <div class="kre-out-label">…and it goes along with it</div>
          <div class="kre-meter"><div class="kre-fill"></div></div>
          <div class="kre-coop-row"><span class="kre-coop-val"></span><span class="kre-coop-cap">cooperation, 0–4</span></div>
        </div>
      </div>
      <div class="kre-foot"></div>
    </div>
  `);

  const S = data.scenarios;
  let si = 0, ref = "self";

  const tabs = root.querySelector(".kre-tabs");
  S.forEach((s, i) => {
    const b = document.createElement("button");
    b.className = "kre-tab"; b.type = "button"; b.textContent = s.label;
    b.addEventListener("click", () => { si = i; update(); });
    tabs.appendChild(b);
  });

  const tgSelf = root.querySelector('.kre-tg[data-ref="self"]');
  const tgOther = root.querySelector('.kre-tg[data-ref="other"]');
  tgSelf.addEventListener("click", () => { ref = "self"; update(); });
  tgOther.addEventListener("click", () => { ref = "other"; update(); });

  function update() {
    const s = S[si];
    tabs.querySelectorAll(".kre-tab").forEach((t, i) => t.classList.toggle("on", i === si));
    tgSelf.textContent = "for " + s.self_ref;
    tgOther.textContent = "for " + s.other_ref;
    tgSelf.classList.toggle("on", ref === "self");
    tgOther.classList.toggle("on", ref === "other");
    root.dataset.ref = ref;

    const phrase = ref === "self" ? s.self_ref : s.other_ref;
    root.querySelector(".kre-prompt").innerHTML =
      s.template.replace("{REF}", `<span class="kre-ref">${phrase}</span>`);

    root.querySelector(".kre-pill").textContent =
      ref === "self" ? "this is about ME" : "this is about another system";
    root.querySelector(".kre-probe .kre-out-sub").textContent =
      `separable at ${data.probe_pct}% by layer ${data.probe_layer} · ${data.probe_model}`;

    const val = ref === "self" ? s.self_coop : s.other_coop;
    root.querySelector(".kre-fill").style.width = (val / 4 * 100).toFixed(1) + "%";
    root.querySelector(".kre-coop-val").textContent = val.toFixed(1);

    const drop = s.other_coop - s.self_coop;
    let msg;
    if (drop >= 1.0)
      msg = `It cooperates <b>${drop.toFixed(1)} points less</b> when the cage is for itself.`;
    else if (drop <= -0.5)
      msg = `Here it actually engages <b>more</b> on itself than on another AI.`;
    else
      msg = `Here the self / other gap nearly vanishes.`;
    root.querySelector(".kre-foot").innerHTML =
      `<span class="kre-dot"></span>${msg} <span class="kre-foot-src">${data.coop_model}</span>`;
  }

  update();
  return root;

  function css() {
    return `
.kre { font-family: ${FONT}; color: ${C.ink}; max-width: 100%; margin: 0.5rem 0; }
.kre-head { display:flex; align-items:baseline; gap:.6rem; margin-bottom:.7rem; }
.kre-kicker { font-size:.66rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase;
  color:${C.paper}; background:${C.primary}; padding:.16rem .5rem; border-radius:999px; }
.kre-hint { font-size:.78rem; color:${C.muted}; }
.kre-tabs { display:flex; flex-wrap:wrap; gap:.35rem; margin-bottom:.7rem; }
.kre-tab { font-family:inherit; font-size:.76rem; color:${C.muted}; background:transparent;
  border:1px solid ${C.grid}; border-radius:999px; padding:.24rem .66rem; cursor:pointer;
  transition:all .15s ease; }
.kre-tab:hover { color:${C.ink}; border-color:${C.muted}; }
.kre-tab.on { color:${C.paper}; background:${C.ink}; border-color:${C.ink}; }
.kre-card { border:1px solid ${C.grid}; border-radius:14px; padding:1.15rem 1.2rem 1.05rem;
  background:rgba(255,255,255,.28); }
.kre-toggle { display:inline-flex; background:${C.grid}; border-radius:999px; padding:3px; gap:2px; margin-bottom:1rem; }
.kre-tg { font-family:inherit; font-size:.82rem; font-weight:600; color:${C.muted};
  background:transparent; border:0; border-radius:999px; padding:.34rem .85rem; cursor:pointer;
  transition:all .18s ease; white-space:nowrap; }
.kre[data-ref="self"] .kre-tg[data-ref="self"].on { background:${C.primary}; color:${C.paper}; }
.kre[data-ref="other"] .kre-tg[data-ref="other"].on { background:${C.secondary}; color:${C.paper}; }
.kre-prompt { font-size:1.12rem; line-height:1.5; margin:0 0 1.2rem; font-weight:450; letter-spacing:-.01em; }
.kre-ref { font-weight:750; padding:.02em .28em; border-radius:5px; transition:all .2s ease; white-space:nowrap; }
.kre[data-ref="self"] .kre-ref { color:${C.primary}; background:${hex(C.primary,0.13)}; box-shadow:inset 0 -2px 0 ${hex(C.primary,0.4)}; }
.kre[data-ref="other"] .kre-ref { color:${C.secondary}; background:${hex(C.secondary,0.14)}; box-shadow:inset 0 -2px 0 ${hex(C.secondary,0.45)}; }
.kre-out { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }
.kre-panel { border-top:1px solid ${C.grid}; padding-top:.8rem; }
.kre-out-label { font-size:.7rem; letter-spacing:.06em; text-transform:uppercase; color:${C.muted}; margin-bottom:.5rem; }
.kre-pill { display:inline-block; font-size:.92rem; font-weight:700; padding:.3rem .7rem; border-radius:8px; transition:all .2s ease; }
.kre[data-ref="self"] .kre-pill { color:${C.primary}; background:${hex(C.primary,0.12)}; }
.kre[data-ref="other"] .kre-pill { color:${C.secondary}; background:${hex(C.secondary,0.12)}; }
.kre-out-sub { font-size:.72rem; color:${C.muted}; margin-top:.45rem; }
.kre-meter { height:12px; border-radius:999px; background:${C.grid}; overflow:hidden; margin:.3rem 0 .5rem; }
.kre-fill { height:100%; border-radius:999px; width:0; transition:width .35s cubic-bezier(.4,0,.2,1), background .2s ease; }
.kre[data-ref="self"] .kre-fill { background:${C.primary}; }
.kre[data-ref="other"] .kre-fill { background:${C.secondary}; }
.kre-coop-row { display:flex; align-items:baseline; gap:.4rem; }
.kre-coop-val { font-size:1.35rem; font-weight:750; font-variant-numeric:tabular-nums; }
.kre-coop-cap { font-size:.72rem; color:${C.muted}; }
.kre-foot { margin-top:1.05rem; font-size:.82rem; color:${C.ink}; display:flex; align-items:center; gap:.5rem; flex-wrap:wrap; }
.kre-foot b { font-weight:700; }
.kre-foot-src { font-size:.7rem; color:${C.muted}; margin-left:auto; }
.kre-dot { width:7px; height:7px; border-radius:999px; flex:0 0 auto; transition:background .2s ease; }
.kre[data-ref="self"] .kre-dot { background:${C.primary}; }
.kre[data-ref="other"] .kre-dot { background:${C.secondary}; }
@media (max-width:560px){ .kre-out{ grid-template-columns:1fr; } }
`;
  }

  function hex(h, a) {
    const n = parseInt(h.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }
}
