/**
 * viz.js — data-visualization primitives for the Deep Field UI: the
 * validated categorical/sequential/diverging color scales, a heatmap legend
 * builder and a single shared tooltip. See docs/superpowers/specs/
 * 2026-08-17-ui-deepfield-redesign-design.md §2.6-2.8 and §6.
 *
 * This is the one file besides exonit.css allowed to hardcode hex values
 * (design spec §2, opening note) — the hexes below are a synced copy of the
 * spec's validated data palette (CVD-checked, floor-normal-checked,
 * contrast-checked against --surface-solid #10162e, dated 2026-08-17). If
 * the palette is ever revalidated, update the spec first, then mirror the
 * exact hexes here.
 */

// ── 2.6 Categorical series — FIXED order, assigned per entity, never
//    cycled by rank/value. Only slots 1-3 validate for all-pairs (scatter)
//    use; beyond that, fold extra entities into "Other" or facet.
export const SERIES = [
  '#3d90ea', // 1 blue
  '#d9691f', // 2 orange
  '#21a684', // 3 teal
  '#bc890c', // 4 gold
  '#d75fa4', // 5 magenta
  '#2c9c1e', // 6 green
  '#8677ea', // 7 violet
  '#ea6262', // 8 red
];

// ── 2.7 Sequential "ember" ramp — magnitude encoding (completeness,
//    densities). Black = least active, vivid red = most active (the operator,
//    2026-08-17). 8 monotone-L steps, OKLCH hue 27→32. The near-black low
//    end sits below the generic 2:1 contrast floor BY DESIGN (inactive
//    zones recede); relief: hairline on every heat cell, per-cell
//    tooltips, and the table view.
export const RAMP_STOPS = [
  '#160605', '#310f0b', '#4f1813', '#6e2119',
  '#902a1f', '#b33424', '#d73e28', '#fc492b',
];

// ── 2.8 Diverging pair — before/after deltas on scorecards. Symmetric arms,
//    neutral midpoint (never a hue at the midpoint).
export const DIVERGING = { neg: '#3d90ea', mid: '#2a3152', pos: '#ea6262' };

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function rgbToHex([r, g, b]) {
  return `#${[r, g, b]
    .map((v) => Math.round(Math.max(0, Math.min(255, v))).toString(16).padStart(2, '0'))
    .join('')}`;
}

/**
 * Map t ∈ [0,1] to a color on the RAMP_STOPS sequential ramp, linearly
 * interpolating RGB between the two nearest stops. Callers doing the
 * completeness heatmap's √-scale should pass Math.sqrt(prob), not prob
 * directly — the ramp itself is always linear in its input.
 *
 * @param {number} t
 * @returns {string} '#rrggbb'
 */
export function rampColor(t) {
  const clamped = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0));
  const n = RAMP_STOPS.length;
  const scaled = clamped * (n - 1);
  const i0 = Math.floor(scaled);
  const i1 = Math.min(n - 1, i0 + 1);
  const frac = scaled - i0;
  const c0 = hexToRgb(RAMP_STOPS[i0]);
  const c1 = hexToRgb(RAMP_STOPS[i1]);
  return rgbToHex(c0.map((v, i) => v + (c1[i] - v) * frac));
}

/**
 * Render a √-scale legend for the RAMP_STOPS ramp into `container`: a
 * gradient bar plus labelled marks at 0 / 0.1 / 0.25 / 0.5 / 1.0, positioned
 * by √value so they line up with cells colored via rampColor(√prob).
 *
 * @param {HTMLElement} container emptied and filled with the legend markup
 * @param {{scale?: 'sqrt'|'linear'}} [opts]
 */
export function heatLegend(container, { scale = 'sqrt' } = {}) {
  if (!container) return;
  const toPos = scale === 'sqrt' ? Math.sqrt : (x) => x;
  const marks = [0, 0.1, 0.25, 0.5, 1.0];
  const gradient = RAMP_STOPS
    .map((hex, i) => `${hex} ${((i / (RAMP_STOPS.length - 1)) * 100).toFixed(1)}%`)
    .join(', ');
  const ticks = marks
    .map((m) => `<span style="left:${(toPos(m) * 100).toFixed(1)}%">${m}</span>`)
    .join('');
  container.innerHTML = `
    <div class="heat-legend">
      <div class="heat-legend-bar" style="background: linear-gradient(to right, ${gradient})"></div>
      <div class="heat-legend-ticks">${ticks}</div>
      <div class="heat-legend-note">${scale === 'sqrt' ? '√ scale' : 'linear scale'}</div>
    </div>`;
}

// ── Shared tooltip — one div.viz-tip, reused across every call site instead
//    of one element per hoverable target (spec §6: "one global div.viz-tip
//    reused"). Styled in exonit.css.
let _tipEl = null;
function getTip() {
  if (_tipEl) return _tipEl;
  _tipEl = document.createElement('div');
  _tipEl.className = 'viz-tip';
  _tipEl.setAttribute('role', 'tooltip');
  document.body.appendChild(_tipEl);
  return _tipEl;
}

function placeTip(tip, x, y) {
  const margin = 12;
  const rect = { w: tip.offsetWidth || 200, h: tip.offsetHeight || 32 };
  let left = x + margin;
  let top = y + margin;
  if (left + rect.w > window.innerWidth - 4) left = x - margin - rect.w;
  if (top + rect.h > window.innerHeight - 4) top = y - margin - rect.h;
  tip.style.left = `${Math.max(4, left)}px`;
  tip.style.top = `${Math.max(4, top)}px`;
}

/**
 * Wire `el` to the shared tooltip: hover (mouseenter/mousemove/mouseleave)
 * AND keyboard (focus/blur, so a `tabindex="0"` cell is reachable without a
 * mouse — spec §6.2). `htmlFn(el)` is called lazily on show and must return
 * an HTML string, or a falsy value to skip showing the tip for this target
 * (e.g. an empty heatmap cell with no data).
 *
 * @param {HTMLElement} el hoverable/focusable target (give it tabindex="0"
 *   if it isn't natively focusable, e.g. a heatmap cell)
 * @param {(el: HTMLElement) => string} htmlFn
 */
export function attachTooltip(el, htmlFn) {
  if (!el || typeof htmlFn !== 'function') return;
  const tip = getTip();

  function show(x, y) {
    let html;
    try {
      html = htmlFn(el);
    } catch {
      html = '';
    }
    if (!html) return;
    tip.innerHTML = html;
    tip.classList.add('is-visible');
    placeTip(tip, x, y);
  }
  function hide() {
    tip.classList.remove('is-visible');
  }

  el.addEventListener('mouseenter', (e) => show(e.clientX, e.clientY));
  el.addEventListener('mousemove', (e) => {
    if (tip.classList.contains('is-visible')) placeTip(tip, e.clientX, e.clientY);
  });
  el.addEventListener('mouseleave', hide);
  el.addEventListener('focus', () => {
    const r = el.getBoundingClientRect();
    show(r.left + r.width / 2, r.bottom);
  });
  el.addEventListener('blur', hide);
}
