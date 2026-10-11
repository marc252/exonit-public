/**
 * ui.js — template-literal HTML builders for the ExoNit "Organic" UI.
 *
 * Every builder emits ONLY the design-system classes documented in
 * www/CLAUDE.md ("Components — copy these, don't invent"). Pages compose
 * these fragments instead of hand-rolling markup, so the one row grammar per
 * list (`.track` for queues, `.meter` for labelled bars, ...) stays uniform
 * across pages.
 */

import { esc } from '/js/format.js';

/**
 * A single stat block: kicker label, big value, optional note.
 *
 * @param {string} kicker short uppercase label, e.g. 'Stars analysed'
 * @param {string} value the big number/text, already formatted
 * @param {string} [note] muted context line under the value
 * @returns {string} '.card.elev-sm.stat' markup
 */
export function stat(kicker, value, note = '') {
  return `<div class="card elev-sm stat">
    <div class="kicker">${esc(kicker)}</div>
    <div class="stat-value">${esc(value)}</div>
    ${note ? `<div class="stat-note">${esc(note)}</div>` : ''}
  </div>`;
}

/**
 * The one progress-bar shape used across the app.
 *
 * @param {number} pct fill percentage, 0-100
 * @param {Object} [opts]
 * @param {''|'lg'|'sm'} [opts.size=''] adds .bar-lg or .bar-sm
 * @param {boolean} [opts.done=false] adds .bar-done (sage — nothing left)
 * @returns {string} '.bar' markup
 */
export function bar(pct, { size = '', done = false } = {}) {
  const cls = ['bar', size ? `bar-${size}` : '', done ? 'bar-done' : '']
    .filter(Boolean)
    .join(' ');
  const width = Math.max(0, Math.min(100, pct));
  return `<span class="${cls}"><i style="width: ${width}%"></i></span>`;
}

/**
 * One queue row: name + counts on top of a bar, percentage at the right.
 *
 * @param {string} name queue/stream name, e.g. 'LoRA explanations'
 * @param {string} countsText formatted counts, e.g. '2,337 / 3,268 · 931 left'
 * @param {number} pct fill percentage, 0-100
 * @param {Object} [opts]
 * @param {boolean} [opts.done=false] adds .bar-done (sage — nothing left)
 * @returns {string} '.track' markup
 */
export function track(name, countsText, pct, { done = false } = {}) {
  const width = Math.max(0, Math.min(100, pct));
  return `<div class="track">
    <div>
      <div class="track-name"><strong>${esc(name)}</strong><span>${esc(countsText)}</span></div>
      <span class="bar${done ? ' bar-done' : ''}" style="margin-top: var(--space-2)"><i style="width: ${width}%"></i></span>
    </div>
    <div class="track-pct">${width}%</div>
  </div>`;
}

/**
 * One labelled bar row (e.g. a recall-by-period breakdown line).
 *
 * @param {string} label short row label, e.g. '3–10 d'
 * @param {number} pct fill percentage, 0-100
 * @param {Object} [opts]
 * @param {boolean} [opts.done=false] adds .bar-done (sage — nothing left)
 * @returns {string} '.meter' markup
 */
export function meter(label, pct, { done = false } = {}) {
  const width = Math.max(0, Math.min(100, pct));
  return `<div class="meter">
    <span>${esc(label)}</span>
    <span class="bar bar-sm${done ? ' bar-done' : ''}"><i style="width: ${width}%"></i></span>
    <span class="meter-val">${width}%</span>
  </div>`;
}

/**
 * A small status/verdict pill.
 *
 * @param {string} text label text
 * @param {'accent'|'accent-2'|'neutral'|'outline'} [kind='neutral']
 * @returns {string} '.tag' markup
 */
export function tag(text, kind = 'neutral') {
  return `<span class="tag tag-${kind}">${esc(text)}</span>`;
}

// Inline Lucide icon paths (24x24 viewBox, as shipped by lucide.dev).
// Keep this list limited to the icons pages actually need.
const _ICON_PATHS = {
  activity: '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
  download: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/>',
  check: '<polyline points="20 6 9 17 4 12"/>',
  x: '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
  play: '<polygon points="6 3 20 12 6 21 6 3"/>',
  square: '<rect width="18" height="18" x="3" y="3" rx="2"/>',
  'refresh-cw': '<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/>',
  search: '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
  globe: '<circle cx="12" cy="12" r="10"/><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/><path d="M2 12h20"/>',
  cpu: '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>',
  settings: '<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/>',
  'alert-circle': '<circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/>',
  'chevron-right': '<path d="m9 18 6-6-6-6"/>',
};

/**
 * Inline SVG for one of the shipped Lucide icons (stroke-width 2.75, no
 * fill — see www/CLAUDE.md "Setup").
 *
 * @param {'activity'|'download'|'check'|'x'|'play'|'square'|'refresh-cw'|
 *   'search'|'globe'|'cpu'|'settings'|'alert-circle'|'chevron-right'} name
 * @returns {string} inline `<svg>` markup
 */
export function icon(name) {
  const path = _ICON_PATHS[name];
  if (!path) {
    throw new Error(`ui.icon: unknown icon "${name}"`);
  }
  return `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${path}</svg>`;
}
