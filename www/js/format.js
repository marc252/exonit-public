/**
 * format.js — value formatting helpers for the ExoNit "Organic" UI.
 *
 * Centralizes number/time formatting so every page shows the same shape for
 * the same kind of value (tabular nums, thousands separators, '—' for
 * missing data, sentinels never shown raw — see www/CLAUDE.md).
 */

const _MISSING = '—'; // '—'

/**
 * HTML-escape a string for safe interpolation into template-literal markup.
 *
 * @param {string} s
 * @returns {string}
 */
export function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[c]));
}

/**
 * Format an integer with thousands separators.
 *
 * @param {number|null|undefined} n
 * @returns {string} '9,917,540'; '—' if n is null/undefined/NaN
 */
export function fmtInt(n) {
  if (n === null || n === undefined || Number.isNaN(n)) return _MISSING;
  return Math.round(n).toLocaleString('en-US');
}

/**
 * Format a fraction (0..1) as a percentage string.
 *
 * @param {number|null|undefined} x
 * @param {number} [digits=1]
 * @returns {string} '71.5%' for 0.715; '—' if x is null/undefined/NaN
 */
export function fmtPct(x, digits = 1) {
  if (x === null || x === undefined || Number.isNaN(x)) return _MISSING;
  return `${(x * 100).toFixed(digits)}%`;
}

/**
 * Format a general number: hides the -1 sentinel, switches to compact
 * exponent notation for very large magnitudes, otherwise fixed-point with
 * thousands separators.
 *
 * @param {number|null|undefined} x
 * @param {number} [digits=2]
 * @returns {string} '8.7e28' for 8.718262196169956e+28; '—' for -1/null/NaN
 */
export function fmtNum(x, digits = 2) {
  if (x === null || x === undefined || Number.isNaN(x) || x === -1) return _MISSING;
  if (Math.abs(x) >= 1e6) {
    return x.toExponential(Math.max(digits - 1, 0)).replace('e+', 'e');
  }
  return x.toLocaleString('en-US', {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits,
  });
}

/**
 * Format an age in seconds as a short, single-unit "time ago" phrase.
 *
 * @param {number} seconds
 * @returns {string} '12 s ago' | '3 min ago' | '2 h ago'
 */
export function fmtAgo(seconds) {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s} s ago`;
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  return `${Math.floor(s / 3600)} h ago`;
}

/**
 * Format a duration in seconds as a compact, human-readable span.
 *
 * @param {number} seconds
 * @returns {string} '4 h 10 min' | '2 min 5 s' | '52 s'
 */
export function fmtDur(seconds) {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s} s`;
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) return `${h} h ${m} min`;
  return sec > 0 ? `${m} min ${sec} s` : `${m} min`;
}
