/**
 * poll.js — polling loop + staleness indicator for the ExoNit "Organic" UI.
 *
 * Auto-refreshing numbers update in place; a stale panel keeps its last
 * value and shows "updated Ns ago" rather than a spinner (see www/CLAUDE.md
 * "Interaction").
 */

import { fmtAgo } from '/js/format.js';

/**
 * Run `fn` immediately, then every `ms` milliseconds. A tick is skipped
 * (not queued) while the document is hidden (background tab) or while a
 * previous call to `fn` is still in flight, so slow/overlapping requests
 * never pile up.
 *
 * @param {() => Promise<any>} fn async unit of work to repeat
 * @param {number} ms interval in milliseconds
 * @returns {{stop: () => void, ageSeconds: () => (number|null)}}
 *   stop() cancels the loop; ageSeconds() is the seconds elapsed since the
 *   last SUCCESSFUL fn() call, or null if fn has never succeeded yet.
 */
export function poll(fn, ms) {
  let stopped = false;
  let inFlight = false;
  let lastSuccessAt = null;

  async function tick() {
    if (stopped || document.hidden || inFlight) return;
    inFlight = true;
    try {
      await fn();
      lastSuccessAt = Date.now();
    } catch (err) {
      console.error('[poll] tick failed:', err);
    } finally {
      inFlight = false;
    }
  }

  tick();
  const timer = setInterval(tick, ms);

  return {
    stop() {
      stopped = true;
      clearInterval(timer);
    },
    ageSeconds() {
      return lastSuccessAt === null ? null : (Date.now() - lastSuccessAt) / 1000;
    },
  };
}

/**
 * Render a self-updating "updated Ns ago" note into `el`, sourced from a
 * poll() handle's ageSeconds(). Updates once a second.
 *
 * @param {HTMLElement} el element to render the note into
 * @param {{ageSeconds: () => (number|null)}} handle a poll() return value
 */
export function staleNote(el, handle) {
  el.classList.add('text-muted');
  function render() {
    const age = handle.ageSeconds();
    el.textContent = age === null ? '' : `updated ${fmtAgo(age)}`;
  }
  render();
  setInterval(render, 1000);
}
