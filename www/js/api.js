/**
 * api.js — shared fetch helpers for the ExoNit "Organic" UI.
 *
 * Every page-level module imports from here instead of calling fetch()
 * directly, so auth headers and error shapes stay consistent across pages.
 */

/**
 * Build the Authorization header used by every /api/* call, merged with any
 * extra headers the caller needs (e.g. Content-Type for a POST body).
 *
 * The token is read from localStorage.exonit_token; if unset, falls back to
 * the well-known local dev token so the UI works against a fresh checkout.
 *
 * @param {Object} [extra] extra headers to merge in
 * @returns {Object} headers object including Authorization
 */
export function authHeaders(extra = {}) {
  const token = localStorage.exonit_token || 'Bearer exonit-default-token';
  return { Authorization: token, ...extra };
}

/**
 * Fetch JSON from `path`, throwing on a non-OK response.
 *
 * @param {string} path request path (e.g. '/api/stats')
 * @returns {Promise<any>} parsed JSON body
 * @throws {Error} message is body.detail || body.error || the HTTP status
 */
export async function getJSON(path) {
  const res = await fetch(path, { headers: authHeaders() });
  if (!res.ok) {
    throw new Error(await _errorMessage(res));
  }
  return res.json();
}

/**
 * Fetch plain text from `path` (e.g. a CSV export endpoint), throwing on a
 * non-OK response.
 *
 * @param {string} path request path
 * @returns {Promise<string>} response body as text
 * @throws {Error} message is body.detail || body.error || the HTTP status
 */
export async function getText(path) {
  const res = await fetch(path, { headers: authHeaders() });
  if (!res.ok) {
    throw new Error(await _errorMessage(res));
  }
  return res.text();
}

/**
 * POST a JSON body to `path` with auth headers, throwing on a non-OK
 * response.
 *
 * @param {string} path request path
 * @param {Object} [body] request body, JSON-encoded
 * @returns {Promise<any>} parsed JSON response body
 * @throws {Error} message is body.detail || body.error || the HTTP status
 */
export async function postJSON(path, body = {}) {
  const res = await fetch(path, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(await _errorMessage(res));
  }
  return res.json();
}

/**
 * Extract a human-readable error message from a failed Response: prefers
 * body.detail, then body.error, then falls back to the HTTP status line.
 *
 * @param {Response} res a Response with !res.ok
 * @returns {Promise<string>}
 */
async function _errorMessage(res) {
  try {
    const body = await res.json();
    return body.detail || body.error || `${res.status} ${res.statusText}`;
  } catch {
    return `${res.status} ${res.statusText}`;
  }
}
