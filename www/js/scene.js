/**
 * scene.js — the Deep Field three.js background: a live star field with
 * mouse parallax, and (Dashboard only) "The Transit" signature — a star with
 * an orbiting planet. See docs/superpowers/specs/2026-08-17-ui-deepfield-
 * redesign-design.md §3 for the design brief this implements.
 *
 * ```js
 * import { initScene, fxEnabled, toggleFx } from '/js/scene.js';
 *
 * const handle = initScene(document.getElementById('bgScene'), { mode: 'ambient' });
 * // ... later, e.g. on page teardown:
 * handle.dispose();
 * ```
 *
 * `initScene(canvas, { mode: 'hero' | 'ambient', seed = 42 })` returns
 * `{ dispose(), setTransitPhase(cb) }`.
 *
 * - `dispose()` tears down the renderer, geometries, materials, textures and
 *   every listener/observer this module attached. Always available, even on
 *   the no-op object returned when FX is off / WebGL failed to init.
 * - `setTransitPhase(cb)` is only meaningful in `mode: 'hero'`. It registers
 *   `cb(t)` to be called once per rendered frame with the current transit
 *   phase `t` in `[0, 1)` — `t = 0` is mid-transit (the planet centred in
 *   front of the star, as seen from the camera). The Dashboard's light-curve
 *   SVG (spec §6.1, drawn by page-level code, not this module) uses this to
 *   sync its "dip" to the 3D planet. In `ambient` mode this is a no-op (no
 *   callback is ever invoked — there is no orbiting planet to phase).
 *
 * Every performance rule in spec §0.3 lives in this file so pages never have
 * to think about it: capped pixel ratio, one Points draw call for the whole
 * field, ~30fps throttle, full pause on `document.hidden`, a single static
 * frame under `prefers-reduced-motion`, and a silent WebGL-failure /
 * `localStorage.exonit_fx === 'off'` fallback to the CSS starfield in
 * exonit.css (`body::before`/`body::after`) — canvas.hidden = true and a
 * fully inert object are returned so the caller never has to branch on it.
 */

import * as THREE from '/static/vendor/three.module.min.js';

const FX_KEY = 'exonit_fx';
const FRAME_INTERVAL_MS = 1000 / 30; // §0.3: throttle to ~30fps inside rAF
const STAR_COUNT_AMBIENT = 2600; // within the 2,000-3,000 budget
const FIELD_RADIUS_MIN = 55;
const FIELD_RADIUS_MAX = 210;
const PARALLAX_MAX_RAD = (3 * Math.PI) / 180; // ±3°
const PARALLAX_LERP = 0.04;
const DRIFT_RAD_PER_SEC = 0.0035; // slow field drift (ambient + touch fallback)
const TRANSIT_PERIOD_S = 14; // one full orbit, seconds — hero mode only

/**
 * Whether background FX are enabled (localStorage 'exonit_fx' !== 'off').
 * Exported so the nav can render the toggle's initial label.
 * @returns {boolean}
 */
export function fxEnabled() {
  try {
    return localStorage.getItem(FX_KEY) !== 'off';
  } catch {
    return true; // localStorage blocked (private mode, etc.) — default on
  }
}

/**
 * Flip the FX on/off flag and reload the page — the simplest correct way to
 * fully tear down and re-init (or skip) a live WebGL context from a single
 * nav click, with no special-casing at each call site.
 */
export function toggleFx() {
  try {
    localStorage.setItem(FX_KEY, fxEnabled() ? 'off' : 'on');
  } catch {
    /* localStorage unavailable — nothing to persist, reload is still safe */
  }
  location.reload();
}

// A tiny deterministic PRNG (mulberry32) seeded per-scene instead of
// Math.random(), so the "random" field is reproducible across reloads for a
// given seed — useful for visual regression / screenshots.
function makeRng(seed) {
  let a = seed >>> 0;
  return function rng() {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Star "temperature" buckets → base color, roughly astrophysical (blue-white
// hot stars are rarer than cool white/amber ones).
const TEMPERATURES = [
  { weight: 0.15, color: new THREE.Color(0x9ec8ff) }, // blue-white
  { weight: 0.55, color: new THREE.Color(0xf3f6ff) }, // white
  { weight: 0.30, color: new THREE.Color(0xffd9a0) }, // amber, tenue
];

function pickTemperature(rng) {
  const r = rng();
  let acc = 0;
  for (const t of TEMPERATURES) {
    acc += t.weight;
    if (r <= acc) return t.color;
  }
  return TEMPERATURES[TEMPERATURES.length - 1].color;
}

function buildStarField(rng, count) {
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const sizes = new Float32Array(count);
  const tmp = new THREE.Color();
  for (let i = 0; i < count; i++) {
    // Uniform-on-sphere-shell placement (Marsaglia), radius jittered per
    // star for depth instead of one flat shell.
    let x, y, z, len;
    do {
      x = rng() * 2 - 1; y = rng() * 2 - 1; z = rng() * 2 - 1;
      len = x * x + y * y + z * z;
    } while (len > 1 || len === 0);
    len = Math.sqrt(len);
    const radius = FIELD_RADIUS_MIN + rng() * (FIELD_RADIUS_MAX - FIELD_RADIUS_MIN);
    positions[i * 3 + 0] = (x / len) * radius;
    positions[i * 3 + 1] = (y / len) * radius;
    positions[i * 3 + 2] = (z / len) * radius;

    tmp.copy(pickTemperature(rng));
    // Slight per-star brightness variation so the field isn't flat.
    const dim = 0.55 + rng() * 0.45;
    colors[i * 3 + 0] = tmp.r * dim;
    colors[i * 3 + 1] = tmp.g * dim;
    colors[i * 3 + 2] = tmp.b * dim;

    sizes[i] = 0.55 + rng() * 1.35;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geo.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

  // Circular (not square) points via a tiny generated alpha texture — cheap,
  // one-time canvas draw, reused by the single shared material.
  const mat = new THREE.PointsMaterial({
    size: 1.6,
    map: dotTexture(),
    vertexColors: true,
    transparent: true,
    depthWrite: false,
    sizeAttenuation: true,
    alphaTest: 0.02,
  });
  return new THREE.Points(geo, mat);
}

let _dotTextureCache = null;
function dotTexture() {
  if (_dotTextureCache) return _dotTextureCache;
  const size = 32;
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d');
  const grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.4, 'rgba(255,255,255,0.9)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  _dotTextureCache = new THREE.CanvasTexture(c);
  return _dotTextureCache;
}

let _glowTextureCache = null;
function glowTexture() {
  if (_glowTextureCache) return _glowTextureCache;
  const size = 128;
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d');
  const grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  grad.addColorStop(0, 'rgba(255,255,255,0.9)');
  grad.addColorStop(0.35, 'rgba(255,255,255,0.35)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  _glowTextureCache = new THREE.CanvasTexture(c);
  return _glowTextureCache;
}

function buildNebulae(rng, group) {
  // 2-3 very faint additive sprites — pure atmosphere, opacity <= 0.05.
  const nebColors = [0x3d4dea, 0x8677ea, 0x2a3b8f];
  const n = 2 + Math.floor(rng() * 2); // 2 or 3
  for (let i = 0; i < n; i++) {
    const mat = new THREE.SpriteMaterial({
      map: glowTexture(),
      color: nebColors[i % nebColors.length],
      transparent: true,
      opacity: 0.03 + rng() * 0.02, // <= 0.05
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const sprite = new THREE.Sprite(mat);
    const scale = 90 + rng() * 70;
    sprite.scale.set(scale, scale, 1);
    sprite.position.set(
      (rng() * 2 - 1) * FIELD_RADIUS_MAX * 0.7,
      (rng() * 2 - 1) * FIELD_RADIUS_MAX * 0.7,
      (rng() * 2 - 1) * FIELD_RADIUS_MAX * 0.7,
    );
    group.add(sprite);
  }
}

function buildHero(rng, group) {
  const heroGroup = new THREE.Group();
  heroGroup.position.set(0, 0, -18);

  // The star: a bright basic-material sphere (no scene lights needed — a lit
  // PBR star would cost a real light + shadows for no visual gain here)
  // plus an additive corona sprite standing in for bloom.
  const starGeo = new THREE.SphereGeometry(2.1, 24, 16);
  const starMat = new THREE.MeshBasicMaterial({ color: 0xfff2d6 });
  const star = new THREE.Mesh(starGeo, starMat);
  heroGroup.add(star);

  const coronaMat = new THREE.SpriteMaterial({
    map: glowTexture(),
    color: 0xffe6ad,
    transparent: true,
    opacity: 0.55,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const corona = new THREE.Sprite(coronaMat);
  corona.scale.set(9, 9, 1);
  heroGroup.add(corona);

  // A dark planet on an inclined orbit that crosses the star's disc from the
  // camera's point of view once per period — the transit.
  const planetGeo = new THREE.SphereGeometry(0.42, 16, 12);
  const planetMat = new THREE.MeshBasicMaterial({ color: 0x1a1f33 });
  const planet = new THREE.Mesh(planetGeo, planetMat);
  heroGroup.add(planet);

  const orbitRadius = 4.4;
  const inclination = (6 + rng() * 3) * (Math.PI / 180); // slight tilt, degrees→rad

  group.add(heroGroup);
  return { heroGroup, planet, orbitRadius, inclination };
}

const NOOP_HANDLE = Object.freeze({ dispose() {}, setTransitPhase() {} });

/**
 * @param {HTMLCanvasElement} canvas
 * @param {{mode?: 'hero'|'ambient', seed?: number}} [opts]
 * @returns {{dispose: () => void, setTransitPhase: (cb: (t:number)=>void) => void}}
 */
export function initScene(canvas, { mode = 'ambient', seed = 42 } = {}) {
  if (!canvas) return NOOP_HANDLE;

  if (!fxEnabled()) {
    canvas.hidden = true;
    return NOOP_HANDLE;
  }

  const reducedMotion = window.matchMedia
    && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: (Math.min(window.devicePixelRatio || 1, 1.5)) <= 1,
      powerPreference: 'low-power',
    });
  } catch {
    canvas.hidden = true;
    return NOOP_HANDLE;
  }
  if (!renderer) {
    canvas.hidden = true;
    return NOOP_HANDLE;
  }

  try {
    const rng = makeRng(seed >>> 0);
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    renderer.setPixelRatio(dpr);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 600);
    camera.position.set(0, 0, 0.01);

    const starGroup = new THREE.Group();
    const starField = buildStarField(rng, STAR_COUNT_AMBIENT);
    starGroup.add(starField);
    buildNebulae(rng, starGroup);
    scene.add(starGroup);

    let hero = null;
    let transitCb = null;
    if (mode === 'hero') {
      hero = buildHero(rng, scene);
    }

    // ── Sizing (ResizeObserver — works for both fixed-fullscreen ambient
    //    canvases and an in-flow hero canvas sized by its container) ──────
    function resize() {
      const w = canvas.clientWidth || canvas.parentElement?.clientWidth || window.innerWidth;
      const h = canvas.clientHeight || canvas.parentElement?.clientHeight || window.innerHeight;
      if (w === 0 || h === 0) return;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h, false);
    }
    resize();
    let resizeObserver = null;
    if (typeof ResizeObserver !== 'undefined') {
      resizeObserver = new ResizeObserver(resize);
      resizeObserver.observe(canvas);
    } else {
      window.addEventListener('resize', resize);
    }

    // ── Mouse parallax (desktop only — touch/no-pointer devices just drift,
    //    per spec) ─────────────────────────────────────────────────────────
    let targetX = 0, targetY = 0; // target camera rotation (radians)
    function onPointerMove(e) {
      const nx = (e.clientX / window.innerWidth) * 2 - 1; // -1..1
      const ny = (e.clientY / window.innerHeight) * 2 - 1;
      targetY = nx * PARALLAX_MAX_RAD;
      targetX = -ny * PARALLAX_MAX_RAD;
    }
    const hasPointer = window.matchMedia && window.matchMedia('(pointer: fine)').matches;
    if (hasPointer && !reducedMotion) {
      window.addEventListener('pointermove', onPointerMove, { passive: true });
    }

    // ── Animation loop ───────────────────────────────────────────────────
    let rafId = null;
    let lastFrameTime = 0;
    let lastElapsed = 0;
    let running = false;

    function renderFrame(elapsedS) {
      const dt = Math.max(0, elapsedS - lastElapsed);
      lastElapsed = elapsedS;

      starGroup.rotation.y += DRIFT_RAD_PER_SEC * dt;
      camera.rotation.x += (targetX - camera.rotation.x) * PARALLAX_LERP;
      camera.rotation.y += (targetY - camera.rotation.y) * PARALLAX_LERP;

      if (hero) {
        const t = (elapsedS % TRANSIT_PERIOD_S) / TRANSIT_PERIOD_S; // 0..1
        const angle = t * Math.PI * 2;
        // t = 0 → planet at its closest point to the camera (mid-transit).
        hero.planet.position.set(
          Math.sin(angle) * hero.orbitRadius * Math.cos(hero.inclination),
          Math.sin(angle) * hero.orbitRadius * Math.sin(hero.inclination),
          Math.cos(angle) * hero.orbitRadius,
        );
        hero.planet.position.add(hero.heroGroup.position);
        if (transitCb) {
          try { transitCb(t); } catch { /* a broken page-side callback must not kill the scene */ }
        }
      }

      renderer.render(scene, camera);
    }

    function animate(now) {
      if (!running) return;
      rafId = requestAnimationFrame(animate);
      if (now - lastFrameTime < FRAME_INTERVAL_MS) return; // §0.3: ~30fps throttle
      lastFrameTime = now;
      renderFrame(now / 1000);
    }

    function start() {
      if (running) return;
      running = true;
      lastFrameTime = 0;
      rafId = requestAnimationFrame(animate);
    }
    function stop() {
      running = false;
      if (rafId !== null) cancelAnimationFrame(rafId);
      rafId = null;
    }

    function onVisibilityChange() {
      if (document.hidden) stop();
      else if (!reducedMotion) start();
    }
    document.addEventListener('visibilitychange', onVisibilityChange);

    if (reducedMotion) {
      // One static frame, no loop, no parallax, no transit callback spam —
      // still call it once so a light-curve listener has an initial phase.
      renderFrame(0);
      if (hero && transitCb) { try { transitCb(0); } catch { /* ignore */ } }
    } else if (!document.hidden) {
      start();
    }

    function dispose() {
      stop();
      document.removeEventListener('visibilitychange', onVisibilityChange);
      if (hasPointer) window.removeEventListener('pointermove', onPointerMove);
      if (resizeObserver) resizeObserver.disconnect();
      else window.removeEventListener('resize', resize);
      scene.traverse((obj) => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
          const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
          mats.forEach((m) => {
            if (m.map) m.map.dispose();
            m.dispose();
          });
        }
      });
      renderer.dispose();
    }

    return {
      dispose,
      setTransitPhase(cb) {
        transitCb = typeof cb === 'function' ? cb : null;
      },
    };
  } catch {
    // Any unexpected failure past context-creation (e.g. a driver quirk
    // mid-setup) must still degrade to the silent CSS fallback, not throw
    // into the caller's page script.
    try { renderer.dispose(); } catch { /* already broken, nothing to free */ }
    canvas.hidden = true;
    return NOOP_HANDLE;
  }
}
