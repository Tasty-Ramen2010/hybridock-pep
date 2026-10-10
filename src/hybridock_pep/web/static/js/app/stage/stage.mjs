// stage.js — the one full-page canvas behind the whole interface.
//
// It draws a protein (soft gray-blue tube) and a peptide (warm orange helix) that float against the
// page background: slow drift, drag to rotate, soft glow, thickness that changes with distance and a
// little depth-of-field. In dark mode it also draws faint twinkling stars.
//
// How screens use it (see main.js / ui/*.js):
//   stage.setSlot(element)        glide to sit centred on that element (null = middle of the page)
//   stage.setProtein(structure)   cross-fade to a new protein
//   stage.setPeptide(length)      make a helix for a peptide of that many residues
//   stage.setPeptideMode(mode)    'float' | 'tumble' | 'dock' | 'hidden'
//   stage.setDockPose(points)     where the peptide settles (flat x,y,z in the protein's own frame)
//   stage.setBox(box), stage.editBox(callbacks)   the translucent search box on the binding-site step
//
// All drawing is plain Canvas 2D (no WebGL, no libraries) so it is light on a laptop battery.

import {
  clamp, lerp, smoothstep, easeInOut, qAxisAngle, qMul, qNormalize, qToMat3,
  mat3Apply, mat3ApplyInverse, convexHull, pointInPolygon,
} from './math.mjs';
import { helixPoints } from '../peptide.mjs';
import { atomsInBox } from '../pdb.mjs';
import { BOX_MIN, BOX_MAX } from '../config.mjs';

const DEPTH_FACTOR = 4.5; // camera distance in protein radii: smaller = stronger perspective
const FOCUS = 0.15; //       depth (−1 far … +1 near) that is perfectly sharp
const TUBE_WIDTH = 1.05; //  protein tube thickness in Å
const QUALITY = [ // the stage starts at level 0 and steps down by itself if frames keep arriving late
  { dpr: 1.5, fps: 32, dof: true },
  { dpr: 1.25, fps: 24, dof: false },
  { dpr: 1, fps: 15, dof: false },
];

// ---- colour helpers ---------------------------------------------------------------------------
const hexToRgb = (hex) => {
  const h = hex.trim().replace('#', '');
  const f = h.length === 3 ? [...h].map((c) => c + c).join('') : h;
  return [parseInt(f.slice(0, 2), 16), parseInt(f.slice(2, 4), 16), parseInt(f.slice(4, 6), 16)];
};
const mix = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
const css = (c) => `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`;

/** Turn a structure into smooth tube geometry (Catmull-Rom through the C-alpha trace). */
function buildTube(structure) {
  const ca = structure.ca;
  const sub = ca.length > 900 ? 2 : ca.length > 450 ? 3 : 4;
  const pts = [], ptCA = [], tone = [], segs = [];
  const chainTone = new Map();
  for (const seg of structure.segments) {
    const t = chainTone.get(ca[seg[0]].chain) ?? chainTone.set(ca[seg[0]].chain, chainTone.size).get(ca[seg[0]].chain);
    for (let k = 0; k < seg.length - 1; k++) {
      const p0 = ca[seg[Math.max(k - 1, 0)]], p1 = ca[seg[k]], p2 = ca[seg[k + 1]], p3 = ca[seg[Math.min(k + 2, seg.length - 1)]];
      for (let s = 0; s < sub; s++) {
        const u = s / sub, u2 = u * u, u3 = u2 * u;
        for (const axis of ['x', 'y', 'z']) {
          pts.push(0.5 * (2 * p1[axis] + (-p0[axis] + p2[axis]) * u + (2 * p0[axis] - 5 * p1[axis] + 4 * p2[axis] - p3[axis]) * u2 + (-p0[axis] + 3 * p1[axis] - 3 * p2[axis] + p3[axis]) * u3));
        }
        ptCA.push(u < 0.5 ? seg[k] : seg[k + 1]);
        tone.push(t);
        if (s > 0 || k > 0) segs.push(pts.length / 3 - 2); // segment from previous point to this one
      }
    }
    const last = ca[seg[seg.length - 1]];
    pts.push(last.x, last.y, last.z);
    ptCA.push(seg[seg.length - 1]);
    tone.push(t);
    segs.push(pts.length / 3 - 2);
  }
  return { pts: Float32Array.from(pts), ptCA: Int32Array.from(ptCA), tone: Uint8Array.from(tone), segs: Int32Array.from(segs), n: pts.length / 3 };
}

export class Stage {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.level = 0; this._slow = 0; this._lastDrawAt = 0; this._hiddenAt = 0; this._occluded = false;
    this.dpr = Math.min(window.devicePixelRatio || 1, QUALITY[0].dpr);
    this.W = 0; this.H = 0;

    this.q = qNormalize(qMul(qAxisAngle([1, 0, 0], -0.5), qAxisAngle([0, 1, 0], 0.6)));
    this.angVel = [0, 0];
    this.dragging = false;
    this.t = 0;

    this.cur = { cx: 0, cy: 0, size: 300 };
    this.tgt = { cx: 0, cy: 0, size: 300 };
    this._laidOut = false;
    this.slot = null;

    this.P = null; // protein geometry
    this.structure = null;
    this.modelAlpha = 0;
    this._pendingProtein = null;

    this.pep = { n: 0, shape: null, mode: 'hidden', dockT: 0, alpha: 0, prev: null, next: null, tp: 1, seed: 0 };
    this.box = null; this.boxAlpha = 0; this.editing = null; this.inside = [];
    this.pulse = null;

    this.reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    this.pickable = false;
    this._settle = 0;
    this._running = false;

    this._bindEvents();
    this.refreshTheme();
    this.resize();
    this.start();
  }

  // ---- public API --------------------------------------------------------------------------
  /** Re-read colours from the CSS custom properties (call when theme or accent changes). */
  refreshTheme() {
    const s = getComputedStyle(document.documentElement);
    const get = (n, d) => hexToRgb(s.getPropertyValue(n) || d);
    this.dark = document.documentElement.dataset.theme === 'dark';
    this.col = {
      bg: get('--stage-haze', '#f7f2ea'), protein: get('--protein', '#7189a3'), peptide: get('--peptide', '#f08a3c'),
      accent: get('--accent', '#0f766e'), warn: get('--warn', '#b45309'), danger: get('--danger', '#dc2626'),
    };
    this.fontFamily = s.getPropertyValue('--font-sans') || 'sans-serif';
    this._buildPalettes();
    this.invalidate();
  }

  _buildPalettes() {
    const { protein, bg, peptide } = this.col;
    const tones = [-0.07, 0.0, 0.07, -0.035, 0.035];
    const white = [255, 255, 255], black = [0, 0, 0];
    this.pal = tones.map((tn) => {
      const base = tn < 0 ? mix(protein, black, -tn) : mix(protein, white, tn);
      const body = [], sheen = [];
      for (let b = 0; b < 16; b++) {
        const depth = b / 15;
        const haze = 0.66 * Math.pow(1 - depth, 1.25);
        body.push(css(mix(base, bg, haze)));
        sheen.push(css(mix(mix(base, white, this.dark ? 0.4 : 0.5), bg, haze)));
      }
      return { body, sheen };
    });
    const pBody = [], pSheen = [];
    for (let b = 0; b < 16; b++) {
      const haze = 0.3 * Math.pow(1 - b / 15, 1.2);
      pBody.push(css(mix(peptide, bg, haze)));
      pSheen.push(css(mix(mix(peptide, white, 0.55), bg, haze)));
    }
    this.pepPal = { body: pBody, sheen: pSheen };
  }

  /** Centre the model on this element (it glides there). Pass null for the middle of the page. */
  setSlot(el) {
    this.slotObserver?.disconnect();
    this.slot = el || null;
    if (el && typeof ResizeObserver !== 'undefined') {
      this.slotObserver = new ResizeObserver(() => this._measure());
      this.slotObserver.observe(el);
    }
    this._measure();
    if (!this._laidOut) { this.cur = { ...this.tgt }; this._laidOut = true; }
    this.invalidate();
  }

  /** Cross-fade to a new protein structure (from pdb.js parsePDB), or null to clear. */
  setProtein(structure) {
    if (structure === this.structure && this.P) return;
    if (!this.P || this.reduced || this.modelAlpha < 0.05) { this._applyProtein(structure); return; }
    this._pendingProtein = { structure };
    this.invalidate();
  }

  _applyProtein(structure) {
    this.structure = structure;
    this._fit = null;
    this.P = structure ? { ...buildTube(structure), center: structure.center, R: structure.radius, fitR: structure.fitRadius } : null;
    this._updateInside();
    this.invalidate();
  }

  /** Make a helix for a peptide of `n` residues. */
  setPeptide(n) {
    const pep = this.pep;
    if (pep.n === n) return;
    pep.n = n;
    pep.shape = n ? helixPoints(n) : null;
    pep.prev = pep.next = null;
    pep.tp = 1;
    this.invalidate();
  }

  /** 'float' (drifts beside the protein), 'tumble' (busy, while a run is going), 'dock', or 'hidden'. */
  setPeptideMode(mode) {
    this.pep.mode = mode;
    if (this.reduced) { this.pep.dockT = mode === 'dock' ? 1 : 0; this.pep.alpha = mode === 'hidden' ? 0 : 1; }
    this.invalidate();
  }

  /** Where the peptide settles: flat [x,y,z,…] in the protein's own coordinates (or null). */
  setDockPose(points) {
    const pep = this.pep;
    if (!points || points.length !== pep.n * 3) { pep.prev = pep.next = null; pep.tp = 1; this.invalidate(); return; }
    pep.prev = this._dockedLocal();
    pep.next = Float32Array.from(points);
    pep.tp = pep.prev && !this.reduced ? 0 : 1;
    this.invalidate();
  }

  /** While dragging the box: slide the settled peptide so it stays centred in it (no pushing out). */
  moveDock(center) {
    const pep = this.pep;
    if (!pep.next) { if (pep.shape) { pep.next = Float32Array.from(pep.shape); pep.prev = null; pep.tp = 1; } else return; }
    const c = [0, 0, 0], n = pep.n;
    for (let i = 0; i < n; i++) { c[0] += pep.next[i * 3]; c[1] += pep.next[i * 3 + 1]; c[2] += pep.next[i * 3 + 2]; }
    const d = [center[0] - c[0] / n, center[1] - c[1] / n, center[2] - c[2] / n];
    for (let i = 0; i < n; i++) { pep.next[i * 3] += d[0]; pep.next[i * 3 + 1] += d[1]; pep.next[i * 3 + 2] += d[2]; }
    pep.prev = null; pep.tp = 1;
    this.invalidate();
  }

  /** Show (or hide with null) the translucent search box: { center:[x,y,z], size, level:'ok'|'warn'|'bad' }. */
  setBox(box) {
    this.box = box ? { center: [...box.center], size: box.size, level: box.level || 'ok' } : null;
    this._updateInside();
    this.invalidate();
  }

  /** Recolour the box without rebuilding it ('ok' accent, 'warn' amber, 'bad' red). */
  setBoxLevel(level) {
    if (this.box && this.box.level !== level) { this.box.level = level; this.invalidate(); }
  }

  /**
   * Let the user drag/resize the box and click the protein to move it.
   * handlers: { onBoxChange({center,size}, {final}), onPick({chain,res,resName,local}) } — or null to stop.
   */
  editBox(handlers) {
    this.editing = handlers || null;
    this.canvas.tabIndex = handlers ? 0 : -1;
    this.canvas.setAttribute('aria-hidden', handlers ? 'false' : 'true');
    this.canvas.setAttribute('role', handlers ? 'application' : 'presentation');
    this.canvas.setAttribute('aria-label', handlers ? 'Binding-site box. Use arrow keys to move it, Page Up and Page Down to move it in depth, and plus and minus to resize it.' : '');
    this.canvas.style.touchAction = handlers ? 'none' : 'pan-y';
    if (!handlers) this.canvas.style.cursor = '';
    this.invalidate();
  }

  /** A soft expanding ring at a point in the protein's frame (feedback for "you clicked here"). */
  pingAt(local) { this.pulse = { local, t0: this.t }; this.invalidate(); }

  invalidate() {
    this._settle = 90;
    if (!this._running) this.start();
  }

  destroy() {
    cancelAnimationFrame(this._raf);
    clearTimeout(this._timer);
    this._running = false;
    removeEventListener('resize', this._onResize);
    removeEventListener('scroll', this._onScroll, true);
    this.slotObserver?.disconnect();
  }

  // ---- layout ------------------------------------------------------------------------------
  resize() {
    this.W = innerWidth; this.H = innerHeight;
    this.canvas.width = Math.round(this.W * this.dpr);
    this.canvas.height = Math.round(this.H * this.dpr);
    this.canvas.style.width = this.W + 'px';
    this.canvas.style.height = this.H + 'px';
    this._measure();
    this.invalidate();
  }

  _measure() {
    if (this.slot && this.slot.isConnected) {
      const r = this.slot.getBoundingClientRect();
      this.tgt = { cx: r.left + r.width / 2, cy: r.top + r.height / 2, size: Math.max(120, Math.min(r.width, r.height)) };
    } else {
      this.tgt = { cx: this.W / 2, cy: this.H * 0.46, size: Math.min(this.W, this.H) * 0.78 };
    }
    this.invalidate();
  }

  // ---- animation loop ----------------------------------------------------------------------
  start() {
    if (this._running) return;
    this._running = true;
    this._last = performance.now();
    this._raf = requestAnimationFrame(this._tick);
  }

  /** Is anything of the stage on screen? Not when a sheet covers the page or the model's slot has scrolled away. */
  _checkOccluded() {
    const sheet = !!document.querySelector('dialog[open]');
    let away = false;
    if (this.slot && this.slot.isConnected) {
      const r = this.slot.getBoundingClientRect();
      away = r.bottom < -40 || r.top > this.H + 40;
    }
    this._occluded = sheet || away;
  }

  /** Step the quality down when frames keep arriving late (a slow phone, a busy machine), so it never stutters. */
  _govern(now) {
    const gap = now - this._lastDrawAt;
    this._lastDrawAt = now;
    if (gap > 500 || this.dragging) return; // a pause (tab switch, sheet) says nothing about speed
    const target = 1000 / QUALITY[this.level].fps;
    this._slow = gap > target * 1.45 ? this._slow + 1 : Math.max(0, this._slow - 1);
    if (this._slow >= 24 && this.level < QUALITY.length - 1) {
      this.level++; this._slow = 0;
      this.dpr = Math.min(window.devicePixelRatio || 1, QUALITY[this.level].dpr);
      this.resize();
    }
  }

  _tick = (now) => {
    if (document.hidden) { this._running = false; return; }
    if (now - (this._checkedAt || 0) > 250) { this._checkedAt = now; this._checkOccluded(); }
    if (this._occluded && !this.dragging) { // nothing visible: idle at 4 checks a second instead of drawing
      this._timer = setTimeout(() => { this._raf = requestAnimationFrame(this._tick); }, 250);
      this._last = now; this._lastDrawAt = 0;
      return;
    }
    const interval = this.dragging ? 0 : 1000 / QUALITY[this.level].fps;
    if (now - this._last >= interval - 2) {
      const dt = Math.min(0.06, (now - this._last) / 1000);
      this._last = now;
      this._govern(now);
      this.t += dt;
      this._update(dt);
      this._draw();
      if (this.reduced) this._settle--;
    }
    if (!this.reduced || this._settle > 0) this._raf = requestAnimationFrame(this._tick);
    else this._running = false;
  };

  _update(dt) {
    const k = this.reduced ? 1 : 1 - Math.exp(-dt * 5);
    for (const key of ['cx', 'cy', 'size']) this.cur[key] += (this.tgt[key] - this.cur[key]) * k;

    // rotation: drag inertia + slow drift (drift pauses while the box is being edited)
    if (!this.dragging) {
      const drift = this.reduced || this.editing ? 0 : 0.11;
      const decay = Math.exp(-dt * 3);
      this.angVel[0] *= decay; this.angVel[1] *= decay;
      const yaw = this.angVel[0] * dt + drift * dt, pitch = this.angVel[1] * dt + drift * 0.18 * dt;
      if (Math.abs(yaw) + Math.abs(pitch) > 1e-6) {
        this.q = qNormalize(qMul(qMul(qAxisAngle([0, 1, 0], yaw), qAxisAngle([1, 0, 0], pitch)), this.q));
      }
    }

    // fades
    const fade = this.reduced ? 1 : 1 - Math.exp(-dt * 6);
    if (this._pendingProtein) {
      this.modelAlpha = Math.max(0, this.modelAlpha - (this.reduced ? 1 : dt * 4));
      if (this.modelAlpha <= 0.02) { this._applyProtein(this._pendingProtein.structure); this._pendingProtein = null; }
    } else if (this.P) {
      this.modelAlpha = Math.min(1, this.modelAlpha + (this.reduced ? 1 : dt * 2.5));
    }
    const pep = this.pep;
    pep.dockT += (((pep.mode === 'dock') ? 1 : 0) - pep.dockT) * (this.reduced ? 1 : 1 - Math.exp(-dt * 2.6));
    pep.alpha += (((pep.mode === 'hidden' || !pep.n) ? 0 : 1) - pep.alpha) * fade;
    if (pep.tp < 1) pep.tp = Math.min(1, pep.tp + dt / 0.7);
    this.boxAlpha += ((this.box ? 1 : 0) - this.boxAlpha) * fade;
  }

  // ---- peptide geometry --------------------------------------------------------------------
  _dockedLocal() {
    const { prev, next, tp } = this.pep;
    if (!next) return prev || null;
    if (!prev || tp >= 1) return next;
    const e = easeInOut(tp), out = new Float32Array(next.length);
    for (let i = 0; i < out.length; i++) out[i] = lerp(prev[i], next[i], e);
    return out;
  }

  /** Peptide C-alpha positions in view space (Å), mixing "floating" and "settled" by dockT. */
  _peptideView(M, c, R) {
    const pep = this.pep, n = pep.n;
    const F = this.P.fitR; // orbit and size follow the visible extent, not the farthest stray atom
    const e = easeInOut(clamp(pep.dockT, 0, 1));
    const tumble = pep.mode === 'tumble';
    const motion = this.reduced ? 0 : 1;
    const a = this.t * (tumble ? 0.9 : 0.28) * motion + 0.8;
    const radius = F * (tumble ? 0.62 : 0.9);
    const oc = [Math.cos(a) * radius, Math.sin(a * 1.3) * F * (tumble ? 0.3 : 0.38) - F * 0.1, Math.sin(a) * radius * 0.9];
    const spin = qToMat3(qAxisAngle([0.3, 1, 0.2], this.t * (tumble ? 3.2 : 0.9) * motion));
    const halfLen = Math.max(1, ((n - 1) * 1.5) / 2);
    const floatScale = clamp((0.42 * F) / halfLen, 1, 3.4); // exaggerated while floating so it reads on screen
    const sc = lerp(floatScale, 1, e);
    const docked = this._dockedLocal();
    const out = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const sx = pep.shape[i * 3], sy = pep.shape[i * 3 + 1], sz = pep.shape[i * 3 + 2];
      const f = mat3Apply(spin, sx * floatScale, sy * floatScale, sz * floatScale);
      const fx = oc[0] + f[0], fy = oc[1] + f[1], fz = oc[2] + f[2];
      let dx, dy, dz;
      if (docked) {
        const d = mat3Apply(M, docked[i * 3] - c[0], docked[i * 3 + 1] - c[1], docked[i * 3 + 2] - c[2]);
        dx = d[0]; dy = d[1]; dz = d[2];
      } else { // no pose yet: settle at the middle of the protein
        const d = mat3Apply(M, sx, sy, sz); dx = d[0]; dy = d[1]; dz = d[2];
      }
      out[i * 3] = lerp(fx, dx, e); out[i * 3 + 1] = lerp(fy, dy, e); out[i * 3 + 2] = lerp(fz, dz, e);
    }
    return { v: out, sc };
  }

  // ---- drawing -----------------------------------------------------------------------------
  _draw() {
    const { ctx, W, H, col } = this;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    // (no star field in the dark appearance: it is plain black, like the rest of the page)
    const P = this.P;
    if (!P || this.modelAlpha < 0.01) return;

    const bob = this.reduced ? 0 : Math.sin(this.t * 0.6) * 5;
    const cx = this.cur.cx, cy = this.cur.cy + bob;
    // Scale so the protein AND the search box fit the slot (a 30 Å box is big next to a small protein). The scale eases
    // towards its target so resizing the box never makes the scene jump.
    let boxFit = 0;
    if (this.box && this.boxAlpha > 0.01) { // farthest box corner from the protein's centre, capped so a stray box can't shrink the protein to a dot
      const b = this.box.center, c0 = P.center;
      const offset = Math.hypot(b[0] - c0[0], b[1] - c0[1], b[2] - c0[2]);
      boxFit = Math.min((offset + this.box.size * 0.87) * 1.05, P.fitR * 3);
    }
    const wantFit = Math.max(P.fitR * 1.1, boxFit);
    this._fit = this._fit == null || this.reduced ? wantFit : this._fit + (wantFit - this._fit) * 0.15;
    const R = P.R, S = (this.cur.size * 0.5) / this._fit, D = R * DEPTH_FACTOR;
    const M = qToMat3(this.q), c = P.center;
    this.view = { cx, cy, S, D, R, M };
    const A = this.modelAlpha;

    // (no halo behind the model: it sits on the plain page, like a product shot)

    // transform the protein tube
    const n = P.n, pts = P.pts;
    if (!this._buf || this._buf.n < n) this._buf = { n, vz: new Float32Array(n), sx: new Float32Array(n), sy: new Float32Array(n), sf: new Float32Array(n), key: new Float32Array(n + 4096) };
    const { vz, sx, sy, sf } = this._buf;
    for (let i = 0; i < n; i++) {
      const v = mat3Apply(M, pts[i * 3] - c[0], pts[i * 3 + 1] - c[1], pts[i * 3 + 2] - c[2]);
      const f = D / (D - v[2]);
      vz[i] = v[2]; sf[i] = f; sx[i] = cx + v[0] * f * S; sy[i] = cy + v[1] * f * S;
    }

    // transform + smooth the peptide
    const pep = this.pep, hasPep = pep.n > 0 && pep.alpha > 0.01 && pep.shape;
    let ps = null;
    if (hasPep) {
      const { v, sc } = this._peptideView(M, c, R);
      const sub = 3, m = (pep.n - 1) * sub + 1;
      ps = { m, sub, sc, x: new Float32Array(m), y: new Float32Array(m), z: new Float32Array(m), sx: new Float32Array(m), sy: new Float32Array(m), sf: new Float32Array(m) };
      for (let k = 0; k < pep.n - 1; k++) {
        const g = (j) => Math.min(Math.max(j, 0), pep.n - 1);
        for (let s = 0; s < sub; s++) {
          const u = s / sub, u2 = u * u, u3 = u2 * u, idx = k * sub + s;
          const w = [0, 0, 0];
          for (let ax = 0; ax < 3; ax++) {
            const p0 = v[g(k - 1) * 3 + ax], p1 = v[k * 3 + ax], p2 = v[g(k + 1) * 3 + ax], p3 = v[g(k + 2) * 3 + ax];
            w[ax] = 0.5 * (2 * p1 + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2 + (-p0 + 3 * p1 - 3 * p2 + p3) * u3);
          }
          ps.x[idx] = w[0]; ps.y[idx] = w[1]; ps.z[idx] = w[2];
        }
      }
      const last = pep.n - 1;
      ps.x[m - 1] = v[last * 3]; ps.y[m - 1] = v[last * 3 + 1]; ps.z[m - 1] = v[last * 3 + 2];
      for (let i = 0; i < m; i++) {
        const f = D / (D - ps.z[i]);
        ps.sf[i] = f; ps.sx[i] = cx + ps.x[i] * f * S; ps.sy[i] = cy + ps.y[i] * f * S;
      }
    }

    // painter's algorithm: far → near, protein and peptide pieces together
    const segs = P.segs, nSeg = segs.length, nPep = ps ? ps.m - 1 : 0, total = nSeg + nPep;
    if (!this._order || this._order.length < total) { this._order = new Int32Array(total); this._keys = new Float32Array(total); }
    const order = this._order.subarray(0, total), keys = this._keys;
    for (let s = 0; s < nSeg; s++) { order[s] = s; keys[s] = vz[segs[s]] + vz[segs[s] + 1]; }
    for (let j = 0; j < nPep; j++) { order[nSeg + j] = nSeg + j; keys[nSeg + j] = ps.z[j] + ps.z[j + 1]; }
    order.sort((a, b) => keys[a] - keys[b]);

    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    const pw = S * TUBE_WIDTH;
    // The peptide is drawn thicker once it settles, so it stays the star next to a big protein.
    const peptideThick = lerp(1.5, 2.7, easeInOut(clamp(pep.dockT, 0, 1)));
    for (let o = 0; o < total; o++) {
      const id = order[o];
      if (id < nSeg) {
        const i = segs[id];
        const zn = clamp((vz[i] + vz[i + 1]) / (2 * R), -1.2, 1.2);
        const depth = clamp((zn + 1) / 2, 0, 1), bucket = Math.round(depth * 15);
        const f = (sf[i] + sf[i + 1]) / 2;
        const w = Math.max(1.3, pw * f);
        const dof = QUALITY[this.level].dof ? smoothstep(0.32, 1.15, Math.abs(zn - FOCUS)) * (this.reduced ? 0.5 : 1) : 0;
        const pal = this.pal[P.tone[i] % 5];
        const a = A * (0.5 + 0.5 * depth);
        ctx.beginPath(); ctx.moveTo(sx[i], sy[i]); ctx.lineTo(sx[i + 1], sy[i + 1]);
        if (dof > 0.03) { // out of focus: a wide, faint stroke underneath
          ctx.globalAlpha = a * 0.16 * dof; ctx.strokeStyle = pal.body[bucket]; ctx.lineWidth = w * (1 + dof * 1.8); ctx.stroke();
        }
        ctx.globalAlpha = a * (1 - 0.5 * dof); ctx.strokeStyle = pal.body[bucket]; ctx.lineWidth = w; ctx.stroke();
      } else {
        const j = id - nSeg;
        const zn = clamp((ps.z[j] + ps.z[j + 1]) / (2 * R), -1.2, 1.2);
        const depth = clamp((zn + 1) / 2, 0, 1), bucket = Math.round(depth * 15);
        const f = (ps.sf[j] + ps.sf[j + 1]) / 2;
        const w = Math.max(3.5, pw * peptideThick * f * Math.sqrt(ps.sc));
        const a = A * pep.alpha * (0.7 + 0.3 * depth);
        ctx.beginPath(); ctx.moveTo(ps.sx[j], ps.sy[j]); ctx.lineTo(ps.sx[j + 1], ps.sy[j + 1]);
        ctx.globalAlpha = a; ctx.strokeStyle = this.pepPal.body[bucket]; ctx.lineWidth = w; ctx.stroke();
        if (j % ps.sub === 0) {
          ctx.globalAlpha = a; ctx.fillStyle = this.pepPal.body[bucket];
          ctx.beginPath(); ctx.arc(ps.sx[j], ps.sy[j], w * 0.62, 0, 6.2832); ctx.fill();
          ctx.globalAlpha = a * 0.7; ctx.fillStyle = this.pepPal.sheen[bucket];
          ctx.beginPath(); ctx.arc(ps.sx[j] - w * 0.18, ps.sy[j] - w * 0.2, w * 0.2, 0, 6.2832); ctx.fill();
        }
      }
    }
    ctx.globalAlpha = 1;

    if (this.boxAlpha > 0.01 && this.box) this._drawBox(A);
    if (this.pulse) this._drawPulse();
  }

  /** Project a point in the protein's frame to the screen: [x, y, depth]. */
  _project(p) {
    const { cx, cy, S, D, M } = this.view;
    const c = this.P.center;
    const v = mat3Apply(M, p[0] - c[0], p[1] - c[1], p[2] - c[2]);
    const f = D / (D - v[2]);
    return [cx + v[0] * f * S, cy + v[1] * f * S, v[2]];
  }

  _boxCorners() {
    const { center: b, size } = this.box, h = size / 2, out = [];
    for (let i = 0; i < 8; i++) out.push(this._project([b[0] + (i & 1 ? h : -h), b[1] + (i & 2 ? h : -h), b[2] + (i & 4 ? h : -h)]));
    return out;
  }

  _drawBox(A) {
    const { ctx, col } = this;
    const tint = { ok: col.accent, warn: col.warn, bad: col.danger }[this.box.level] || col.accent;
    const rgb = tint.map((v) => v | 0).join(',');
    const k = this.boxAlpha * A;
    const corners = this._boxCorners();
    const hull = convexHull(corners.map((p) => [p[0], p[1]]));
    this._boxHull = hull; this._boxCornersScreen = corners;

    // faint fill + the dots of protein atoms that are inside the box
    ctx.globalAlpha = k * 0.1; ctx.fillStyle = `rgb(${rgb})`;
    ctx.beginPath(); hull.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]))); ctx.closePath(); ctx.fill();
    if (this.inside.length) {
      ctx.globalAlpha = k * 0.3; ctx.fillStyle = `rgb(${rgb})`;
      const S = this.structure;
      const step = Math.ceil(this.inside.length / 500);
      for (let m = 0; m < this.inside.length; m += step) {
        const i = this.inside[m], p = this._project([S.x[i], S.y[i], S.z[i]]);
        ctx.fillRect(p[0] - 1, p[1] - 1, 2, 2);
      }
    }

    // edges: the far ones fade so the box reads as 3D
    const edges = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
    const zs = corners.map((p) => p[2]);
    const zMid = (Math.min(...zs) + Math.max(...zs)) / 2;
    ctx.lineWidth = 1.5; ctx.strokeStyle = `rgb(${rgb})`;
    for (const [a, b] of edges) {
      const near = (corners[a][2] + corners[b][2]) / 2 > zMid;
      ctx.globalAlpha = k * (near ? 0.95 : 0.4);
      ctx.setLineDash(near ? [] : [4, 4]);
      ctx.beginPath(); ctx.moveTo(corners[a][0], corners[a][1]); ctx.lineTo(corners[b][0], corners[b][1]); ctx.stroke();
    }
    ctx.setLineDash([]);

    if (this.editing) { // corner handles
      for (const p of corners) {
        ctx.globalAlpha = k; ctx.fillStyle = this.dark ? '#0b1224' : '#fff';
        ctx.beginPath(); ctx.arc(p[0], p[1], 6, 0, 6.2832); ctx.fill();
        ctx.lineWidth = 2; ctx.strokeStyle = `rgb(${rgb})`; ctx.stroke();
      }
    }
    // size label above the box
    const top = Math.min(...hull.map((p) => p[1])), midX = hull.reduce((s, p) => s + p[0], 0) / hull.length;
    ctx.globalAlpha = k; ctx.fillStyle = `rgb(${rgb})`;
    ctx.font = `600 12px ${this.fontFamily}`; ctx.textAlign = 'center';
    ctx.fillText(`${Math.round(this.box.size)} Å box`, midX, top - 12);
    ctx.globalAlpha = 1;
  }

  _drawPulse() {
    const age = (this.t - this.pulse.t0) / 0.7;
    if (age >= 1) { this.pulse = null; return; }
    const p = this._project(this.pulse.local), ctx = this.ctx;
    ctx.globalAlpha = (1 - age) * 0.8; ctx.strokeStyle = `rgb(${this.col.accent.map((v) => v | 0).join(',')})`;
    ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(p[0], p[1], 6 + age * 26, 0, 6.2832); ctx.stroke();
    ctx.globalAlpha = 1;
  }

  _updateInside() {
    this.inside = this.box && this.structure ? atomsInBox(this.structure, this.box.center, this.box.size) : [];
  }

  // ---- interaction -------------------------------------------------------------------------
  _bindEvents() {
    const cv = this.canvas;
    this._onResize = () => this.resize();
    this._onScroll = () => { if (this._scrollQueued) return; this._scrollQueued = true; requestAnimationFrame(() => { this._scrollQueued = false; this._measure(); }); };
    addEventListener('resize', this._onResize);
    addEventListener('scroll', this._onScroll, true);
    document.addEventListener('visibilitychange', () => { if (!document.hidden) this.invalidate(); });
    matchMedia('(prefers-reduced-motion: reduce)').addEventListener?.('change', (e) => { this.reduced = e.matches; this.invalidate(); });
    cv.addEventListener('pointerdown', (e) => this._down(e));
    cv.addEventListener('pointermove', (e) => this._move(e));
    cv.addEventListener('pointerup', (e) => this._up(e));
    cv.addEventListener('pointercancel', (e) => this._up(e));
    cv.addEventListener('keydown', (e) => this._key(e));
  }

  _hit(x, y) {
    if (!this.editing || !this.box || !this._boxCornersScreen) return null;
    const corners = this._boxCornersScreen;
    for (let i = 0; i < corners.length; i++) if (Math.hypot(x - corners[i][0], y - corners[i][1]) < 13) return { kind: 'corner' };
    if (this._boxHull && pointInPolygon(x, y, this._boxHull)) return { kind: 'inside' };
    return null;
  }

  _nearestBackbone(x, y) {
    const P = this.P, b = this._buf;
    if (!P || !b) return null;
    // Among backbone points within 18 px of the pointer, prefer the closest one that faces the viewer.
    let best = -1, bestScore = Infinity;
    for (let i = 0; i < P.n; i++) {
      const d2 = (b.sx[i] - x) ** 2 + (b.sy[i] - y) ** 2;
      if (d2 > 324) continue;
      const score = Math.sqrt(d2) - (b.vz[i] / P.R) * 8;
      if (score < bestScore) { bestScore = score; best = i; }
    }
    if (best < 0) return null;
    const ca = this.structure.ca[P.ptCA[best]];
    return { chain: ca.chain, res: ca.res, resName: ca.resName, local: [P.pts[best * 3], P.pts[best * 3 + 1], P.pts[best * 3 + 2]] };
  }

  _down(e) {
    if (e.button !== 0 || !this.P) return;
    const hit = this._hit(e.clientX, e.clientY);
    this.drag = { x: e.clientX, y: e.clientY, sx: e.clientX, sy: e.clientY, mode: hit ? (hit.kind === 'corner' ? 'resize' : 'move') : 'rotate', moved: 0, size0: this.box?.size };
    if (this.drag.mode === 'resize') {
      const cs = this._project(this.box.center);
      this.drag.d0 = Math.max(10, Math.hypot(e.clientX - cs[0], e.clientY - cs[1]));
    }
    this.dragging = this.drag.mode === 'rotate';
    this.angVel = [0, 0];
    this.canvas.setPointerCapture?.(e.pointerId);
    this.invalidate();
  }

  _move(e) {
    if (!this.drag) { this._hover(e); return; }
    const d = this.drag, dx = e.clientX - d.x, dy = e.clientY - d.y;
    d.moved += Math.abs(dx) + Math.abs(dy);
    d.x = e.clientX; d.y = e.clientY;
    if (d.mode === 'rotate') {
      this.q = qNormalize(qMul(qMul(qAxisAngle([0, 1, 0], dx * 0.008), qAxisAngle([1, 0, 0], dy * 0.008)), this.q));
      this.angVel = [dx * 0.5, dy * 0.5];
    } else if (d.mode === 'move') {
      const { S, D, M } = this.view, cs = this._project(this.box.center);
      const f = D / (D - cs[2]);
      const local = mat3ApplyInverse(M, dx / (S * f), dy / (S * f), 0);
      this._nudge(local, false);
    } else if (d.mode === 'resize') {
      const cs = this._project(this.box.center);
      const dist = Math.max(10, Math.hypot(e.clientX - cs[0], e.clientY - cs[1]));
      this._resize(d.size0 * (dist / d.d0), false);
    }
    this.invalidate();
  }

  _up(e) {
    const d = this.drag;
    if (!d) return;
    this.drag = null; this.dragging = false;
    try { this.canvas.releasePointerCapture?.(e.pointerId); } catch { /* already released */ }
    if (d.mode === 'rotate' && d.moved < 5 && this.editing) {
      const hit = this._nearestBackbone(e.clientX, e.clientY);
      if (hit) { this.pingAt(hit.local); this.editing.onPick?.(hit); }
    } else if (d.mode !== 'rotate') {
      this.editing?.onBoxChange?.({ center: [...this.box.center], size: this.box.size }, { final: true });
    }
    this.invalidate();
  }

  _hover(e) {
    if (!this.editing) { this.canvas.style.cursor = ''; return; }
    const hit = this._hit(e.clientX, e.clientY);
    this.canvas.style.cursor = hit ? (hit.kind === 'corner' ? 'nwse-resize' : 'move') : this._nearestBackbone(e.clientX, e.clientY) ? 'pointer' : 'grab';
  }

  _nudge(localDelta, final) {
    const b = this.box;
    b.center = [b.center[0] + localDelta[0], b.center[1] + localDelta[1], b.center[2] + localDelta[2]];
    this._updateInside();
    this.moveDock(b.center);
    this.editing?.onBoxChange?.({ center: [...b.center], size: b.size }, { final });
  }

  _resize(size, final) {
    const b = this.box;
    b.size = clamp(size, BOX_MIN, BOX_MAX);
    this._updateInside();
    this.editing?.onBoxChange?.({ center: [...b.center], size: b.size }, { final });
  }

  _key(e) {
    if (!this.editing || !this.box || !this.view) return;
    const step = e.shiftKey ? 5 : 1;
    const v = { ArrowLeft: [-step, 0, 0], ArrowRight: [step, 0, 0], ArrowUp: [0, -step, 0], ArrowDown: [0, step, 0], PageUp: [0, 0, step], PageDown: [0, 0, -step] }[e.key];
    if (v) { e.preventDefault(); this._nudge(mat3ApplyInverse(this.view.M, ...v), true); this.invalidate(); return; }
    if (e.key === '+' || e.key === '=') { e.preventDefault(); this._resize(this.box.size + 2 * step, true); this.invalidate(); }
    if (e.key === '-' || e.key === '_') { e.preventDefault(); this._resize(this.box.size - 2 * step, true); this.invalidate(); }
  }
}
