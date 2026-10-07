// placement.js — put the decorative helix next to a binding site without sitting inside the protein.
// Used for the "settled in the box" look and for the demo poses. Not a docking algorithm.

import { helixPoints } from './peptide.mjs';
import { nearestAtom } from './pdb.mjs';
import { mat3Apply, mat3Mul, rotationBetween, qAxisAngle, qToMat3 } from './stage/math.mjs';

/** Unit vector pointing from the protein towards the open space around `center`. */
function outwardDirection(structure, center) {
  let sx = 0, sy = 0, sz = 0, count = 0;
  const r2 = 12 * 12;
  for (let i = 0; i < structure.n; i += 3) {
    const dx = structure.x[i] - center[0], dy = structure.y[i] - center[1], dz = structure.z[i] - center[2];
    if (dx * dx + dy * dy + dz * dz < r2) { sx += dx; sy += dy; sz += dz; count++; }
  }
  let v = count ? [-sx / count, -sy / count, -sz / count] : [0, 0, 0];
  if (Math.hypot(...v) < 0.5) v = [center[0] - structure.center[0], center[1] - structure.center[1], center[2] - structure.center[2]];
  const l = Math.hypot(...v) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

/**
 * A helix for `n` residues sitting near `center`, lying along the protein surface and pushed
 * out of any overlap. `rng` is a seeded random function so the result is repeatable.
 * @returns {Float32Array} flat coordinates in the protein's own frame
 */
export function placeHelixAt(structure, n, center, rng, { clearance = 3.4 } = {}) {
  const shape = helixPoints(n);
  const out = outwardDirection(structure, center);

  // Choose a helix axis perpendicular to "outward", then a random roll about it.
  let t = [rng() - 0.5, rng() - 0.5, rng() - 0.5];
  const dot = t[0] * out[0] + t[1] * out[1] + t[2] * out[2];
  t = [t[0] - dot * out[0], t[1] - dot * out[1], t[2] - dot * out[2]];
  const tl = Math.hypot(...t) || 1;
  const axis = [t[0] / tl, t[1] / tl, t[2] / tl];
  const toAxis = rotationBetween([1, 0, 0], axis);
  const roll = qToMat3(qAxisAngle(axis, rng() * Math.PI * 2));
  const m = mat3Mul(roll, toAxis);

  const pts = new Float32Array(shape.length);
  for (let i = 0; i < n; i++) {
    const p = mat3Apply(m, shape[i * 3], shape[i * 3 + 1], shape[i * 3 + 2]);
    pts[i * 3] = p[0] + center[0] + out[0] * 3;
    pts[i * 3 + 1] = p[1] + center[1] + out[1] * 3;
    pts[i * 3 + 2] = p[2] + center[2] + out[2] * 3;
  }

  // Nudge away from any protein atom closer than `clearance` Å.
  for (let iter = 0; iter < 30; iter++) {
    let px = 0, py = 0, pz = 0, hits = 0;
    for (let i = 0; i < n; i++) {
      const hit = nearestAtom(structure, pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2], clearance);
      if (!hit) continue;
      const dx = pts[i * 3] - structure.x[hit.index], dy = pts[i * 3 + 1] - structure.y[hit.index], dz = pts[i * 3 + 2] - structure.z[hit.index];
      const d = Math.max(hit.distance, 0.3), push = (clearance - hit.distance) / d;
      px += dx * push; py += dy * push; pz += dz * push; hits++;
    }
    if (!hits) break;
    const k = 0.7 / hits;
    for (let i = 0; i < n; i++) { pts[i * 3] += px * k; pts[i * 3 + 1] += py * k; pts[i * 3 + 2] += pz * k; }
  }
  return pts;
}
