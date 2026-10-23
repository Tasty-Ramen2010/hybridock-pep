// collide.mjs — keep the drawn peptide out of the drawn protein.
// The floating, tumbling and settling peptide is moved as one rigid piece, just far enough that no part of it is closer
// than `clearance` Å to the protein's backbone. Works in the protein's own frame, so it does not care how the view is turned.

const KEY = 1024, HALF = 512;

/** A lookup grid over flat [x,y,z,…] points; `cell` should be the largest clearance you will ask for. */
export function makeGrid(pts, cell) {
  const cells = new Map();
  const n = pts.length / 3;
  for (let i = 0; i < n; i++) {
    const k = cellKey(Math.floor(pts[i * 3] / cell), Math.floor(pts[i * 3 + 1] / cell), Math.floor(pts[i * 3 + 2] / cell));
    const bucket = cells.get(k);
    if (bucket) bucket.push(i); else cells.set(k, [i]);
  }
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < n; i++) { cx += pts[i * 3]; cy += pts[i * 3 + 1]; cz += pts[i * 3 + 2]; }
  return { pts, cell, cells, centre: n ? [cx / n, cy / n, cz / n] : [0, 0, 0] };
}

const cellKey = (ix, iy, iz) => ((ix + HALF) * KEY + (iy + HALF)) * KEY + (iz + HALF);

/** The deepest overlap of one point with the grid: { d, dx, dy, dz } to the nearest too-close point, or null. */
function nearest(grid, x, y, z, clearance) {
  const { pts, cell, cells } = grid;
  const ix = Math.floor(x / cell), iy = Math.floor(y / cell), iz = Math.floor(z / cell);
  let best = null, bd = clearance;
  for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let c = -1; c <= 1; c++) {
    const bucket = cells.get(cellKey(ix + a, iy + b, iz + c));
    if (!bucket) continue;
    for (const j of bucket) {
      const dx = x - pts[j * 3], dy = y - pts[j * 3 + 1], dz = z - pts[j * 3 + 2];
      const d = Math.hypot(dx, dy, dz);
      if (d < bd) { bd = d; best = { d, dx, dy, dz }; }
    }
  }
  return best;
}

/** Points to test along a peptide: every C-alpha and the middle of every link between two. */
function probes(pep, n) {
  const out = new Float32Array((2 * n - 1) * 3);
  for (let i = 0; i < n; i++) {
    out[i * 3] = pep[i * 3]; out[i * 3 + 1] = pep[i * 3 + 1]; out[i * 3 + 2] = pep[i * 3 + 2];
  }
  for (let i = 0; i < n - 1; i++) for (let k = 0; k < 3; k++) out[(n + i) * 3 + k] = (pep[i * 3 + k] + pep[(i + 1) * 3 + k]) / 2;
  return out;
}

/** Closest approach (Å) of the peptide (flat C-alpha positions) to the protein points; Infinity if nothing is within `within`. */
export function closestApproach(grid, pep, n, within = 12) {
  const p = probes(pep, n), m = p.length / 3;
  let best = Infinity;
  for (let i = 0; i < m; i++) {
    const hit = nearest(grid, p[i * 3], p[i * 3 + 1], p[i * 3 + 2], within);
    if (hit && hit.d < best) best = hit.d;
  }
  return best;
}

/**
 * The rigid shift [x,y,z] that puts the peptide at least `clearance` Å from every protein point (zero when it already is).
 * It pushes away from the deepest overlap and tries again; if that has not settled in a few rounds it backs the whole peptide
 * straight out from the protein's middle, which always ends in clear space.
 */
export function clearShift(grid, pep, n, clearance) {
  const shift = [0, 0, 0];
  if (!grid || n < 1 || clearance <= 0) return shift;
  const p = probes(pep, n), m = p.length / 3;
  for (let round = 0; round < 14; round++) {
    let worst = null;
    for (let i = 0; i < m; i++) {
      const hit = nearest(grid, p[i * 3] + shift[0], p[i * 3 + 1] + shift[1], p[i * 3 + 2] + shift[2], clearance);
      if (hit && (!worst || clearance - hit.d > clearance - worst.d)) worst = hit;
    }
    if (!worst) return shift;
    const d = Math.max(worst.d, 0.05), push = (clearance - worst.d + 0.05) / d;
    shift[0] += worst.dx * push; shift[1] += worst.dy * push; shift[2] += worst.dz * push;
  }
  // fall back: straight out from the middle of the protein until clear
  let ox = 0, oy = 0, oz = 0;
  for (let i = 0; i < n; i++) { ox += pep[i * 3]; oy += pep[i * 3 + 1]; oz += pep[i * 3 + 2]; }
  const [gx, gy, gz] = grid.centre;
  let ux = ox / n + shift[0] - gx, uy = oy / n + shift[1] - gy, uz = oz / n + shift[2] - gz;
  const l = Math.hypot(ux, uy, uz) || 1;
  ux /= l; uy /= l; uz /= l;
  for (let step = 0; step < 80; step++) {
    let clear = true;
    for (let i = 0; i < m && clear; i++) clear = !nearest(grid, p[i * 3] + shift[0], p[i * 3 + 1] + shift[1], p[i * 3 + 2] + shift[2], clearance);
    if (clear) break;
    shift[0] += ux * 2; shift[1] += uy * 2; shift[2] += uz * 2;
  }
  return shift;
}
