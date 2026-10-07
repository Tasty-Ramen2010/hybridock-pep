// pdb.js — read a PDB file in the browser and answer simple questions about it.
// A "structure" is just typed arrays of atom positions plus the C-alpha backbone trace.

const THREE_TO_ONE = {
  ALA: 'A', ARG: 'R', ASN: 'N', ASP: 'D', CYS: 'C', GLN: 'Q', GLU: 'E', GLY: 'G', HIS: 'H', ILE: 'I',
  LEU: 'L', LYS: 'K', MET: 'M', PHE: 'F', PRO: 'P', SER: 'S', THR: 'T', TRP: 'W', TYR: 'Y', VAL: 'V',
  MSE: 'M',
};

/** CA–CA distance above which we treat the chain as broken (ideal is 3.8 Å). */
const CHAIN_BREAK = 4.6;

/**
 * Parse PDB text (first model only, heavy atoms only, first alt-location).
 * @returns {{
 *   n:number, x:Float32Array, y:Float32Array, z:Float32Array,
 *   chain:string[], res:Int32Array, resName:string[],
 *   ca:{chain:string,res:number,resName:string,x:number,y:number,z:number}[],
 *   segments:number[][], center:number[], radius:number, bounds:{min:number[],max:number[]}
 * }}
 */
export function parsePDB(text) {
  const xs = [], ys = [], zs = [], chain = [], res = [], resName = [], ca = [];
  for (const line of text.split(/\r?\n/)) {
    if (line.startsWith('ENDMDL')) break;
    const isAtom = line.startsWith('ATOM');
    if (!isAtom && !(line.startsWith('HETATM') && line.slice(17, 20) === 'MSE')) continue;
    const alt = line[16];
    if (alt !== ' ' && alt !== 'A' && alt !== undefined) continue;
    const name = line.slice(12, 16).trim();
    const element = (line.slice(76, 78).trim() || name.replace(/[0-9]/g, '').slice(0, 1)).toUpperCase();
    if (element === 'H' || element === 'D') continue;
    const x = parseFloat(line.slice(30, 38));
    const y = parseFloat(line.slice(38, 46));
    const z = parseFloat(line.slice(46, 54));
    if (!Number.isFinite(x + y + z)) continue;
    xs.push(x); ys.push(y); zs.push(z);
    chain.push(line[21]);
    res.push(parseInt(line.slice(22, 26), 10));
    resName.push(line.slice(17, 20).trim());
    if (name === 'CA') ca.push({ chain: line[21], res: res[res.length - 1], resName: resName[resName.length - 1], x, y, z });
  }
  const n = xs.length;
  if (n === 0) throw new Error('No protein atoms found. Is this a PDB file?');
  const s = {
    n, x: Float32Array.from(xs), y: Float32Array.from(ys), z: Float32Array.from(zs),
    chain, res: Int32Array.from(res), resName, ca, segments: [],
  };
  // Backbone pieces: consecutive C-alphas of the same chain with no big gap.
  let seg = [];
  for (let i = 0; i < ca.length; i++) {
    const a = ca[i], p = ca[i - 1];
    const broken = !p || p.chain !== a.chain || Math.hypot(a.x - p.x, a.y - p.y, a.z - p.z) > CHAIN_BREAK;
    if (broken && seg.length) { s.segments.push(seg); seg = []; }
    seg.push(i);
  }
  if (seg.length) s.segments.push(seg);
  s.segments = s.segments.filter((g) => g.length >= 2);

  // Centre and size come from the backbone when we have one, otherwise from all atoms.
  const pts = ca.length ? ca : xs.map((_, i) => ({ x: xs[i], y: ys[i], z: zs[i] }));
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  let cx = 0, cy = 0, cz = 0;
  for (const p of pts) {
    cx += p.x; cy += p.y; cz += p.z;
    min[0] = Math.min(min[0], p.x); min[1] = Math.min(min[1], p.y); min[2] = Math.min(min[2], p.z);
    max[0] = Math.max(max[0], p.x); max[1] = Math.max(max[1], p.y); max[2] = Math.max(max[2], p.z);
  }
  s.center = [cx / pts.length, cy / pts.length, cz / pts.length];
  const dists = pts.map((p) => Math.hypot(p.x - s.center[0], p.y - s.center[1], p.z - s.center[2]));
  s.radius = Math.max(1, ...dists);
  // "Fit radius": ignores the farthest 15% so long thin shapes (fibrils) still fill the view.
  s.fitRadius = Math.max(1, dists.slice().sort((a, b) => a - b)[Math.floor(dists.length * 0.85)] || 1);
  s.bounds = { min, max };
  return s;
}

/** One-letter sequence of the C-alpha trace (optionally one chain). Unknown residues become X. */
export function sequenceOf(structure, chainId) {
  return structure.ca
    .filter((a) => !chainId || a.chain === chainId)
    .map((a) => THREE_TO_ONE[a.resName] || 'X')
    .join('');
}

// ---- simple spatial grid so "what is near this point?" is fast ---------------------------------
const CELL = 5; // Å
const cellKey = (ix, iy, iz) => ((ix + 1024) * 2048 + (iy + 1024)) * 2048 + (iz + 1024);

function grid(structure) {
  if (structure._grid) return structure._grid;
  const g = new Map();
  for (let i = 0; i < structure.n; i++) {
    const k = cellKey(Math.floor(structure.x[i] / CELL), Math.floor(structure.y[i] / CELL), Math.floor(structure.z[i] / CELL));
    const bucket = g.get(k);
    if (bucket) bucket.push(i); else g.set(k, [i]);
  }
  structure._grid = g;
  return g;
}

/** Nearest protein atom within maxR Å of a point, or null. */
export function nearestAtom(structure, x, y, z, maxR = 8) {
  const g = grid(structure);
  const r = Math.ceil(maxR / CELL);
  const cx = Math.floor(x / CELL), cy = Math.floor(y / CELL), cz = Math.floor(z / CELL);
  let best = null, bestD2 = maxR * maxR;
  for (let a = -r; a <= r; a++) for (let b = -r; b <= r; b++) for (let c = -r; c <= r; c++) {
    const bucket = g.get(cellKey(cx + a, cy + b, cz + c));
    if (!bucket) continue;
    for (const i of bucket) {
      const d2 = (structure.x[i] - x) ** 2 + (structure.y[i] - y) ** 2 + (structure.z[i] - z) ** 2;
      if (d2 < bestD2) { bestD2 = d2; best = { index: i, distance: Math.sqrt(d2) }; }
    }
  }
  return best;
}

/** Indices of protein atoms inside an axis-aligned box (the backend's box is axis-aligned). */
export function atomsInBox(structure, center, size) {
  const h = size / 2, out = [];
  for (let i = 0; i < structure.n; i++) {
    if (Math.abs(structure.x[i] - center[0]) <= h && Math.abs(structure.y[i] - center[1]) <= h && Math.abs(structure.z[i] - center[2]) <= h) out.push(i);
  }
  return out;
}

/**
 * Is the box on the protein? Plain-language verdict used by the binding-site step.
 * @returns {{level:'ok'|'warn'|'bad', atomsInside:number, message:string}}
 */
export function assessBox(structure, center, size) {
  const inside = atomsInBox(structure, center, size).length;
  // A box packed solid with protein holds about 0.065 heavy atoms per Å³ (rough rule of thumb).
  const packed = 0.065 * size ** 3;
  if (inside === 0) {
    return { level: 'bad', atomsInside: 0, message: 'This box is in empty space, so the peptide would have nothing to grab. Click the protein to move the box onto it.' };
  }
  if (inside < Math.max(12, packed * 0.03)) {
    return { level: 'warn', atomsInside: inside, message: 'Only the very edge of the protein is inside this box. Try moving it a little closer in.' };
  }
  if (size < 15) {
    return { level: 'warn', atomsInside: inside, message: 'This box is quite small, so a peptide may not fit. Try making it bigger.' };
  }
  const span = Math.max(...structure.bounds.max.map((m, i) => m - structure.bounds.min[i]));
  if (size > Math.max(60, span * 1.3)) {
    return { level: 'warn', atomsInside: inside, message: 'This box is much bigger than the protein, so the search will be unfocused. Try shrinking it.' };
  }
  return { level: 'ok', atomsInside: inside, message: 'Good: the box is on the protein.' };
}

/** Pull the C-alpha positions (flat [x,y,z,…]) out of a peptide PDB. */
export function caPoints(text) {
  const s = parsePDB(text);
  const out = new Float32Array(s.ca.length * 3);
  s.ca.forEach((a, i) => { out[i * 3] = a.x; out[i * 3 + 1] = a.y; out[i * 3 + 2] = a.z; });
  return out;
}
