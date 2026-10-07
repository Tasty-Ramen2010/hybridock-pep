// peptide.js — checking what the user typed, plus the decorative helix we draw for it.

export const STANDARD = 'ACDEFGHIKLMNPQRSTVWY';

// Average residue masses (Da). A peptide's mass is the sum plus one water.
const MASS = {
  A: 71.0788, R: 156.1875, N: 114.1038, D: 115.0886, C: 103.1388, E: 129.1155, Q: 128.1307, G: 57.0519,
  H: 137.1411, I: 113.1594, L: 113.1594, K: 128.1741, M: 131.1926, F: 147.1766, P: 97.1167, S: 87.0782,
  T: 101.1051, W: 186.2132, Y: 163.176, V: 99.1326,
};

/** Strip spaces, line breaks, digits and a FASTA ">header" line; upper-case the rest. */
export function cleanSequence(raw) {
  return String(raw || '')
    .split(/\r?\n/)
    .filter((l) => !l.trim().startsWith('>'))
    .join('')
    .replace(/[\s\d*-]/g, '')
    .toUpperCase();
}

/**
 * Friendly validation. `level` is 'empty' | 'error' | 'warn' | 'ok'.
 * Only the 20 standard one-letter amino acids are accepted.
 */
export function validatePeptide(raw) {
  const seq = cleanSequence(raw);
  if (!seq) return { ok: false, level: 'empty', seq, message: 'Type or paste a peptide to begin.', bad: [] };
  const bad = [...new Set([...seq].filter((c) => !STANDARD.includes(c)))];
  if (bad.length) {
    const list = bad.map((c) => `“${c}”`).join(', ');
    return {
      ok: false, level: 'error', seq, bad,
      message: `${list} ${bad.length > 1 ? "aren't" : "isn't"} one of the 20 standard amino acids. Use only these letters: ${STANDARD}.`,
    };
  }
  if (seq.length < 2) return { ok: false, level: 'error', seq, bad: [], message: 'A peptide needs at least 2 amino acids.' };
  if (seq.length > 30) {
    return { ok: true, level: 'warn', seq, bad: [], message: `${seq.length} amino acids is long. It will run slower and the result is less reliable.` };
  }
  return { ok: true, level: 'ok', seq, bad: [], message: 'Looks good.' };
}

/** Length, approximate weight (Da) and approximate net charge at neutral pH (K/R +1, D/E −1). */
export function peptideStats(seq) {
  let mass = 18.0153, charge = 0;
  for (const c of seq) {
    mass += MASS[c] || 0;
    if (c === 'K' || c === 'R') charge += 1;
    if (c === 'D' || c === 'E') charge -= 1;
  }
  return { length: seq.length, mass, charge };
}

/**
 * Ideal α-helix C-alpha trace along the x axis, centred on the origin.
 * (radius 2.3 Å, 1.5 Å rise and 100° turn per residue). Decorative: the real peptide shape
 * comes from the backend's poses.
 * @returns {Float32Array} flat [x,y,z,…]
 */
export function helixPoints(n) {
  const out = new Float32Array(n * 3);
  const mid = ((n - 1) * 1.5) / 2;
  for (let i = 0; i < n; i++) {
    const a = (i * 100 * Math.PI) / 180;
    out[i * 3] = i * 1.5 - mid;
    out[i * 3 + 1] = 2.3 * Math.cos(a);
    out[i * 3 + 2] = 2.3 * Math.sin(a);
  }
  return out;
}

export function centroid(points) {
  const n = points.length / 3;
  let x = 0, y = 0, z = 0;
  for (let i = 0; i < n; i++) { x += points[i * 3]; y += points[i * 3 + 1]; z += points[i * 3 + 2]; }
  return [x / n, y / n, z / n];
}
