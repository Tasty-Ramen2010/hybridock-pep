// config.js — the knobs a student is most likely to want to change.
// Nothing in here talks to the backend; that lives only in adapter.js.

/** Where the app's own files live (…/static/). Every bundled PDB / JSON is loaded relative to this. */
export const ASSET_BASE = (import.meta.url.match(/^(.*\/static\/)/) || [])[1] || '/static/'; // works from js/app/ and from dist/
export const assetUrl = (rel) => ASSET_BASE + rel;

/**
 * Where the API lives. Normally on the same site as the page (`hybridock-pep serve`). On the hosted website it is
 * another computer: the team's, reached through a tunnel, found by reading LIVE_LINK (see docs/SHARING.md).
 */
export const backend = { url: '', remote: false, offline: false };
export const apiUrl = (path) => backend.url + path;
export const LIVE_LINK = 'https://raw.githubusercontent.com/Tasty-Ramen2010/hybridock-pep/live-link/backend.json';
/**
 * The top bar's memory meter: how full memory is, as a colour and a level.
 *   ok < 75 %      busy 75-88 %      low >= 88 % or under 1 GB free      full >= 97 % (the liquid spills over)
 * `hue` slides from green through amber to red as the tank fills; `fill` is 0-100 for the height of the liquid.
 */
const HUE_STOPS = [[0, 150], [55, 142], [75, 48], [88, 26], [100, 2]];
export function memoryLevel(percent, availableGb = null) {
  if (typeof percent !== 'number' || !Number.isFinite(percent)) return { id: 'unknown', hue: 215, fill: 0, percent: null };
  const pct = Math.min(100, Math.max(0, percent));
  let hue = HUE_STOPS[HUE_STOPS.length - 1][1];
  for (let i = 1; i < HUE_STOPS.length; i++) {
    const [p1, h1] = HUE_STOPS[i];
    if (pct <= p1) { const [p0, h0] = HUE_STOPS[i - 1]; hue = h0 + ((h1 - h0) * (pct - p0)) / (p1 - p0); break; }
  }
  const low = pct >= 88 || (typeof availableGb === 'number' && availableGb < 1);
  const id = pct >= 97 ? 'full' : low ? 'low' : pct >= 75 ? 'busy' : 'ok';
  return { id, hue: Math.round(hue), fill: pct, percent: Math.round(pct) };
}

/** A shared computer limits what one visitor can ask for (it tells us in /api/env). Infinity when there is no limit. */
export const sharedLimits = (env) => env?.public || null;
export const poseCap = (env, compare = false) => { const p = sharedLimits(env); return p ? (compare ? p.max_compare_samples : p.max_samples) : Infinity; };

/** Where the project lives. The home page leads with this link. */
export const GITHUB_URL = 'https://github.com/Tasty-Ramen2010/hybridock-pep';

/**
 * The "How many poses?" choice. The backend setting behind it is the number of poses (`--n-samples`). Every run on the
 * live demo uses one of these three, so a prediction finishes in minutes. Times come from the catalog's measured table
 * (see TIMING below), not from a guess.
 */
export const THOROUGHNESS = [
  { id: 'quick', label: '5 poses', poses: 5, hint: 'Fastest' },
  { id: 'half', label: '10 poses', poses: 10, hint: 'Balanced' },
  { id: 'full', label: '25 poses', poses: 25, hint: 'Most thorough' },
];
export const DEFAULT_THOROUGHNESS = 'half';

/**
 * Measured seconds for a whole run on the demo computer, by peptide length and pose count (filled in by `useTiming` from the
 * server's /api/env, or from data/catalog.json). Run time is not a straight line in length (from 13 residues the longer-peptide
 * model is used), so `timingFor` joins the measured points with straight lines. The same rule is in hybridock_pep/web/timing.py.
 */
export const TIMING = { grid: null, seconds: { 5: null, 10: null, 25: null }, refLength: 12, machine: '' };

const interp = (pts, x) => {
  if (pts.length === 1) return pts[0][1];
  let i = pts.findIndex((p, k) => k < pts.length - 1 && x <= pts[k + 1][0]);
  if (x <= pts[0][0]) i = 0; else if (i < 0) i = pts.length - 2;
  const [[x0, y0], [x1, y1]] = [pts[i], pts[i + 1]];
  return y0 + ((y1 - y0) * (x - x0)) / (x1 - x0);
};

/** Seconds for a whole prediction, from the measured grid; null when this computer has not been measured. */
export function timingFor(poses, peptideLength = TIMING.refLength) {
  if (!TIMING.grid) return null;
  const rows = Object.entries(TIMING.grid)
    .map(([L, row]) => [Number(L), Object.entries(row).map(([n, t]) => [Number(n), Number(t)]).sort((a, b) => a[0] - b[0])])
    .filter(([, row]) => row.length).sort((a, b) => a[0] - b[0]);
  if (!rows.length) return null;
  const atLength = rows.map(([L, row]) => [L, interp(row, poses)]);
  const floor = 0.5 * Math.min(...rows.flatMap(([, row]) => row.map((p) => p[1])));
  return Math.max(10, Math.floor(Math.max(floor, interp(atLength, peptideLength)) + 0.5));
}

/** Search box (grid box) edge length in Å. */
export const BOX_DEFAULT = 30;
export const BOX_MIN = 10;
export const BOX_MAX = 60; // the backend refuses anything larger (hybridock_pep/ui/tui.py: --box is an integer in [10, 60])

/** The backend's accepted ranges for the settings the UI exposes (tui.FIELDS validators). */
export const LIMITS = {
  box: [10, 60],
  longCheckpointThreshold: [3, 30],
  poses: [1, 500],
  refineTopK: [0, 50],
  ultraK: [0, 256],
  seed: [0, 2 ** 31 - 1],
};

/** True unless the live server says autogrid4 is missing (the "vina + AD4" scoring option cannot run without it). */
export const ad4Available = (env) => env?.checks?.autogrid?.ok !== false;

/** True unless the live server says the optional long-peptide model is missing (then the threshold has no effect). */
export const longModelAvailable = (env) => env?.checks?.long_model?.ok !== false;

/**
 * Problems with the numeric settings, in plain words (an empty list = fine). The server's own validator stays the
 * authority; this catches the same mistakes before a request is made, in demo mode too.
 */
export function settingProblems({ box, expert = {} }) {
  const out = [];
  const int = (v) => (v === '' || v == null ? null : Number(v));
  const check = (label, v, [lo, hi], { optional = false } = {}) => {
    if ((v === '' || v == null) && optional) return;
    const n = Number(v);
    if (v === '' || v == null || !Number.isInteger(n)) out.push(`${label} must be a whole number.`);
    else if (n < lo || n > hi) out.push(`${label} must be between ${lo} and ${hi}.`);
  };
  check('Box size', box, LIMITS.box);
  check('Long-peptide model threshold', expert.longCheckpointThreshold, LIMITS.longCheckpointThreshold);
  check('Refine the top poses', int(expert.refineTopK), LIMITS.refineTopK, { optional: true });
  if (expert.ultra) check('Ultra mode K', expert.ultraK, LIMITS.ultraK);
  check('Random seed', int(expert.seed), LIMITS.seed, { optional: true });
  return out;
}

/** The one-click example on the home screen. */
export const EXAMPLE = { proteinKey: 'mdm2', peptide: 'ETFSDLWKLLPE' }; // the p53 peptide on MDM2: a famous, named pair (also the live server's own example)

/** Plain-language stage names shown while a run is going. `tech` is the small text under each. */
export const RUN_STAGES = [
  { id: 'sample', label: 'Generating shapes', tech: 'Pose sampling' },
  { id: 'relax', label: 'Relaxing clashes', tech: 'Energy minimization' },
  { id: 'rank', label: 'Ranking poses', tech: 'Clustering & ranking' },
  { id: 'score', label: 'Scoring binding', tech: 'ΔG prediction' },
];

/** Expert-only settings and their backend defaults (see `hybridock-pep dock --help`). */
export const EXPERT_DEFAULTS = {
  longCheckpointThreshold: 13, // --long-checkpoint-threshold
  scoring: 'vina', //             --scoring
  refineTopK: '', //              --refine-topk   ('' = off)
  ultra: false, //                --ultra [K]
  ultraK: 32,
  seed: '', //                    --seed
  inputPoses: '', //              --input-poses
  noMinimize: false, //           --no-minimize
  ensemble: false, //             --ensemble
  calibration: 'data/calibration_v1_2_production_entropy.json', // --calibration
  outputDir: '', //              --output-dir  ('' = automatic: runs/studio/<run name>)
};

/** Accent presets for the small colour control. All are far from the peptide orange. */
export const ACCENTS = [
  // fill: the colour of the primary button (white text on it is always >= 4.5:1); link: text links on that appearance
  { id: 'blue', name: 'Blue', light: '#0071e3', dark: '#0071e3', linkLight: '#0066cc', linkDark: '#2997ff' },
  { id: 'purple', name: 'Purple', light: '#8944ab', dark: '#a050c8', linkLight: '#8944ab', linkDark: '#d18bf5' },
  { id: 'green', name: 'Green', light: '#1f7a37', dark: '#1f7a37', linkLight: '#1f7a37', linkDark: '#5fd47a' },
  { id: 'graphite', name: 'Graphite', light: '#636366', dark: '#6e6e73', linkLight: '#636366', linkDark: '#aeaeb2' },
];

/** Our own rough wording for ΔG. NOT backend output — the UI labels it as such. */
export const ROUGH_GUIDE = [
  { max: -9.0, id: 'strong', label: 'Strong' },
  { max: -6.5, id: 'moderate', label: 'Moderate' },
  { max: Infinity, id: 'weak', label: 'Weak' },
];
