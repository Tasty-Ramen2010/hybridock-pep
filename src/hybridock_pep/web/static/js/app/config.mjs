// config.js — the knobs a student is most likely to want to change.
// Nothing in here talks to the backend; that lives only in adapter.js.

/** Where the app's own files live (…/static/). Every bundled PDB / JSON is loaded relative to this. */
export const ASSET_BASE = (import.meta.url.match(/^(.*\/static\/)/) || [])[1] || '/static/'; // works from js/app/ and from dist/
export const assetUrl = (rel) => ASSET_BASE + rel;

/** How far off a single ΔG prediction typically is (kcal/mol). Shown next to every ΔG. */
export const TYPICAL_ERROR = 1.6;

/**
 * The "How thorough?" choice. The backend setting behind it is the number of poses
 * (`--n-samples`, default 100). The Quick/Half/Full pose counts are OUR choice for the
 * UI, not something the backend defines — change them here if the team decides otherwise.
 */
export const THOROUGHNESS = [
  { id: 'quick', label: 'Quick', poses: 25, hint: 'A fast first look' },
  { id: 'half', label: 'Half', poses: 50, hint: 'A good balance' },
  { id: 'full', label: 'Full', poses: 100, hint: 'The default: most reliable' },
];
export const DEFAULT_THOROUGHNESS = 'full';

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
export const EXAMPLE = { proteinKey: 'tau', peptide: 'LIYKWVNK' };

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
  { id: 'teal', name: 'Teal', light: '#0f766e', dark: '#2dd4bf' },
  { id: 'indigo', name: 'Indigo', light: '#4f46e5', dark: '#818cf8' },
  { id: 'plum', name: 'Plum', light: '#9333ea', dark: '#c084fc' },
  { id: 'forest', name: 'Forest', light: '#15803d', dark: '#4ade80' },
];

/** Our own rough wording for ΔG. NOT backend output — the UI labels it as such. */
export const ROUGH_GUIDE = [
  { max: -9.0, id: 'strong', label: 'Strong' },
  { max: -6.5, id: 'moderate', label: 'Moderate' },
  { max: Infinity, id: 'weak', label: 'Weak' },
];
