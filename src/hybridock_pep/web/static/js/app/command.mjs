// command.js — builds the exact `hybridock-pep` command shown in Expert mode.
// Flags and defaults come straight from `hybridock-pep dock|selectivity|crystal-score --help`.
// Pure text; this file never runs anything.

import { EXPERT_DEFAULTS } from './config.mjs';

/** Quote a value for a shell only when it needs it. */
export function shq(value) {
  const s = String(value);
  return /^[\w./:=,@%+-]+$/.test(s) ? s : `'${s.replace(/'/g, `'\\''`)}'`;
}

const num = (v) => (Number.isInteger(v) ? String(v) : String(Math.round(v * 100) / 100));

/** Join [flag, value…] groups into a readable multi-line command. */
function format(head, groups) {
  const lines = [head, ...groups.map((g) => '  ' + g.map(shq).join(' '))];
  return lines.join(' \\\n');
}

/**
 * `hybridock-pep dock …`
 * @param job {{peptide:string, receptor:string, blind:boolean, site:{x,y,z}, box:number,
 *              poses:number, expert:object}}
 */
export function buildDockCommand(job) {
  const e = { ...EXPERT_DEFAULTS, ...(job.expert || {}) };
  const g = [['--peptide', job.peptide], ['--receptor', job.receptor]];
  if (job.blind) g.push(['--blind']);
  else g.push(['--site', num(job.site.x), num(job.site.y), num(job.site.z)], ['--box', num(job.box)]);
  g.push(['--n-samples', job.poses]);
  if (Number(e.longCheckpointThreshold) !== EXPERT_DEFAULTS.longCheckpointThreshold) g.push(['--long-checkpoint-threshold', e.longCheckpointThreshold]);
  if (e.scoring && e.scoring !== EXPERT_DEFAULTS.scoring) g.push(['--scoring', e.scoring]);
  if (e.refineTopK !== '' && e.refineTopK != null) g.push(['--refine-topk', e.refineTopK]);
  if (e.ultra) g.push(['--ultra', e.ultraK || EXPERT_DEFAULTS.ultraK]);
  if (e.seed !== '' && e.seed != null) g.push(['--seed', e.seed]);
  if (e.inputPoses) g.push(['--input-poses', e.inputPoses]);
  if (e.noMinimize) g.push(['--no-minimize']);
  if (e.ensemble) g.push(['--ensemble']);
  if (e.calibration && e.calibration !== EXPERT_DEFAULTS.calibration) g.push(['--calibration', e.calibration]);
  g.push(['--output-dir', job.outputDir || e.outputDir || 'runs/studio/run']);
  return format('hybridock-pep dock', g);
}

/** `hybridock-pep selectivity …` (Compare two proteins). */
export function buildCompareCommand(job) {
  const side = (p, s) => [
    [`--${p}-receptor`, s.receptor],
    [`--${p}-site`, num(s.site.x), num(s.site.y), num(s.site.z)],
    [`--${p}-box`, num(s.box)],
  ];
  const g = [['--peptide', job.peptide], ...side('target', job.target), ...side('offtarget', job.offTarget),
    ['--n-samples', job.poses], ['--output-dir', job.outputDir || 'runs/studio/compare']];
  return format('hybridock-pep selectivity', g);
}

/** `hybridock-pep crystal-score …` (Score a structure). */
export function buildScoreCommand(job) {
  const g = [['--receptor', job.receptor], ['--peptide-pdb', job.peptidePdbName], ['--peptide', job.peptide]];
  if (job.allowClashes) g.push(['--allow-clashes']);
  return format('hybridock-pep crystal-score', g);
}
