// jobs.js — turns what the user chose into the "job" object the adapter and command builder expect.

import { THOROUGHNESS } from './config.mjs';

export const findProtein = (proteins, key) => proteins.find((p) => p.key === key) || null;
export const poseCount = (thoroughId) => (THOROUGHNESS.find((t) => t.id === thoroughId) || THOROUGHNESS[2]).poses;

/** Path shown in the command preview: bundled file for presets, the file name for uploads. */
export const receptorPath = (ref) => (ref.custom ? ref.file || `${ref.name}.pdb` : ref.file);

const slug = (s) => String(s || 'run').toLowerCase().replace(/[^a-z0-9]+/g, '').slice(0, 24) || 'run';

/** A unique output folder per run, unless Expert mode names one: runs/studio/<kind>_<peptide>_<stamp>. */
export const outputDirFor = (kind, peptide, stamp, custom) => (custom && custom.trim()) || `runs/studio/${kind}_${slug(peptide)}_${stamp}`;

/** A "dock" job (Predict binding). */
export function dockJob({ proteinRef, peptide, siteMode, site, box, thorough, expert, runStamp = Date.now().toString(36) }) {
  const blind = siteMode === 'find';
  return {
    kind: 'dock',
    peptide,
    protein: proteinRef,
    receptor: receptorPath(proteinRef),
    blind,
    site: blind ? null : { ...site },
    box,
    poses: poseCount(thorough),
    expert: { ...expert },
    outputDir: outputDirFor('dock', peptide, runStamp, expert?.outputDir),
  };
}

/** One side of a comparison, with the default site filled from the protein's suggested site. */
export function compareSide(side) {
  return { protein: side.proteinRef, receptor: receptorPath(side.proteinRef), site: { ...side.site }, box: side.box };
}

export function compareJob(state) {
  return {
    kind: 'compare',
    peptide: state.peptide,
    target: compareSide(state.target),
    offTarget: compareSide(state.offTarget),
    poses: poseCount(state.thorough || 'full'),
    outputDir: outputDirFor('compare', state.peptide, state.runStamp || Date.now().toString(36)),
  };
}

/** Initials for the avatar: "Ram" → "R", "Ada Lovelace" → "AL". */
export function initials(name) {
  const parts = String(name || '').trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return '?';
  return (parts[0][0] + (parts.length > 1 ? parts[parts.length - 1][0] : '')).toUpperCase();
}

export function greeting(date = new Date()) {
  const hour = date.getHours();
  return hour < 5 ? 'Good evening' : hour < 12 ? 'Good morning' : hour < 17 ? 'Good afternoon' : 'Good evening';
}
