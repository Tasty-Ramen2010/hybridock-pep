// catalog.mjs — the preloaded results.
//
// data/catalog.json is written by scripts/precompute_catalog.py: real predictions, run ahead of time on the demo computer,
// so a visitor sees finished results the moment the page opens. It also carries the measured run times (TIMING), which is
// where every "about N minutes" in the app comes from.
//
//   { generated, machine, poses,
//     timing: { seconds: { "5": s, "10": s, "25": s }, perResidue, refLength },
//     binders: { <proteinKey>: [ { seq, deltaG, known, poses: [ { rank, deltaG, ca: [x,y,z,…] } ] } ] } }

import { EXPERT_DEFAULTS, TIMING, assetUrl, timingFor } from './config.mjs';
import { buildDockCommand } from './command.mjs';
import { dockJob } from './jobs.mjs';
import { peptideTitle } from './peptides.mjs';
import { fetchRetry } from './structures.mjs';

let cached = null;

/** The catalog, or an empty one when the file is missing (the page then simply has nothing preloaded). */
export async function loadCatalog() {
  if (cached) return cached;
  let data = { binders: {}, timing: null, poses: 25, machine: '' };
  try {
    const res = await fetchRetry(assetUrl('data/catalog.json'));
    if (res.ok) data = { ...data, ...(await res.json()) };
  } catch { /* nothing preloaded: the carousel still lets people run things live */ }
  useTiming(data.timing ? { ...data.timing, machine: data.machine } : null);
  return (cached = data);
}

/**
 * Make `profile` (the measured table: { machine, seconds, perResidue, refLength }) the one every "about N minutes" reads
 * from, or clear it. A page talking to a live server uses what that server says it measured (its /api/env `timing`),
 * so a copy running on someone else's laptop never shows the demo computer's times as if they were its own.
 */
export function useTiming(profile) {
  TIMING.grid = null; TIMING.machine = '';
  for (const k of Object.keys(TIMING.seconds)) TIMING.seconds[k] = null;
  if (!profile?.grid) return;
  TIMING.grid = profile.grid;
  TIMING.refLength = profile.refLength ?? 12;
  TIMING.machine = profile.machine || '';
  for (const k of Object.keys(TIMING.seconds)) TIMING.seconds[k] = timingFor(Number(k), TIMING.refLength); // the three sizes at the reference length
}

/** Strongest first. */
export const bindersFor = (catalog, key) => [...(catalog?.binders?.[key] || [])].sort((a, b) => a.deltaG - b.deltaG);

export const binderKey = (proteinKey, seq) => `${proteinKey}:${seq}`;

/**
 * A preloaded binder as the DockResult the results screen already knows how to show. It is a real run that was done
 * ahead of time, so it is not marked Demo; `preloaded` only tells the screen that there is no run folder to download.
 */
export function binderResult(protein, binder, catalog) {
  const job = dockJob({
    proteinRef: protein, peptide: binder.seq, siteMode: 'known', site: protein.site, box: protein.box,
    thorough: 'full', expert: { ...EXPERT_DEFAULTS }, runStamp: 'catalog',
  });
  return {
    id: `cat_${protein.key}_${binder.seq}`, kind: 'dock', demo: false, preloaded: true,
    createdAt: catalog.generated || new Date().toISOString(),
    name: `${peptideTitle(binder.seq)} on ${protein.name}`,
    protein, peptide: binder.seq, site: { ...protein.site }, box: protein.box, blind: false,
    nPoses: catalog.poses || 25, deltaG: binder.deltaG,
    poses: (binder.poses || []).map((p, i) => ({ rank: p.rank ?? i + 1, file: '', deltaG: p.deltaG, rankScore: null, nClash: 0, cluster: 0, ca: p.ca })),
    command: buildDockCommand(job), job,
  };
}

/** Where a binder sits on the strength bar: 0 (weak) to 1 (very tight), over the range that real peptides fall in. */
export const strength = (dg) => Math.max(0.04, Math.min(1, (-dg - 3) / 9));
