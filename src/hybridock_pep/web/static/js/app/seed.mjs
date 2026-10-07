// seed.js — a few example entries so the dashboard and History aren't empty on first launch.
// They exist ONLY in the demo adapter, are flagged demo + seeded, and are labelled "Demo" in the UI.
// Opening one re-creates its (simulated) poses on the fly; nothing here is a real prediction.

import { demoDeltaG } from './adapter.mjs';
import { dockJob, findProtein } from './jobs.mjs';
import { EXPERT_DEFAULTS } from './config.mjs';

const SEEDS = [
  { key: 'tau', peptide: 'LIYKWVNK', daysAgo: 1 },
  { key: 'mdm2', peptide: 'SQETFSDLWKLLP', daysAgo: 4 },
  { key: 'kras', peptide: 'LVVVGACGV', daysAgo: 9 },
];

export function seedHistory(proteins) {
  const out = [];
  for (const [i, s] of SEEDS.entries()) {
    const protein = findProtein(proteins, s.key);
    if (!protein) continue;
    const job = dockJob({ proteinRef: protein, peptide: s.peptide, siteMode: 'known', site: protein.site, box: protein.box, thorough: 'full', expert: { ...EXPERT_DEFAULTS } });
    out.push({
      id: `seed_${i}`, kind: 'dock', demo: true, seeded: true,
      name: `${s.peptide} → ${protein.name}`,
      createdAt: new Date(Date.now() - s.daysAgo * 86400000).toISOString(),
      headline: { label: 'ΔG', value: demoDeltaG(protein.key, s.peptide) },
      proteinNames: [protein.name], job, result: undefined,
    });
  }
  return out;
}
