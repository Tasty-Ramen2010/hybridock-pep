// Regenerates tests/fixtures/web_requests.json from the browser UI's own request builders.
//   node tests/web_js/regen_fixtures.mjs
// The JS test asserts the UI still produces exactly this; tests/test_web_ui.py feeds the same
// requests to the real server code. If you change what the UI sends, regenerate and re-run both.
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { dockValues, compareValues, crystalValues } from '../../src/hybridock_pep/web/static/js/app/adapter.mjs';
import { EXPERT_DEFAULTS } from '../../src/hybridock_pep/web/static/js/app/config.mjs';

export const R = '__RECEPTOR__', R2 = '__OFFTARGET__', P = '__PEPTIDE_POSE__';
const site = { x: 25.2, y: -25.61, z: -7.97 };
const base = { peptide: 'ETFSDLWKLLPE', blind: false, site, box: 30, poses: 25, outputDir: 'runs/studio/dock_test', expert: { ...EXPERT_DEFAULTS } };

export function build() {
  return {
    dock: { mode: 'dock', scoring: 'vina', values: dockValues(base, R) },
    dock_expert: { mode: 'dock', scoring: 'vina', values: dockValues({ ...base, expert: { ...EXPERT_DEFAULTS, refineTopK: 3, seed: 7, noMinimize: true, ensemble: true, longCheckpointThreshold: 10 } }, R) },
    dock_blind: { mode: 'dock', scoring: 'vina', values: dockValues({ ...base, blind: true, site: null }, R) },
    compare: { mode: 'selectivity', scoring: 'vina', values: compareValues({ peptide: 'LISDAELEAIFEADC', target: { site, box: 30 }, offTarget: { site, box: 30 }, poses: 25, outputDir: 'runs/studio/compare_test' }, R, R2) },
    crystal: { mode: 'crystal', values: crystalValues({ peptide: 'ETFSDLWKLLPE' }, R, P) },
  };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const out = join(dirname(fileURLToPath(import.meta.url)), '..', 'fixtures', 'web_requests.json');
  writeFileSync(out, JSON.stringify(build(), null, 2) + '\n');
  console.log('wrote', out);
}
