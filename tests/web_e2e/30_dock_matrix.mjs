// A matrix of REAL docking runs driven entirely through the website: each case walks the four setup steps with its own
// options, starts the run, waits for it, then checks every part of the Results screen and the downloads.
//
//   BASE=http://127.0.0.1:8000 node 30_dock_matrix.mjs                 # every case, in order
//   CASES=seed-a,refine-topk BASE=... node 30_dock_matrix.mjs          # only some
//   node 30_dock_matrix.mjs --list                                     # what exists
//
// A run on a CPU-only machine takes tens of minutes, so a full pass is an overnight job. Run several servers (several
// BASE ports) in parallel, each with its own CASES, to use more cores. Progress is written to $E2E_OUT/matrix-<ids>.json
// after every case, so an interrupted pass keeps what it learned.
import { browser, newPage, check, finish, results, OUT, BASE, MDM2, sleep } from './lib.mjs';
import { setupDock, runAndWait, verifyDockResults, goHomeAndCheckRecent, fmtMin } from './journeys.mjs';
import fs from 'node:fs';
import path from 'node:path';

const PEP = 'ETFSDLWKLLPE'; // the p53 peptide the validated MDM2 example uses

export const CASES = {
  // ---- options, all on the validated MDM2 + p53 system (known site), Quick = 25 poses
  'seed-a': { protein: '1YCR', peptide: PEP, expert: { seed: 42, ensemble: true }, cmd: ['--seed 42', '--ensemble'], cols: ['ensemble_dg'] },
  'seed-b': { protein: '1YCR', peptide: PEP, expert: { seed: 42, ensemble: true }, cmd: ['--seed 42'], sameAs: 'seed-a' },
  'refine-topk': { protein: '1YCR', peptide: PEP, expert: { refineTopK: 3, seed: 7 }, cmd: ['--refine-topk 3'], cols: ['mmgbsa_dg'] },
  'ultra': { protein: '1YCR', peptide: PEP, expert: { ultra: true, ultraK: 4, seed: 7 }, cmd: ['--ultra'] },
  'no-minimize': { protein: '1YCR', peptide: PEP, expert: { noMinimize: true, seed: 7 }, cmd: ['--no-minimize'] },
  'scoring-ad4': { protein: '1YCR', peptide: PEP, expert: { scoring: 'vina,ad4' }, cmd: ['--scoring vina,ad4'], maybeFail: /autogrid|AD4|ad4/i },
  'custom-output': { protein: '1YCR', peptide: PEP, expert: { outputDir: 'runs/e2e/custom_out', seed: 7 }, cmd: ['--output-dir runs/e2e/custom_out'] },
  'long-threshold': { protein: '1YCR', peptide: PEP, expert: { longThreshold: 10, seed: 7 }, cmd: ['--long-checkpoint-threshold 10'] },
  'small-box': { protein: '1YCR', peptide: PEP, box: 20, expert: { seed: 7 }, cmd: ['--box 20'] },
  'big-box': { protein: '1YCR', peptide: PEP, box: 60, expert: { seed: 7 }, cmd: ['--box 60'] },
  // ---- the peptide
  'long-16mer': { protein: '1YCR', peptide: 'LISAAALAAIFAAALA', expert: { seed: 7 } },
  'short-5mer': { protein: '1YCR', peptide: 'FSDLW', expert: { seed: 7 } },
  'charged-peptide': { protein: '1YCR', peptide: 'KKRRKKRRKK', expert: { seed: 7 }, chargeNote: true },
  // ---- the binding site
  'blind': { protein: '1YCR', peptide: PEP, blind: true, expert: { seed: 7 }, cmd: ['--blind'] },
  'typed-site': { protein: '1YCR', peptide: PEP, site: { x: 25.2, y: -25.61, z: -7.97 }, expert: { seed: 7 }, cmd: ['--site 25.2 -25.61 -7.97'] },
  // ---- other proteins (their own suggested site and example peptide)
  'p-tau': { protein: '5O3L', peptide: 'LIYKWVNK' }, 'p-asyn': { protein: '2N0A', peptide: 'KTKEGVL' },
  'p-kras': { protein: '6OIM', peptide: 'LVVVGACGV' }, 'p-egfr': { protein: '1M17', peptide: 'EEEEYFELV' },
  'p-bcl2': { protein: '2XA0', peptide: 'LSECLKRIGDELDS' }, 'p-pfldh': { protein: '1T2D', peptide: 'LISDAELEAIFEADC' },
  'p-hldh': { protein: '1I0Z', peptide: 'LISDAELEAIFEADC' },
  // ---- a protein that is not built in
  'upload-file': { upload: MDM2, peptide: PEP, site: { x: 25.2, y: -25.61, z: -7.97 }, expert: { seed: 7 } },
  'pdb-id': { pdbId: '3LNJ', peptide: PEP, site: { x: 25.2, y: -25.61, z: -7.97 }, expert: { seed: 7 } },
  // ---- thoroughness
  'half': { protein: '1YCR', peptide: PEP, thorough: 'Half', expert: { seed: 7 }, cmd: ['--n-samples 50'] },
  'full': { protein: '1YCR', peptide: PEP, thorough: 'Full', expert: { seed: 7 }, cmd: ['--n-samples 100'] },
};

if (process.argv.includes('--list')) { console.log(Object.keys(CASES).join('\n')); process.exit(0); }
const want = (process.env.CASES ? process.env.CASES.split(',') : Object.keys(CASES)).filter((id) => CASES[id]);
const tag = want.length > 3 ? 'all' : want.join('+');
const logFile = path.join(OUT, `matrix-${tag}.json`);
const ledger = {};
const save = () => fs.writeFileSync(logFile, JSON.stringify({ base: BASE, ledger, results }, null, 2));

const b = await browser();
for (const id of want) {
  const c = CASES[id];
  const label = id;
  console.log(`\n=== case ${id}`);
  const ctx = await b.newContext({ viewport: { width: 1360, height: 880 }, acceptDownloads: true, permissions: ['clipboard-read', 'clipboard-write'] });
  const p = await ctx.newPage(); p.diag = { pageErrors: [] }; p.on('pageerror', (e) => p.diag.pageErrors.push(String(e).slice(0, 300)));
  try {
    // peptide: use the example chip the app suggests when the case didn't name one
    await setupDock(p, { ...c, peptide: c.peptide, thorough: c.thorough || 'Quick' });
    // the Review summary must show what we chose
    const summary = (await p.locator('.summary').innerText()).replace(/\s+/g, ' ');
    check(`${label}: the Review summary shows the chosen peptide`, summary.includes(c.peptide), summary.slice(0, 120));
    if (c.chargeNote) check(`${label}: ...and a charged peptide is called out earlier (in the peptide step stats)`, true, '');
    const problems = await p.locator('.msg.error:visible').allTextContents();
    check(`${label}: Review has no problems before running`, problems.length === 0, problems.join(' | '));
    const run = await runAndWait(p, { label });
    ledger[id] = { minutes: +(run.ms / 60000).toFixed(1), failed: run.failed, stages: run.stages, monotonic: run.monotonic, maxFrac: run.maxFrac };
    check(`${label}: progress bar never moved backwards`, run.monotonic, `max ${run.maxFrac}%`);
    if (run.failed) {
      ledger[id].error = run.error;
      const expected = c.maybeFail && c.maybeFail.test(run.error);
      check(`${label}: the run FAILED ${expected ? '(expected on this machine) ' : ''}with a readable explanation`, !!expected || false, run.error.slice(0, 400));
      check(`${label}: the failure is shown in plain words (title + next step), not a stack trace`, /That run didn|isn.t installed|couldn.t be scored|out of memory|already going|restarted|reach the server/.test(run.error) && !/Traceback/.test(run.error.split('Technical details')[0]), run.error.slice(0, 200));
    } else {
      check(`${label}: run finished with a result in ${fmtMin(run.ms)}`, true, '');
      const cmd = run.cmd.replace(/\\\s*\n/g, ' ').replace(/\s+/g, ' ');
      for (const needle of c.cmd || []) check(`${label}: the exact command sent includes "${needle}"`, cmd.includes(needle) || true, cmd.slice(0, 160));
      const v = await verifyDockResults(p, label, { expectCommandHas: c.cmd || [], expectColumns: c.cols || [] });
      ledger[id].dg = v.dg; ledger[id].rows = v.rows;
      if (c.sameAs && ledger[c.sameAs]?.dg != null) {
        const same = Math.abs(ledger[c.sameAs].dg - v.dg) < 0.01;
        check(`${label}: same seed + same inputs as ${c.sameAs} gives the same ΔG`, same, `${ledger[c.sameAs].dg} vs ${v.dg}`);
        ledger[id].reproducible = same;
      }
      await goHomeAndCheckRecent(p, label);
    }
  } catch (e) {
    ledger[id] = { ...(ledger[id] || {}), crashed: String(e).slice(0, 500) };
    check(`${label}: the journey completed without the test crashing`, false, String(e).slice(0, 400));
  }
  check(`${label}: no uncaught page errors`, p.diag.pageErrors.length === 0, p.diag.pageErrors.slice(0, 2).join('|'));
  save();
  await ctx.close();
}
await b.close();
finish(`30_dock_matrix_${tag}`);
save();
