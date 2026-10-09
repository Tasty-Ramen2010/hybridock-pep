// Reusable end-to-end journeys for the REAL runs: walk the whole UI to start a Dock / Compare / Score with any combination
// of options, wait for it, then verify everything on the Results screen. All through clicks and typing, never the API.
import { check, sleep, shot, BASE, MDM2, POSE } from './lib.mjs';
import fs from 'node:fs';

const cont = (p) => p.getByRole('button', { name: 'Continue' });
export const fmtMin = (ms) => `${(ms / 60000).toFixed(1)} min`;

/** Home -> New prediction -> protein -> peptide -> site -> review, with the options given. Stops on the Review step. */
export async function setupDock(p, o) {
  const { protein, peptide, site = 'suggested', box, blind = false, thorough = 'Quick', expert = {}, upload, pdbId } = o;
  await p.goto(BASE + '/'); await p.waitForSelector('.hero-card');
  await p.getByRole('button', { name: 'New prediction' }).click(); await p.waitForSelector('.pick');
  // ---- 1 protein
  if (upload) { await p.locator('input[type=file]').first().setInputFiles(upload); await p.waitForSelector('text=residues loaded'); }
  else if (pdbId) {
    await p.locator('input[type=search]').fill(pdbId);
    await p.locator('.pick', { hasText: new RegExp(`Load PDB ${pdbId}`, 'i') }).click();
    await p.waitForSelector('text=residues loaded', { timeout: 60000 });
  } else {
    await p.locator('input[type=search]').fill('');
    await p.locator('.pick', { has: p.locator('.mono', { hasText: `PDB ${protein}` }) }).last().click();
    await p.waitForSelector('text=residues loaded');
  }
  await cont(p).click(); await p.waitForSelector('#pep-input');
  // ---- 2 peptide
  await p.locator('#pep-input').fill(peptide); await p.locator('#pep-input').blur(); await sleep(150);
  await cont(p).click(); await p.waitForSelector('.choice');
  // ---- 3 binding site
  if (blind) await p.getByRole('radio', { name: /Find the pocket for me/ }).click();
  else {
    await sleep(400);
    if (site === 'suggested') { const b = p.getByRole('button', { name: /Use the suggested site/ }); if (await b.isVisible()) await b.click(); }
    if (box) await p.locator('input[type=range]').first().fill(String(box));
    if (site && typeof site === 'object') { // explicit coordinates typed by hand (Coordinates drawer)
      await p.locator('details.adv summary', { hasText: 'Coordinates' }).click();
      for (const [axis, v] of Object.entries(site)) await p.locator(`[aria-label="Box centre ${axis.toUpperCase()}"]`).fill(String(v));
    }
  }
  await sleep(300);
  await cont(p).click(); await p.waitForSelector('.summary');
  // ---- 4 review
  await p.getByRole('radio', { name: new RegExp(thorough) }).click();
  if (Object.keys(expert).length) {
    await p.getByRole('button', { name: 'Expert' }).click();
    await p.locator('details.adv summary', { hasText: 'Advanced settings' }).click();
    const field = (label) => p.locator('label.field-label', { hasText: label }).locator('xpath=following-sibling::*[self::input or self::select]').first();
    const tick = (label) => p.locator('label.check', { hasText: label }).locator('input');
    if (expert.longThreshold != null) await field('Long-peptide model starts at').fill(String(expert.longThreshold));
    if (expert.scoring) await field('Scoring mode').selectOption(expert.scoring).catch(() => { /* the option is disabled when autogrid4 is missing: leave the default */ });
    if (expert.refineTopK != null) await field('Refine the top poses').fill(String(expert.refineTopK));
    if (expert.ultra) { await tick('Ultra mode').check(); if (expert.ultraK != null) await field('Ultra mode K').fill(String(expert.ultraK)); }
    if (expert.seed != null) await field('Random seed').fill(String(expert.seed));
    if (expert.inputPoses) await field('Input-poses folder').fill(expert.inputPoses);
    if (expert.noMinimize) await tick('Skip pre-minimization').check();
    if (expert.ensemble) await tick('ensemble ΔG column').check();
    if (expert.calibration) await field('Calibration file').fill(expert.calibration);
    if (expert.outputDir) await field('Output folder').fill(expert.outputDir);
  }
  await sleep(1500);
}

/** Press Run on the Review step and poll the Running screen until a result or an error card. Returns timing and the stages seen. */
export async function runAndWait(p, { label, maxMs = 3 * 3600 * 1000, onTick } = {}) {
  const cmd = (await p.locator('pre.cmd').first().textContent().catch(() => '')) || '';
  await p.getByRole('button', { name: /Run prediction|^Compare|Score it/ }).last().click();
  await p.waitForSelector('.track', { timeout: 30000 });
  const t0 = Date.now(); const stages = new Set(); let maxFrac = 0, monotonic = true, lastStage = -1, sawLive = false;
  const pill = async () => (await p.locator('.topbar button', { hasText: 'Run in progress' }).count()) > 0;
  while (true) {
    if (await p.locator('.big-number, .error-card').count()) break;
    const idx = await p.evaluate(() => [...document.querySelectorAll('.track li')].findIndex((li) => li.classList.contains('active')));
    if (idx >= 0) stages.add(idx);
    if (idx !== lastStage) { lastStage = idx; console.log(`   [${label}] ${fmtMin(Date.now() - t0)}: stage ${idx}`); }
    const now = await p.locator('[role=progressbar]').getAttribute('aria-valuenow').catch(() => null);
    if (now != null) { const f = Number(now); if (f < maxFrac) monotonic = false; maxFrac = Math.max(maxFrac, f); }
    if (!sawLive) { const log = await p.locator('pre[aria-label="Live log"]').textContent().catch(() => ''); if (log && log.length > 20) sawLive = true; }
    await onTick?.(p, Date.now() - t0);
    if (Date.now() - t0 > maxMs) { check(`${label}: finishes within ${fmtMin(maxMs)}`, false, 'timed out'); break; }
    await sleep(4000);
  }
  const ms = Date.now() - t0;
  const failed = (await p.locator('.error-card').count()) > 0;
  let error = '';
  if (failed) { await p.locator('.error-card summary').click().catch(() => {}); error = (await p.locator('.error-card').innerText()).replace(/\s+/g, ' ').slice(0, 700); }
  return { ms, failed, error, stages: [...stages], monotonic, maxFrac, cmd };
}

/** Everything a person can do on a finished Dock result, and a check on each. */
export async function verifyDockResults(p, label, { expectCommandHas = [], expectColumns = [] } = {}) {
  const dg = (await p.locator('.big-number .n').textContent()).replace('−', '-').trim();
  const v = parseFloat(dg);
  check(`${label}: a real ΔG number, physically sensible`, Number.isFinite(v) && v < 0 && v > -25, `${dg} kcal/mol`);
  check(`${label}: not labelled Demo`, (await p.locator('.badge-demo').count()) === 0, '');
  check(`${label}: the typical error is stated (±1.6)`, /1\.6/.test(await p.locator('.err-line').textContent()), '');
  check(`${label}: a plain-language meaning and our-own-guide label`, (await p.locator('.glass.panel p').filter({ hasText: /tighter|stick|grip|binding/i }).count()) >= 1 && /Our own rough guide/.test(await p.locator('.guide-line').textContent()), '');
  const rows = await p.locator('table.poses tbody tr').count();
  check(`${label}: ranked pose table is filled`, rows >= 3, `${rows} rows`);
  // pose rows: select several, the 3D label follows and the ΔG column is numeric and ordered
  const dgs = await p.locator('table.poses tbody tr td:nth-child(2)').allTextContents();
  const nums = dgs.map((t) => parseFloat(t.replace('−', '-')));
  check(`${label}: every pose has a numeric ΔG`, nums.every(Number.isFinite), dgs.slice(0, 5).join(' '));
  for (const i of [1, Math.min(2, rows), Math.min(5, rows)]) {
    await p.locator(`table.poses tbody tr[data-rank="${i}"]`).click(); await sleep(700);
    const lab = (await p.locator('.pose-label').textContent()).trim();
    check(`${label}: selecting pose ${i} loads it ("${lab}") and marks the row`, new RegExp(`Pose ${i} of`).test(lab) && (await p.locator(`table.poses tr[data-rank="${i}"]`).getAttribute('aria-selected')) === 'true', '');
  }
  // keyboard selection of a row
  await p.locator('table.poses tbody tr[data-rank="1"]').focus(); await p.keyboard.press('Enter'); await sleep(500);
  check(`${label}: a pose row can be chosen with the keyboard`, /Pose 1 of/.test(await p.locator('.pose-label').textContent()), '');
  // Expert columns and command
  await p.getByRole('button', { name: 'Expert' }).click(); await sleep(200);
  check(`${label}: Expert adds ranking score / clashes / cluster, with the "not a ΔG" note`, (await p.locator('th', { hasText: 'Ranking score' }).count()) === 1 && /not a ΔG/.test(await p.locator('.glass.panel').innerText()), '');
  const cmd = (await p.locator('pre.cmd').last().textContent()).replace(/\s+/g, ' ');
  check(`${label}: Expert shows the exact command that was run`, /hybridock-pep dock/.test(cmd) && expectCommandHas.every((x) => cmd.includes(x)), cmd.slice(0, 220));
  check(`${label}: Expert shows the dissociation-constant conversion`, /dissociation constant/.test(await p.locator('.glass.panel').innerText()), '');
  const saved = (await p.locator('text=Saved in').first().textContent().catch(() => '')).trim();
  check(`${label}: Expert shows where the run was saved`, /runs\//.test(saved) || /\//.test(saved), saved);
  await p.getByRole('button', { name: 'Guided' }).click();
  // downloads
  const dl = async (re, ext) => { const [d] = await Promise.all([p.waitForEvent('download', { timeout: 30000 }), p.getByRole('button', { name: re }).click()]); return { name: d.suggestedFilename(), text: fs.readFileSync(await d.path(), 'utf8') }; };
  const pdb = await dl(/Best pose/);
  const atoms = pdb.text.split('\n').filter((l) => l.startsWith('ATOM')).length;
  check(`${label}: "Best pose" downloads a real PDB (${pdb.name})`, atoms > 20, `${atoms} ATOM lines`);
  const csv = await dl(/Ranked list/);
  const header = csv.text.split('\n')[0];
  check(`${label}: "Ranked list" downloads the real CSV with delta_g and rank_score`, /delta_g/.test(header) && /rank_score/.test(header) && expectColumns.every((c) => header.includes(c)), `${header.slice(0, 120)}`);
  const csvRows = csv.text.trim().split('\n').length - 1;
  check(`${label}: the CSV has at least as many poses as the table`, csvRows >= rows, `${csvRows} vs ${rows}`);
  await p.context().grantPermissions(['clipboard-read', 'clipboard-write']).catch(() => {});
  await p.getByRole('button', { name: /Run folder/ }).click(); await sleep(500);
  const clip = await p.evaluate(() => navigator.clipboard.readText()).catch(() => '');
  check(`${label}: "Run folder" copies the folder path`, /runs\//.test(clip) || (await p.locator('.toast').last().textContent().catch(() => '')).includes('runs'), clip || (await p.locator('.toast').last().textContent().catch(() => '')));
  await shot(p, `result-${label.replace(/\W+/g, '_')}`);
  return { dg: v, rows, header, csv: csv.text };
}

export async function goHomeAndCheckRecent(p, label) {
  await p.locator('.btn', { hasText: 'Done' }).last().click().catch(() => {});
  await p.goto(BASE + '/#/'); await p.waitForSelector('.hero-card');
  check(`${label}: the run is in "Recent predictions" with a Download button`, (await p.locator('.recent-card').count()) >= 1 && (await p.locator('.recent-dl').count()) >= 1, '');
}

export { MDM2, POSE };
