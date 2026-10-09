import { browser, newPage, check, finish, shot, BASE, sleep } from './lib.mjs';
import fs from 'node:fs';
// A real Dock without the sampling engine: it re-scores saved poses (--input-poses). Point E2E_INPUT_POSES at a folder, ON THE SERVER'S
// machine, holding poses of ETFSDLWKLLPE (for example the first 12 poses_*/pose_N.pdb of a finished MDM2 run).
const POSES = process.env.E2E_INPUT_POSES;
if (!POSES) { console.log('SKIP 12_real_dock: set E2E_INPUT_POSES to a folder of saved ETFSDLWKLLPE poses on the server'); process.exit(0); }
const b = await browser(); const p = await newPage(b);
await p.goto(BASE + '/'); await p.waitForSelector('.hero-card');
await p.getByRole('button', { name: 'New prediction' }).click(); await p.waitForSelector('.pick');
await p.locator('.pick', { hasText: 'MDM2' }).first().click(); await p.getByRole('button', { name: 'Continue' }).click();
await p.waitForSelector('#pep-input'); await p.locator('#pep-input').fill('ETFSDLWKLLPE'); await p.getByRole('button', { name: 'Continue' }).click();
await p.waitForSelector('.choice');
const useSuggested = p.getByRole('button', { name: /Use the suggested site/ }); if (await useSuggested.count()) await useSuggested.click();
await p.getByRole('button', { name: 'Continue' }).click(); await p.waitForSelector('.summary');
await p.getByRole('button', { name: 'Expert' }).click();
await p.locator('details.adv summary', { hasText: 'Advanced settings' }).click();
await p.locator('label.field-label', { hasText: 'Input-poses folder' }).locator('xpath=following-sibling::input').fill(POSES);
await p.locator('label.check', { hasText: 'Skip pre-minimization' }).locator('input').check();
await p.waitForTimeout(1500);
const cmd = (await p.locator('pre.cmd').first().textContent()).replace(/\s+/g, ' ');
check('Expert command shows --input-poses and --no-minimize', cmd.includes('--input-poses ' + POSES) && /--no-minimize/.test(cmd), cmd.slice(0, 220));
await shot(p, 's12-review');
const t0 = Date.now();
await p.getByRole('button', { name: /Run prediction/ }).click(); await p.waitForSelector('.track', { timeout: 20000 });
check('Running screen shows the four plain stages', (await p.locator('.track li').count()) === 4, '');
await sleep(4000); await p.locator('details.adv summary', { hasText: 'Live log' }).click().catch(() => {});
await sleep(1500); const log = ((await p.locator('pre[aria-label="Live log"]').textContent().catch(() => '')) || '').trim();
check('Expert live log streams real program output', log.length > 20, `${log.length} chars; first: ${log.split('\n')[0]?.slice(0, 100)}`);
await shot(p, 's12-running');
await p.waitForSelector('.big-number, .error-card', { timeout: 900000 });
const dt = ((Date.now() - t0) / 1000).toFixed(0);
if (await p.locator('.error-card').count()) {
  await p.locator('.error-card summary').click().catch(() => {});
  const why = (await p.locator('.error-card').textContent()).replace(/\s+/g, ' ').slice(0, 500);
  await shot(p, 's12-error'); check('real Dock (input poses) finishes with a result', false, `${dt}s: ${why}`);
} else {
  const dg = (await p.locator('.big-number .n').textContent()).replace('−', '-').trim();
  check('real Dock (input poses) finishes with a result', !isNaN(parseFloat(dg)), `ΔG ${dg} kcal/mol after ${dt}s`);
  check('Dock ΔG is physically sensible', parseFloat(dg) < 0 && parseFloat(dg) > -20, dg);
  check('result is not labelled Demo', (await p.locator('.badge-demo').count()) === 0, '');
  const rows = await p.locator('table.poses tbody tr').count();
  check('ranked poses table is filled', rows >= 1, `${rows} rows`);
  const rs = await p.locator('table.poses tbody tr').first().locator('td').allTextContents();
  check('Expert shows rank_score separately from ΔG, labelled as not a ΔG', (await p.locator('th', { hasText: 'Ranking score' }).count()) === 1, rs.join(' | '));
  // click another pose; the 3D pose changes and the label updates
  if (rows > 1) { await p.locator('table.poses tbody tr').nth(1).click(); await sleep(800); check('selecting a row loads that pose in 3D', /Pose 2 of/.test(await p.locator('.pose-label').textContent()), await p.locator('.pose-label').textContent()); }
  // real downloads from the server
  const [d1] = await Promise.all([p.waitForEvent('download'), p.getByRole('button', { name: /Best pose/ }).click()]);
  const pdb = fs.readFileSync(await d1.path(), 'utf8'); check('Download "Best pose" returns a real PDB', /ATOM/.test(pdb) && pdb.length > 1000, `${d1.suggestedFilename()} ${pdb.length} bytes`);
  const [d2] = await Promise.all([p.waitForEvent('download'), p.getByRole('button', { name: /Ranked list/ }).click()]);
  const csv = fs.readFileSync(await d2.path(), 'utf8'); check('Download "Ranked list" returns the real CSV with delta_g and rank_score columns', /delta_g/.test(csv.split('\n')[0]) && /rank_score/.test(csv.split('\n')[0]), `${d2.suggestedFilename()} ${csv.split('\n').length - 1} rows`);
  const saved = (await p.locator('text=Saved in').first().textContent().catch(() => '')).trim(); check('Expert shows where the run was saved', /runs\/studio/.test(saved) || saved.length > 8, saved);
  await shot(p, 's12-result');
  // it appears in Recent + History
  await p.goto(BASE + '/#/'); await p.waitForSelector('.recent-card'); check('the run appears in Recent predictions with a Download button', (await p.locator('.recent-card .recent-dl').count()) >= 1, '');
  const [d3] = await Promise.all([p.waitForEvent('download'), p.locator('.recent-dl').first().click()]); check('Recent → Download gives the real ranked list', d3.suggestedFilename() === 'ranked_poses.csv', d3.suggestedFilename());
}
check('no page errors', p.diag.pageErrors.length === 0, p.diag.pageErrors.slice(0, 2).join('|'));
await b.close(); finish('12_real_dock');
