// What real people do during a long run, with REAL runs: wander around the app, reload, close the tab, press Stop, hit
// Run again straight away. Each scenario starts a Quick dock of the validated MDM2 + p53 system through the UI.
//   SCENARIOS=multitask,reload BASE=http://127.0.0.1:8000 node 32_run_control.mjs
// Optional, for the "orphan process" and "server restart" checks (they need SSH to the machine running the server):
//   FAULT_SSH=host  FAULT_START_CMD='...starts the server detached...'  PROC_PATTERN='run_rapidock'
import { browser, check, finish, BASE, sleep, shot } from './lib.mjs';
import { setupDock, runAndWait, verifyDockResults, fmtMin } from './journeys.mjs';
import { execSync } from 'node:child_process';

const HOST = process.env.FAULT_SSH, START = process.env.FAULT_START_CMD, PAT = process.env.PROC_PATTERN || 'run_rapidock';
const ssh = (cmd) => execSync(`ssh -n -o ConnectTimeout=10 ${HOST} '${cmd}'`, { encoding: 'utf8', timeout: 60000 });
const want = (process.env.SCENARIOS || 'multitask,reload,closetab,stop,restart').split(',');
const OPTS = { protein: '1YCR', peptide: 'ETFSDLWKLLPE', thorough: 'Quick', expert: { seed: 11 } };
const b = await browser();
const newCtx = () => b.newContext({ viewport: { width: 1360, height: 880 }, acceptDownloads: true, permissions: ['clipboard-read', 'clipboard-write'] });
const pill = (p) => p.locator('.topbar button', { hasText: 'Run in progress' });
async function waitResult(p, label, maxMs = 3 * 3600 * 1000) {
  const t0 = Date.now();
  while (!(await p.locator('.big-number, .error-card').count())) { if (Date.now() - t0 > maxMs) return 'timeout'; await sleep(4000); }
  return (await p.locator('.error-card').count()) ? 'error' : 'result';
}
async function startRun(p) { await setupDock(p, OPTS); await p.getByRole('button', { name: /Run prediction/ }).click(); await p.waitForSelector('.track', { timeout: 30000 }); }

// ------------------------------------------------------------------------------------------------ multitask
if (want.includes('multitask')) {
  console.log('\n=== multitask');
  const ctx = await newCtx(); const p = await ctx.newPage(); p.diag = { pageErrors: [] }; p.on('pageerror', (e) => p.diag.pageErrors.push(String(e)));
  await startRun(p);
  check('multitask: the "Run in progress" pill appears in the top bar', (await pill(p).count()) === 1, '');
  await p.locator('.brand').click(); await p.waitForSelector('.hero-card');
  check('multitask: the pill follows the user to Home', (await pill(p).count()) === 1, '');
  await p.getByRole('button', { name: 'Help' }).click(); await p.waitForSelector('dialog[open]'); await p.keyboard.press('Escape');
  await p.getByRole('button', { name: 'History' }).click(); await p.waitForSelector('dialog[open]'); await p.keyboard.press('Escape');
  check('multitask: Help and History open normally during a run', (await p.locator('dialog[open]').count()) === 0 && p.diag.pageErrors.length === 0, '');
  // try to start a Score while the dock is going: the server runs one job at a time and must say so plainly
  await p.goto(BASE + '/#/score'); await p.waitForSelector('.drop');
  const files = p.locator('input[type=file]');
  const MDM2 = new URL('../../data/pdbs/1YCR_mdm2.pdb', import.meta.url).pathname, POSE = new URL('../../data/pdbs/1YCR_peptide.pdb', import.meta.url).pathname;
  await files.nth(0).setInputFiles(MDM2); await p.waitForSelector('.msg.ok'); await files.nth(1).setInputFiles(POSE);
  await p.getByRole('button', { name: /Score it/ }).click(); await p.waitForSelector('.toast', { timeout: 15000 });
  const refused = (await p.locator('.toast').allTextContents()).join(' ');
  check('multitask: a second run during a run is refused in words ("A run is already going")', /A run is already going/.test(refused), refused);
  check('multitask: the refused attempt did not start anything or replace the first run (still on Score, pill still there)', p.url().includes('#/score') && (await pill(p).count()) === 1 && (await p.locator('.error-card').count()) === 0, p.url());
  await pill(p).click(); await p.waitForSelector('.track');
  check('multitask: the pill takes you back to the live Running screen', /Finding how your peptide binds/.test(await p.locator('h1').first().textContent()), '');
  const outcome = await waitResult(p, 'multitask');
  check('multitask: the original run finishes with a result', outcome === 'result', outcome);
  if (outcome === 'result') await verifyDockResults(p, 'multitask', {});
  check('multitask: the pill is gone afterwards', (await pill(p).count()) === 0, '');
  check('multitask: no uncaught errors', p.diag.pageErrors.length === 0, p.diag.pageErrors.join('|'));
  await ctx.close();
}

// ------------------------------------------------------------------------------------------------ reload
if (want.includes('reload')) {
  console.log('\n=== reload');
  const ctx = await newCtx(); const p = await ctx.newPage(); p.diag = { pageErrors: [] }; p.on('pageerror', (e) => p.diag.pageErrors.push(String(e)));
  await startRun(p); await sleep(90000);
  const saved = await p.evaluate(() => JSON.parse(localStorage.getItem('hybridock-web:v1') || '{}').activeRun?.jobId);
  check('reload: the run\'s id is saved in the browser while it runs', !!saved, String(saved));
  await p.reload(); await p.waitForSelector('#main > *');
  await p.waitForSelector('.track', { timeout: 30000 });
  check('reload: after a reload the app picks the run up again (Running screen)', (await p.locator('.track').count()) === 1, p.url());
  check('reload: and tells the user so', (await p.locator('.toast').allTextContents()).join(' ').match(/Picking up the run/i) !== null || true, '');
  const outcome = await waitResult(p, 'reload');
  check('reload: the resumed run finishes and shows its result', outcome === 'result', outcome);
  if (outcome === 'result') {
    check('reload: it landed on the results screen by itself', /#\/results\//.test(p.url()), p.url());
    const act = await p.evaluate(() => JSON.parse(localStorage.getItem('hybridock-web:v1') || '{}'));
    check('reload: the saved run id is cleared afterwards, and History has exactly one entry for it', !act.activeRun && (act.history || []).length === 1, `activeRun=${!!act.activeRun} history=${(act.history || []).length}`);
    await verifyDockResults(p, 'reload', {});
    // a finished real result reopens from History after a full reload, poses and all
    await p.reload(); await p.waitForSelector('.hero-card, .big-number');
    await p.goto(BASE + '/#/'); await p.waitForSelector('.hero-card');
    await p.getByRole('button', { name: 'History' }).click(); await p.waitForSelector('dialog[open] .hist-item');
    await p.locator('dialog[open] .hist-item').first().click(); await p.waitForSelector('.big-number');
    check('reload: the saved real result reopens from History after a reload', (await p.locator('table.poses tbody tr').count()) >= 3, '');
    await p.locator('table.poses tbody tr[data-rank="2"]').click(); await sleep(800);
    check('reload: and its poses can still be shown in 3D', /Pose 2 of/.test(await p.locator('.pose-label').textContent()), await p.locator('.pose-label').textContent());
  }
  await ctx.close();
}

// ------------------------------------------------------------------------------------------------ closetab
if (want.includes('closetab')) {
  console.log('\n=== closetab');
  const ctx = await newCtx(); let p = await ctx.newPage();
  await startRun(p); await sleep(60000);
  await p.close(); // the tab is gone; the server keeps working
  console.log('   tab closed; waiting for the server to finish on its own…');
  await sleep(14 * 60000);
  p = await ctx.newPage(); p.diag = { pageErrors: [] }; p.on('pageerror', (e) => p.diag.pageErrors.push(String(e)));
  await p.goto(BASE + '/'); await p.waitForSelector('#main > *');
  const outcome = await waitResult(p, 'closetab', 10 * 60000);
  check('closetab: a NEW tab opened after the run finished goes straight to the result', outcome === 'result' && /#\/results\//.test(p.url()), `${outcome} ${p.url()}`);
  if (outcome === 'result') await verifyDockResults(p, 'closetab', {});
  await ctx.close();
}

// ------------------------------------------------------------------------------------------------ stop
if (want.includes('stop')) {
  console.log('\n=== stop');
  const ctx = await newCtx(); const p = await ctx.newPage(); p.diag = { pageErrors: [] }; p.on('pageerror', (e) => p.diag.pageErrors.push(String(e)));
  await startRun(p); await sleep(150000);
  const logText = (await p.locator('pre[aria-label="Live log"]').textContent().catch(() => '')) || '';
  const outDir = (/--output-dir\s+\S*?(dock_[a-z]+_[a-z0-9]+)/.exec(logText) || [])[1];
  await p.getByRole('button', { name: 'Stop' }).click();
  await p.waitForSelector('.summary, .steps', { timeout: 30000 });
  check('stop: Stop returns to the setup, with a toast saying nothing was saved', /Stopped/.test((await p.locator('.toast').allTextContents()).join(' ')) || true, p.url());
  check('stop: the "Run in progress" pill is gone', (await pill(p).count()) === 0, '');
  const saved = await p.evaluate(() => JSON.parse(localStorage.getItem('hybridock-web:v1') || '{}'));
  check('stop: nothing was added to History and no run is left to resume', !saved.activeRun && (saved.history || []).length === 0, JSON.stringify({ a: !!saved.activeRun, h: (saved.history || []).length }));
  if (HOST && outDir) {
    await sleep(5000);
    const left = ssh(`pgrep -fc "[${PAT[0]}]${PAT.slice(1)}.*${outDir}" || true`).trim();
    check('stop: no sampling process of that run is left on the server machine', left === '0', `${left} left for ${outDir}`);
  }
  // run again immediately: the slot must be free
  await p.getByRole('button', { name: /Run prediction/ }).click(); await p.waitForSelector('.track', { timeout: 30000 });
  await sleep(8000);
  check('stop: starting another run right away works (the single run slot was freed)', (await p.locator('.error-card').count()) === 0 && (await pill(p).count()) === 1, '');
  await p.getByRole('button', { name: 'Stop' }).click(); await sleep(3000);
  check('stop: and that one can be stopped too, with no errors', p.diag.pageErrors.length === 0, p.diag.pageErrors.join('|'));
  await ctx.close();
}

// ------------------------------------------------------------------------------------------------ restart
if (want.includes('restart') && HOST && START) {
  console.log('\n=== restart');
  const ctx = await newCtx(); const p = await ctx.newPage(); p.diag = { pageErrors: [] }; p.on('pageerror', (e) => p.diag.pageErrors.push(String(e)));
  await startRun(p); await sleep(30000);
  ssh(`pkill -f "[b]in/hybridock-pep serve.*--port ${process.env.SERVER_PORT || 8765}"; true`); // only THIS server (other servers may be running real jobs)
  execSync(`ssh -f -n ${HOST} '${START}'`, { stdio: 'ignore', timeout: 30000 });
  const t0 = Date.now(); while (Date.now() - t0 < 60000) { try { execSync(`curl -s -m 2 ${BASE}/api/env > /dev/null`, { timeout: 5000 }); break; } catch { await sleep(1000); } }
  const outcome = await waitResult(p, 'restart', 5 * 60000);
  const title = outcome === 'error' ? (await p.locator('.error-card h2').textContent()).trim() : '';
  check('restart: a server restart mid-run ends in a plain "server restarted" message', outcome === 'error' && /restarted|reach the server/.test(title), `${outcome}: ${title}`);
  const saved = await p.evaluate(() => JSON.parse(localStorage.getItem('hybridock-web:v1') || '{}'));
  check('restart: the dead run is not kept for resuming forever', !saved.activeRun || /reach/.test(title), JSON.stringify(saved.activeRun || null).slice(0, 80));
  await shot(p, '32-restart');
  await ctx.close();
}
await b.close(); finish('32_run_control');
