import { browser, newPage, check, finish, shot, BASE, MDM2, POSE, sleep } from './lib.mjs';
import { execSync } from 'node:child_process';
// Fault injection needs control of the machine that runs the server (kill / freeze / restart it), over SSH:
//   FAULT_SSH=myhost  FAULT_START_CMD='cd ~/hybridock-pep && setsid nohup hybridock-pep serve --no-browser > serve.log 2>&1 < /dev/null &'
const HOST = process.env.FAULT_SSH, START = process.env.FAULT_START_CMD;
if (!HOST || !START) { console.log('SKIP 09_faults: set FAULT_SSH (ssh host of the server) and FAULT_START_CMD (starts it detached)'); process.exit(0); }
const SSH = (cmd) => execSync(`ssh -n -o ConnectTimeout=10 ${HOST} '${cmd}'`, { encoding: 'utf8', timeout: 60000 });
const PORT = process.env.SERVER_PORT || '';  // when several servers share a machine, name THIS one's port
const SRV = `[b]in/hybridock-pep serve${PORT ? '.*--port ' + PORT : ''}`;
const srvPid = () => SSH(`pgrep -f "${SRV}" | head -1`).trim();
const startServer = () => { execSync(`ssh -f -n ${HOST} '${START}'`, { timeout: 30000, stdio: 'ignore' }); };
async function waitUp(maxS = 40) { for (let i = 0; i < maxS; i++) { try { execSync(`curl -s -m 2 ${BASE}/api/env > /dev/null`, { timeout: 5000 }); return true; } catch { await sleep(1000); } } return false; }
const b = await browser();
async function fillScore(p) {
  await p.goto(`${BASE}/#/score`); await p.waitForSelector('.drop');
  const inputs = p.locator('input[type=file]');
  await inputs.nth(0).setInputFiles(MDM2); await p.waitForSelector('.msg.ok');
  await inputs.nth(1).setInputFiles(POSE); await p.locator('#sc-pep').fill('ETFSDLWKLLPE');
}
const outcome = async (p, ms = 120000) => { await p.waitForSelector('.big-number, .error-card', { timeout: ms }); return (await p.locator('.big-number').count()) ? 'result ' + (await p.locator('.big-number .n').textContent()).trim() : 'error: ' + (await p.locator('.error-card h2').textContent()).trim(); };

// A. the server is FROZEN (SIGSTOP: accepts connections, never answers) for 15 s mid-run, then thawed
{
  const p = await newPage(b); await fillScore(p);
  await p.getByRole('button', { name: /Score it/ }).click(); await p.waitForSelector('.track');
  const pid = srvPid(); SSH(`kill -STOP ${pid}`); const t0 = Date.now();
  await sleep(15000);
  const stillRunning = await p.locator('.track').count();
  const elapsedTxt = (await p.locator('.times b').last().textContent()).trim();
  check('while the server is frozen the Running screen stays up (no crash, timer ticking)', stillRunning === 1 && elapsedTxt !== '0:00', `elapsed shows ${elapsedTxt}`);
  await shot(p, 's9-frozen');
  SSH(`kill -CONT ${pid}`);
  const o = await outcome(p, 90000);
  check('after the server thaws the run completes or ends in a clear message', /result|error: (Couldn.t reach|Another run|The server restarted|That run didn)/.test(o), `${o} (${((Date.now() - t0) / 1000).toFixed(0)}s)`);
  await p.context().close();
}
// B. Stop while the server is frozen: the Stop button must still work (client-side abort)
{
  const p = await newPage(b); await fillScore(p);
  await p.getByRole('button', { name: /Score it/ }).click(); await p.waitForSelector('.track');
  const pid = srvPid(); SSH(`kill -STOP ${pid}`); await sleep(2500);
  await p.getByRole('button', { name: 'Stop' }).click();
  await p.waitForSelector('.drop, .hero-card', { timeout: 15000 });
  check('Stop works even while the server is unresponsive', true);
  SSH(`kill -CONT ${pid}`); await sleep(2000);
  await p.context().close();
}
// C. the server is KILLED mid-run and stays down 40 s: friendly "can't reach" + Try again works after it is back
{
  const p = await newPage(b); await fillScore(p);
  await p.getByRole('button', { name: /Score it/ }).click(); await p.waitForSelector('.track');
  SSH(`pkill -f "${SRV}"; true`); const t0 = Date.now();
  const o = await outcome(p, 120000);
  check('server killed mid-run: UI ends in a clear message, not a hang', /error: Couldn.t reach the server/.test(o) || /result/.test(o), `${o} after ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  await shot(p, 's9-killed');
  startServer(); const up = await waitUp(); check('server restarts cleanly after the kill', up, '');
  if (/error/.test(o)) { await p.getByRole('button', { name: 'Try again' }).click(); const o2 = await outcome(p, 90000); check('Try again after the server is back gives a result', /result/.test(o2), o2); }
  await p.context().close();
}
// D. the server is killed and restarted within 3 s mid-run (the job id is forgotten): "server restarted" wording
{
  const p = await newPage(b); await fillScore(p);
  await p.getByRole('button', { name: /Score it/ }).click(); await p.waitForSelector('.track');
  SSH(`pkill -f "${SRV}"; true`); startServer(); await waitUp();
  const o = await outcome(p, 120000);
  check('quick restart mid-run: friendly outcome (result, restarted, or unreachable)', /result|error: (The server restarted|Couldn.t reach)/.test(o), o);
  await p.context().close();
}
// E. the BROWSER goes offline mid-run, then comes back
{
  const p = await newPage(b); await fillScore(p);
  await p.getByRole('button', { name: /Score it/ }).click(); await p.waitForSelector('.track');
  await p.context().setOffline(true);
  const o = await outcome(p, 120000);
  check('browser offline mid-run: friendly message', /error: Couldn.t reach the server/.test(o), o);
  await p.context().setOffline(false);
  await p.getByRole('button', { name: 'Try again' }).click(); const o2 = await outcome(p, 90000);
  check('back online: Try again produces a result', /result/.test(o2), o2);
  await p.context().close();
}
// F. reload the page mid-run: lands on Home, and the next run works once the first finishes
{
  const p = await newPage(b); await fillScore(p);
  await p.getByRole('button', { name: /Score it/ }).click(); await p.waitForSelector('.track');
  await p.reload(); await p.waitForSelector('#main > *');
  check('reload mid-run does not break the app', p.diag.pageErrors.length === 0 && (await p.locator('#topbar').count()) === 1, p.url());
  await sleep(6000); await fillScore(p); await p.getByRole('button', { name: /Score it/ }).click();
  const o = await outcome(p, 90000); check('a new run after the reload works', /result/.test(o), o);
  await p.context().close();
}
await b.close(); finish('09_faults');
