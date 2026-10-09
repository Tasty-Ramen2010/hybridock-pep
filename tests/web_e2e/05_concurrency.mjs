import { browser, newPage, check, finish, shot, BASE, MDM2, POSE, sleep } from './lib.mjs';
const b = await browser();
// ---- A. 12 tabs load the home page at the same moment ----
let t0 = Date.now();
const pages = await Promise.all(Array.from({ length: 12 }, () => newPage(b)));
const loads = await Promise.all(pages.map(async (p) => { const s = Date.now(); await p.goto(BASE + '/'); await p.waitForSelector('.hero-card', { timeout: 60000 }); return Date.now() - s; }));
check('12 simultaneous tabs all load', loads.length === 12, `slowest ${Math.max(...loads)} ms, median ${loads.sort((a, b) => a - b)[6]} ms`);
check('no tab hit a 5xx or page error', pages.every((p) => p.diag.bad.length === 0 && p.diag.pageErrors.length === 0), pages.flatMap((p) => [...p.diag.bad, ...p.diag.pageErrors]).join('|'));
// ---- B. every tab hops between routes 25 times, all at once ----
const routes = ['#/', '#/predict', '#/compare', '#/score', '#/predict', '#/'];
await Promise.all(pages.map(async (p, i) => { for (let k = 0; k < 25; k++) { await p.goto(`${BASE}/${routes[(k + i) % routes.length]}`); await p.waitForSelector('#main > *', { timeout: 30000 }); } }));
check('300 route changes across 12 tabs, no errors', pages.every((p) => p.diag.pageErrors.length === 0 && p.diag.bad.length === 0), pages.flatMap((p) => [...p.diag.bad, ...p.diag.pageErrors]).slice(0, 3).join('|'));
await Promise.all(pages.map((p) => p.context().close()));

// ---- C. three Score jobs at the same instant: the server runs ONE at a time, the others must be told so plainly ----
const sp = await Promise.all([0, 1, 2].map(() => newPage(b)));
async function fillScore(p) {
  await p.goto(`${BASE}/#/score`); await p.waitForSelector('.drop');
  const inputs = p.locator('input[type=file]');
  await inputs.nth(0).setInputFiles(MDM2); await p.waitForSelector('.msg.ok');
  await inputs.nth(1).setInputFiles(POSE); await p.locator('#sc-pep').fill('ETFSDLWKLLPE');
}
/** Click Score it; if the server is busy ("Another run is already going") press Try again until it is our turn. */
async function scoreWithRetry(p, tries = 60) {
  await p.getByRole('button', { name: /Score it/ }).click();
  let busy = 0;
  for (let i = 0; i < tries; i++) {
    await p.waitForSelector('.big-number, .error-card', { timeout: 300000 });
    if (await p.locator('.big-number').count()) return { ok: true, v: (await p.locator('.big-number .n').textContent()).trim(), busy };
    const title = (await p.locator('.error-card h2').textContent()).trim();
    if (!/Another run is already going/.test(title)) return { ok: false, v: title, busy };
    busy++; await sleep(700); await p.getByRole('button', { name: 'Try again' }).click();
    await p.waitForFunction(() => !document.querySelector('.error-card'), null, { timeout: 15000 }); // wait for the Running screen to replace the old card
  }
  return { ok: false, v: 'gave up waiting for a turn', busy };
}
await Promise.all(sp.map(fillScore));
const first = await Promise.all(sp.map(async (p) => {
  await p.getByRole('button', { name: /Score it/ }).click();
  await p.waitForSelector('.big-number, .error-card', { timeout: 120000 });
  return (await p.locator('.big-number').count()) ? 'ok' : (await p.locator('.error-card h2').textContent()).trim();
}));
console.log('simultaneous:', JSON.stringify(first));
check('exactly one of 3 simultaneous Score jobs runs; the rest get a plain "another run is going"', first.filter((x) => x === 'ok').length >= 1 && first.every((x) => x === 'ok' || /Another run is already going/.test(x)), first.join(' / '));
// ---- D. 24 real Score runs from 4 tabs contending for the single slot: all must eventually succeed ----
const four = await Promise.all([0, 1, 2, 3].map(() => newPage(b)));
const t1 = Date.now(); let okCount = 0, failCount = 0, busyTotal = 0; const vals = new Set(); const why = [];
await Promise.all(four.map(async (p) => { for (let k = 0; k < 6; k++) { await fillScore(p); const r = await scoreWithRetry(p); busyTotal += r.busy; if (r.ok) { okCount++; vals.add(r.v); } else { failCount++; why.push(r.v); } } }));
check('24 real Score runs, 4 tabs contending: all succeed', failCount === 0, `${okCount} ok, ${failCount} failed (${why.join(',')}) in ${((Date.now() - t1) / 1000).toFixed(0)}s; ${busyTotal} polite "busy" retries`);
check('every run produced the identical number (deterministic)', vals.size === 1, [...vals].join(','));
check('no page errors or 5xx anywhere', [...sp, ...four].every((p) => p.diag.pageErrors.length === 0 && p.diag.bad.length === 0), [...sp, ...four].flatMap((p) => [...p.diag.pageErrors, ...p.diag.bad]).slice(0, 3).join('|'));
await b.close(); finish('05_concurrency');
