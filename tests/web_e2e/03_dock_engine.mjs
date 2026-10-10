import { browser, newPage, check, finish, shot, BASE, sleep } from './lib.mjs';
const b = await browser(); const p = await newPage(b);
await p.goto(BASE + '/'); await p.waitForSelector('.hero-card');
await p.getByRole('button', { name: 'Run example' }).click();
await p.waitForSelector('.track', { timeout: 20000 });
check('Running screen appears after Run example', true);
const t0 = Date.now();
await p.waitForSelector('.big-number, .error-card', { timeout: Number(process.env.E2E_DOCK_WAIT_MIN || 90) * 60000 });
const failed = await p.locator('.error-card').count();
const dt = ((Date.now() - t0) / 1000).toFixed(1);
await shot(p, 's3-dock-error');
if (failed) {
  const title = (await p.locator('.error-card h2').textContent()).trim();
  const body = (await p.locator('.error-card').textContent()).replace(/\s+/g, ' ').slice(0, 500);
  check('failed dock ends in an error card within the wait limit', true, `${dt}s: ${title}`);
  check('error card offers a way out (Back / Try again)', await p.locator('.error-card .btn').count() >= 1, '');
  check('error text is readable (no stack-trace / [object Object])', !/Traceback|\[object|undefined/.test(body), body);
  console.log('ERROR CARD TEXT:', body);
} else {
  check('dock produced a result', true, (await p.locator('.big-number .n').textContent()));
}
check('server stayed up (page can navigate home)', true);
await p.goto(BASE + '/#/'); await p.waitForSelector('.hero-card');
check('no JS errors during failed run', p.diag.pageErrors.length === 0, p.diag.pageErrors.join('|'));
await b.close(); finish('03_dock_engine');
