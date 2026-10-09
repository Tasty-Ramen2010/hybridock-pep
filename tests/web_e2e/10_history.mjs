import { browser, newPage, check, finish, shot, BASE, MDM2, POSE, sleep } from './lib.mjs';
const b = await browser(); const p = await newPage(b);
async function oneScore() {
  await p.goto(`${BASE}/#/score`); await p.waitForSelector('.drop');
  const inputs = p.locator('input[type=file]');
  await inputs.nth(0).setInputFiles(MDM2); await p.waitForSelector('.msg.ok');
  await inputs.nth(1).setInputFiles(POSE); await p.locator('#sc-pep').fill('ETFSDLWKLLPE');
  await p.getByRole('button', { name: /Score it/ }).click();
  await p.waitForSelector('.big-number, .error-card', { timeout: 120000 });
  return (await p.locator('.big-number').count()) === 1;
}
// fresh browser: history is seeded with 3 labelled Demo examples
await p.goto(BASE + '/#/'); await p.waitForSelector('.hero-card');
const seeded = await p.locator('.recent-card').count();
check('a fresh visitor sees at most 4 recent runs', seeded <= 4, `${seeded} cards`);
const t0 = Date.now(); let ok = 0;
for (let i = 0; i < 40; i++) { if (await oneScore()) ok++; }
check('40 real Score runs through the UI all succeed', ok === 40, `${ok}/40 in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
await p.goto(BASE + '/#/'); await p.waitForSelector('.recent-card');
const strip = await p.locator('.recent-card').count();
check('Home "Recent predictions" strip shows exactly 4', strip === 4, String(strip));
const stats = await p.locator('.stat .value').allTextContents();
check('"Predictions run" counts every run (40), not just the 30 kept in History', /^\d+$/.test(stats[0].trim()) && Number(stats[0]) === 40, stats.join(' | '));
const ls = await p.evaluate(() => { const raw = localStorage.getItem('hybridock-web:v1') || ''; const j = JSON.parse(raw || '{}'); return { kb: Math.round(raw.length / 1024), n: (j.history || []).length }; });
check('history is capped and fits in browser storage', ls.n > 0 && ls.n <= 100 && ls.kb < 4000, `${ls.n} entries, ${ls.kb} KB`);
await p.getByRole('button', { name: 'History' }).click(); await p.waitForSelector('dialog[open] .hist-item');
const listed = await p.locator('dialog[open] .hist-item').count();
check('History drawer lists the saved runs', listed === ls.n, `${listed} listed vs ${ls.n} stored`);
await shot(p, 's10-history');
// reopen an OLD entry (the last one) after a full reload
await p.keyboard.press('Escape'); await p.reload(); await p.waitForSelector('.recent-card');
await p.getByRole('button', { name: 'History' }).click(); await p.waitForSelector('dialog[open] .hist-item');
await p.locator('dialog[open] .hist-item').last().click(); await p.waitForSelector('.big-number, .error-card', { timeout: 20000 });
check('an old saved run reopens after a reload', (await p.locator('.big-number').count()) === 1, p.url());
// Clear all
await p.goto(BASE + '/#/'); await p.getByRole('button', { name: 'History' }).click(); await p.getByRole('button', { name: /Clear all/ }).click();
check('Clear all empties History', (await p.locator('dialog[open] .hist-item').count()) === 0, '');
await p.keyboard.press('Escape'); await p.reload(); await p.waitForSelector('.hero-card');
check('after Clear all and a reload the Recent strip is gone', (await p.locator('.recent:not([hidden]) .recent-card').count()) === 0, '');
check('no page errors', p.diag.pageErrors.length === 0, p.diag.pageErrors.slice(0, 2).join('|'));
await b.close(); finish('10_history');
