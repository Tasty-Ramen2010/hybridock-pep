import { browser, newPage, check, finish, shot, BASE } from './lib.mjs';
const b = await browser();
const p = await newPage(b);
let t0 = Date.now();
await p.goto(BASE + '/'); await p.waitForSelector('.hero-card', { timeout: 30000 });
check('home loads from the AGX', true, `${Date.now() - t0} ms`);
const chip = (await p.locator('.topbar .chip').first().textContent()).trim();
check('status chip says Live (not Demo)', /Live|Setup needed/.test(chip) && !(await p.locator('.topbar .badge-demo').count()), chip);
await p.locator('.topbar .chip').first().click();
const rows = await p.locator('.env-row').allTextContents();
check('status popover lists machine checks', rows.length >= 5, rows.join(' | ').slice(0, 300));
check('no stale ADFRsuite "licensed download" warning', !rows.join(' ').match(/licensed/i), '');
await p.keyboard.press('Escape');
check('hero is a real backend example (Known binder chip)', await p.locator('.hero-card .chip', { hasText: 'Known binder' }).count() === 1, '');
await shot(p, 's1-home-live');
// every route loads and shows its heading
for (const [route, sel] of [['#/predict', '.steps'], ['#/compare', '.pcard'], ['#/score', '.drop'], ['#/', '.hero-card']]) {
  await p.goto(`${BASE}/${route}`); await p.waitForSelector(sel, { timeout: 15000 });
  check(`route ${route} renders`, true);
}
// guided/expert + theme persistence across a reload
await p.getByRole('button', { name: 'Expert' }).click();
await p.getByRole('button', { name: /Switch to dark mode|Switch to light mode/ }).click();
await p.reload(); await p.waitForSelector('.hero-card');
const attrs = await p.evaluate(() => [document.documentElement.dataset.mode, document.documentElement.dataset.theme]);
check('mode + theme persist across reload', attrs[0] === 'expert' && attrs[1] === 'dark', attrs.join('/'));
// fonts + assets actually loaded (no fallback 404s)
const fonts = await p.evaluate(async () => { await document.fonts.ready; return [...document.fonts].filter(f => f.status === 'loaded').map(f => f.family); });
check('bundled fonts load', fonts.length >= 1, fonts.join(','));
check('no console errors', p.diag.errors.length === 0 && p.diag.pageErrors.length === 0, [...p.diag.errors, ...p.diag.pageErrors].join(' | '));
check('no 5xx responses', p.diag.bad.length === 0, p.diag.bad.join(' | '));
await b.close(); finish('01_smoke');
