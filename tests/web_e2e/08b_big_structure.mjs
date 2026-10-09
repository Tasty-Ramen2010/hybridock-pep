import { browser, newPage, check, finish, shot, bigPdb, BASE } from './lib.mjs';
import fs from 'node:fs';
const b = await browser(); const p = await newPage(b);
await p.addInitScript(() => { window.__long = 0; new PerformanceObserver((l) => { for (const e of l.getEntries()) window.__long += e.duration; }).observe({ entryTypes: ['longtask'] }); });
await p.goto(`${BASE}/#/predict`); await p.waitForSelector('.pick');
const big = bigPdb('big_30mb.pdb', 600);
const mb = (fs.statSync(big).size / 1048576).toFixed(1);
const t0 = Date.now(); await p.locator('input[type=file]').first().setInputFiles(big);
await Promise.race([p.waitForSelector('text=residues loaded', { timeout: 120000 }), p.waitForSelector('.toast', { timeout: 120000 })]);
const loaded = await p.locator('text=residues loaded').count(); const dt = Date.now() - t0;
const msg = loaded ? 'loaded' : (await p.locator('.toast').last().textContent()).trim();
const long = await p.evaluate(() => Math.round(window.__long));
check(`${mb} MB / 400k-atom upload: loads or is refused politely`, loaded || msg.length > 10, `${msg} in ${dt} ms; ${long} ms of long tasks`);
check('big upload: main thread blocked < 8 s total', long < 8000, `${long} ms`);
if (loaded) {
  await p.getByRole('button', { name: 'Continue' }).click(); await p.waitForSelector('#pep-input');
  await p.locator('.chip.mono').first().click(); await p.getByRole('button', { name: 'Continue' }).click(); await p.waitForSelector('.choice');
  await p.waitForTimeout(2000);
  // measure frames per second of the 3D stage on this big structure
  const fps = await p.evaluate(() => new Promise((res) => { let n = 0; const t = performance.now(); (function f() { n++; if (performance.now() - t < 2000) requestAnimationFrame(f); else res(n / 2); })(); }));
  check('3D view of the big structure still animates (≥ 5 fps)', fps >= 5, `${fps.toFixed(0)} fps`);
  await shot(p, 's8b-big-site');
  const t1 = Date.now(); await p.locator('input[type=range]').first().fill('40'); check('box slider responsive on big structure', Date.now() - t1 < 3000, `${Date.now() - t1} ms`);
}
check('no page errors', p.diag.pageErrors.length === 0, p.diag.pageErrors.slice(0, 2).join('|'));
await b.close(); finish('08b_big_structure');
