import { browser, newPage, check, finish, shot, bigPdb, BASE } from './lib.mjs';
import fs from 'node:fs';
const b = await browser(); const p = await newPage(b);
await p.addInitScript(() => { window.__long = 0; new PerformanceObserver((l) => { for (const e of l.getEntries()) window.__long += e.duration; }).observe({ entryTypes: ['longtask'] }); });
await p.goto(`${BASE}/#/predict`); await p.waitForSelector('.pick');
for (const [file, label] of [[bigPdb('big_small.pdb', 6), '≈ 4k atoms'], [bigPdb('big_medium.pdb', 80), '≈ 56k atoms']]) {
  const sizeMB = (fs.statSync(file).size / 1048576).toFixed(1);
  await p.evaluate(() => { window.__long = 0; });
  const t0 = Date.now();
  await p.locator('input[type=file]').first().setInputFiles(file);
  // either it loads ("residues loaded") or we get a friendly toast; either way the page must stay responsive
  await Promise.race([p.waitForSelector('text=residues loaded', { timeout: 90000 }), p.waitForSelector('.toast', { timeout: 90000 })]);
  const dt = Date.now() - t0;
  const loaded = await p.locator('text=residues loaded').count();
  const toast = loaded ? '' : (await p.locator('.toast').last().textContent()).trim();
  const blocked = await p.evaluate(() => Math.round(window.__long));
  check(`upload ${label} (${sizeMB} MB): loads or is refused politely`, loaded || (toast.length > 10 && !/undefined|TypeError/.test(toast)), `${loaded ? 'loaded' : 'refused: ' + toast} in ${dt} ms`);
  check(`upload ${label}: main thread not frozen for long`, blocked < 6000, `${blocked} ms of long tasks`);
  // the page still responds to input
  const t1 = Date.now(); await p.locator('input[type=search]').fill('mdm'); const rt = Date.now() - t1;
  check(`upload ${label}: UI still responds afterwards`, rt < 3000, `${rt} ms`);
  await p.locator('input[type=search]').fill('');
  await shot(p, `s8-${sizeMB}mb`);
}
check('no page errors', p.diag.pageErrors.length === 0, p.diag.pageErrors.slice(0, 2).join('|'));
await b.close(); finish('08_uploads');
