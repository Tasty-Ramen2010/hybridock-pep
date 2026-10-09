// The one-click "Run example" path: a REAL Dock (the server's validated MDM2 + p53 example, Quick = 25 poses).
// Slow on a CPU-only machine (minutes to an hour); the point is that every stage reports honestly and the result is complete.
import { browser, newPage, check, finish, shot, BASE, sleep } from './lib.mjs';
import fs from 'node:fs';
const b = await browser(); const p = await newPage(b);
await p.goto(BASE + '/'); await p.waitForSelector('.hero-card');
check('Home offers the validated MDM2 example as a known binder', await p.locator('.hero-card .chip', { hasText: 'Known binder' }).count() === 1, '');
await p.getByRole('button', { name: 'Run example' }).click(); await p.waitForSelector('.track');
const t0 = Date.now(); const stages = new Set(); let last = '', maxFrac = 0, monotonic = true;
while (true) {
  if (await p.locator('.big-number, .error-card').count()) break;
  const idx = await p.evaluate(() => [...document.querySelectorAll('.track li')].findIndex((li) => li.classList.contains('active')));
  if (idx >= 0) stages.add(idx);
  const now = await p.locator('[role=progressbar]').getAttribute('aria-valuenow').catch(() => null);
  if (now != null) { const f = Number(now); if (f < maxFrac) monotonic = false; maxFrac = Math.max(maxFrac, f); }
  const eta = (await p.locator('.run-meta .times b').first().textContent().catch(() => '')) || '';
  const line = `${Math.round((Date.now() - t0) / 1000)}s stage ${idx} ${now}% eta="${eta.trim()}"`;
  if (line.split(' ').slice(1).join(' ') !== last.split(' ').slice(1).join(' ')) { console.log('   ' + line); last = line; }
  if (Date.now() - t0 > 3 * 3600 * 1000) { check('example finishes within 3 hours', false, 'timeout'); break; }
  await sleep(5000);
}
const secs = Math.round((Date.now() - t0) / 1000);
await shot(p, '13-example-end');
if (await p.locator('.error-card').count()) {
  await p.locator('.error-card summary').click().catch(() => {});
  check('real example run finishes with a result', false, (await p.locator('.error-card').textContent()).replace(/\s+/g, ' ').slice(0, 600));
} else {
  const dg = (await p.locator('.big-number .n').textContent()).replace('−', '-').trim();
  check('real example run finishes with a result', !isNaN(parseFloat(dg)), `ΔG ${dg} kcal/mol in ${secs}s`);
  check('progress bar never moved backwards', monotonic, `max ${maxFrac}%`);
  check('Running screen passed through the stages', stages.size >= 2, [...stages].join(','));
  const rows = await p.locator('table.poses tbody tr').count(); check('ranked poses table filled', rows >= 5, `${rows} rows`);
}
await b.close(); finish('13_example_run');
