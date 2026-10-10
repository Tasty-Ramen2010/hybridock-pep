// Regenerates the screenshots in docs/guide-img/ from a LIVE server, by driving the real UI (it starts one real Quick
// dock and one real Compare, so allow 15 to 30 minutes on a CPU-only machine).
//
//   BASE=http://127.0.0.1:8000 node guide_shots.mjs            # writes ../../docs/guide-img/*.png
//   OUT=/tmp/shots BASE=... node guide_shots.mjs               # write somewhere else
//   ONLY=home,menu BASE=... node guide_shots.mjs               # just some, no real runs (home, menu, history, score, compare need none)
//
// Then `python3 scripts/build_web.py` copies them into the app (the --check in CI fails if you forget).
import { browser, BASE, MDM2, POSE, sleep } from './lib.mjs';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const OUT = process.env.OUT || path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..', 'docs', 'guide-img');
const ONLY = process.env.ONLY ? new Set(process.env.ONLY.split(',')) : null;
const want = (n) => !ONLY || ONLY.has(n);
fs.mkdirSync(OUT, { recursive: true });

const b = await browser();
const ctx = await b.newContext({ viewport: { width: 1100, height: 820 }, deviceScaleFactor: 1, colorScheme: 'light' });
const p = await ctx.newPage();
const errors = []; p.on('pageerror', (e) => errors.push(String(e)));
const shot = async (name, opts = {}) => { if (!want(name)) return; await sleep(1200); await p.screenshot({ path: path.join(OUT, `${name}.png`), ...opts }); console.log('  shot', name); };
const cont = () => p.getByRole('button', { name: 'Continue' });
const waitResult = async (ms = 3 * 3600 * 1000) => { const t0 = Date.now(); while (Date.now() - t0 < ms) { if (await p.locator('.big-number').count()) return true; if (await p.locator('.error-card').count()) throw new Error('the run failed: ' + (await p.locator('.error-card').innerText()).slice(0, 300)); await sleep(5000); } throw new Error('timed out'); };

// ---- Predict: the four steps (no run yet) ----
await p.goto(BASE + '/#/predict'); await p.waitForSelector('.pick');
await p.locator('.pick', { has: p.locator('.mono', { hasText: 'PDB 1YCR' }) }).click(); await p.waitForSelector('text=residues loaded');
await shot('protein');
await cont().click(); await p.waitForSelector('#pep-input');
await p.locator('#pep-input').fill('ETFSDLWKLLPE'); await p.locator('#pep-input').blur();
await shot('peptide');
await cont().click(); await p.waitForSelector('.choice'); await sleep(600);
const suggested = p.getByRole('button', { name: /Use the suggested site/ }); if (await suggested.isVisible()) await suggested.click();
await shot('site');
await cont().click(); await p.waitForSelector('.summary');
await p.getByRole('radio', { name: /Quick/ }).click();
await shot('review');
await p.getByRole('button', { name: 'Expert' }).click(); await p.locator('details.adv summary', { hasText: 'Advanced settings' }).click(); await sleep(400);
if (want('expert')) { await p.evaluate(() => document.querySelector('details.adv').scrollIntoView({ block: 'start' })); await shot('expert'); }
await p.getByRole('button', { name: 'Guided' }).click();
await p.evaluate(() => scrollTo(0, 0));

// ---- the real dock ----
if (want('running') || want('result') || want('poses') || want('history') || want('home')) {
  await p.getByRole('button', { name: /Run prediction/ }).click(); await p.waitForSelector('.track', { timeout: 30000 });
  await sleep(20000);
  await shot('running');
  await waitResult();
  await sleep(2500);
  await shot('result');
  await p.locator('.panel').evaluate((el) => el.querySelector('.table-wrap')?.scrollIntoView({ block: 'center' }));
  await shot('poses');
}

// ---- Compare: the form, then a real comparison ----
await p.goto(BASE + '/#/compare'); await p.waitForSelector('.pcard'); await sleep(1500);
await p.locator('select[aria-label="Target protein"]').selectOption({ label: 'PfLDH (1T2D)' }).catch(() => {});
await p.locator('select[aria-label="Off-target protein"]').selectOption({ label: 'Human LDH (1I0Z)' }).catch(() => {});
await p.locator('#cmp-pep').fill('LISDAELEAIFEADC'); await sleep(800);
await shot('compare');
if (want('compare-result')) {
  await p.getByRole('radio', { name: /Quick/ }).click();
  await p.getByRole('button', { name: /^Compare/ }).last().click(); await p.waitForSelector('.track', { timeout: 30000 });
  await waitResult(); await sleep(2500);
  await shot('compare-result');
}

// ---- Score: the form filled in (the real score itself takes seconds) ----
await p.goto(BASE + '/#/score'); await p.waitForSelector('.drop');
await p.locator('input[type=file]').nth(0).setInputFiles(MDM2); await p.waitForSelector('.msg.ok');
await p.locator('input[type=file]').nth(1).setInputFiles(POSE); await sleep(800);
await p.locator('#sc-pep').fill('ETFSDLWKLLPE'); await sleep(500);
await shot('score');

// ---- Home, History and the menu, with real runs behind them ----
await p.goto(BASE + '/#/'); await p.waitForSelector('.hero-card'); await sleep(800);
await shot('home');
await p.getByRole('button', { name: 'Appearance and name' }).click(); await shot('menu'); await p.keyboard.press('Escape');
await p.getByRole('button', { name: 'History' }).click(); await p.waitForSelector('dialog[open]'); await shot('history');

console.log(errors.length ? 'page errors: ' + errors.join(' | ') : 'no page errors');
await b.close();
