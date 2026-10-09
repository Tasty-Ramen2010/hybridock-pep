import { browser, newPage, check, finish, shot, scratch, BASE, MDM2, POSE, HLDH, sleep } from './lib.mjs';
import fs from 'node:fs';
const b = await browser(); const p = await newPage(b);
async function openScore() { await p.goto(`${BASE}/#/score`); await p.waitForSelector('.drop'); }
async function fillScore({ protein = MDM2, pose = POSE, pep = 'ETFSDLWKLLPE' } = {}) {
  const inputs = p.locator('input[type=file]');
  await inputs.nth(0).setInputFiles(protein); await p.waitForSelector('.msg.ok');
  await inputs.nth(1).setInputFiles(pose);
  await p.locator('#sc-pep').fill(pep);
}
// 1. the Score button is disabled until everything is provided
await openScore();
check('Score it is disabled with nothing provided', await p.getByRole('button', { name: /Score it/ }).isDisabled());
// 2. a non-PDB file as the protein is rejected in plain words (no crash)
await p.locator('input[type=file]').nth(0).setInputFiles(scratch('not_a_pdb.txt', 'hello this is not a structure\n'));
await p.waitForSelector('.toast', { timeout: 5000 });
const toast = (await p.locator('.toast').last().textContent()).trim();
check('non-PDB upload gives a friendly error', toast.length > 10 && !/undefined|TypeError|\[object/.test(toast), toast);
// 3. an empty file
await p.locator('input[type=file]').nth(0).setInputFiles(scratch('empty.pdb', '')); await sleep(600);
check('empty file does not break the page', await p.locator('.drop').count() >= 2, '');
// 4. the happy path, for real (AGX CPU)
await openScore(); await fillScore();
check('Score it enables when protein + pose + sequence are given', await p.getByRole('button', { name: /Score it/ }).isEnabled());
let t0 = Date.now();
await p.getByRole('button', { name: /Score it/ }).click();
await p.waitForSelector('.big-number, .error-card', { timeout: 180000 });
const dt = ((Date.now() - t0) / 1000).toFixed(1);
const ok = await p.locator('.big-number').count();
const n = ok ? (await p.locator('.big-number .n').textContent()).replace('−', '-').trim() : (await p.locator('.error-card').textContent()).slice(0, 300);
await shot(p, 's2-score-result');
check('real Score run finishes with a number', ok && !isNaN(parseFloat(n)), `${n} kcal/mol in ${dt}s`);
if (ok) {
  const v = parseFloat(n);
  check('score is physically sensible (-20 < dG < 0)', v < 0 && v > -20, String(v));
  check('result is not labelled Demo', (await p.locator('.big-number').locator('xpath=ancestor::*[contains(@class,"panel")]//*[contains(@class,"badge-demo")]').count()) === 0, '');
  check('typical error is shown', (await p.locator('.err-line').textContent()).includes('1.6'), '');
}
// 5. the same inputs twice give the same number (scoring is deterministic)
await openScore(); await fillScore();
await p.getByRole('button', { name: /Score it/ }).click();
await p.waitForSelector('.big-number, .error-card', { timeout: 180000 });
const n2 = (await p.locator('.big-number .n').textContent().catch(() => 'ERR')).replace('−', '-').trim();
check('scoring the same pose twice is reproducible', n2 === n, `${n} vs ${n2}`);
// 6. a different protein (human LDH) with the MDM2 pose: should still return something or a clear error, never hang
await openScore(); await fillScore({ protein: HLDH });
await p.getByRole('button', { name: /Score it/ }).click();
await p.waitForSelector('.big-number, .error-card', { timeout: 180000 });
const r = (await p.locator('.big-number .n, .error-card').first().textContent()).trim().slice(0, 200);
check('mismatched protein/pose ends in a number or a clear message (no hang)', r.length > 0, r);
check('no console/page errors', p.diag.errors.length === 0 && p.diag.pageErrors.length === 0, [...p.diag.errors, ...p.diag.pageErrors].join(' | '));
check('no 5xx', p.diag.bad.length === 0, p.diag.bad.join(' | '));
await b.close(); finish('02_score');
