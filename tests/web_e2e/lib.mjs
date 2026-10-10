// Shared helpers for the browser end-to-end suites. Everything is driven through a real browser (Playwright/Chromium)
// against the web UI that `hybridock-pep serve` (or any copy of it) is serving; nothing here calls the API directly.
//
//   BASE=http://127.0.0.1:8000  node 01_smoke.mjs        # which server to test (default: a local `hybridock-pep serve`)
//   E2E_OUT=/some/folder                                  # where screenshots, results JSON and scratch files go
import { chromium } from 'playwright';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const BASE = (process.env.BASE || 'http://127.0.0.1:8000').replace(/\/$/, '');
export const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
export const OUT = process.env.E2E_OUT || path.join(os.tmpdir(), 'hybridock-e2e');
fs.mkdirSync(path.join(OUT, 'shots'), { recursive: true });
// The example structures that ship with the repository.
export const MDM2 = path.join(REPO, 'data/pdbs/1YCR_mdm2.pdb');
export const POSE = path.join(REPO, 'data/pdbs/1YCR_peptide.pdb');
export const HLDH = path.join(REPO, 'data/pdbs/1I0Z.pdb');
export const results = [];

export function check(name, ok, detail = '') {
  results.push({ name, ok: !!ok, detail: String(detail) });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
}
export async function browser() { return chromium.launch(); }

/** Open the avatar menu (appearance, accent colour, your name) if it is closed. */
export async function openMenu(p) {
  const a = p.locator('.avatar');
  if ((await a.getAttribute('aria-expanded')) !== 'true') await a.click();
}
/** Switch light <-> dark through the avatar menu, the way a person does. */
export async function flipTheme(p) {
  await openMenu(p);
  const dark = (await p.evaluate(() => document.documentElement.dataset.theme)) === 'dark';
  await p.getByRole('button', { name: dark ? 'Light appearance' : 'Dark appearance' }).click();
}

/** A page that records console errors, page errors, 5xx responses and 404s. */
export async function newPage(b, opts = {}) {
  const ctx = await b.newContext({ viewport: { width: 1360, height: 880 }, acceptDownloads: true, ...opts });
  const page = await ctx.newPage();
  page.diag = { errors: [], bad: [], pageErrors: [], notFound: [] };
  page.on('console', (m) => { if (m.type() === 'error') page.diag.errors.push(m.text().slice(0, 300)); });
  page.on('pageerror', (e) => page.diag.pageErrors.push(String(e).slice(0, 300)));
  page.on('response', (r) => { if (r.status() >= 500) page.diag.bad.push(`${r.status()} ${r.url()}`); else if (r.status() === 404) page.diag.notFound.push(r.url()); });
  return page;
}
export const shot = (page, name) => page.screenshot({ path: path.join(OUT, 'shots', `${name}.png`) }).catch(() => {});
export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Write a scratch file under OUT and return its path. */
export function scratch(name, text) { const f = path.join(OUT, name); fs.writeFileSync(f, text); return f; }

/** A large synthetic protein, made by tiling copies of the MDM2 structure through space. */
export function bigPdb(name, copies) {
  const f = path.join(OUT, name);
  if (fs.existsSync(f)) return f;
  const atoms = fs.readFileSync(MDM2, 'utf8').split('\n').filter((l) => l.startsWith('ATOM'));
  const out = []; let serial = 0;
  for (let k = 0; k < copies; k++) {
    for (const l of atoms) {
      serial++;
      const x = (parseFloat(l.slice(30, 38)) + (k % 10) * 40).toFixed(3).padStart(8);
      const y = (parseFloat(l.slice(38, 46)) + (Math.floor(k / 10) % 10) * 40).toFixed(3).padStart(8);
      const z = (parseFloat(l.slice(46, 54)) + Math.floor(k / 100) * 40).toFixed(3).padStart(8);
      out.push(l.slice(0, 6) + String(serial % 100000).padStart(5) + l.slice(11, 30) + x + y + z + l.slice(54));
    }
  }
  fs.writeFileSync(f, out.join('\n') + '\nEND\n');
  return f;
}

export function finish(name) {
  const failed = results.filter((r) => !r.ok);
  fs.writeFileSync(path.join(OUT, `${name}.json`), JSON.stringify(results, null, 2));
  console.log(`\n== ${name}: ${results.length - failed.length}/${results.length} passed${failed.length ? ', FAILED: ' + failed.map((f) => f.name).join('; ') : ''}`);
  if (failed.length) process.exitCode = 1;
}
