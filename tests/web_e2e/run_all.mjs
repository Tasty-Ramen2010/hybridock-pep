// Run every suite in order and print one summary. Exit code is non-zero if any suite failed.
//   BASE=http://127.0.0.1:8000 node run_all.mjs [01 05 ...]     (optionally name the suites to run by their number)
import { spawnSync } from 'node:child_process';
import { readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const want = process.argv.slice(2);
const suites = readdirSync(here).filter((f) => /^\d\d.*\.mjs$/.test(f)).sort().filter((f) => !want.length || want.some((w) => f.startsWith(w)));
// The app runs ONE job at a time, and a suite that starts a real run (a random click, "Run example") can leave it going: wait for
// the slot to be free before the next suite starts (a demo server has no /api/jobs and returns at once).
const BASE = (process.env.BASE || 'http://127.0.0.1:8000').replace(/\/$/, '');
async function waitIdle() {
  for (let t0 = Date.now(); Date.now() - t0 < 25 * 60000;) {
    try { const jobs = (await (await fetch(`${BASE}/api/jobs`)).json()).jobs; if (!jobs?.some((j) => j.state === 'running')) return; } catch { return; }
    await new Promise((r) => setTimeout(r, 5000));
  }
}
let failed = 0;
for (const s of suites) {
  await waitIdle();
  console.log(`\n######## ${s}`);
  const r = spawnSync(process.execPath, [join(here, s)], { stdio: 'inherit', env: process.env });
  if (r.status !== 0) { failed++; console.log(`!! ${s} exited with ${r.status}`); }
}
console.log(`\n${suites.length - failed}/${suites.length} suites passed`);
process.exit(failed ? 1 : 0);
