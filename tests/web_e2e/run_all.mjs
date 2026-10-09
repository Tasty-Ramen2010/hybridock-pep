// Run every suite in order and print one summary. Exit code is non-zero if any suite failed.
//   BASE=http://127.0.0.1:8000 node run_all.mjs [01 05 ...]     (optionally name the suites to run by their number)
import { spawnSync } from 'node:child_process';
import { readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const want = process.argv.slice(2);
const suites = readdirSync(here).filter((f) => /^\d\d.*\.mjs$/.test(f)).sort().filter((f) => !want.length || want.some((w) => f.startsWith(w)));
let failed = 0;
for (const s of suites) {
  console.log(`\n######## ${s}`);
  const r = spawnSync(process.execPath, [join(here, s)], { stdio: 'inherit', env: process.env });
  if (r.status !== 0) { failed++; console.log(`!! ${s} exited with ${r.status}`); }
}
console.log(`\n${suites.length - failed}/${suites.length} suites passed`);
process.exit(failed ? 1 : 0);
