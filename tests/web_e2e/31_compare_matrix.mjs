// REAL "Compare two proteins" runs (each is two dockings) driven only through the website, each followed by a check of every
// part of the comparison result: ΔΔG, its 95% interval, the three-way verdict, both ΔG cards, the 3D tabs, the command.
//   CASES=cmp-ldh BASE=http://127.0.0.1:8000 node 31_compare_matrix.mjs
import { browser, check, finish, results, OUT, BASE, sleep, shot } from './lib.mjs';
import { runAndWait, fmtMin } from './journeys.mjs';
import fs from 'node:fs';
import path from 'node:path';

const CASES = {
  'cmp-ldh': { target: 'pfldh', off: 'hldh', peptide: 'LISDAELEAIFEADC', thorough: 'Quick' }, // the project's own selectivity question
  'cmp-mdm2-bcl2': { target: 'mdm2', off: 'bcl2', peptide: 'ETFSDLWKLLPE', thorough: 'Quick' },
  'cmp-default': { target: null, off: null, peptide: null, thorough: 'Quick' }, // what a newcomer gets by just pressing Compare
  'cmp-half': { target: 'mdm2', off: 'bcl2', peptide: 'ETFSDLWKLLPE', thorough: 'Half' },
};
if (process.argv.includes('--list')) { console.log(Object.keys(CASES).join('\n')); process.exit(0); }
const want = (process.env.CASES ? process.env.CASES.split(',') : Object.keys(CASES)).filter((id) => CASES[id]);
const tag = process.env.TAG || want.join('+');
const ledger = {}; const save = () => fs.writeFileSync(path.join(OUT, `compare-${tag}.json`), JSON.stringify({ base: BASE, ledger, results }, null, 2));

async function verifyCompare(p, label, expect) {
  const ddgTxt = (await p.locator('.big-number .n').textContent()).replace('−', '-').trim();
  const ddg = parseFloat(ddgTxt);
  check(`${label}: ΔΔG is a real number`, Number.isFinite(ddg) && Math.abs(ddg) < 30, `${ddgTxt} kcal/mol`);
  const ci = (await p.locator('.err-line b').textContent()).replace(/−/g, '-').match(/-?\d+(\.\d+)?/g)?.map(Number) || [];
  check(`${label}: the 95% interval is shown and contains the estimate`, ci.length === 2 && ci[0] <= ddg + 0.011 && ddg - 0.011 <= ci[1], `[${ci.join(', ')}]`);
  check(`${label}: the ΔΔG bar draws a band, an estimate mark and the zero line`, (await p.locator('.ddg-bar .ci').count()) === 1 && (await p.locator('.ddg-bar .est').count()) === 1 && (await p.locator('.ddg-bar .zero').count()) === 1, '');
  const verdict = (await p.locator('.verdict h3').textContent()).trim();
  check(`${label}: one of the three plain verdicts`, /selective|no clear|prefers/i.test(verdict), verdict);
  const loSide = ci[1] < 0 ? 'target' : ci[0] > 0 ? 'off' : 'none';
  check(`${label}: the verdict agrees with where the interval sits relative to zero`, (loSide === 'target' && /target/i.test(verdict) && !/off/i.test(verdict.replace(/off-target/i, 'X'))) || (loSide === 'off' && /off-target/i.test(verdict)) || (loSide === 'none' && /no clear/i.test(verdict)), `${verdict} for [${ci}]`);
  const cards = await p.locator('.two-dg .card .v').allTextContents();
  const [dt, doff] = cards.map((t) => parseFloat(t.replace('−', '-')));
  check(`${label}: both ΔG cards are shown`, cards.length === 2 && Number.isFinite(dt) && Number.isFinite(doff), cards.join(' / '));
  check(`${label}: ΔΔG = ΔG(target) − ΔG(off-target) as displayed`, Math.abs((dt - doff) - ddg) < 0.06, `${dt} - ${doff} = ${(dt - doff).toFixed(2)} vs ${ddg}`);
  check(`${label}: the page explains why comparing is more trustworthy`, /more trustworthy than a single number/.test(await p.locator('.glass.panel').innerText()), '');
  check(`${label}: it says which score both ΔG values come from`, /using the .* (score|ΔG)|mean of the top/.test(await p.locator('.glass.panel').innerText()), '');
  if (Math.abs(ddg) < 1) check(`${label}: |ΔΔG| below 1 carries the "direction, not a measurement" note`, /direction, not a measurement/.test(await p.locator('.glass.panel').innerText()), '');
  check(`${label}: not labelled Demo`, (await p.locator('.badge-demo').count()) === 0, '');
  // 3D tabs
  const tabs = p.locator('[aria-label="Which protein to show in 3D"] button');
  check(`${label}: Target / Off-target tabs exist`, (await tabs.count()) === 2, '');
  await tabs.nth(1).click(); await sleep(700);
  check(`${label}: the Off-target tab switches the 3D view (hint and pressed state)`, /Off-target:/.test(await p.locator('.hint').first().textContent()) && (await tabs.nth(1).getAttribute('aria-pressed')) === 'true', await p.locator('.hint').first().textContent());
  await tabs.nth(0).click(); await sleep(500);
  check(`${label}: the Target tab switches back`, /Target:/.test(await p.locator('.hint').first().textContent()), '');
  // Expert command
  await p.getByRole('button', { name: 'Expert' }).click();
  const cmd = (await p.locator('pre.cmd').last().textContent()).replace(/\s+/g, ' ');
  check(`${label}: Expert shows the exact selectivity command`, /selectivity|offtarget|off-target/i.test(cmd) && cmd.includes(expect.peptide), cmd.slice(0, 200));
  await p.getByRole('button', { name: 'Guided' }).click();
  await shot(p, `compare-${label}`);
  return { ddg, ci, verdict, dt, doff };
}

const b = await browser();
for (const id of want) {
  const c = CASES[id]; console.log(`\n=== case ${id}`);
  const ctx = await b.newContext({ viewport: { width: 1360, height: 880 }, acceptDownloads: true, permissions: ['clipboard-read', 'clipboard-write'] });
  const p = await ctx.newPage(); p.diag = { pageErrors: [] }; p.on('pageerror', (e) => p.diag.pageErrors.push(String(e).slice(0, 300)));
  try {
    await p.goto(BASE + '/'); await p.waitForSelector('.hero-card');
    await p.locator('.start-card', { hasText: 'Compare two proteins' }).click(); await p.waitForSelector('.pcard');
    await sleep(1200);
    if (c.target) await p.locator('select[aria-label="Target protein"]').selectOption(c.target);
    if (c.off) await p.locator('select[aria-label="Off-target protein"]').selectOption(c.off);
    if (c.peptide) await p.locator('#cmp-pep').fill(c.peptide);
    await p.getByRole('radio', { name: new RegExp(c.thorough) }).click();
    await sleep(2500);
    const peptide = await p.locator('#cmp-pep').inputValue();
    const cmpBtn = p.getByRole('button', { name: /^Compare/ }).last();
    check(`${id}: Compare is enabled with the chosen pair`, await cmpBtn.isEnabled(), '');
    await p.getByRole('button', { name: 'Expert' }).click(); await sleep(200);
    check(`${id}: Expert shows both sites on the cards`, (await p.locator('.pcard').allInnerTexts()).every((t) => /at \(-?\d/.test(t)), '');
    await p.getByRole('button', { name: 'Guided' }).click();
    const run = await runAndWait(p, { label: id });
    ledger[id] = { minutes: +(run.ms / 60000).toFixed(1), failed: run.failed, stages: run.stages, monotonic: run.monotonic };
    check(`${id}: progress never moved backwards`, run.monotonic, `max ${run.maxFrac}%`);
    if (run.failed) { ledger[id].error = run.error; check(`${id}: the comparison finished with a result`, false, run.error.slice(0, 500)); }
    else {
      check(`${id}: the comparison finished in ${fmtMin(run.ms)}`, true, '');
      const v = await verifyCompare(p, id, { peptide }); Object.assign(ledger[id], v);
      await p.getByRole('button', { name: 'Compare again' }).click(); await p.waitForSelector('.pcard');
      check(`${id}: "Compare again" returns to the form`, /compare/.test(p.url()), p.url());
      await p.goto(BASE + '/#/'); await p.waitForSelector('.recent-card');
      check(`${id}: the comparison is in Recent predictions (ΔΔG, no ranked list to download)`, /ΔΔG/.test(await p.locator('.recent-card').first().textContent()) && (await p.locator('.recent-card').first().locator('.recent-dl').count()) === 0, '');
    }
  } catch (e) { ledger[id] = { ...(ledger[id] || {}), crashed: String(e).slice(0, 500) }; check(`${id}: the journey completed without the test crashing`, false, String(e).slice(0, 400)); }
  check(`${id}: no uncaught page errors`, p.diag.pageErrors.length === 0, p.diag.pageErrors.slice(0, 2).join('|'));
  save(); await ctx.close();
}
await b.close(); finish(`31_compare_matrix_${tag}`); save();
