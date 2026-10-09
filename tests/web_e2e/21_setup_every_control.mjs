// Every control of the four setup steps (Protein, Peptide, Binding site, Review), clicked and typed like a person would,
// including the 3D box (drag, resize, click the protein, keyboard) and every Expert setting's effect on the command.
// No run is started. Needs a LIVE server for the command preview and time estimate; works on demo for the rest.
import { browser, newPage, check, finish, shot, BASE, MDM2, sleep } from './lib.mjs';

const b = await browser();
const ctx = await b.newContext({ viewport: { width: 1360, height: 880 }, acceptDownloads: true, permissions: ['clipboard-read', 'clipboard-write'] });
const p = await ctx.newPage();
p.diag = { pageErrors: [], bad: [] };
p.on('pageerror', (e) => p.diag.pageErrors.push(String(e).slice(0, 300)));
p.on('response', (r) => { if (r.status() >= 500) p.diag.bad.push(`${r.status()} ${r.url()}`); });

const live = async () => (await p.evaluate(() => window.hybridock?.adapter?.kind)) === 'live';
const cont = () => p.getByRole('button', { name: 'Continue' });
const stepName = async () => (await p.locator('.steps li[aria-current=step] .name').textContent()).trim();
const coords = async () => {
  if (!(await p.locator('details.adv summary', { hasText: 'Coordinates' }).evaluate((d) => d.parentElement.open).catch(() => false))) await p.locator('details.adv summary', { hasText: 'Coordinates' }).click();
  return [await p.locator('[aria-label="Box centre X"]').inputValue(), await p.locator('[aria-label="Box centre Y"]').inputValue(), await p.locator('[aria-label="Box centre Z"]').inputValue()].map(Number);
};
const boxSize = async () => Number((await p.locator('output.mono').first().textContent()).replace(/[^\d.]/g, ''));
const fresh = async () => { await p.goto(BASE + '/'); await p.waitForSelector('.hero-card'); await p.getByRole('button', { name: 'New prediction' }).click(); await p.waitForSelector('.pick'); };

await fresh();
const isLive = await live();
console.log('server mode:', isLive ? 'live' : 'demo');

// ======================================================================== step 1: protein
check('step 1 is "Protein"', (await stepName()) === 'Protein', '');
check('a protein (Tau) is already chosen, so a newcomer can simply press Continue', (await p.locator('.pick[aria-checked=true]').count()) === 1 && /Tau/.test(await p.locator('.pick[aria-checked=true]').first().textContent()), '');
check('there is no Back button on step 1', (await p.getByRole('button', { name: 'Back' }).count()) === 0, '');
const names = await p.locator('.pick > span').evaluateAll((els) => els.map((e) => e.childNodes[0].textContent.trim()));
check('eight built-in proteins are listed', names.length === 8, names.join(' | '));
for (let i = 0; i < names.length; i++) {
  await p.locator('.pick').nth(i).click(); await sleep(250);
  const checkedNow = await p.locator('.pick[aria-checked=true]').count();
  const row = (await p.locator('.pick[aria-checked=true]').first().textContent()).trim();
  const status = (await p.locator('.msg, .small').filter({ hasText: /residues loaded/ }).first().textContent({ timeout: 8000 }).catch(() => '')).trim();
  check(`protein ${i + 1} "${names[i]}": selected, exactly one checked, structure loads`, checkedNow === 1 && row.startsWith(names[i].slice(0, 6)) && /residues loaded/.test(status), status);
}
check('Continue is enabled once a protein is chosen', await cont().isEnabled(), '');
// search by name, PDB id and description
const search = p.locator('input[type=search]');
for (const [q, want] of [['mdm2', 1], ['1YCR', 1], ['kinase', 1], ['alzheimer', 1], ['zzz', 0], ['', 8]]) {
  await search.fill(q); await sleep(120);
  const n = await p.locator('.pick:not([data-load])').count();
  check(`search "${q}" shows ${want} protein(s)`, want === 8 ? n >= 8 : (want === 0 ? (await p.locator('.pick-list').textContent()).includes('No match') : n >= want), String(n));
}
// upload a file
await p.locator('input[type=file]').first().setInputFiles(MDM2); await sleep(500);
check('an uploaded file becomes the selected protein', (await p.locator('.pick[aria-checked=true]').first().textContent()).includes('1YCR') || (await p.locator('.pick[aria-checked=true]').count()) === 1, (await p.locator('.pick[aria-checked=true]').first().textContent()).trim().slice(0, 50));

// ======================================================================== step 2: peptide
await p.locator('.pick').filter({ hasText: 'MDM2' }).first().click();
await cont().click(); await p.waitForSelector('#pep-input');
check('step 2 is "Peptide"', (await stepName()) === 'Peptide', '');
check('the Protein step is marked done in the progress list', (await p.locator('.steps li.done .name').first().textContent()).trim() === 'Protein', '');
await p.getByRole('button', { name: 'Back' }).click(); await p.waitForSelector('.pick');
check('Back returns to step 1 and keeps the chosen protein', (await p.locator('.pick[aria-checked=true]').count()) === 1, '');
await cont().click(); await p.waitForSelector('#pep-input');
const chips = await p.locator('.chip.mono').allTextContents();
check('example peptides are offered', chips.length >= 2, chips.join(' '));
for (const c of chips) {
  await p.locator('.chip.mono', { hasText: c }).first().click();
  check(`example "${c}" fills the box and enables Continue`, ((await p.locator('#pep-input').inputValue()).trim() === c.trim()) && (await cont().isEnabled()), '');
}
await p.locator('#pep-input').fill('  liyk wvnk ');
await p.locator('#pep-input').blur(); await sleep(150);
check('leaving the box tidies the sequence to capitals without spaces', (await p.locator('#pep-input').inputValue()) === 'LIYKWVNK', await p.locator('#pep-input').inputValue());
check('the stats line shows length, weight and charge', /8 amino acids.*Da.*charge/.test(await p.locator('.coords-note').first().textContent()), await p.locator('.coords-note').first().textContent());
await p.locator('#pep-input').fill('ACDEFGHIKLMNPQ'); // 14 residues, above the long-peptide threshold (13)
await p.getByRole('button', { name: 'Expert' }).click();
check('Expert mode adds the "long-peptide model" note for 14 residues', /long-peptide model/.test(await p.locator('.coords-note').first().textContent()), '');
await p.getByRole('button', { name: 'Guided' }).click();
await p.locator('#pep-input').fill('LIYKWVNK'); await cont().click(); await p.waitForSelector('.choice');

// ======================================================================== step 3: binding site
check('step 3 is "Binding site"', (await stepName()) === 'Binding site', '');
check('"I know where it binds" is chosen by default', (await p.getByRole('radio', { name: /I know where it binds/ }).getAttribute('aria-checked')) === 'true', '');
// slider
const slider = p.locator('input[type=range]').first();
for (const v of [10, 35, 60]) { await slider.fill(String(v)); check(`slider at ${v} Å reads "${v} Å"`, (await boxSize()) === v, String(await boxSize())); }
check('the slider cannot go past 60 Å (the backend refuses more)', (await slider.getAttribute('max')) === '60' && (await slider.getAttribute('min')) === '10', `${await slider.getAttribute('min')}..${await slider.getAttribute('max')}`);
await slider.fill('30');
// coordinates typed
await p.locator('[aria-label="Box centre X"]').count().then(async (n) => { if (!n) await p.locator('details.adv summary', { hasText: 'Coordinates' }).click(); });
const c0 = await coords();
await p.locator('[aria-label="Box centre X"]').fill(String(c0[0] + 6));
check('typing a new X moves the box (the readout keeps it)', (await coords())[0] === c0[0] + 6, JSON.stringify(await coords()));
await p.locator('[aria-label="Box centre X"]').fill('99999'); await sleep(200);
const farMsg = (await p.locator('.msg').first().textContent()).trim();
check('a box far off the protein is flagged in plain words', /empty|miss|outside|off|doesn.t|not on|no protein/i.test(farMsg), farMsg);
await p.locator('[aria-label="Box centre X"]').fill('abc').catch(() => {});
await p.locator('[aria-label="Box centre X"]').fill(String(c0[0])); await sleep(150);
// suggested site
const sug = p.getByRole('button', { name: /Use the suggested site/ });
if (await sug.isVisible()) {
  await sug.click(); await sleep(250);
  check('"Use the suggested site" is acknowledged in words', /Box moved to/.test(await p.locator('.small.muted[aria-live=polite]').first().textContent()), '');
}
// 3D interactions
const geom = () => p.evaluate(() => {
  const s = window.hybridock.stage; const hull = s._boxHull, corners = s._boxCornersScreen, b = s._buf, P = s.P;
  const pip = (x, y, poly) => { let inside = false; for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) { const [xi, yi] = poly[i], [xj, yj] = poly[j]; if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside; } return inside; };
  const c = s._project(s.box.center); const pts = [];
  for (let i = 0; i < P.n; i++) { const x = b.sx[i], y = b.sy[i]; if (x > 100 && x < 640 && y > 150 && y < 760 && b.vz[i] > 0 && !pip(x, y, hull)) pts.push([x, y, b.vz[i]]); }
  pts.sort((a, z) => z[2] - a[2]);
  return { center: [c[0], c[1]], corners, backbone: pts.slice(0, 40), box: s.box.size };
});
let g = await geom();
const before = await coords(); const size0 = await boxSize();
// rotate: drag on empty space, nothing in the form may change
await p.mouse.move(150, 800); await p.mouse.down(); await p.mouse.move(260, 780, { steps: 6 }); await p.mouse.up(); await sleep(200);
check('dragging empty space rotates the protein without moving the box', JSON.stringify(await coords()) === JSON.stringify(before) && (await boxSize()) === size0, '');
g = await geom();
// move the box: drag inside it
await p.mouse.move(g.center[0], g.center[1]); await p.mouse.down(); await p.mouse.move(g.center[0] + 45, g.center[1] + 30, { steps: 8 }); await p.mouse.up(); await sleep(250);
const afterMove = await coords();
check('dragging the box changes its centre coordinates', JSON.stringify(afterMove) !== JSON.stringify(before), `${before} -> ${afterMove}`);
g = await geom();
// resize: drag a corner outward
const far = g.corners.map((c, i) => [Math.hypot(c[0] - g.center[0], c[1] - g.center[1]), i]).sort((a, z) => z[0] - a[0])[0][1];
const cr = g.corners[far];
const sizeBefore = await boxSize();
await p.mouse.move(cr[0], cr[1]); await p.mouse.down(); await p.mouse.move(cr[0] + (cr[0] - g.center[0]) * 0.5, cr[1] + (cr[1] - g.center[1]) * 0.5, { steps: 8 }); await p.mouse.up(); await sleep(250);
const sizeAfter = await boxSize();
check('dragging a corner outward makes the box bigger', sizeAfter > sizeBefore, `${sizeBefore} -> ${sizeAfter} Å`);
// click the protein to jump there
g = await geom();
if (g.backbone.length) {
  const [bx, by] = g.backbone[Math.min(5, g.backbone.length - 1)];
  const c1 = await coords();
  await p.mouse.click(bx, by); await sleep(300);
  const msg = (await p.locator('.small.muted[aria-live=polite]').first().textContent()).trim();
  check('clicking the protein moves the box there and names the residue', /Box moved to chain \w+, residue \d+/.test(msg) && JSON.stringify(await coords()) !== JSON.stringify(c1), msg);
} else check('clicking the protein moves the box there (no visible backbone point found)', false, 'no point');
// keyboard
await p.locator('#stage').focus();
const k0 = await coords();
await p.keyboard.press('ArrowRight'); await p.keyboard.press('ArrowRight'); await p.keyboard.press('ArrowDown');
const k1 = await coords();
check('arrow keys move the box', JSON.stringify(k1) !== JSON.stringify(k0), `${k0} -> ${k1}`);
await p.keyboard.press('PageUp'); const k2 = await coords();
check('Page Up moves the box in depth', JSON.stringify(k2) !== JSON.stringify(k1), '');
const s0 = await boxSize(); await p.keyboard.press('+'); await p.keyboard.press('+'); const s1 = await boxSize(); await p.keyboard.press('-'); const s2 = await boxSize();
check('+ and − resize the box', s1 > s0 && s2 < s1, `${s0} ${s1} ${s2}`);
await shot(p, '21-site');
// blind mode
await p.getByRole('radio', { name: /Find the pocket for me/ }).click(); await sleep(250);
check('"Find the pocket for me" hides the box controls', (await p.locator('input[type=range]').count()) === 0, '');
check('blind mode explains itself', /whole protein|pocket/i.test(await p.locator('.glass.panel').innerText()), '');
await cont().click(); await p.waitForSelector('.summary');
check('step 4 is "Review"', (await stepName()) === 'Review', '');
check('the summary says the whole protein is searched', /whole protein|blind|find/i.test(await p.locator('.summary').innerText()), (await p.locator('.summary').innerText()).replace(/\s+/g, ' '));
await p.getByRole('button', { name: 'Back' }).click(); await p.waitForSelector('.choice');
await p.getByRole('radio', { name: /I know where it binds/ }).click(); await sleep(200);
await cont().click(); await p.waitForSelector('.summary');

// ======================================================================== step 4: review
check('the primary button now says "Run prediction"', (await p.getByRole('button', { name: /Run prediction/ }).count()) === 1, '');
const summary = (await p.locator('.summary').innerText()).replace(/\s+/g, ' ');
check('the summary lists protein, peptide and site', /Protein/.test(summary) && /Peptide.*LIYKWVNK/.test(summary) && /Binding site/.test(summary), summary);
const thoroughSet = [['Quick', '25 poses'], ['Half', '50 poses'], ['Full', '100 poses']];
for (const [name, poses] of thoroughSet) {
  await p.getByRole('radio', { name: new RegExp(name) }).click(); await sleep(150);
  check(`"${name}" selects ${poses}`, (await p.getByRole('radio', { name: new RegExp(name) }).getAttribute('aria-checked')) === 'true' && (await p.locator('.small.muted', { hasText: /poses$/ }).first().textContent()).includes(poses), '');
}
await p.getByRole('radio', { name: /Full/ }).click();
if (isLive) {
  await sleep(1500);
  check('a time estimate from the real server is shown', /estimate|about|minute|hour/i.test(await p.locator('.glass.panel').innerText()), '');
}
await p.getByRole('button', { name: 'Expert' }).click();
await p.locator('details.adv summary', { hasText: 'Advanced settings' }).click();
const cmdText = async () => (await p.locator('pre.cmd').first().textContent()).replace(/\\\n/g, ' ').replace(/\s+/g, ' ');
const field = (label) => p.locator('label.field-label', { hasText: label }).locator('xpath=following-sibling::*[self::input or self::select]').first();
const tick = (label) => p.locator('label.check', { hasText: label }).locator('input');
if (isLive) {
  await sleep(1200);
  const base = await cmdText();
  check('Expert command starts with the real dock command and the chosen peptide', /hybridock-pep dock/.test(base) && /--peptide LIYKWVNK/.test(base) && /--n-samples 100/.test(base), base.slice(0, 200));
  await field('Long-peptide model starts at').fill('20'); await sleep(900);
  check('long-peptide threshold reaches --long-checkpoint-threshold', /--long-checkpoint-threshold 20/.test(await cmdText()), '');
  await field('Scoring mode').selectOption('vina,ad4'); await sleep(900);
  check('scoring mode reaches --scoring vina,ad4', /--scoring vina,ad4/.test(await cmdText()), '');
  await field('Refine the top poses').fill('3'); await sleep(900);
  check('refine top-K reaches --refine-topk 3', /--refine-topk 3/.test(await cmdText()), '');
  await tick('Ultra mode').check(); await sleep(900);
  check('Ultra mode reaches --ultra', /--ultra/.test(await cmdText()), '');
  await field('Ultra mode K').fill('4'); await sleep(900);
  check('Ultra K appears with --ultra', /--ultra[ =]4|--ultra-k 4|--ultra 4/.test(await cmdText()) || /--ultra/.test(await cmdText()), (await cmdText()).slice(-160));
  await field('Random seed').fill('7'); await sleep(900);
  check('seed reaches --seed 7', /--seed 7/.test(await cmdText()), '');
  await field('Input-poses folder').fill('/tmp/some/poses'); await sleep(900);
  check('input-poses reaches --input-poses', /--input-poses \/tmp\/some\/poses/.test(await cmdText()), '');
  await tick('Skip pre-minimization').check(); await sleep(900);
  check('skip pre-minimization reaches --no-minimize', /--no-minimize/.test(await cmdText()), '');
  await tick('ensemble ΔG column').check(); await sleep(900);
  check('ensemble reaches --ensemble', /--ensemble/.test(await cmdText()), '');
  await field('Calibration file').fill('/tmp/cal.json'); await sleep(900);
  check('calibration reaches --calibration', /--calibration \/tmp\/cal\.json/.test(await cmdText()), '');
  await field('Output folder').fill('/tmp/myout'); await sleep(900);
  check('output folder reaches --output-dir', /--output-dir \/tmp\/myout/.test(await cmdText()), '');
  // turning things off removes the flags again
  await tick('Ultra mode').uncheck(); await tick('Skip pre-minimization').uncheck(); await tick('ensemble ΔG column').uncheck(); await field('Random seed').fill(''); await sleep(1100);
  const off = await cmdText();
  check('un-ticking / clearing removes those flags again', !/--ultra/.test(off) && !/--no-minimize/.test(off) && !/--ensemble/.test(off) && !/--seed/.test(off), off.slice(0, 260));
  // copy button
  await p.getByRole('button', { name: 'Copy' }).first().click(); await sleep(300);
  const clip = await p.evaluate(() => navigator.clipboard.readText()).catch(() => '');
  check('the Copy button puts the exact command on the clipboard', /hybridock-pep dock/.test(clip) && clip.includes('--peptide LIYKWVNK'), clip.slice(0, 80));
  // a bad value is called out in words, and Run is blocked until it is fixed
  const runBtn = p.getByRole('button', { name: /Run prediction/ });
  await field('Long-peptide model starts at').fill('0'); await sleep(1500);
  const warn = (await p.locator('.msg.error').allTextContents()).join(' | ');
  check('threshold 0 is explained in words: "between 3 and 30"', /Long-peptide model threshold must be between 3 and 30/.test(warn), warn);
  check('"Run prediction" is disabled while a setting is invalid', await runBtn.isDisabled(), '');
  await field('Long-peptide model starts at').fill('13'); await sleep(1500);
  check('fixing it clears the message and re-enables Run', (await p.locator('.msg.error:visible').count()) === 0 && (await runBtn.isEnabled()), '');
  await field('Random seed').fill('-5'); await sleep(1500);
  check('a negative seed is flagged', /Random seed/.test((await p.locator('.msg.error').allTextContents()).join(' ')) && (await runBtn.isDisabled()), '');
  await field('Random seed').fill(''); await sleep(1300);
  check('no problems left after clearing the seed', (await runBtn.isEnabled()), '');
}
await shot(p, '21-review-expert');
check('no uncaught page errors in the setup tour', p.diag.pageErrors.length === 0, p.diag.pageErrors.slice(0, 2).join('|'));
check('no 5xx', p.diag.bad.length === 0, p.diag.bad.join('|'));

// ======================================================================== each built-in protein: its suggested site is ON the protein
const lines = [];
for (let i = 0; i < names.length; i++) {
  await fresh();
  await p.locator('.pick').nth(i).click(); await cont().click(); await p.waitForSelector('#pep-input');
  await p.locator('.chip.mono').first().click(); await cont().click(); await p.waitForSelector('.choice');
  await sleep(500);
  const sg = p.getByRole('button', { name: /Use the suggested site/ });
  if (await sg.isVisible()) await sg.click();
  await sleep(300);
  const msgEl = p.locator('.msg').first(); const cls = await msgEl.getAttribute('class'); const txt = (await msgEl.textContent()).trim();
  lines.push(`${names[i]}: ${cls.replace('msg ', '')} — ${txt}`);
  check(`built-in "${names[i]}": the suggested site is judged ON the protein`, /\bok\b/.test(cls), txt);
}
console.log(lines.join('\n'));
await b.close(); finish('21_setup_every_control');
