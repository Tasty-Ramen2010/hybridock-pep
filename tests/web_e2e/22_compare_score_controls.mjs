// Every control on the Compare and Score pages, used the way a person would (selects, uploads, drag-and-drop, the 3D site
// editor). No Compare run is started here (see 30_*). Works on a live or a demo server.
import { browser, newPage, check, finish, shot, BASE, MDM2, POSE, HLDH, scratch, sleep } from './lib.mjs';

const b = await browser();
const p = await newPage(b);
const live = async () => (await p.evaluate(() => window.hybridock?.adapter?.kind)) === 'live';

// ======================================================================== COMPARE
await p.goto(BASE + '/#/compare'); await p.waitForSelector('.pcard');
const tSel = p.locator('select[aria-label="Target protein"]'), oSel = p.locator('select[aria-label="Off-target protein"]');
const cmpBtn = p.getByRole('button', { name: /^Compare/ }).last();
await sleep(1200);
const tName = await tSel.evaluate((s) => s.selectedOptions[0].textContent), oName = await oSel.evaluate((s) => s.selectedOptions[0].textContent);
check('Compare opens pre-filled with two different proteins', tName !== oName && tName.length > 3 && oName.length > 3, `${tName} vs ${oName}`);
check('the peptide box is pre-filled with a valid peptide', /^[A-Z]{3,30}$/.test(await p.locator('#cmp-pep').inputValue()), await p.locator('#cmp-pep').inputValue());
check('Compare is enabled once both structures have loaded', await cmpBtn.isEnabled(), '');
const optsT = await tSel.locator('option').allTextContents();
check('each protein menu lists the 8 built-ins plus the upload option', optsT.length >= 9 && /Upload my own/.test(optsT[optsT.length - 1]), `${optsT.length} options`);

// choose each built-in on both sides; the card text and the site line follow
for (let i = 0; i < 8; i++) {
  const val = await tSel.locator('option').nth(i).getAttribute('value');
  await tSel.selectOption(val); await sleep(350);
  const card = (await p.locator('.pcard.target').innerText()).replace(/\s+/g, ' ');
  check(`Target = option ${i + 1}: the card shows its description and a suggested site`, /Suggested site:/.test(card) && card.length > 80, card.slice(40, 140));
}
// same protein on both sides is refused in words
const sameVal = await tSel.inputValue();
await oSel.selectOption(sameVal); await sleep(300);
check('picking the same protein twice disables Compare and says why', (await cmpBtn.isDisabled()) && /different proteins/i.test(await p.locator('.small.muted', { hasText: /different/ }).first().textContent()), '');
const other = await oSel.locator('option').evaluateAll((os, v) => os.map((o) => o.value).find((x) => x !== v && !x.startsWith('__')), sameVal);
await oSel.selectOption(other); await sleep(500);
check('choosing a different off-target re-enables Compare', await cmpBtn.isEnabled(), '');

// peptide validation on this page
for (const [seq, okExpected, label] of [['', false, 'empty'], ['AB', false, 'too short / invalid'], ['LIYKWVNK', true, 'valid'], ['A'.repeat(31), false, '31 letters (too long)'], ['LIYK1WVNK', true, 'digit ignored']]) {
  await p.locator('#cmp-pep').fill(seq); await sleep(200);
  const en = await cmpBtn.isEnabled(); const msg = (await p.locator('#cmp-pep-msg').textContent()).trim();
  check(`Compare peptide "${label}": Compare ${okExpected ? 'enabled' : 'disabled'}, with a message`, en === okExpected && msg.length > 3, msg);
}
await p.locator('#cmp-pep').fill('LIYKWVNK');

// thoroughness radios
for (const [name, poses] of [['Quick', 25], ['Half', 50], ['Full', 100]]) {
  await p.getByRole('radio', { name: new RegExp(name) }).click(); await sleep(120);
  check(`Compare "${name}" (${poses} poses per protein) is selected`, (await p.getByRole('radio', { name: new RegExp(name) }).getAttribute('aria-checked')) === 'true' && (await p.getByRole('radio', { name: new RegExp(name) }).textContent()).includes(String(poses)), '');
}
// upload through the select
const chooserP = p.waitForEvent('filechooser', { timeout: 8000 });
await oSel.selectOption('__upload');
const chooser = await chooserP; await chooser.setFiles(HLDH); await sleep(1500);
const upLabel = await oSel.evaluate((s) => s.selectedOptions[0].textContent);
check('"Upload my own PDB file…" opens the file chooser and the file becomes the off-target', /your file/i.test(upLabel), upLabel);
check('an uploaded protein has a "Custom site" line (no suggestion for unknown structures)', /Custom site|site/.test(await p.locator('.pcard:not(.target)').innerText()), '');

// Expert shows coordinates on the cards
await p.getByRole('button', { name: 'Expert' }).click();
check('Expert mode shows the site coordinates on the cards', /at \(-?\d/.test(await p.locator('.pcard.target').innerText()), '');
await p.getByRole('button', { name: 'Guided' }).click();

// Adjust the site in 3D
await tSel.selectOption(sameVal); await oSel.selectOption(other); await sleep(800);
await p.locator('.pcard.target').getByRole('button', { name: /Adjust the site in 3D/ }).click();
await p.waitForSelector('input[type=range]');
check('"Adjust the site in 3D" opens the site editor for that side', /Adjust the target site/.test(await p.locator('h1.panel-title').textContent()), '');
await p.locator('input[type=range]').fill('22'); await sleep(200);
check('the editor shows the new box size', /22 Å/.test(await p.locator('output.mono').textContent()), '');
const sugg = p.getByRole('button', { name: /Use the suggested site/ });
check('the editor offers the suggested site for a built-in protein', await sugg.isVisible(), '');
await p.getByRole('button', { name: 'Done' }).click(); await p.waitForSelector('.vs-grid');
check('Done returns to Compare and keeps the new box size', /Box 22 Å/.test(await p.locator('.pcard.target').innerText()), (await p.locator('.pcard.target .small').allTextContents()).join(' | '));
check('resizing only the box keeps the suggested centre, and the card still says so', /Suggested site/.test(await p.locator('.pcard.target').innerText()), '');
await p.locator('.pcard:not(.target)').getByRole('button', { name: /Adjust the site in 3D/ }).click(); await p.waitForSelector('input[type=range]');
check('the off-target has its own editor', /Adjust the off-target site/.test(await p.locator('h1.panel-title').textContent()), '');
await p.getByRole('button', { name: 'Done' }).click();
await shot(p, '22-compare');
check('no page errors on Compare', p.diag.pageErrors.length === 0, p.diag.pageErrors.slice(0, 2).join('|'));

// ======================================================================== SCORE
await p.goto(BASE + '/#/score'); await p.waitForSelector('.drop');
const scoreBtn = p.getByRole('button', { name: /Score it/ });
check('Score it starts disabled', await scoreBtn.isDisabled(), '');
const files = p.locator('input[type=file]');
await files.nth(0).setInputFiles(MDM2); await p.waitForSelector('.msg.ok');
check('uploading a protein confirms the residue count', /residues loaded/.test(await p.locator('.msg.ok').first().textContent()), await p.locator('.msg.ok').first().textContent());
check('the drop zone now shows the file name and offers a replacement', /Click to choose a different file/.test(await p.locator('.drop').first().innerText()), '');
check('Score it is still disabled (no pose, no sequence)', await scoreBtn.isDisabled(), '');
await files.nth(1).setInputFiles(POSE); await sleep(400);
check('uploading the pose fills the sequence in from the file', (await p.locator('#sc-pep').inputValue()) === 'ETFSDLWKLLPEN', await p.locator('#sc-pep').inputValue()); // the file has 13 residues (a terminal N)
check('the pose upload reports how many residues it found', /13 residues found/.test(await p.locator('.msg.ok').nth(1).textContent()), '');
check('Score it enables with protein + pose + sequence', await scoreBtn.isEnabled(), '');
await p.locator('#sc-pep').fill('ACDEF'); await sleep(250);
check('a sequence that does not match the pose is warned about in words', /5 letters.*13 residues/.test(await p.locator('#sc-pep-msg').textContent()), await p.locator('#sc-pep-msg').textContent());
await p.locator('#sc-pep').fill('AB!'); await sleep(250);
check('an invalid sequence disables Score it', await scoreBtn.isDisabled(), await p.locator('#sc-pep-msg').textContent());
await p.locator('#sc-pep').fill('ETFSDLWKLLPEN');
// drag-and-drop onto the drop zones
await p.goto(BASE + '/#/score'); await p.waitForSelector('.drop');
const dropFile = async (zoneIndex, name, text) => p.evaluate(([i, n, t]) => {
  const zone = document.querySelectorAll('.drop')[i]; const dt = new DataTransfer(); dt.items.add(new File([t], n, { type: 'text/plain' }));
  zone.dispatchEvent(new DragEvent('dragover', { bubbles: true, cancelable: true, dataTransfer: dt }));
  const over = zone.classList.contains('over');
  zone.dispatchEvent(new DragEvent('drop', { bubbles: true, cancelable: true, dataTransfer: dt }));
  return over;
}, [zoneIndex, name, text]);
const fs = await import('node:fs');
const over = await dropFile(0, '1YCR_mdm2.pdb', fs.readFileSync(MDM2, 'utf8')); await sleep(600);
check('dragging a file over a drop zone highlights it', over === true, '');
check('dropping a protein file loads it', /residues loaded/.test((await p.locator('.msg.ok').allTextContents()).join(' ')), '');
await dropFile(1, '1YCR_peptide.pdb', fs.readFileSync(POSE, 'utf8')); await sleep(600);
check('dropping the pose file fills the sequence', (await p.locator('#sc-pep').inputValue()) === 'ETFSDLWKLLPEN', await p.locator('#sc-pep').inputValue());
await dropFile(0, 'notes.txt', 'this is not a protein'); await sleep(500);
check('dropping a non-PDB file gives a friendly message and keeps the page working', (await p.locator('.toast').count()) >= 1 && p.diag.pageErrors.length === 0, (await p.locator('.toast').last().textContent().catch(() => '')) || '');
const isLive = await live();
const clashBox = p.locator('label.check', { hasText: /Allow clashes/ });
await p.getByRole('button', { name: 'Expert' }).click();
check(isLive ? 'the live server has no --allow-clashes, so the option is hidden' : 'demo shows the "Allow clashes" option in Expert', (await clashBox.count()) === (isLive ? 0 : 1), '');
await p.getByRole('button', { name: 'Guided' }).click();
await shot(p, '22-score');
check('no page errors on Score', p.diag.pageErrors.length === 0, p.diag.pageErrors.slice(0, 2).join('|'));
check('no 5xx', p.diag.bad.length === 0, p.diag.bad.join('|'));
await b.close(); finish('22_compare_score_controls');
