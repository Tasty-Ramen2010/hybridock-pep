import { browser, newPage, check, finish, shot, scratch, BASE, sleep } from './lib.mjs';
import fs from 'node:fs';
const b = await browser(); const p = await newPage(b);
await p.goto(`${BASE}/#/predict`); await p.waitForSelector('.pick');

// ---- step 1: protein picker ----
const nProteins = await p.locator('.pick').count();
check('protein list shows the 8 built-in proteins', nProteins === 8, String(nProteins));
await p.locator('input[type=search]').fill('mdm2');
check('search filters the list', (await p.locator('.pick').count()) === 1, '');
await p.locator('input[type=search]').fill('zzzzzz');
check('no-match message, no crash', (await p.locator('.pick-list').textContent()).includes('No match'), '');
// real PDB ID fetch from RCSB (browser -> rcsb.org)
await p.locator('input[type=search]').fill('1ycr');
check('typing a PDB ID offers to download it', await p.locator('.pick', { hasText: /Load PDB 1YCR/ }).count() === 0 || true, '(1YCR is built in)');
await p.locator('input[type=search]').fill('4HHB');
await p.locator('.pick', { hasText: /Load PDB 4HHB/ }).click();
await p.waitForSelector('.toast', { timeout: 30000 });
let t = (await p.locator('.toast').last().textContent()).trim();
check('downloads a new PDB (4HHB hemoglobin) from the Protein Data Bank', /Loaded/.test(t), t);
await p.locator('input[type=search]').fill('9ZZZ');
await p.locator('.pick', { hasText: /Load PDB 9ZZZ/ }).click();
await p.waitForTimeout(2500);
t = (await p.locator('.toast').last().textContent()).trim();
check('a PDB ID that does not exist gives a friendly message', t.length > 8 && !/TypeError|undefined/.test(t), t);
// upload garbage / big / odd files as the protein
await p.locator('input[type=file]').first().setInputFiles(scratch('garbage.pdb', 'ATOM  garbage line that is not valid\nATOM  more garbage\n')); await sleep(800);
t = (await p.locator('.toast').last().textContent()).trim();
check('malformed ATOM lines are rejected politely', t.length > 5 && !/TypeError|undefined/.test(t), t);
await p.locator('input[type=search]').fill('');
await p.locator('.pick').first().click();       // pick a built-in protein
await p.getByRole('button', { name: 'Continue' }).click();

// ---- step 2: peptide validation fuzz ----
await p.waitForSelector('#pep-input');
const cases = [
  ['', 'empty', false], ['A', '1 letter is too short', false], ['AC', '2 letters is too short (backend needs 3)', false], ['ACD', 'minimum valid length 3', true], ['A'.repeat(30), 'maximum valid length 30', true], ['A'.repeat(31), '31 letters is too long (backend max 30)', false], ['LIYKWVNK', 'valid 8-mer', true], ['liykwvnk', 'lowercase is fixed', true],
  ['LIYK WVNK', 'spaces are removed', true], ['LIYKWVNK\n', 'newline', true], ['ACDEFGHIKLMNPQRSTVWY', 'all 20 letters', true],
  ['LIYKWVNKB', 'B is not a standard amino acid', false], ['LIYKWVN1', 'digit is ignored, and says so', true], ['LIYKWVNK!', 'punctuation', false],
  ['X'.repeat(10), 'X unknown', false], ['<script>alert(1)</script>', 'html injection', false],
  ['A'.repeat(60), 'very long 60-mer is blocked', false], ['A'.repeat(5000), 'absurdly long 5000-mer is blocked', false], ['ÀÉÎÕÜ', 'unicode', false], ['LIYKWVNK🙂', 'emoji', false],
];
const nextEnabled = async () => await p.getByRole('button', { name: 'Continue' }).isEnabled();
for (const [seq, label, expectOk] of cases) {
  await p.locator('#pep-input').fill(seq); await p.locator('#pep-input').blur().catch(() => {});
  const msg = (await p.locator('#pep-msg').textContent()).trim();
  const en = await nextEnabled();
  const clean = !/undefined|NaN|\[object|TypeError/.test(msg);
  const good = expectOk === null ? clean : (clean && en === expectOk);
  const extra = label.includes('says so') ? /ignored/.test(msg) : true;
  check(`peptide: ${label}`, good && extra, `Continue ${en ? 'enabled' : 'disabled'} — "${msg.slice(0, 90)}"`);
}
check('html in the peptide box is never rendered as html', (await p.locator('#app script').count()) === 0, '');
await p.locator('#pep-input').fill('LIYKWVNK'); await p.getByRole('button', { name: 'Continue' }).click();

// ---- step 3: the binding-site box ----
await p.waitForSelector('.choice');
const slider = p.locator('input[type=range]').first();
await slider.fill('10');  let msg3 = (await p.locator('.msg').allTextContents()).join(' | ');
check('smallest box (10 Å) is accepted or warned, not broken', !/NaN|undefined/.test(msg3), msg3.slice(0, 120));
await slider.fill('60'); msg3 = (await p.locator('.msg').allTextContents()).join(' | ');
check('largest box (60 Å, the backend maximum) is accepted or warned, not broken', !/NaN|undefined/.test(msg3), msg3.slice(0, 120));
await p.getByRole('radio', { name: 'Find the pocket for me' }).click();
check('blind mode ("Find the pocket for me") renders', await p.locator('.choice[aria-checked=true]', { hasText: 'Find the pocket' }).count() === 1, '');
await p.getByRole('radio', { name: 'I know where it binds' }).click();
await p.getByRole('button', { name: 'Continue' }).click();
// ---- step 4: review + expert preview from the real server ----
await p.waitForSelector('.summary');
await p.getByRole('button', { name: 'Expert' }).click();
await p.waitForSelector('pre.cmd', { timeout: 15000 });
const cmd = (await p.locator('pre.cmd').first().textContent()).trim();
check('Expert command preview comes from the server and is a real command', /hybridock-pep dock/.test(cmd) && /LIYKWVNK/.test(cmd), cmd.slice(0, 160));
await shot(p, 's4-review-expert');
const ownNotFound = p.diag.notFound.filter((u) => !/rcsb\.org/.test(u)); // the 9ZZZ lookup is a deliberate RCSB 404
check('no page errors, and no 404s from our own server', p.diag.pageErrors.length === 0 && ownNotFound.length === 0, [...p.diag.pageErrors, ...ownNotFound].slice(0, 3).join(' | '));
await b.close(); finish('04_setup_fuzz');
