import { browser, newPage, check, finish, BASE, MDM2, POSE, OUT, sleep } from './lib.mjs';
import fs from 'node:fs';
const AXE = fs.readFileSync(new URL('./node_modules/axe-core/axe.min.js', import.meta.url), 'utf8');
const b = await browser();
const all = [];
async function audit(p, label) {
  await p.waitForTimeout(1600); // let the entrance animation settle (axe reads real colours)
  await p.evaluate(AXE);
  const r = await p.evaluate(async () => (await axe.run(document, { runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa', 'wcag21aa', 'best-practice'] }, resultTypes: ['violations'] })).violations.map((v) => ({ id: v.id, impact: v.impact, help: v.help, n: v.nodes.length, sample: v.nodes[0].html.slice(0, 140), target: v.nodes[0].target.join(' ') })));
  const bad = r.filter((v) => v.impact === 'serious' || v.impact === 'critical');
  all.push({ label, violations: r });
  check(`axe ${label}: no serious/critical violations`, bad.length === 0, bad.map((v) => `${v.id}×${v.n} (${v.target})`).join('; '));
  const minor = r.filter((v) => !bad.includes(v)); if (minor.length) console.log(`   (${label} minor/moderate: ${minor.map((v) => `${v.id}×${v.n}`).join(', ')})`);
}
for (const scheme of ['light', 'dark']) {
  const p = await newPage(b, { colorScheme: scheme });
  await p.goto(BASE + '/'); await p.waitForSelector('.recent-card, .hero-card'); await audit(p, `${scheme} home`);
  await p.getByRole('button', { name: 'Expert' }).click(); await audit(p, `${scheme} home (Expert)`); await p.getByRole('button', { name: 'Guided' }).click();
  await p.goto(BASE + '/#/predict'); await p.waitForSelector('.pick'); await audit(p, `${scheme} setup 1 protein`);
  await p.locator('.pick').first().click(); await p.getByRole('button', { name: 'Continue' }).click(); await p.waitForSelector('#pep-input'); await audit(p, `${scheme} setup 2 peptide`);
  await p.locator('.chip.mono').first().click(); await p.getByRole('button', { name: 'Continue' }).click(); await p.waitForSelector('.choice'); await audit(p, `${scheme} setup 3 binding site`);
  await p.getByRole('button', { name: 'Continue' }).click(); await p.waitForSelector('.summary'); await audit(p, `${scheme} setup 4 review`);
  await p.goto(BASE + '/#/compare'); await p.waitForSelector('.pcard'); await audit(p, `${scheme} compare`);
  await p.goto(BASE + '/#/score'); await p.waitForSelector('.drop'); await audit(p, `${scheme} score`);
  await p.getByRole('button', { name: 'History' }).click(); await p.waitForSelector('dialog[open]'); await audit(p, `${scheme} history drawer`); await p.keyboard.press('Escape');
  await p.getByRole('button', { name: 'Help' }).click(); await p.waitForSelector('dialog[open]'); await audit(p, `${scheme} help`); await p.keyboard.press('Escape');
  // a finished result (real Score run on the AGX)
  await p.goto(BASE + '/#/score'); await p.waitForSelector('.drop');
  const inputs = p.locator('input[type=file]'); await inputs.nth(0).setInputFiles(MDM2); await p.waitForSelector('.msg.ok');
  await inputs.nth(1).setInputFiles(POSE); await p.locator('#sc-pep').fill('ETFSDLWKLLPE');
  await p.getByRole('button', { name: /Score it/ }).click(); await p.waitForSelector('.big-number', { timeout: 120000 }); await audit(p, `${scheme} result`);
  await p.context().close();
}
fs.writeFileSync(`${OUT}/axe_all.json`, JSON.stringify(all, null, 1));
await b.close(); finish('07_accessibility');
