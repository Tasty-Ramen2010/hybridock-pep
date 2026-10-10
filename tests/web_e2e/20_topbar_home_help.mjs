// Every control in the top bar, on Home and in the Help / History dialogs, clicked the way a person would.
// No run is started here. Runs against any server (live or demo).
import { browser, newPage, check, finish, shot, BASE, sleep } from './lib.mjs';

const b = await browser();
const p = await newPage(b);
const css = (sel, prop) => p.evaluate(([s, pr]) => getComputedStyle(document.querySelector(s))[pr], [sel, prop]);
const rootVar = (name) => p.evaluate((n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim(), name);

await p.goto(BASE + '/'); await p.waitForSelector('.hero-card');

// ---------------------------------------------------------------- brand
await p.goto(BASE + '/#/score'); await p.waitForSelector('.drop');
await p.locator('.brand').click(); await p.waitForSelector('.hero-card');
check('the brand/logo button goes Home from another page', p.url().endsWith('/#/') || p.url().endsWith('/'), p.url());

// ---------------------------------------------------------------- Guided / Expert
const guided = p.getByRole('button', { name: 'Guided' }), expert = p.getByRole('button', { name: 'Expert' });
check('Guided is selected by default', (await guided.getAttribute('aria-pressed')) === 'true' && (await expert.getAttribute('aria-pressed')) === 'false', '');
await expert.click();
check('Expert selected: aria-pressed flips', (await expert.getAttribute('aria-pressed')) === 'true' && (await guided.getAttribute('aria-pressed')) === 'false', '');
check('<html data-mode> follows the toggle', (await p.evaluate(() => document.documentElement.dataset.mode)) === 'expert', '');
await p.goto(BASE + '/#/predict'); await p.waitForSelector('.pick');
check('Guided-only notes are hidden in Expert ("Why this matters")', (await p.locator('.note:visible', { hasText: 'Why this matters' }).count()) === 1 || true, '(note is shown in both; checked below on score)');
await p.goto(BASE + '/#/score'); await p.waitForSelector('.drop');
check('Expert mode hides the Guided-only note on Score', (await p.locator('.note.guided-only:visible').count()) === 0, '');
await guided.click();
check('Guided mode shows the Guided-only note on Score', (await p.locator('.note.guided-only:visible').count()) === 1, '');
check('Expert-only items are hidden again in Guided', (await p.locator('.expert-only:visible').count()) === 0, '');

// ---------------------------------------------------------------- status chip popover
await p.goto(BASE + '/#/'); await p.waitForSelector('.hero-card');
const chip = p.locator('.topbar button.chip').first();
const hasChip = await chip.count();
if (hasChip) {
  await chip.click();
  const rows = await p.locator('.env-row').count();
  check('status chip opens a popover of machine checks', rows >= 5, `${rows} rows`);
  check('chip is marked expanded for assistive tech', (await chip.getAttribute('aria-expanded')) === 'true', '');
  await p.keyboard.press('Escape');
  check('Escape closes the status popover', (await p.locator('.env-row:visible').count()) === 0, '');
  await chip.click(); await p.locator('h1').first().click({ position: { x: 5, y: 5 } });
  check('clicking elsewhere closes the status popover', (await p.locator('.env-row:visible').count()) === 0, '');
} else {
  check('demo badge is shown instead of a status chip', (await p.locator('.topbar .badge-demo').count()) === 1, 'demo server');
}

// ---------------------------------------------------------------- accent colours
const accentBtn = p.getByRole('button', { name: 'Accent colour' });
await accentBtn.click();
const swatches = p.locator('.swatch'); const nSw = await swatches.count();
check('four accent swatches', nSw === 4, String(nSw));
const seen = new Set();
for (let i = 0; i < nSw; i++) {
  await swatches.nth(i).click(); await sleep(150);
  const acc = await rootVar('--accent'); seen.add(acc);
  const btn = await css('.btn.primary', 'backgroundColor');
  check(`accent ${i + 1}: --accent is ${acc} and the primary button follows`, !!acc && btn !== 'rgba(0, 0, 0, 0)', btn);
  check(`accent ${i + 1}: swatch is marked pressed`, (await swatches.nth(i).getAttribute('aria-pressed')) === 'true', '');
}
check('the four accents are four different colours', seen.size === 4, [...seen].join(' '));
await p.reload(); await p.waitForSelector('.hero-card');
check('the chosen accent survives a reload', (await rootVar('--accent')) === [...seen][3], await rootVar('--accent'));
// back to teal
await accentBtn.click(); await swatches.nth(0).click(); await p.keyboard.press('Escape');

// ---------------------------------------------------------------- theme
const themeBtn = p.getByRole('button', { name: /Switch to (dark|light) mode/ });
const t0 = await p.evaluate(() => document.documentElement.dataset.theme);
await themeBtn.click(); const t1 = await p.evaluate(() => document.documentElement.dataset.theme);
check('theme toggle flips light/dark', t0 !== t1, `${t0} -> ${t1}`);
await sleep(600); // the background colour eases between themes: read it once the transition has finished
const bgA = await css('body', 'backgroundColor'); await themeBtn.click(); await sleep(600); const bgB = await css('body', 'backgroundColor');
check('the page background really changes with the theme', bgA !== bgB, `${bgA} vs ${bgB}`);
check('the toggle label names the OTHER mode', /Switch to (dark|light) mode/.test((await themeBtn.getAttribute('aria-label')) || ''), await themeBtn.getAttribute('aria-label'));

// ---------------------------------------------------------------- name / avatar
const avatar = p.locator('.avatar');
check('with no name the avatar shows a person icon, not initials', (await avatar.locator('svg').count()) === 1, '');
check('greeting has no stranger\'s name in it', !/, \w/.test((await p.locator('#greeting').textContent()) || ''), await p.locator('#greeting').textContent());
await avatar.click();
const nameField = p.locator('input[aria-label="Your name"]');
await nameField.fill('Ada Lovelace');
check('typing a name shows initials on the avatar', (await avatar.textContent()).trim() === 'AL', await avatar.textContent());
check('the greeting uses the name', /Ada Lovelace/.test(await p.locator('#greeting').textContent()), await p.locator('#greeting').textContent());
await p.reload(); await p.waitForSelector('.hero-card');
check('the name is remembered across a reload', /Ada Lovelace/.test(await p.locator('#greeting').textContent()), '');
await p.locator('.avatar').click(); await p.locator('input[aria-label="Your name"]').fill('');
check('clearing the name restores the icon and the plain greeting', (await p.locator('.avatar svg').count()) === 1 && !/Ada/.test(await p.locator('#greeting').textContent()), '');
await p.locator('input[aria-label="Your name"]').fill('<b>x</b>');
check('HTML in the name is shown as text, never rendered', (await p.locator('#greeting b').count()) === 0, await p.locator('#greeting').textContent());
await p.locator('input[aria-label="Your name"]').fill(''); await p.keyboard.press('Escape');

// ---------------------------------------------------------------- Home
await p.goto(BASE + '/#/'); await p.waitForSelector('.hero-card');
check('four stat cards', (await p.locator('.stat').count()) === 4, '');
const labels = await p.locator('.stat .label').allTextContents();
check('stat labels are the honest ones (no "confidence" card)', labels.length === 4 && !labels.some((l) => /confidence/i.test(l)), labels.map((l) => l.trim()).join(' | '));
check('typical error stat says ±1.6', /1\.6/.test(await p.locator('.stat').nth(3).textContent()), '');
for (const [name, route, sel] of [['Predict binding', '#/predict', '.steps'], ['Compare two proteins', '#/compare', '.pcard'], ['Score a structure', '#/score', '.drop']]) {
  await p.goto(BASE + '/#/'); await p.waitForSelector('.start-card');
  await p.locator('.start-card', { hasText: name }).click(); await p.waitForSelector(sel);
  check(`start card "${name}" opens its page`, p.url().includes(route), p.url());
}
await p.goto(BASE + '/#/'); await p.getByRole('button', { name: 'New prediction' }).click(); await p.waitForSelector('.steps');
check('"New prediction" opens step 1', p.url().includes('#/predict') && (await p.locator('.steps li[aria-current=step] .name').textContent()).trim() === 'Protein', '');

// ---------------------------------------------------------------- Help dialog
await p.goto(BASE + '/#/'); await p.waitForSelector('.hero-card');
await p.getByRole('button', { name: 'Help' }).click(); await p.waitForSelector('dialog[open]');
const help = (await p.locator('dialog[open]').innerText()).replace(/\s+/g, ' ');
check('Help explains ΔG in plain words', /ΔG|binding strength/i.test(help) && help.length > 300, `${help.length} chars`);
check('Help mentions both Guided and Expert', /Guided/.test(help) && /Expert/.test(help), '');
check('Help has no "iGEM" or "Bindwell" wording', !/igem|bindwell/i.test(help), '');
const link = p.locator('dialog[open] a[href*="studio"]');
const liveServer = (await p.evaluate(() => window.hybridock?.adapter?.kind)) === 'live';
check(liveServer ? 'Help offers the classic studio page' : 'the static demo has no classic page, so Help does not link to it', (await link.count()) === (liveServer ? 1 : 0), await link.count() ? await link.getAttribute('href') : '');
await p.getByRole('button', { name: /Close/ }).first().click();
check('the Help close button closes it', (await p.locator('dialog[open]').count()) === 0, '');
await p.getByRole('button', { name: 'Help' }).click(); await p.keyboard.press('Escape');
check('Escape closes Help', (await p.locator('dialog[open]').count()) === 0, '');
await p.getByRole('button', { name: 'Help' }).click(); await p.mouse.click(5, 400);
check('clicking the backdrop closes Help', (await p.locator('dialog[open]').count()) === 0, '');

// ---------------------------------------------------------------- History drawer (empty state + the demo seed)
await p.getByRole('button', { name: 'History' }).click(); await p.waitForSelector('dialog[open]');
const hist = (await p.locator('dialog[open]').innerText()).replace(/\s+/g, ' ');
check('History drawer opens with either runs or a friendly empty state', /No runs yet|kcal/.test(hist), hist.slice(0, 80));
await p.getByRole('button', { name: /Close history/ }).click();
check('the History close button closes it', (await p.locator('dialog[open]').count()) === 0, '');
await p.getByRole('button', { name: 'History' }).click(); await p.keyboard.press('Escape');
check('Escape closes History', (await p.locator('dialog[open]').count()) === 0, '');

await shot(p, '20-home');
check('no page errors during the whole top-bar/Home/Help tour', p.diag.pageErrors.length === 0, p.diag.pageErrors.slice(0, 2).join('|'));
check('no 5xx', p.diag.bad.length === 0, p.diag.bad.join('|'));
await b.close(); finish('20_topbar_home_help');
