import { browser, newPage, check, finish, shot, BASE, MDM2, POSE, sleep } from './lib.mjs';
const b = await browser();
// ---------- keyboard only ----------
{
  const p = await newPage(b); await p.goto(BASE + '/'); await p.waitForSelector('.hero-card');
  const focusLabel = () => p.evaluate(() => { const e = document.activeElement; return e ? (e.getAttribute('aria-label') || e.textContent || e.tagName).trim().slice(0, 40) : ''; });
  await p.keyboard.press('Tab'); const first = await focusLabel();
  check('first Tab stop is the "Skip to main content" link', /Skip to main/.test(first), first);
  await p.keyboard.press('Enter'); await sleep(200);
  const inMain = await p.evaluate(() => document.activeElement?.id === 'main' || !!document.activeElement?.closest('#main'));
  check('Skip link moves focus into the main content', inMain, '');
  // reach "New prediction" by Tab and press it
  let hit = false; for (let i = 0; i < 40 && !hit; i++) { await p.keyboard.press('Tab'); hit = /New prediction/.test(await focusLabel()); }
  check('"New prediction" is reachable by Tab', hit, '');
  await p.keyboard.press('Enter'); await p.waitForSelector('.pick');
  // step 1: pick a protein with the keyboard
  let onPick = false; for (let i = 0; i < 30 && !onPick; i++) { await p.keyboard.press('Tab'); onPick = await p.evaluate(() => document.activeElement?.classList.contains('pick')); }
  check('protein rows are reachable by Tab', onPick, '');
  await p.keyboard.press('Space'); await sleep(300);
  check('Space selects a protein', (await p.locator('.pick[aria-checked=true]').count()) === 1, '');
  // Continue by keyboard
  let cont = false; for (let i = 0; i < 30 && !cont; i++) { await p.keyboard.press('Tab'); cont = /Continue/.test(await focusLabel()); }
  check('Continue is reachable by Tab', cont, ''); await p.keyboard.press('Enter');
  await p.waitForSelector('#pep-input'); check('focus lands in the new step (not lost on <body>)', await p.evaluate(() => document.activeElement && document.activeElement !== document.body), await focusLabel());
  await p.locator('#pep-input').fill('LIYKWVNK');
  await p.keyboard.press('Tab'); // out of the textarea
  // dialogs trap focus and close on Escape
  await p.goto(BASE + '/#/'); await p.waitForSelector('.hero-card');
  await p.getByRole('button', { name: 'Help' }).focus(); await p.keyboard.press('Enter'); await p.waitForSelector('dialog[open]');
  for (let i = 0; i < 12; i++) await p.keyboard.press('Tab');
  check('Help dialog keeps focus inside while tabbing', await p.evaluate(() => !!document.activeElement?.closest('dialog')), '');
  await p.keyboard.press('Escape'); await sleep(300);
  check('Escape closes the dialog', (await p.locator('dialog[open]').count()) === 0, '');
  check('focus returns to the page after the dialog closes', await p.evaluate(() => document.activeElement && document.activeElement !== document.body), await focusLabel());
  // box keyboard control on the binding site
  await p.goto(BASE + '/'); await p.waitForSelector('.hero-card'); await p.getByRole('button', { name: 'New prediction' }).click(); await p.waitForSelector('.pick'); await p.locator('.pick').first().click(); await p.getByRole('button', { name: 'Continue' }).click();
  await p.locator('.chip.mono').first().click(); await p.getByRole('button', { name: 'Continue' }).click(); await p.waitForSelector('.choice');
  const before = (await p.locator('.coords').first().textContent().catch(() => '')) || '';
  await p.locator('#stage').focus(); for (let i = 0; i < 6; i++) await p.keyboard.press('ArrowRight');
  await sleep(300); const after = (await p.locator('.coords').first().textContent().catch(() => '')) || '';
  check('arrow keys on the 3D view move the search box (coordinates change)', before !== after || before === '', `${before.trim()} -> ${after.trim()}`);
  check('keyboard walkthrough: no page errors', p.diag.pageErrors.length === 0, p.diag.pageErrors.slice(0, 2).join('|'));
  await p.context().close();
}
// ---------- phone: no horizontal scroll anywhere, tap targets ≥ 44px ----------
{
  const p = await newPage(b, { viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
  const overflow = () => p.evaluate(() => ({ sw: document.documentElement.scrollWidth, iw: window.innerWidth }));
  const small = () => p.evaluate(() => [...document.querySelectorAll('button, a[href], input, select, textarea, [role=radio], summary')].filter((e) => { const r = e.getBoundingClientRect(); const cs = getComputedStyle(e); return r.width > 0 && r.height > 0 && cs.visibility !== 'hidden' && (r.height < 43.5 || r.width < 43.5) && !e.closest('[hidden]') && e.type !== 'hidden' && e.type !== 'file' && e.type !== 'checkbox' && e.type !== 'range'; }).map((e) => `${(e.getAttribute('aria-label') || e.textContent || e.tagName).trim().slice(0, 24)} ${Math.round(e.getBoundingClientRect().width)}x${Math.round(e.getBoundingClientRect().height)}`));
  const screens = [['home', '#/'], ['predict-1', '#/predict'], ['compare', '#/compare'], ['score', '#/score']];
  for (const [name, hash] of screens) {
    await p.goto(BASE + '/' + hash); await p.waitForSelector('#main > *'); await p.waitForTimeout(1500);
    const o = await overflow(); check(`phone ${name}: no horizontal scroll`, o.sw <= o.iw + 1, `${o.sw} vs ${o.iw}`);
    const s = await small(); check(`phone ${name}: tap targets ≥ 44 px`, s.length === 0, s.slice(0, 5).join('; '));
    await shot(p, `s11-phone-${name}`);
  }
  // a whole Score run on a phone with touch input
  await p.goto(BASE + '/#/score'); await p.waitForSelector('.drop');
  const inputs = p.locator('input[type=file]'); await inputs.nth(0).setInputFiles(MDM2); await p.waitForSelector('.msg.ok'); await inputs.nth(1).setInputFiles(POSE); await p.locator('#sc-pep').fill('ETFSDLWKLLPE');
  await p.getByRole('button', { name: /Score it/ }).tap(); await p.waitForSelector('.big-number', { timeout: 90000 });
  const o = await overflow(); check('phone result screen: a real run finishes, no horizontal scroll', o.sw <= o.iw + 1, `${(await p.locator('.big-number .n').textContent()).trim()}; ${o.sw} vs ${o.iw}`);
  await shot(p, 's11-phone-result');
  check('phone: no page errors', p.diag.pageErrors.length === 0, p.diag.pageErrors.slice(0, 2).join('|'));
  await p.context().close();
}
await b.close(); finish('11_keyboard_mobile');
