import { browser, newPage, check, finish, shot, BASE, sleep, flipTheme, openMenu } from './lib.mjs';
const b = await browser();
const p = await newPage(b);
const cdp = await p.context().newCDPSession(p);
async function heapMB() { await cdp.send('HeapProfiler.collectGarbage'); const m = await cdp.send('Performance.getMetrics'); return m.metrics.find((x) => x.name === 'JSHeapUsedSize').value / 1048576; }
await cdp.send('Performance.enable');
await p.goto(BASE + '/'); await p.waitForSelector('.hero-card');
const h0 = await heapMB();

// ---- 1. 400 route changes in one tab: does memory (JS heap) grow without bound? ----
const routes = ['#/', '#/predict', '#/compare', '#/score', '#/predict', '#/'];
const t0 = Date.now();
for (let i = 0; i < 400; i++) { await p.evaluate((r) => { location.hash = r; }, routes[i % routes.length]); await p.waitForTimeout(25); }
await p.waitForSelector('#main > *');
const h1 = await heapMB();
check('400 route changes: JS heap does not balloon', h1 - h0 < 40, `${h0.toFixed(1)} -> ${h1.toFixed(1)} MB in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
check('400 route changes: no page errors', p.diag.pageErrors.length === 0, p.diag.pageErrors.slice(0, 2).join('|'));

// ---- 2. 200 flips of theme / guided-expert / accent ----
await p.goto(BASE + '/#/'); await p.waitForSelector('.hero-card');
for (let i = 0; i < 100; i++) {
  await flipTheme(p);
  await p.getByRole('button', { name: i % 2 ? 'Guided' : 'Expert' }).click();
}
await openMenu(p);
const sw = p.locator('.swatch'); const n = await sw.count();
for (let i = 0; i < 40; i++) await sw.nth(i % n).click();
check('200 theme/mode/accent flips: still responsive, no page errors', p.diag.pageErrors.length === 0 && (await p.locator('.hero-card').count()) === 1, p.diag.pageErrors.slice(0, 2).join('|'));

// ---- 3. the monkey: 700 random clicks/typing/keys over every screen ----
let seed = 12345; const rnd = () => (seed = (seed * 1664525 + 1013904223) % 4294967296) / 4294967296;
const pick = (a) => a[Math.floor(rnd() * a.length)];
const junk = ['', 'A', 'LIYKWVNK', '🙂🙂', '<img src=x onerror=alert(1)>', ' '.repeat(5), '9'.repeat(40), '../../etc/passwd', 'ACDEFGHIKLMNPQRSTVWY'.repeat(3), '-1', '1e999', 'NaN'];
let actions = 0, clicksOnRun = 0;
let dialogs = 0; p.on('dialog', (d) => { dialogs++; d.dismiss().catch(() => {}); });
await p.goto(BASE + '/#/'); await p.waitForSelector('.hero-card');
const t1 = Date.now();
while (actions < 700) {
  actions++;
  try {
    if (actions % 15 === 0) { await p.evaluate((r) => { location.hash = r; }, pick(routes)); await p.waitForTimeout(80); continue; }
    if (!(await p.evaluate(() => !!document.querySelector('dialog[open]'))) === false && rnd() < 0.5) { await p.keyboard.press('Escape'); continue; }
    const kind = rnd();
    if (kind < 0.12) { // type junk into a random visible text field
      const f = p.locator('input[type=text], input[type=search], textarea, input:not([type])').filter({ visible: true });
      const c = await f.count(); if (c) await f.nth(Math.floor(rnd() * c)).fill(pick(junk), { timeout: 800 });
    } else if (kind < 0.18) { await p.keyboard.press(pick(['Tab', 'Shift+Tab', 'Enter', 'Space', 'ArrowDown', 'ArrowUp', 'ArrowLeft', 'ArrowRight', 'PageUp', 'PageDown', 'Escape'])); }
    else if (kind < 0.24) { // drag on the 3D stage
      const x = 100 + rnd() * 900, y = 120 + rnd() * 600; await p.mouse.move(x, y); await p.mouse.down(); await p.mouse.move(x + rnd() * 200 - 100, y + rnd() * 200 - 100, { steps: 4 }); await p.mouse.up();
    } else if (kind < 0.27) { await p.mouse.wheel(0, rnd() * 800 - 400); }
    else { // click a random visible, enabled button/link/radio/chip
      const els = p.locator('button:not([disabled]), a[href], [role=radio], summary, tr[data-rank]').filter({ visible: true });
      const c = await els.count(); if (!c) continue;
      const el = els.nth(Math.floor(rnd() * c)); const label = ((await el.textContent({ timeout: 800 })) || '').trim();
      if (/Run prediction|Run example|Score it|Compare/i.test(label)) clicksOnRun++;
      await el.click({ timeout: 1000, noWaitAfter: true });
    }
  } catch { /* an element vanished mid-action: that's a normal part of a UI changing under a monkey */ }
}
await sleep(1500);
check('700 random actions: zero uncaught page errors', p.diag.pageErrors.length === 0, `${clicksOnRun} run-starting clicks; ${p.diag.pageErrors.slice(0, 3).join(' | ')}`);
check('700 random actions: no HTTP 5xx', p.diag.bad.length === 0, p.diag.bad.slice(0, 3).join('|'));
check('700 random actions: no alert()/XSS dialogs fired', dialogs === 0, String(dialogs));
await shot(p, 's6-after-monkey');
const leftApp = !p.url().startsWith(BASE + '/?') && !p.url().startsWith(BASE + '/#') && p.url() !== BASE + '/'; // the Help dialog legitimately links to the classic studio page
if (leftApp) await p.goto(BASE + '/#/');
await p.waitForSelector('#main > *', { timeout: 10000 });
const alive = await p.evaluate(() => !!document.querySelector('#main') && !!document.querySelector('#topbar button'));
check('the app is still alive and rendering afterwards', alive, `${((Date.now() - t1) / 1000).toFixed(0)}s${leftApp ? ' (monkey had wandered to ' + 'the classic page)' : ''}`);
// stop any run the monkey started, so the AGX is idle again
await p.goto(BASE + '/#/'); await sleep(500);
const h2 = await heapMB(); check('heap after the monkey is sane', h2 < 250, `${h2.toFixed(1)} MB`);
// the monkey may have started real runs: stop the one still going (through the UI, as a person would) so it cannot block the next suite
const pillNow = p.locator('.topbar button', { hasText: 'Run in progress' }).filter({ visible: true });
if (await pillNow.count()) { await pillNow.click().catch(() => {}); await p.getByRole('button', { name: 'Stop' }).click({ timeout: 5000 }).catch(() => {}); await sleep(1500); }
await b.close(); finish('06_monkey');
