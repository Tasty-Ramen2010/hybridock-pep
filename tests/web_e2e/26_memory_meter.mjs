// The top bar's memory meter: a dot while idle, a liquid tank while a prediction runs, a warning bubble when memory is low.
// Memory readings are fed in through the page's own network layer, so every level can be looked at; one check uses the real server.
import { browser, newPage, check, finish, shot, BASE, sleep } from './lib.mjs';

const b = await browser();
let stats = { ok: true, total_gb: 16, available_gb: 9, percent: 44, run_gb: 3.1 };
async function open(opts = {}) {
  const p = await newPage(b, opts);
  await p.route('**/api/stats', (r) => r.fulfill({ json: stats }));
  await p.route('**/api/run', () => {}); // never really starts: the request hangs, which keeps the Running screen up
  await p.goto(BASE + '/#/'); await p.waitForSelector('.hero-card');
  return p;
}
const lvl = (p) => p.locator('.ram').evaluate((e) => ({ level: e.dataset.level, on: e.dataset.on, w: e.getBoundingClientRect().width, liq: getComputedStyle(e).getPropertyValue('--liq'), fill: getComputedStyle(e).getPropertyValue('--fill') }));
const settle = async (p) => { await sleep(2600); };

let p = await open();
const demo = (await p.locator('.topbar .badge-demo').count()) > 0;
if (demo) { check('on the demo site there is no meter (nothing real to measure)', (await p.locator('.ram').count()) === 0, ''); await b.close(); finish('26_memory_meter'); process.exit(0); }

// ---- idle: a small dot, no warning ----
await p.waitForSelector('.status-chip'); await settle(p);
let a = await lvl(p);
check('idle: the meter is a small dot', a.on === 'false' && a.w <= 12, JSON.stringify(a));
check('the chip says GPU or CPU, nothing more', /^(GPU|CPU)$/.test((await p.locator('.status-chip .chip-text').textContent()).trim()), '');
check('no warning when memory is fine', await p.locator('.ram-bubble').isHidden(), '');
const idleColour = a.liq;

// ---- running: the dot widens into a tank ----
await p.getByRole('button', { name: /Watch a (live|demo) run/ }).click(); await p.waitForSelector('.track'); await settle(p);
a = await lvl(p);
check('while a run goes the dot widens into a tank', a.on === 'true' && a.w >= 30, JSON.stringify(a));
check('the percentage appears beside it', (await p.locator('.ram-pct').textContent()).trim() === '44%', '');
check('the liquid is as high as memory is full (44 %)', a.fill.trim() === '44%', a.fill);
const green = a.liq;

// ---- the colour follows the percentage ----
const seen = { 44: green };
for (const [pc, av] of [[78, 3.5], [91, 1.4], [99, 0.2]]) {
  stats = { ...stats, percent: pc, available_gb: av }; await settle(p);
  const s = await lvl(p); seen[pc] = s.liq;
  check(`${pc} % fills the tank to ${pc} %`, s.fill.trim() === `${pc}%`, s.fill);
  if (pc === 78) check('78 %: busy, amber, and still no warning', s.level === 'busy' && (await p.locator('.ram-bubble').isHidden()), s.level);
  if (pc === 91) {
    check('91 %: low, and a warning bubble appears', s.level === 'low' && (await p.locator('.ram-bubble').isVisible()), s.level);
    check('...which says what to do', /close other apps/i.test(await p.locator('.ram-bubble').innerText()), '');
    await shot(p, '26-low');
  }
  if (pc === 99) { check('99 %: full, and the liquid spills over', s.level === 'full' && (await p.locator('.ram-foam').evaluate((e) => getComputedStyle(e).opacity)) === '1', s.level); await shot(p, '26-full'); }
}
check('the colour changes with the percentage (all four are different)', new Set(Object.values(seen)).size === 4, JSON.stringify(seen));
const hue = (c) => Number((c.match(/hsl\(\s*([\d.]+)/) || [])[1]);
check('green at low, red at high', hue(seen[44]) > 100 && hue(seen[99]) < 20, `${hue(seen[44])} -> ${hue(seen[99])}`);

// ---- the bubble: dismiss, and it comes back next time ----
await p.locator('.ram-bubble-x').click();
check('the warning can be dismissed', await p.locator('.ram-bubble').isHidden(), '');
await sleep(2200);
check('...and stays dismissed while memory stays low', await p.locator('.ram-bubble').isHidden(), '');
stats = { ...stats, percent: 50, available_gb: 8 }; await settle(p);
stats = { ...stats, percent: 95, available_gb: 0.8 }; await settle(p);
check('...but comes back the next time memory runs low', await p.locator('.ram-bubble').isVisible(), '');
check('low free memory alone (under 1 GB) also warns, even at 80 %', await (async () => { stats = { ...stats, percent: 80, available_gb: 0.6 }; await settle(p); return (await lvl(p)).level === 'low'; })(), '');

// ---- details are one click away ----
await p.locator('.status-chip').click(); await sleep(400);
const pop = (await p.locator('.topbar .popover:visible').innerText()).replace(/\s+/g, ' ');
check('the popover gives memory in numbers and what the run is using', /\d+% used, [\d.]+ of 16 GB/.test(pop) && /This prediction is using about 3\.1 GB/.test(pop), pop.slice(0, 200));
check('the warning bubble steps aside while the popover is open', await p.locator('.ram-bubble').isHidden(), '');
await p.keyboard.press('Escape');

// ---- the run ends: back to a dot ----
stats = { ...stats, percent: 40, available_gb: 9.6 };
await p.goto(BASE + '/#/running'); await p.waitForSelector('.track');
await p.getByRole('button', { name: /^Stop/ }).click();
await p.getByRole('button', { name: /Stop|Yes|Confirm/ }).last().click({ timeout: 2000 }).catch(() => {});
await p.waitForFunction(() => document.querySelector('.ram')?.dataset.on === 'false', null, { timeout: 15000 }).catch(() => {});
const idleAgain = await lvl(p);
check('when the run ends the tank shrinks back to a dot', idleAgain.on === 'false' && idleAgain.w <= 12, JSON.stringify(idleAgain));
check('no uncaught page errors', p.diag.pageErrors.length === 0, p.diag.pageErrors.slice(0, 2).join('|'));
await p.close();

// ---- when the server cannot say, the meter stays quiet ----
stats = { ok: false, total_gb: null, available_gb: null, percent: null, run_gb: null };
p = await open(); await p.waitForSelector('.status-chip'); await settle(p);
check('no reading: still a dot, no warning, no errors', (await lvl(p)).level === 'unknown' && await p.locator('.ram-bubble').isHidden() && p.diag.pageErrors.length === 0, '');
await p.close();

// ---- the real server's numbers ----
const real = await (await fetch(BASE + '/api/stats')).json().catch(() => null);
check('the real server reports memory', real && real.ok && real.percent > 0 && real.percent < 100 && real.total_gb > 1, JSON.stringify(real));

// ---- phone: it fits, and the bubble goes across the top ----
stats = { ok: true, total_gb: 16, available_gb: 1, percent: 93, run_gb: 3 };
const m = await open({ viewport: { width: 390, height: 800 }, hasTouch: true, isMobile: true });
await m.getByRole('button', { name: /Watch a (live|demo) run/ }).tap(); await m.waitForSelector('.track'); await sleep(3000);
const fit = await m.evaluate(() => { const r = document.querySelector('.avatar').getBoundingClientRect(); return { right: Math.round(r.right), iw: innerWidth, sw: document.documentElement.scrollWidth }; });
check('phone: the top bar fits while a run is going (avatar inside the screen)', fit.right <= fit.iw && fit.sw <= fit.iw + 1, JSON.stringify(fit));
const bb = await m.locator('.ram-bubble').boundingBox();
check('phone: the warning spans the width under the bar', bb && bb.x >= 0 && bb.x + bb.width <= 390 && bb.width > 300, JSON.stringify(bb));
await shot(m, '26-phone');
await b.close(); finish('26_memory_meter');
