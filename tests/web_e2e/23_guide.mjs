// The in-app guide: navigation, contents, deep links, code-copy buttons, images, phone layout. Works on a live or a demo server.
import { browser, newPage, check, finish, shot, BASE, sleep } from './lib.mjs';

const b = await browser();
const p = await newPage(b, { permissions: ['clipboard-read', 'clipboard-write'] });

// ---- the way in: four destinations in the top bar ----
await p.goto(BASE + '/#/'); await p.waitForSelector('.hero-card');
const links = await p.locator('.nav-links a').allTextContents();
check('the top bar has Predict, Compare, Score and Guide', links.join(',') === 'Predict,Compare,Score,Guide', links.join(','));
await p.locator('.nav-links a', { hasText: 'Guide' }).click(); await p.waitForSelector('.guide-body h2');
check('Guide opens the guide and is marked as the current page', /#\/guide/.test(p.url()) && (await p.locator('.nav-links a[aria-current="page"]').textContent()).trim() === 'Guide', p.url());
check('the guide has a title and a lead sentence', /guide/i.test(await p.locator('.guide-body h1').textContent()) && (await p.locator('.guide-body h1 + p').textContent()).length > 40, '');
const heads = await p.locator('.guide-body h2').count();
check('the guide has its main sections (at least 12)', heads >= 12, `${heads} sections`);

// ---- contents ----
const tocLinks = await p.locator('.guide-toc a[data-toc]').count();
check('every section is in the table of contents', tocLinks >= heads, `${tocLinks} entries for ${heads} sections`);
await p.locator('.guide-toc a', { hasText: 'Troubleshooting' }).first().click(); await sleep(900);
const inView = await p.evaluate(() => { const r = document.getElementById('troubleshooting').getBoundingClientRect(); return r.top >= 0 && r.top < innerHeight * 0.5; });
check('a contents entry scrolls to its section', inView, '');
check('and updates the address so it can be shared', /#\/guide\/troubleshooting$/.test(p.url()), p.url());
await sleep(500);
check('the contents marks the section being read', (await p.locator('.guide-toc a[aria-current="true"]').count()) === 1, '');

// ---- deep links ----
await p.goto(BASE + '/#/guide/faq'); await p.waitForSelector('.guide-body h2'); await sleep(900);
check('a deep link opens at that section', await p.evaluate(() => { const r = document.getElementById('faq').getBoundingClientRect(); return r.top >= 0 && r.top < innerHeight * 0.5; }), '');
await p.goto(BASE + '/#/guide/no-such-section'); await p.waitForSelector('.guide-body h2');
check('an unknown section still shows the guide (from the top)', (await p.locator('.guide-body h1').count()) === 1, '');

// ---- in-guide links scroll instead of reloading the screen ----
await p.goto(BASE + '/#/guide'); await p.waitForSelector('.guide-body h2');
const internal = p.locator('.guide-body a[data-jump]').first();
check('links between sections exist', (await internal.count()) === 1, '');
await internal.click(); await sleep(800);
check('following one keeps the guide open', (await p.locator('.guide-body h2').count()) >= 12 && /#\/guide\//.test(p.url()), p.url());

// ---- external links are safe ----
const ext = await p.locator('.guide-body a[href^="http"]').evaluateAll((as) => as.map((a) => [a.target, a.rel]));
check('external links open in a new tab with noopener', ext.length > 0 && ext.every(([t, r]) => t === '_blank' && /noopener/.test(r)), `${ext.length} links`);

// ---- code samples can be copied ----
await p.goto(BASE + '/#/guide/install'); await p.waitForSelector('.guide-body .cmd-wrap'); await sleep(600);
const wrap = p.locator('.guide-body .cmd-wrap').first();
const sample = (await wrap.locator('pre').textContent()).trim();
await wrap.getByRole('button', { name: /Copy/ }).click(); await sleep(300);
check('a code sample has a Copy button that copies exactly the sample', (await p.evaluate(() => navigator.clipboard.readText())) === sample, sample.slice(0, 40));

// ---- images: the screenshots load ----
await p.goto(BASE + '/#/guide'); await p.waitForSelector('.guide-body figure img', { state: 'attached' });
const total = await p.locator('.guide-body figure img').count();
for (const img of await p.locator('.guide-body figure img').all()) { await img.scrollIntoViewIfNeeded(); await sleep(120); }
await sleep(800);
const broken = await p.locator('.guide-body figure img').evaluateAll((is) => is.filter((i) => !i.complete || i.naturalWidth === 0).map((i) => i.getAttribute('src')));
check(`all ${total} screenshots load`, total >= 10 && broken.length === 0, broken.join(' '));
check('every screenshot has a caption', (await p.locator('.guide-body figure figcaption').count()) === total, '');

// ---- no stray markup from the text ----
check('no raw Markdown marks leak onto the page', !/\*\*|\]\(|^#{1,3} /m.test(await p.locator('.guide-body').innerText()), '');
await shot(p, '23-guide');
check('no uncaught page errors', p.diag.pageErrors.length === 0, p.diag.pageErrors.slice(0, 2).join('|'));

// ---- phone ----
const m = await newPage(b, { viewport: { width: 390, height: 800 }, hasTouch: true });
await m.goto(BASE + '/#/guide'); await m.waitForSelector('.guide-body h2');
check('phone: no sideways scrolling', await m.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), '');
check('phone: the contents is a collapsible block at the top', (await m.locator('.guide-toc details').count()) === 1, '');
const tabs = await m.locator('.tabbar a').allTextContents();
check('phone: the tab bar has four tabs and Guide is current', tabs.length === 4 && (await m.locator('.tabbar a[aria-current="page"]').textContent()).includes('Guide'), tabs.join(','));
check('phone: tables scroll inside themselves instead of widening the page', await m.evaluate(() => [...document.querySelectorAll('.table-scroll')].every((t) => t.parentElement.scrollWidth <= innerWidth + 1)), '');
await b.close(); finish('23_guide');
