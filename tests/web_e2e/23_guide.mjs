// The in-app guide, used the way a person uses it: open it from the top bar, pick a lesson, step through, come back.
// Works on a live or a demo server (the wording differs a little between them, and the checks follow that).
import { browser, newPage, check, finish, shot, BASE, sleep } from './lib.mjs';

const b = await browser();
const p = await newPage(b, { permissions: ['clipboard-read', 'clipboard-write'] });

// ---- the way in ----
await p.goto(BASE + '/#/'); await p.waitForSelector('.hero-card');
const live = (await p.locator('.get-app').count()) === 0;
const links = await p.locator('.nav-links a').allTextContents();
check('the top bar has Predict, Compare, Score and Guide', links.join(',') === 'Predict,Compare,Score,Guide', links.join(','));
await p.locator('.nav-links a', { hasText: 'Guide' }).click(); await p.waitForSelector('.lesson');
check('Guide opens and is marked as the current page', /#\/guide$/.test(p.url()) && (await p.locator('.nav-links a[aria-current="page"]').textContent()).trim() === 'Guide', p.url());

// ---- the lessons are tiles, not a wall of text ----
const tiles = await p.locator('.lesson').count();
check('there are a handful of lessons (8 to 10)', tiles >= 8 && tiles <= 10, `${tiles}`);
const words = (await p.locator('.screen.guide').innerText()).split(/\s+/).length;
check('the guide home is short (under 220 words)', words < 220, `${words} words`);
check('no long paragraphs, no tables, no contents list on the home', (await p.locator('.screen.guide table, .guide-toc, .guide-body').count()) === 0, '');
const firstTitle = (await p.locator('.lesson .lesson-title').first().textContent()).trim();
check('the guide opens with the live demo lesson', firstTitle === 'Try the live demo', firstTitle);
const text = await p.locator('.screen.guide').innerText();
check('nothing about GPUs, memory sizes or the internet is on the guide home', !/GPU|VRAM|gigabyte|internet|requirement/i.test(text), '');
check('no greeting and no "iGEM" or "AI" wording', !/good (morning|evening)|iGEM|\bAI\b/i.test(text), '');
await shot(p, '23-guide-home');

// ---- a lesson: step through with the buttons ----
await p.locator('.lesson', { hasText: 'Try the live demo' }).click(); await p.waitForSelector('.step-card');
check('a lesson opens at step 1 and the address says so', /#\/guide\/live-demo\/1$/.test(p.url()) && /Step 1 of \d/.test(await p.locator('.step-count').textContent()), p.url());
check('step 1 has no Back button', await p.getByRole('button', { name: 'Back' }).isHidden(), '');
const total = await p.locator('.step-bar').count();
check('the progress bars match the steps', total >= 5 && total <= 8, `${total} steps`);
const img = p.locator('.step-shot img');
await img.waitFor();
await p.waitForFunction(() => { const i = document.querySelector('.step-shot img'); return i && i.complete && i.naturalWidth > 0; }, null, { timeout: 8000 });
check('step 1 shows its screenshot', (await img.evaluate((i) => i.naturalWidth)) > 400, '');
const bodyWords = (await p.locator('.step-body').textContent()).trim().split(/\s+/).length;
check('a step is one short idea (under 45 words)', bodyWords < 45, `${bodyWords} words`);
const first = (await p.locator('.step-text h2').textContent()).trim();
await p.getByRole('button', { name: /^Next/ }).click(); await sleep(400);
check('Next goes to step 2 with a new headline', /Step 2 of/.test(await p.locator('.step-count').textContent()) && (await p.locator('.step-text h2').textContent()).trim() !== first, '');
check('the address follows the step (and Back in the browser is not trapped)', /#\/guide\/live-demo\/2$/.test(p.url()), p.url());
check('the bars fill up to the current step', (await p.locator('.step-bar.on').count()) === 2, '');
await p.keyboard.press('ArrowRight'); await sleep(300);
check('the right arrow key goes forward', /Step 3 of/.test(await p.locator('.step-count').textContent()), '');
await p.keyboard.press('ArrowLeft'); await sleep(300);
check('the left arrow key goes back', /Step 2 of/.test(await p.locator('.step-count').textContent()), '');
await p.getByRole('button', { name: /^Back/ }).click(); await sleep(300);
check('Back goes back to step 1', /Step 1 of/.test(await p.locator('.step-count').textContent()), '');
await p.locator('.step-bar').nth(total - 1).click(); await sleep(300);
check('a progress bar jumps to its step', new RegExp(`Step ${total} of ${total}`).test(await p.locator('.step-count').textContent()), '');
check('the last step says Done instead of Next', (await p.locator('.step-nav .btn.primary').textContent()).trim() === 'Done', '');
await shot(p, '23-guide-last-step');
await p.locator('.step-nav .btn.primary').click(); await p.waitForSelector('.lesson');
check('Done returns to the lessons and ticks the lesson off', (await p.locator('.lesson-done').count()) === 1 && /1 of \d+ done/.test(await p.locator('.guide-progress').textContent()), '');

// ---- a deep link opens at that step; bad links still show the guide ----
await p.goto(BASE + '/#/guide/results/3'); await p.waitForSelector('.step-card');
check('a deep link opens at the right step', /Step 3 of/.test(await p.locator('.step-count').textContent()), '');
await p.goto(BASE + '/#/guide/results/99'); await p.waitForSelector('.step-card');
check('a step number that is too big opens the last step', new RegExp('Step (\\d+) of \\1').test(await p.locator('.step-count').textContent()), '');
await p.goto(BASE + '/#/guide/no-such-lesson'); await p.waitForSelector('.lesson');
check('an unknown lesson shows the lessons', (await p.locator('.lesson').count()) >= 8, '');

// ---- the install lesson: the commands can be copied ----
await p.goto(BASE + '/#/guide/own-computer'); await p.waitForSelector('.step-card .cmd-wrap');
const sample = (await p.locator('.cmd-wrap pre').textContent()).trim();
check('the install step shows the real commands', /git clone --recurse-submodules .*hybridock-pep/.test(sample) && /cd hybridock-pep/.test(sample), sample.slice(0, 50));
await p.locator('.cmd-wrap').getByRole('button', { name: /Copy/ }).click(); await sleep(300);
check('Copy puts exactly those commands on the clipboard', (await p.evaluate(() => navigator.clipboard.readText())) === sample, '');
await p.getByRole('button', { name: /^Next/ }).click(); await sleep(300);
check('the next step is ./install.sh', (await p.locator('.cmd-wrap pre').textContent()).trim() === './install.sh', '');
await p.getByRole('button', { name: /^Next/ }).click(); await sleep(300);
check('and then ./launch_web.sh to start it again', (await p.locator('.cmd-wrap pre').textContent()).trim() === './launch_web.sh', '');
const installText = await p.evaluate(async () => { const out = []; for (const b of document.querySelectorAll('.step-bar')) { b.click(); await new Promise((r) => setTimeout(r, 60)); out.push(document.querySelector('.step-card').innerText); } return out.join('\n'); });
check('the install lesson leaves out GPU, memory and internet requirements', !/GPU|VRAM|memory|internet|disk space/i.test(installText), installText.slice(0, 60));

// ---- list lessons ----
await p.goto(BASE + '/#/guide/troubleshooting'); await p.waitForSelector('.qa');
const qa = await p.locator('.qa').count();
check('troubleshooting is a list of questions that open one by one', qa >= 5 && (await p.locator('.qa[open]').count()) === 0, `${qa}`);
await p.locator('.qa summary').first().click(); await sleep(200);
check('opening a question shows its answer', (await p.locator('.qa[open] p').first().isVisible()), '');
check('the demo and the real app each show only their own entries', live ? !/numbers aren’t real/.test(await p.locator('.qa-list').innerText()) : !/Couldn’t reach the server/.test(await p.locator('.qa-list').innerText()), '');
await p.goto(BASE + '/#/guide/words'); await p.waitForSelector('.info-row');
check('the words lesson explains ΔG', /ΔG/.test(await p.locator('.info-list').innerText()), '');
await p.locator('.back-link').click(); await p.waitForSelector('.lesson');
check('the Guide link goes back to the lessons', /#\/guide$/.test(p.url()), p.url());

// ---- every lesson opens and every screenshot loads ----
const ids = await p.locator('.lesson').evaluateAll((as) => as.map((a) => a.dataset.topic));
const broken = [];
for (const id of ids) {
  await p.goto(BASE + `/#/guide/${id}`); await p.waitForSelector('.step-card, .info-list, .qa-list');
  const n = await p.locator('.step-bar').count();
  for (let i = 0; i < n; i++) {
    const im = p.locator('.step-shot img');
    if (await im.count()) { try { await p.waitForFunction(() => { const x = document.querySelector('.step-shot img'); return x.complete && x.naturalWidth > 0; }, null, { timeout: 6000 }); } catch { broken.push(`${id}/${i + 1}`); } }
    if (i < n - 1) await p.getByRole('button', { name: /^Next/ }).click();
  }
}
check(`every lesson opens (${ids.length}) and no screenshot is broken`, broken.length === 0, broken.join(' '));
check('no uncaught page errors', p.diag.pageErrors.length === 0, p.diag.pageErrors.slice(0, 2).join('|'));

// ---- night mode applies here too ----
await p.goto(BASE + '/#/guide'); await p.waitForSelector('.lesson');
await p.locator('button[aria-label^="Switch to"]').click(); await sleep(400);
check('the guide follows night mode', await p.evaluate(() => getComputedStyle(document.querySelector('.lesson')).backgroundColor !== 'rgb(245, 245, 247)'), '');
await shot(p, '23-guide-dark');

// ---- phone ----
const m = await newPage(b, { viewport: { width: 390, height: 800 }, hasTouch: true });
await m.goto(BASE + '/#/guide'); await m.waitForSelector('.lesson');
check('phone: one lesson per row and no sideways scrolling', (await m.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)) && (await m.evaluate(() => new Set([...document.querySelectorAll('.lesson')].map((e) => Math.round(e.getBoundingClientRect().left))).size)) === 1, '');
const tabs = await m.locator('.tabbar a').allTextContents();
check('phone: the tab bar has four tabs and Guide is current', tabs.length === 4 && (await m.locator('.tabbar a[aria-current="page"]').textContent()).includes('Guide'), tabs.join(','));
await m.locator('.lesson', { hasText: 'Try the live demo' }).tap(); await m.waitForSelector('.step-card');
check('phone: a step card fits the screen width', await m.evaluate(() => { const r = document.querySelector('.step-card').getBoundingClientRect(); return r.left >= 0 && r.right <= innerWidth; }), '');
const nav = await m.locator('.step-nav .btn.primary').boundingBox();
check('phone: the Next button is a big tap target', nav && nav.height >= 44, JSON.stringify(nav));
await m.locator('.step-nav .btn.primary').tap(); await sleep(300);
check('phone: tapping Next moves on', /Step 2 of/.test(await m.locator('.step-count').textContent()), '');
await shot(m, '23-guide-phone');
await b.close(); finish('23_guide');
