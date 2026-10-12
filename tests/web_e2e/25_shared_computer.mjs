// The hosted website running REAL predictions on the team's computer, the way a judge would use it.
//
//   BASE   = the static site (python3 scripts/build_pages.py --out DIR; python3 -m http.server 8010 --directory DIR)
//   SHARED = the locked-down server it reaches (hybridock-pep serve --public --allow-origin <BASE> --port 8765)
//   REAL=1 also runs one real Quick prediction through the page (a few minutes on an M3).
//
// Page and server are on different origins, exactly as on GitHub Pages and a tunnel, so cross-site rules are exercised.
import { browser, newPage, check, finish, shot, BASE, sleep } from './lib.mjs';

const SHARED = (process.env.SHARED || 'http://127.0.0.1:8765').replace(/\/$/, '');
const REAL = process.env.REAL === '1';
// DISCOVER=1: no ?backend= in the address, so the page must find the computer itself (the way a judge's browser does)
const QS = process.env.DISCOVER === '1' ? '' : `?backend=${encodeURIComponent(SHARED)}`;
const b = await browser();

// ---- 1. connected: the page becomes the real app ----
const p = await newPage(b);
await p.goto(`${BASE}/${QS}`); await p.waitForSelector('.hero-card');
await p.waitForSelector('.chip-text', { timeout: 20000 });
const chip = (await p.locator('.chip-text').first().textContent()).trim();
check('the top bar says Live and names the computer', /^Live · /.test(chip), chip);
check('no "Demo mode" badge anywhere on the page', (await p.locator('.badge-demo').count()) === 0, '');
const home = await p.locator('.hero-card').innerText();
check('Home carries no banner about the live computer', !/team’s own computer|Live\. /.test(home) && (await p.locator('.hero-card .notice').count()) === 0, home.replace(/\s+/g, ' ').slice(0, 120));
check('Home no longer shows the install-the-preview notice', !/This is the preview/.test(home), '');
const apiOrigins = new Set(); p.on('request', (r) => { const u = new URL(r.url()); if (u.pathname.startsWith('/api/')) apiOrigins.add(u.origin); });

// the Predict screen follows the limits
await p.getByRole('button', { name: 'New prediction' }).click(); await p.waitForSelector('.pick');
await p.locator('.pick').first().click(); await p.waitForSelector('text=residues loaded', { timeout: 20000 });
await p.getByRole('button', { name: 'Continue' }).click(); await p.waitForSelector('#pep-input');
await p.locator('#pep-input').fill('ETFSDLWKLLPE'); await p.locator('#pep-input').blur();
await p.getByRole('button', { name: 'Continue' }).click(); await p.waitForSelector('.choice');
const find = p.getByRole('radio', { name: /Find the pocket/ });
check('"Find the pocket for me" is off on a shared computer, and says why', (await find.isDisabled()) && /shared computer/.test(await find.innerText()), '');
await p.getByRole('button', { name: 'Continue' }).click().catch(() => {}); await p.waitForSelector('.summary', { timeout: 20000 });
check('Full is not offered', await p.getByRole('radio', { name: /Full/ }).isDisabled(), '');
check('Quick and Half are', !(await p.getByRole('radio', { name: /Quick/ }).isDisabled()) && !(await p.getByRole('radio', { name: /Half/ }).isDisabled()), '');
check('the page explains the limit', /limited to 50 poses/.test(await p.locator('.screen').innerText()), '');
await p.getByRole('button', { name: 'Expert' }).click(); await sleep(300);
check('Expert mode has no advanced settings that could name files on the computer', (await p.locator('details.adv').count()) === 0, '');
await p.getByRole('button', { name: 'Guided' }).click();

// the guide speaks to a person using the shared computer
await p.goto(`${BASE}/${QS}#/guide`); await p.waitForSelector('.lesson');
check('the guide opens with the first prediction and has no install steps in the way', (await p.locator('.lesson-title').first().textContent()) === 'Your first prediction', '');
await p.goto(`${BASE}/${QS}#/guide/troubleshooting`); await p.waitForSelector('.qa');
const qa = await p.locator('.qa-list').innerText();
check('troubleshooting is about the shared computer, not about ./launch_web.sh', /shared computer is busy/.test(qa) && !/launch_web|install\.sh/.test(qa), '');

// ---- 2. a real prediction ----
if (REAL) {
  await p.goto(`${BASE}/${QS}`); await p.waitForSelector('.hero-card');
  await p.getByRole('button', { name: 'Run example' }).click(); await p.waitForSelector('.track', { timeout: 30000 });
  const t0 = Date.now();
  while (!(await p.locator('.big-number, .error-card').count())) {
    if (Date.now() - t0 > 20 * 60000) break;
    await sleep(4000);
  }
  await shot(p, '25-shared-result');
  if (await p.locator('.error-card').count()) check('a real prediction finishes through the hosted page', false, (await p.locator('.error-card').innerText()).replace(/\s+/g, ' ').slice(0, 300));
  else {
    const dg = parseFloat((await p.locator('.big-number .n').textContent()).replace('−', '-'));
    check('a real prediction finishes through the hosted page, with a believable ΔG', dg < -6 && dg > -13, `ΔG ${dg} in ${Math.round((Date.now() - t0) / 1000)}s`);
    check('the result is not labelled Demo', (await p.locator('.badge-demo').count()) === 0, '');
    const rows = await p.locator('table.poses tbody tr').count();
    check('the ranked pose table is filled', rows >= 5, `${rows} rows`);
    const dl = await p.locator('a[download], .downloads a, .downloads button').count();
    check('download buttons are there', dl >= 2, `${dl}`);
    const text = await p.locator('body').innerText();
    check('no folder of the computer shows on the result page', !/\/Users\/|\/home\/|\/private\//.test(text), '');
    await p.getByRole('button', { name: 'Expert' }).click(); await sleep(400);
    const body = await p.locator('body').innerText();
    check('even the Expert command and run folder name no folder of the computer', !/\/Users\/|\/home\/|\/private\//.test(body), '');
  }
}
check('every API call went to the shared computer, none to the page’s own site', [...apiOrigins].every((o) => o === new URL(SHARED).origin) && apiOrigins.size >= 1, [...apiOrigins].join(','));
check('no uncaught page errors', p.diag.pageErrors.length === 0, p.diag.pageErrors.slice(0, 2).join('|'));

// ---- 3. the computer is off: the page says so and stays a demo ----
const off = await newPage(b);
await off.goto(`${BASE}/?backend=${encodeURIComponent('http://127.0.0.1:9')}`); await off.waitForSelector('.hero-card');
const offHome = await off.locator('.hero-card').innerText();
check('with the computer off, the page says so and shows the demo', /live computer is offline/i.test(offHome) && (await off.locator('.badge-demo').count()) >= 1, offHome.replace(/\s+/g, ' ').slice(0, 100));
await off.getByRole('button', { name: 'Run example' }).click(); await off.waitForSelector('.big-number', { timeout: 60000 });
check('a demo result is still labelled Demo', (await off.locator('.badge-demo').count()) >= 1, '');

// ---- 4. a page can only be pointed at a trusted computer ----
const evil = await newPage(b); const sent = [];
evil.on('request', (r) => { if (new URL(r.url()).hostname === 'evil.example') sent.push(r.url()); });
await evil.goto(`${BASE}/?backend=${encodeURIComponent('https://evil.example')}`); await evil.waitForSelector('.hero-card');
check('a link that names some other computer is ignored: demo, nothing sent to it', (await evil.locator('.badge-demo').count()) >= 1 && sent.length === 0, sent.join(','));

// ---- 5. a computer that is not a shared server is never used ----
const notShared = await newPage(b);
await notShared.goto(`${BASE}/?backend=${encodeURIComponent(process.env.UNSHARED || 'http://127.0.0.1:8000')}`); await notShared.waitForSelector('.hero-card');
if (process.env.UNSHARED !== 'skip') check('a server that is not in public mode is refused (it would accept any file path)', (await notShared.locator('.badge-demo').count()) >= 1, '');
await b.close(); finish('25_shared_computer');
