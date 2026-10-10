// What the app says for each kind of computer, by replaying /api/env with different answers (the page cannot tell the
// difference). Needs a LIVE server (it skips on a demo one). No run is started.
import { browser, newPage, check, finish, BASE, sleep } from './lib.mjs';

const b = await browser();
const probe = await newPage(b);
await probe.goto(BASE + '/'); await probe.waitForSelector('.hero-card');
if ((await probe.evaluate(() => window.hybridock?.adapter?.kind)) !== 'live') { console.log('SKIP 24_env_states: needs a live server'); await b.close(); process.exit(0); }
const real = await probe.evaluate(() => fetch('/api/env').then((r) => r.json()));

/** Open Home as a computer whose /api/env answers `patch(env)`. */
async function asMachine(patch, path = '/') {
  const p = await newPage(b);
  await p.route('**/api/env', async (route) => { const env = JSON.parse(JSON.stringify(real)); patch(env); await route.fulfill({ json: env }); });
  await p.goto(BASE + path); await p.waitForSelector('.hero-card, .steps');
  return p;
}
const chip = (p) => p.locator('.topbar .chip').first().textContent().then((t) => t.trim());

// ---- the status chip names the hardware ----
let p = await asMachine((e) => { e.checks.gpu = { ok: true, kind: 'metal', detail: 'Apple M3 (Metal)', fix: '' }; });
check('an Apple Silicon Mac shows its chip and Metal, not "CPU only"', /Apple M3 \(Metal\)/.test(await chip(p)) && !/CPU only/.test(await chip(p)), await chip(p));
const home = (await p.locator('.hero-copy').innerText()).replace(/\s+/g, ' ');
check('...and the example does not claim the machine has no GPU', !/no GPU/i.test(home), home.slice(0, 160));
await p.close();

p = await asMachine((e) => { e.checks.gpu = { ok: true, kind: 'cuda', detail: 'NVIDIA GeForce RTX 5070', fix: '' }; });
check('an NVIDIA machine shows the card', /RTX 5070/.test(await chip(p)), await chip(p));
await p.close();

p = await asMachine((e) => { e.checks.gpu = { ok: false, kind: null, detail: 'no GPU detected', fix: 'CPU works' }; });
check('a computer without a GPU says CPU only', /CPU only/.test(await chip(p)), await chip(p));
check('...and warns that a run takes tens of minutes', /no GPU.*tens of minutes/i.test((await p.locator('.hero-copy').innerText()).replace(/\s+/g, ' ')), '');
await p.close();

p = await asMachine((e) => { e.checks.gpu = { ok: true, kind: 'rocm', detail: 'AMD GPU (ROCm)', fix: '' }; });
check('an AMD GPU is recognised, and no countdown is promised for it (we have not measured one)', /AMD GPU/.test(await chip(p)) && !/roughly/i.test(await p.locator('.hero-copy').innerText()), await chip(p));
await p.close();

// ---- first prediction: tell people about the download before it happens ----
p = await asMachine((e) => { e.checks.first_run = { ok: false, detail: 'Model files still to download (first prediction)', fix: 'about 2.5 GB' }; });
check('before the first prediction, Home mentions the 2.5 GB download', /2\.5 GB/.test(await p.locator('.hero-copy').innerText()), '');
await p.getByRole('button', { name: 'Run example' }).click(); await p.waitForSelector('.track, .toast, .error-card', { timeout: 20000 }).catch(() => {});
const running = (await p.locator('.notice').allTextContents()).join(' ');
check('...and so does the running screen, in a notice', /First prediction on this computer/.test(running) && /10 minutes/.test(running), running.slice(0, 120));
await p.close();   // (the example run is refused or started on the real server; either way this page does not wait for it)
await sleep(300);

p = await asMachine((e) => { e.checks.first_run = { ok: true, detail: 'Model files are ready', fix: '' }; });
check('once the models are cached, the first-run note is gone', !/2\.5 GB/.test(await p.locator('.hero-copy').innerText()), '');
await p.close();

// ---- setup that is not finished ----
p = await asMachine((e) => { e.ready = false; e.checks.cli = { ok: false, detail: 'hybridock-pep on PATH', fix: 'conda activate score-env' }; e.checks.vina = { ok: false, detail: 'AutoDock Vina', fix: 'conda install -c conda-forge vina' }; });
const notice = p.locator('.hero-copy .notice');
check('a machine that is missing something says so on Home', (await notice.count()) === 1 && /Setup isn’t finished/.test(await notice.textContent()), '');
const fixes = await notice.locator('code').allTextContents();
check('...names each missing piece with the command that fixes it', fixes.includes('conda activate score-env') && fixes.includes('conda install -c conda-forge vina'), fixes.join(' | '));
check('...and links to the troubleshooting guide', (await notice.locator('a[href="#/guide/troubleshooting"]').count()) === 1, '');
check('the status chip says Setup needed', /Setup needed/.test(await chip(p)), await chip(p));
await p.close();

p = await asMachine(() => {});
check('a healthy machine shows no setup notice', (await p.locator('.hero-copy .notice').count()) === 0, '');
await p.close();

await b.close(); finish('24_env_states');
