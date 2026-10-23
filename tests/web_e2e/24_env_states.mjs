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
  // start from a healthy machine whatever this one is (a CI runner or a laptop may lack pieces), then apply the case's change
  await p.route('**/api/env', async (route) => {
    const env = JSON.parse(JSON.stringify(real));
    env.ready = true; for (const k of ['cli', 'vina', 'receptor_prep', 'scorer', 'models']) env.checks[k] = { ok: true, detail: k, fix: '' };
    patch(env); await route.fulfill({ json: env });
  });
  await p.route('**/api/run', () => {}); // a run is never really started here: the request just hangs, which keeps the Running screen up
  await p.goto(BASE + path); await p.waitForSelector('.hero-card, .steps');
  return p;
}
const chip = (p) => p.locator('.topbar .chip').first().evaluate((e) => `${(e.querySelector('.chip-text') || e).textContent.trim()} [${e.title}]`);

// ---- the status chip names the hardware ----
let p = await asMachine((e) => { e.checks.gpu = { ok: true, kind: 'metal', detail: 'Apple M3 (Metal)', fix: '' }; });
check('an Apple Silicon Mac says GPU (its chip and Metal in the tooltip), not CPU', /GPU/.test(await chip(p)) && /Apple M3 \(Metal\)/.test(await chip(p)) && !/CPU/.test(await chip(p)), await chip(p));
const home = (await p.locator('.hero-copy').innerText()).replace(/\s+/g, ' ');
check('...and the example does not claim the machine has no GPU', !/no GPU/i.test(home), home.slice(0, 160));
await p.close();

p = await asMachine((e) => { e.checks.gpu = { ok: true, kind: 'cuda', detail: 'NVIDIA GeForce RTX 5070', fix: '' }; });
check('an NVIDIA machine says GPU and names the card', /GPU/.test(await chip(p)) && /RTX 5070/.test(await chip(p)), await chip(p));
await p.close();

p = await asMachine((e) => { e.checks.gpu = { ok: false, kind: null, detail: 'no GPU detected', fix: 'CPU works' }; });
check('a computer without a GPU says CPU', /^CPU/.test(await chip(p)) && /CPU only/.test(await chip(p)), await chip(p));
check('...and warns that a run takes tens of minutes', /no GPU.*tens of minutes/i.test((await p.locator('.hero-copy').innerText()).replace(/\s+/g, ' ')), '');
await p.close();

p = await asMachine((e) => { e.checks.gpu = { ok: true, kind: 'rocm', detail: 'AMD GPU (ROCm)', fix: '' }; });
check('an AMD GPU is recognised, and no countdown is promised for it (we have not measured one)', /AMD GPU/.test(await chip(p)) && !/roughly/i.test(await p.locator('.hero-copy').innerText()), await chip(p));
await p.close();

// ---- first prediction: tell people about the download before it happens ----
p = await asMachine((e) => { e.checks.first_run = { ok: false, detail: 'Model files still to download (first prediction)', fix: 'about 2.5 GB' }; });
check('before the first prediction, Home mentions the 2.5 GB download', /2\.5 GB/.test(await p.locator('.hero-copy').innerText()), '');
await p.getByRole('button', { name: /Watch a (live|demo) run/ }).click(); await p.waitForSelector('.track', { timeout: 20000 });
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

// ---- which mode: a slow or dead server must never turn into pretend results ----
const modeOf = (pg) => pg.evaluate(() => window.hybridock?.adapter?.kind);
{
  const slow = await newPage(b);
  let first = true;
  await slow.route('**/api/env', async (route) => { if (first) { first = false; await sleep(7500); } await route.fulfill({ json: real }); }); // the first answer comes after the 6 s limit
  await slow.goto(BASE + '/'); await slow.waitForSelector('.hero-card', { timeout: 40000 });
  check('a server that answers late (after 6 s) still opens in live mode, not Demo', (await modeOf(slow)) === 'live' && (await slow.locator('.badge-demo').count()) === 0, await modeOf(slow));
  await slow.close();

  const dead = await newPage(b);
  await dead.route('**/api/env', (route) => route.abort('connectionrefused'));
  await dead.goto(BASE + '/'); await dead.waitForSelector('.hero-card', { timeout: 40000 });
  check('a server that does not answer stays live and says "Server not reached"', (await modeOf(dead)) === 'live' && /Server not reached/.test(await chip(dead)), await chip(dead));
  const alert = dead.locator('.hero-copy .notice[role=alert]');
  check('...with a plain "isn’t answering" notice and a Try again button', (await alert.count()) === 1 && /isn’t answering/.test(await alert.textContent()) && (await alert.getByRole('button', { name: 'Try again' }).count()) === 1, '');
  check('...and no "Demo" label anywhere', (await dead.locator('.badge-demo').count()) === 0, '');
  await dead.close();

  const plain = await newPage(b);
  await plain.route('**/api/env', (route) => route.fulfill({ status: 404, body: 'not found' }));
  await plain.goto(BASE + '/'); await plain.waitForSelector('.hero-card');
  check('a plain web host (no API: 404) is Demo, and says so', (await modeOf(plain)) === 'demo' && (await plain.locator('.badge-demo').count()) >= 1, await modeOf(plain));
  await plain.close();
}

await b.close(); finish('24_env_states');
