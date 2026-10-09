// Demo mode end to end: every kind of run (Dock example, Dock from the 4 steps, Compare, Score) with the simulated adapter.
// Works against any server or the static demo site: `?demo` forces the Demo adapter. Regression guard: the demo adapter
// once threw "jobId is not defined" after a change meant for the live adapter, and nothing else in the suites ran it.
import { browser, newPage, check, finish, shot, BASE, MDM2, POSE, sleep } from './lib.mjs';
import { setupDock } from './journeys.mjs';

const b = await browser();
const demoUrl = (hash) => `${BASE}/?demo${hash}`;

// ---- the one-click example
let p = await newPage(b);
await p.goto(demoUrl('#/')); await p.waitForSelector('.hero-card');
check('demo: the page is labelled Demo', (await p.locator('.topbar .badge-demo').count()) >= 1, '');
await p.getByRole('button', { name: 'Run example' }).click(); await p.waitForSelector('.big-number, .error-card', { timeout: 60000 });
check('demo: Run example finishes with a result (no error card)', (await p.locator('.big-number').count()) === 1 && (await p.locator('.error-card').count()) === 0, (await p.locator('.error-card').allTextContents()).join(' ').slice(0, 200));
check('demo: the result says Demo', (await p.locator('.badge-demo').count()) >= 1, '');
check('demo: the ranked list is filled and rows can be chosen', (await p.locator('table.poses tbody tr').count()) >= 3, '');
await p.locator('table.poses tbody tr[data-rank="2"]').click(); await sleep(500);
check('demo: choosing a pose shows it', /Pose 2 of/.test(await p.locator('.pose-label').textContent()), '');
const [d] = await Promise.all([p.waitForEvent('download'), p.getByRole('button', { name: /Best pose/ }).click()]);
check('demo: downloads are named *_DEMO so they cannot pass for real output', /DEMO/.test(d.suggestedFilename()), d.suggestedFilename());
check('demo: no page errors on the Dock example', p.diag.pageErrors.length === 0, p.diag.pageErrors.join('|'));
await p.context().close();

// ---- the four setup steps, with Expert options
p = await newPage(b);
await p.goto(demoUrl('#/')); await p.waitForSelector('.hero-card');
await p.getByRole('button', { name: 'New prediction' }).click(); await p.waitForSelector('.pick');
await p.locator('.pick', { has: p.locator('.mono', { hasText: 'PDB 1YCR' }) }).last().click(); await p.getByRole('button', { name: 'Continue' }).click();
await p.waitForSelector('#pep-input'); await p.locator('#pep-input').fill('ETFSDLWKLLPE'); await p.getByRole('button', { name: 'Continue' }).click();
await p.waitForSelector('.choice'); await sleep(500); await p.getByRole('button', { name: /Use the suggested site/ }).click(); await p.getByRole('button', { name: 'Continue' }).click(); await p.waitForSelector('.summary');
await p.getByRole('radio', { name: /Quick/ }).click();
await p.getByRole('button', { name: /Run prediction/ }).click(); await p.waitForSelector('.big-number, .error-card', { timeout: 60000 });
check('demo: a Dock from the four steps finishes', (await p.locator('.big-number').count()) === 1, (await p.locator('.error-card').allTextContents()).join(' ').slice(0, 200));
check('demo: no page errors on the four-step Dock', p.diag.pageErrors.length === 0, p.diag.pageErrors.join('|'));
await p.context().close();

// ---- Compare
p = await newPage(b);
await p.goto(demoUrl('#/compare')); await p.waitForSelector('.pcard'); await sleep(1200);
await p.getByRole('radio', { name: /Quick/ }).click();
await p.getByRole('button', { name: /^Compare/ }).last().click(); await p.waitForSelector('.big-number, .error-card', { timeout: 60000 });
check('demo: a Compare finishes with ΔΔG, an interval and a verdict', (await p.locator('.big-number').count()) === 1 && (await p.locator('.verdict').count()) === 1 && (await p.locator('.ddg-bar').count()) === 1, (await p.locator('.error-card').allTextContents()).join(' ').slice(0, 200));
check('demo: no page errors on Compare', p.diag.pageErrors.length === 0, p.diag.pageErrors.join('|'));
await p.context().close();

// ---- Score
p = await newPage(b);
await p.goto(demoUrl('#/score')); await p.waitForSelector('.drop');
const f = p.locator('input[type=file]'); await f.nth(0).setInputFiles(MDM2); await p.waitForSelector('.msg.ok'); await f.nth(1).setInputFiles(POSE);
await p.getByRole('button', { name: /Score it/ }).click(); await p.waitForSelector('.big-number, .error-card', { timeout: 60000 });
check('demo: a Score finishes with a number', (await p.locator('.big-number').count()) === 1, (await p.locator('.error-card').allTextContents()).join(' ').slice(0, 200));
await shot(p, '14-demo-score');
check('demo: no page errors on Score', p.diag.pageErrors.length === 0, p.diag.pageErrors.join('|'));
await p.context().close();

// ---- the runner's own guards in demo too: a run in progress cannot be doubled
p = await newPage(b);
await p.goto(demoUrl('#/')); await p.waitForSelector('.hero-card');
await p.getByRole('button', { name: 'Run example' }).click(); await p.waitForSelector('.track');
await p.evaluate(() => window.hybridock.runner.start('dock', window.hybridock.store.get().run.job));
await sleep(300);
check('demo: starting a second run during a run is refused with a message', /already going/.test((await p.locator('.toast').allTextContents()).join(' ')), '');
await p.waitForSelector('.big-number', { timeout: 60000 });
check('demo: the first run still finishes normally', (await p.locator('.big-number').count()) === 1, '');
await p.context().close();
await b.close(); finish('14_demo_runs');
