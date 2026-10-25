// home.mjs — the front door. Plain words on the left for someone who has never heard of a peptide, the project's GitHub
// link as the main button, and on the right a carousel of diseases the team has already worked on, with finished results
// ready to open and a button to run any of them again live.

import { fill, h } from './dom.mjs';
import { icon } from './icons.mjs';
import { fmt, fmtSigned } from '../interpret.mjs';
import { EXAMPLE, EXPERT_DEFAULTS, GITHUB_URL, TIMING, backend, timingFor } from '../config.mjs';
import { dockJob, findProtein } from '../jobs.mjs';
import { freshCompare, freshScore, freshSetup } from '../state.mjs';
import { adapter, adapterFor, demoAdapter, firstRunPending, noGpu } from '../adapter.mjs';
import { fmtDate, fmtDuration, saveBlob } from './dom.mjs';
import { openHistory } from './history.mjs';
import { toast } from './toast.mjs';
import { createCatalog, etaLabel } from './catalog.mjs';

export function mountHome(ctx) {
  const { store, stage, go, runner, catalog } = ctx;
  let alive = true;
  const startNew = (reset, path) => () => { store.set(reset()); go(path); };

  // Recent predictions: the last four runs, each reopenable. Runs are saved automatically in this browser.
  const recentList = h('ul', { class: 'recent-list' });
  const recentSection = h('section', { class: 'recent', 'aria-labelledby': 'recent-title', hidden: true },
    h('div', { class: 'section-head' },
      h('h2', { id: 'recent-title' }, 'Your recent runs'),
      h('button', { class: 'btn sm ghost', type: 'button', onClick: () => openHistory(ctx) }, 'See all')),
    recentList);

  async function downloadRanked(entry) {
    try {
      let result = entry.result;
      if (!result && entry.demo && entry.job) result = await demoAdapter.runDock(entry.job, { instant: true });
      if (!result?.poses?.length) throw new Error('This run’s pose list wasn’t saved, so there is nothing to download.');
      const file = await adapterFor(result).download(result, 'ranked_csv');
      saveBlob(file.blob, file.filename);
    } catch (err) { toast(err.message, 6000); }
  }

  function renderRecent(hist) {
    recentSection.hidden = !hist.length;
    fill(recentList, ...hist.slice(0, 4).map((e) => h('li', { class: 'recent-card' },
      h('button', { class: 'recent-open', type: 'button', onClick: () => go(`/results/${e.id}`) },
        h('span', { class: 'name' }, e.name),
        h('span', { class: 'when' }, fmtDate(e.createdAt), e.demo && ' ', e.demo && h('span', { class: 'badge-demo' }, 'Demo')),
        h('span', { class: 'val nums' }, e.headline.label === 'ΔΔG' ? fmtSigned(e.headline.value) : fmt(e.headline.value), h('small', {}, `${e.headline.label} kcal/mol`))),
      e.kind === 'dock' && !e.result?.preloaded && h('button', { class: 'btn sm ghost recent-dl', type: 'button', 'aria-label': `Download the ranked list for ${e.name}`, onClick: () => downloadRanked(e) }, 'Download'))));
  }

  // ---- the one-click live run: the p53 peptide on MDM2 at 5 poses, the fastest real thing the page can show ----------
  const live = adapter.kind === 'live';
  const proteins = store.get().proteins;
  const exProtein = findProtein(proteins, EXAMPLE.proteinKey);
  const exSeq = EXAMPLE.peptide;
  const exJob = () => dockJob({
    proteinRef: exProtein, peptide: exSeq, siteMode: 'known', site: exProtein.site, box: exProtein.box,
    thorough: 'quick', expert: { ...EXPERT_DEFAULTS },
  });
  const exTime = h('span', { class: 'btn-sub' }, etaLabel(5, exSeq.length));
  let exExact = null; // the server's figure for this very run, which includes how long it really took last time
  function runExample() {
    if (!exProtein) return;
    runner.start('dock', { ...exJob(), estimateSeconds: exExact ?? timingFor(5, exSeq.length) }, { backTo: '/' });
  }
  if (live && adapter.env?.timing && exProtein) {
    adapter.preview(exJob()).then(({ estimateSeconds }) => {
      if (!alive || estimateSeconds == null) return;
      exExact = estimateSeconds; exTime.textContent = `about ${fmtDuration(estimateSeconds)}`;
    }).catch(() => {});
  }

  // A live server that is missing something: say exactly what and how to fix it, instead of letting the first run fail.
  const MISSING = ['cli', 'vina', 'receptor_prep', 'scorer', 'models'];
  const broken = live && adapter.env ? MISSING.map((k) => [k, adapter.env.checks?.[k]]).filter(([, c]) => c && !c.ok) : [];
  const setupNotice = broken.length ? h('div', { class: 'notice', role: 'status' },
    h('b', {}, 'Setup isn’t finished on this computer.'),
    h('ul', {}, broken.map(([, c]) => h('li', {}, c.detail, ' is missing. Fix: ', h('code', { class: 'mono' }, c.fix)))),
    h('a', { class: 'tap', href: '#/guide/troubleshooting' }, 'Open the install guide')) : null;
  const offlineNotice = live && !adapter.env ? h('div', { class: 'notice', role: 'alert' },
    h('b', {}, 'The lab computer isn’t answering. '), 'It may still be starting. Wait a few seconds, then ',
    h('button', { class: 'btn sm', type: 'button', onClick: () => location.reload() }, 'try again'), '.') : null;

  const where = live
    ? ''
    : (backend.offline ? 'Our lab computer is offline for a moment, so new runs here are simulated and marked Demo. The results in the carousel are real.' : 'New runs in this preview are simulated and marked Demo. The results in the carousel are real.');

  // Only a computer nobody has measured, or one that has not run a prediction yet, has anything to add: the demo computer has neither.
  const measured = TIMING.seconds[5] != null;
  const extra = [
    live && noGpu(adapter.env) && !measured ? 'This computer has no GPU, so a run usually takes tens of minutes.' : null,
    live && firstRunPending(adapter.env) ? 'The first prediction on this computer also downloads about 2.5 GB of model files, once.' : null,
  ].filter(Boolean);

  const slotEl = h('div', { class: 'stage-slot catalog-slot', 'data-stage-slot': '' });
  const carousel = catalog && proteins.length ? createCatalog({ ctx, proteins, catalog, slotEl }) : null;

  const el = h('section', { class: 'screen home', 'aria-labelledby': 'home-title' },
    h('article', { class: 'hero-card' },
      h('div', { class: 'hero-copy' },
        h('h1', { class: 'display home-title', id: 'home-title' }, 'Helping doctors find new cures, faster.'),
        h('p', { class: 'lede' }, 'Every medicine and vaccine starts by finding the one small molecule that can grab onto what is making people sick. Finding it used to take scientists years. We find the strongest matches on a computer, in minutes.'),
        h('div', { class: 'row cta-row' },
          h('button', { class: 'btn ghost lg', type: 'button', onClick: runExample }, icon('play', 16),
            live ? 'Watch a live run' : 'Watch a demo run', exTime)),
        where || extra.length ? h('p', { class: 'small muted where' }, where, where && extra.length ? ' ' : '', extra.join(' ')) : null,
        offlineNotice,
        setupNotice),
      carousel ? carousel.el : slotEl),

    h('section', { class: 'why', 'aria-labelledby': 'why-title' },
      h('h2', { id: 'why-title' }, 'How this helps doctors'),
      h('ol', { class: 'why-steps' },
        h('li', {}, h('b', {}, 'Every illness has a culprit.'), ' Most diseases trace back to one protein in the body that misbehaves.'),
        h('li', {}, h('b', {}, 'A medicine has to grab it.'), ' A short chain called a peptide can latch onto the culprit and switch it off, like a key in a lock.'),
        h('li', {}, h('b', {}, 'We find the best key fast.'), ' Testing candidates in a lab takes months. We rank them on a computer first, so scientists start from the strongest.'))),

    recentSection,

    h('div', { class: 'section-head' }, h('h2', {}, 'Try it yourself')),
    h('div', { class: 'start-row' },
      startCard('Test a peptide', 'Pick a protein, type a peptide, and see how hard it grips.', 'Start', startNew(() => ({ setup: freshSetup() }), '/predict'), true),
      startCard('Compare two proteins', 'See which of two proteins a peptide prefers.', 'Compare', startNew(() => ({ compare: freshCompare() }), '/compare')),
      startCard('Score a structure', 'Already have a bound pose? Rate it directly.', 'Score', startNew(() => ({ score: freshScore() }), '/score'))),
  );

  // Each card carries a real button (that is what people look for); a click anywhere else on the card presses it too.
  function startCard(title, text, buttonLabel, onClick, primary = false) {
    const btn = h('button', { class: 'btn', type: 'button', onClick }, buttonLabel, icon('chevron', 16));
    return h('article', { class: `start-card${primary ? ' dark' : ''}`, onClick: (e) => { if (!e.target.closest('button')) btn.click(); } },
      h('h3', {}, title), h('p', {}, text), btn);
  }

  const unsub = store.subscribe((s) => renderRecent(s.history || []));
  renderRecent(store.get().history || []);

  // the floating model: the carousel sets the protein; until it does, a peptide drifts beside the example
  stage.setDockPose(null);
  stage.setPeptideMode('float');
  if (!carousel) stage.setPeptide(exSeq.length);

  return { el, destroy() { alive = false; unsub(); carousel?.destroy(); } };
}
