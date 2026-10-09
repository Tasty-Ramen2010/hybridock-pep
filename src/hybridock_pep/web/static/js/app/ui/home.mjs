// home.js — the dashboard. The "Try an example" card is the hero: the floating protein sits right on it.

import { h } from './dom.mjs';
import { icon } from './icons.mjs';
import { loadProtein } from '../structures.mjs';
import { fmt, fmtSigned } from '../interpret.mjs';
import { EXAMPLE, EXPERT_DEFAULTS, TYPICAL_ERROR } from '../config.mjs';
import { dockJob, findProtein, greeting } from '../jobs.mjs';
import { freshCompare, freshScore, freshSetup } from '../state.mjs';
import { adapter, adapterFor, demoAdapter } from '../adapter.mjs';
import { fmtDate, fmtDuration, saveBlob } from './dom.mjs';
import { openHistory } from './history.mjs';
import { toast } from './toast.mjs';

export function mountHome(ctx) {
  const { store, stage, go, runner } = ctx;
  let alive = true;

  const nameEl = h('span');
  const stats = [
    { label: 'Predictions run', tech: null, ref: h('span') },
    { label: 'Proteins targeted', tech: null, ref: h('span') },
    { label: 'Best binding strength', tech: 'ΔG, kcal/mol', ref: h('span') },
    { label: 'Typical error', tech: 'of a single ΔG', ref: h('span') },
  ];
  const statsNote = h('p', { class: 'stats-note' });

  const startNew = (reset, path) => () => { store.set(reset()); go(path); };

  // Recent predictions: the last four runs, each reopenable and (for a dock run) downloadable.
  // Runs are saved automatically in this browser; "See all" opens the full History drawer.
  const recentList = h('ul', { class: 'recent-list' });
  const recentSection = h('section', { class: 'recent', 'aria-labelledby': 'recent-title', hidden: true },
    h('div', { class: 'section-head' },
      h('h2', { class: 'eyebrow', id: 'recent-title' }, 'Recent predictions'),
      h('button', { class: 'btn sm ghost', type: 'button', onClick: () => openHistory(ctx) }, icon('history', 15), 'See all')),
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
    recentList.replaceChildren(...hist.slice(0, 4).map((e) => h('li', { class: 'recent-card' },
      h('button', { class: 'recent-open', type: 'button', onClick: () => go(`/results/${e.id}`) },
        h('span', { class: 'name' }, e.name),
        h('span', { class: 'when' }, fmtDate(e.createdAt), e.demo && ' · ', e.demo && h('span', { class: 'badge-demo' }, 'Demo')),
        h('span', { class: 'val nums' }, e.headline.label === 'ΔΔG' ? fmtSigned(e.headline.value) : fmt(e.headline.value), h('small', {}, `${e.headline.label} kcal/mol`))),
      e.kind === 'dock' && h('button', { class: 'btn sm ghost recent-dl', type: 'button', 'aria-label': `Download the ranked list for ${e.name}`, onClick: () => downloadRanked(e) }, icon('download', 14), 'Download'))));
  }

  // The example: the spec's Tau example in demo mode; with the live server, its own validated example (known binder).
  const live = adapter.kind === 'live';
  const proteins = store.get().proteins;
  const backendEx = live ? adapter.examples?.[0] : null;
  const exProtein = (backendEx && proteins.find((p) => p.backendExample === backendEx.id)) || findProtein(proteins, EXAMPLE.proteinKey);
  const exPeptide = backendEx && exProtein?.backendExample ? backendEx.peptide : EXAMPLE.peptide;
  const exJob = () => dockJob({
    proteinRef: exProtein, peptide: exPeptide, siteMode: 'known', site: exProtein.site, box: exProtein.box,
    thorough: 'quick', expert: { ...EXPERT_DEFAULTS },
  });
  const timeNote = h('span', { class: 'small muted' }, live ? 'A real quick run (25 poses)' : 'Takes about 7 seconds in demo mode');

  let exEstimate = null;
  function runExample() {
    if (!exProtein) return;
    runner.start('dock', { ...exJob(), estimateSeconds: exEstimate }, { backTo: '/' });
  }
  if (live && exProtein) {
    adapter.preview(exJob()).then(({ estimateSeconds }) => {
      exEstimate = estimateSeconds;
      if (estimateSeconds) timeNote.textContent = `A real quick run (25 poses): roughly ${fmtDuration(estimateSeconds)} (estimate)`;
    }).catch(() => {});
  }

  const el = h('section', { class: 'screen home', 'aria-labelledby': 'greeting' },
    h('div', { class: 'home-head' },
      h('div', {},
        h('h1', { class: 'display', id: 'greeting' }, `${greeting()}, `, nameEl),
        h('p', { class: 'lede' }, 'Let’s find out how tightly a peptide sticks. Pick a protein, type a sequence, and watch it dock.')),
      h('button', { class: 'btn primary', type: 'button', onClick: startNew(() => ({ setup: freshSetup() }), '/predict') }, icon('plus', 18), 'New prediction')),

    h('div', { class: 'stats', role: 'list' },
      stats.map((s) => h('article', { class: 'card stat', role: 'listitem' },
        h('div', { class: 'label' }, s.label, s.tech && h('span', { class: 'tech' }, s.tech)),
        h('div', { class: 'value nums' }, s.ref))),
      statsNote),

    h('article', { class: 'hero-card', 'aria-labelledby': 'hero-title' },
      h('div', { class: 'hero-copy' },
        live
          ? h('div', { class: 'row' }, h('span', { class: 'chip' }, 'Known binder'), h('span', { class: 'small muted' }, backendEx?.expect ? `Expect ${backendEx.expect}` : 'A real run'))
          : h('div', { class: 'row' }, h('span', { class: 'badge-demo' }, 'Demo'), h('span', { class: 'small muted' }, 'Simulated result')),
        h('h2', { id: 'hero-title' }, 'Try an example prediction'),
        live && backendEx
          ? h('p', { class: 'muted' }, 'See a real result in one click: ', h('b', {}, backendEx.name), '. ', backendEx.blurb, ' The peptide is ', h('b', { class: 'mono pep' }, exPeptide), '.')
          : h('p', { class: 'muted' }, 'See a full result in one click: the peptide ', h('b', { class: 'mono pep' }, EXAMPLE.peptide), ' docked against the Tau VQIVYK stretch, a piece of the protein that clumps in Alzheimer’s disease.'),
        h('div', { class: 'row' },
          h('button', { class: 'btn primary', type: 'button', onClick: runExample }, 'Run example', icon('arrow', 18)),
          timeNote)),
      h('div', { class: 'hero-slot stage-slot', 'data-stage-slot': '' })),

    recentSection,

    h('div', { class: 'section-head' }, h('h2', { class: 'eyebrow' }, 'Start something new')),
    h('div', { class: 'start-row' },
      startCard('Predict binding', 'Dock a peptide to a protein and get a binding strength.', 'Docking + ΔG', 'zap', startNew(() => ({ setup: freshSetup() }), '/predict')),
      startCard('Compare two proteins', 'See which of two proteins a peptide prefers.', 'Selectivity, ΔΔG', 'columns', startNew(() => ({ compare: freshCompare() }), '/compare')),
      startCard('Score a structure', 'Already have a bound pose? Score it directly.', 'Score an existing pose', 'badge', startNew(() => ({ score: freshScore() }), '/score'))),
  );

  function startCard(title, text, tech, ic, onClick) {
    return h('button', { class: 'card start-card', type: 'button', onClick },
      h('span', { class: 'icon-tile', 'aria-hidden': 'true' }, icon(ic, 20)),
      h('h3', {}, title), h('p', {}, text), h('span', { class: 'tech' }, tech),
      h('span', { class: 'go' }, 'Open', icon('arrow', 16)));
  }

  function update() {
    const s = store.get(), hist = s.history || [];
    nameEl.textContent = s.userName;
    const dgs = hist.filter((e) => e.kind !== 'compare').map((e) => e.headline.value);
    stats[0].ref.textContent = String(hist.length);
    stats[1].ref.textContent = String(new Set(hist.flatMap((e) => e.proteinNames)).size);
    stats[2].ref.textContent = dgs.length ? fmt(Math.min(...dgs)) : '—';
    stats[3].ref.replaceChildren(`±${TYPICAL_ERROR}`, h('small', {}, 'kcal/mol'));
    statsNote.textContent = hist.some((e) => e.demo) ? 'These numbers include simulated Demo runs.' : '';
    renderRecent(hist);
  }
  const unsub = store.subscribe(update);
  update();

  // the floating model: Tau with the example peptide drifting beside it
  stage.setPeptide(exPeptide.length);
  stage.setDockPose(null);
  stage.setPeptideMode('float');
  const tau = exProtein;
  if (tau) loadProtein(tau).then(({ structure }) => { if (alive) stage.setProtein(structure); }).catch(() => {});

  return { el, destroy() { alive = false; unsub(); } };
}
