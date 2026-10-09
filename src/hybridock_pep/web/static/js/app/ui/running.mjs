// running.js — a calm full-width progress track while a run is going. Never a blank spinner.
// The floating molecule behind it is decoration only; the stages and bar are what the run reports.

import { h, mmss } from './dom.mjs';
import { icon } from './icons.mjs';
import { loadProtein } from '../structures.mjs';
import { RUN_STAGES } from '../config.mjs';
import { adapter } from '../adapter.mjs';

const COPY = {
  dock: { title: 'Finding how your peptide binds', sub: (j) => `${j.peptide} → ${j.protein.name} · ${j.poses} poses${j.blind ? ' · searching the whole protein' : ''}` },
  compare: { title: 'Comparing the two proteins', sub: (j) => `${j.peptide}: ${j.target.protein.name} vs ${j.offTarget.protein.name}` },
  score: { title: 'Scoring your structure', sub: (j) => `${j.peptide} on ${j.protein.name}` },
};

export function mountRunning(ctx) {
  const { store, stage, go, runner } = ctx;
  let alive = true, timer = null;
  const run = store.get().run;

  if (!run.job) { queueMicrotask(() => go('/')); return { el: h('div'), destroy() {} }; }

  const { kind, job } = run;
  const copy = COPY[kind];
  const live = h('p', { class: 'visually-hidden', 'aria-live': 'polite' });
  const bar = h('i');
  const barWrap = h('div', { class: 'bar busy', role: 'progressbar', 'aria-label': 'Run progress', 'aria-valuemin': '0', 'aria-valuemax': '100' }, bar);
  const steps = RUN_STAGES.map((s) => h('li', {}, s.label, h('span', { class: 'tech' }, s.tech)));
  const etaEl = h('b', { class: 'nums' }), elapsedEl = h('b', { class: 'nums' });
  const counterEl = h('span', { class: 'small muted nums' });
  const lateEl = h('p', { class: 'small muted', hidden: true }, 'This is taking longer than the estimate. That is normal on a machine without a GPU, and the run is still going. You can stop it at any time.');
  const logEl = h('pre', { class: 'cmd', tabindex: '0', 'aria-label': 'Live log', style: { maxHeight: '220px', overflow: 'auto', margin: 0, paddingRight: '16px' } });
  let logLines = [], lastProgress = null;
  const stopBtn = h('button', { class: 'btn', type: 'button', onClick: () => runner.stop() }, icon('stop', 16), 'Stop');
  const body = h('div', { class: 'glass run-panel' });
  const el = h('section', { class: 'screen running', 'aria-labelledby': 'run-title' },
    h('div', { class: 'stage-slot run-slot', 'data-stage-slot': '' }), body, live);

  let lastStage = -1, showingError = false;
  function startTimer() {
    clearInterval(timer);
    timer = setInterval(() => { elapsedEl.textContent = mmss((Date.now() - store.get().run.startedAt) / 1000); }, 500);
    elapsedEl.textContent = mmss((Date.now() - store.get().run.startedAt) / 1000);
  }
  function paint() {
    const r = store.get().run;
    if (!alive || r.status === 'idle') return; // the runner is about to navigate away
    if (r.status === 'error') { showingError = true; paintError(r); return; }
    if (showingError) { // "Try again" restarted the run on this same screen: bring the progress view back
      showingError = false;
      logLines = []; logEl.textContent = ''; lastProgress = null; lastStage = -1;
      paintBody(); startTimer(); stage.setPeptideMode('tumble');
    }
    const { stageIndex, fraction, etaSeconds, counter, lines, overdue } = r.progress;
    steps.forEach((li, i) => { li.className = i < stageIndex ? 'done' : i === stageIndex ? 'active' : ''; li.toggleAttribute('aria-current', i === stageIndex); });
    bar.style.width = `${Math.round(fraction * 100)}%`;
    barWrap.setAttribute('aria-valuenow', String(Math.round(fraction * 100)));
    counterEl.textContent = counter ? `${RUN_STAGES[stageIndex].label}: ${counter}` : '';
    if (lines?.length && r.progress !== lastProgress) { // (paint also runs on unrelated store changes: take each batch once)
      lastProgress = r.progress;
      // Expert mode shows the server's own log; keep the last few hundred lines
      logLines = logLines.concat(lines).slice(-300);
      const pinned = logEl.scrollTop + logEl.clientHeight >= logEl.scrollHeight - 8;
      logEl.textContent = logLines.join('\n');
      if (pinned) logEl.scrollTop = logEl.scrollHeight;
    }
    etaEl.textContent = etaSeconds == null ? 'Still working…' : etaSeconds <= 1 ? 'Almost done' : `about ${mmss(etaSeconds)}`;
    lateEl.hidden = !overdue;
    if (stageIndex !== lastStage) { lastStage = stageIndex; live.textContent = `Step ${stageIndex + 1} of ${RUN_STAGES.length}: ${RUN_STAGES[stageIndex].label}`; }
  }

  function paintBody() {
    body.replaceChildren(
      h('div', { class: 'row', style: { justifyContent: 'space-between' } },
        h('div', {}, h('h1', { id: 'run-title' }, copy.title), h('p', { class: 'muted', style: { marginTop: '6px' } }, h('span', { class: 'mono' }, copy.sub(job)))),
        adapter.kind === 'demo' && h('span', { class: 'badge-demo' }, 'Demo')),
      h('ol', { class: 'track', 'aria-label': 'Stages' }, steps),
      barWrap,
      counterEl,
      lateEl,
      h('div', { class: 'run-meta' },
        h('div', { class: 'times' }, h('div', {}, h('span', { class: 'small muted' }, 'Estimated time left'), etaEl), h('div', {}, h('span', { class: 'small muted' }, 'Time so far'), elapsedEl)),
        stopBtn),
      h('details', { class: 'adv expert-only' }, h('summary', {}, h('span', {}, 'Live log ', h('span', { class: 'tech' }, 'What the program is printing'))), h('div', { class: 'stack' }, logEl)),
      h('p', { class: 'small muted' },
        adapter.kind === 'demo' ? 'Demo run: the progress and the result are simulated, and short on purpose. ' : 'A real run can take several minutes. You can leave this tab open. ',
        'The floating molecule is just decoration; the stages above show where the run really is.'));
  }

  function paintError(r) {
    clearInterval(timer);
    body.replaceChildren(h('div', { class: 'error-card', role: 'alert' },
      h('h2', {}, r.error.title),
      h('p', {}, r.error.body),
      r.error.detail && h('details', { class: 'adv' }, h('summary', {}, 'Technical details'), h('pre', { class: 'cmd', tabindex: '0', 'aria-label': 'Technical details' }, r.error.detail)),
      h('div', { class: 'row' },
        h('button', { class: 'btn primary', type: 'button', onClick: () => runner.start(kind, job, { backTo: r.backTo }) }, 'Try again'),
        h('button', { class: 'btn', type: 'button', onClick: () => { store.set({ run: { status: 'idle' } }); go(r.backTo || '/'); } }, 'Go back'))));
    stage.setPeptideMode('float');
  }

  paintBody();
  paint();
  startTimer();
  const unsub = store.subscribe(paint);

  // backdrop: the chosen protein, with the peptide tumbling around it
  const proteinRef = kind === 'compare' ? job.target.protein : job.protein;
  stage.setPeptide(job.peptide.length);
  stage.setPeptideMode('tumble');
  loadProtein(proteinRef).then(({ structure }) => { if (alive) stage.setProtein(structure); }).catch(() => {});

  return { el, destroy() { alive = false; clearInterval(timer); unsub(); } };
}
