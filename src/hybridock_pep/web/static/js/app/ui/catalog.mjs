// catalog.mjs (ui) — the carousel on the home page: one card per protein the team has already worked on, with its best
// peptide matches ready to open, and a button to run it again live. The 3D protein floats above the card and changes as
// the cards slide. Pin any matches with "Compare" and they line up side by side underneath.
//
// The track is a plain scroll-snap row, so swiping, trackpads and the arrow keys all work without any script.

import { fill, h } from './dom.mjs';
import { icon } from './icons.mjs';
import { fmtDuration } from './dom.mjs';
import { fmt } from '../interpret.mjs';
import { THOROUGHNESS, DEFAULT_THOROUGHNESS, EXPERT_DEFAULTS, timingFor } from '../config.mjs';
import { adapter } from '../adapter.mjs';
import { dockJob } from '../jobs.mjs';
import { loadProtein } from '../structures.mjs';
import { namedPeptide } from '../peptides.mjs';
import { historyEntryFor } from '../state.mjs';
import { bindersFor, binderKey, binderResult, strength } from '../catalog.mjs';

const MAX_PINS = 4;
const reduceMotion = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

/** "about 3 minutes" for a run of `poses` poses of a peptide this long, from the measured table. */
export function etaLabel(poses, length) {
  const s = timingFor(poses, length);
  return s == null ? 'a few minutes' : `about ${fmtDuration(s)}`;
}

export function createCatalog({ ctx, proteins, catalog, slotEl }) {
  const { store, stage, go, runner } = ctx;
  let alive = true, active = -1, token = 0;
  const pins = []; // [{ proteinKey, seq }]

  const slides = proteins.map((p) => buildSlide(p));
  const track = h('div', { class: 'catalog-track', role: 'group', 'aria-roledescription': 'carousel', 'aria-label': 'Diseases and their best peptide matches', tabindex: '0' },
    slides.map((s) => s.el));
  const counter = h('span', { class: 'catalog-count nums', 'aria-live': 'polite' });
  const prev = h('button', { class: 'btn icon circle', type: 'button', 'aria-label': 'Previous protein', onClick: () => step(-1) }, icon('chevronL', 18));
  const next = h('button', { class: 'btn icon circle', type: 'button', 'aria-label': 'Next protein', onClick: () => step(1) }, icon('chevron', 18));
  const dots = h('div', { class: 'catalog-dots', role: 'tablist', 'aria-label': 'Choose a protein' },
    proteins.map((p, i) => h('button', { class: 'dot-btn', type: 'button', role: 'tab', tabindex: '-1', 'aria-label': `${p.name}, ${p.disease}`, onClick: () => show(i) })));
  // tabs are a single Tab stop; the arrow keys move between them (the usual tab pattern)
  dots.addEventListener('keydown', (e) => {
    const d = { ArrowRight: 1, ArrowLeft: -1, Home: -99, End: 99 }[e.key];
    if (d == null) return;
    e.preventDefault(); step(d); dots.children[active]?.focus();
  });
  const tray = h('section', { class: 'tray', hidden: true, 'aria-label': 'Side-by-side comparison' });

  const el = h('div', { class: 'catalog' },
    h('div', { class: 'catalog-head' },
      h('div', {}, h('h2', { class: 'catalog-title' }, 'Diseases we’ve worked on'), h('p', { class: 'catalog-sub' }, 'Swipe through, open a result, or pin matches to compare.')),
      h('div', { class: 'catalog-nav' }, prev, counter, next)),
    slotEl,
    track, dots, tray);

  // ---- one card -------------------------------------------------------------------------------------------------
  function buildSlide(p) {
    const list = bindersFor(catalog, p.key);
    const seqs = [...new Set([...list.map((b) => b.seq), ...(p.panel || [])])];
    let thorough = DEFAULT_THOROUGHNESS, seq = list[0]?.seq || seqs[0];

    const rows = list.map((b) => {
      const named = namedPeptide(b.seq);
      const pinBtn = h('button', { class: 'pin', type: 'button', 'aria-pressed': 'false', 'data-key': binderKey(p.key, b.seq), onClick: () => togglePin(p, b) },
        icon('plus', 14), h('span', {}, 'Compare'));
      return h('li', { class: 'binder' },
        h('button', { class: 'binder-open', type: 'button', onClick: () => open(p, b), 'aria-label': `Open the result for ${named?.name || b.seq} on ${p.name}` },
          h('span', { class: 'binder-top' },
            h('span', { class: 'binder-name' }, named?.name || 'Peptide', b.known && h('span', { class: 'chip known' }, 'Known binder')),
            h('span', { class: 'binder-dg nums' }, fmt(b.deltaG), h('small', {}, 'kcal/mol'))),
          h('span', { class: 'binder-seq mono' }, b.seq),
          h('span', { class: 'meter', 'aria-hidden': 'true' }, h('i', { style: { width: `${Math.round(strength(b.deltaG) * 100)}%` } }))),
        pinBtn);
    });

    const etaEl = h('span', { class: 'run-eta' });
    const runBtn = h('button', { class: 'btn primary', type: 'button', onClick: () => runLive(p, seq, thorough, exact) }, icon('play', 14), 'Run it live');
    const seg = h('div', { class: 'seg poses-seg', role: 'radiogroup', 'aria-label': 'Number of poses' },
      THOROUGHNESS.map((t) => h('button', { type: 'button', role: 'radio', 'aria-checked': String(t.id === thorough), onClick: () => { thorough = t.id; paint(); } }, `${t.poses}`)));
    const pick = seqs.length > 1 ? h('select', { class: 'field pep-select', 'aria-label': 'Peptide to run', onChange: (e) => { seq = e.target.value; paint(); } },
      seqs.map((s) => h('option', { value: s, selected: s === seq }, `${namedPeptide(s)?.name || s}`))) : null;
    let askToken = 0, exact = null; // exact: what the server quotes for this peptide at this size (it remembers real run times)
    function paint() {
      seg.querySelectorAll('button').forEach((b, i) => b.setAttribute('aria-checked', String(THOROUGHNESS[i].id === thorough)));
      const poses = THOROUGHNESS.find((t) => t.id === thorough).poses;
      exact = null;
      etaEl.textContent = `${poses} poses · ${etaLabel(poses, seq?.length)}`;
      if (adapter.kind === 'live' && adapter.env?.timing && seq) { // refine the table's figure with the server's, once it answers
        const mine = ++askToken;
        const job = dockJob({ proteinRef: p, peptide: seq, siteMode: 'known', site: p.site, box: p.box, thorough, expert: { ...EXPERT_DEFAULTS } });
        adapter.preview(job).then(({ estimateSeconds }) => {
          if (mine !== askToken || estimateSeconds == null) return;
          exact = estimateSeconds;
          etaEl.textContent = `${poses} poses · about ${fmtDuration(estimateSeconds)}`;
        }).catch(() => {});
      }
    }
    paint();

    const el = h('article', { class: 'slide', 'aria-roledescription': 'slide', 'aria-label': `${p.name}, ${p.disease}` },
      h('header', { class: 'slide-head' },
        h('span', { class: 'chip disease' }, p.disease),
        h('h3', {}, p.name),
        h('p', { class: 'plain' }, p.plain)),
      rows.length
        ? h('div', {}, h('p', { class: 'slide-label' }, rows.length > 1 ? 'Best peptide matches' : 'Best peptide match'), h('ul', { class: 'binders' }, rows))
        : h('p', { class: 'plain', style: { margin: 0 } }, 'Pick a peptide below and run it on our computer.'),
      h('div', { class: 'run-row' },
        h('div', { class: 'run-opts' }, pick, h('div', { class: 'run-poses' }, h('span', { class: 'slide-label' }, 'Poses'), seg)),
        h('div', { class: 'run-go' }, runBtn, etaEl)));
    return { el, protein: p };
  }

  // ---- moving between cards -------------------------------------------------------------------------------------
  const step = (d) => show(Math.min(slides.length - 1, Math.max(0, active + d)));
  function show(i, { instant = false } = {}) {
    const w = track.clientWidth;
    track.scrollTo({ left: i * w, behavior: instant || reduceMotion() ? 'auto' : 'smooth' });
    setActive(i);
  }
  function setActive(i) {
    if (i === active || i < 0 || i >= slides.length) return;
    active = i;
    const my = ++token;
    counter.textContent = `${i + 1} of ${slides.length}`;
    prev.disabled = i === 0;
    next.disabled = i === slides.length - 1;
    [...dots.children].forEach((b, k) => { b.setAttribute('aria-selected', String(k === i)); b.setAttribute('tabindex', k === i ? '0' : '-1'); b.classList.toggle('on', k === i); });
    // only the card in view can be reached with the keyboard or a screen reader; the arrows and dots move between cards
    slides.forEach((sl, k) => { sl.el.inert = k !== i; });
    const p = slides[i].protein;
    stage.setPeptide((bindersFor(catalog, p.key)[0]?.seq || p.examplePeptide || 'ETFSDLWKLLPE').length);
    loadProtein(p).then(({ structure }) => { if (alive && my === token) stage.setProtein(structure); }).catch(() => {});
  }
  let settle = null;
  track.addEventListener('scroll', () => {
    clearTimeout(settle);
    settle = setTimeout(() => setActive(Math.round(track.scrollLeft / (track.clientWidth || 1))), 90);
  }, { passive: true });
  track.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight') { e.preventDefault(); step(1); } else if (e.key === 'ArrowLeft') { e.preventDefault(); step(-1); }
  });
  const onResize = () => track.scrollTo({ left: Math.max(0, active) * track.clientWidth, behavior: 'auto' });
  addEventListener('resize', onResize);

  // ---- open a preloaded result, or run it live -------------------------------------------------------------------
  function open(p, b) {
    const result = binderResult(p, b, catalog);
    store.addHistory(historyEntryFor(result));
    go(`/results/${result.id}`);
  }
  function runLive(p, seq, thoroughId, exact = null) {
    const job = dockJob({
      proteinRef: p, peptide: seq, siteMode: 'known', site: p.site, box: p.box, thorough: thoroughId, expert: { ...EXPERT_DEFAULTS },
    });
    runner.start('dock', { ...job, estimateSeconds: exact ?? timingFor(job.poses, seq.length) }, { backTo: '/' });
  }

  // ---- compare tray ----------------------------------------------------------------------------------------------
  function togglePin(p, b) {
    const i = pins.findIndex((x) => x.proteinKey === p.key && x.seq === b.seq);
    if (i >= 0) pins.splice(i, 1);
    else if (pins.length < MAX_PINS) pins.push({ proteinKey: p.key, seq: b.seq });
    else pins.shift(), pins.push({ proteinKey: p.key, seq: b.seq });
    paintPins();
  }
  function paintPins() {
    el.querySelectorAll('.pin').forEach((btn) => {
      const on = pins.some((x) => binderKey(x.proteinKey, x.seq) === btn.dataset.key);
      btn.setAttribute('aria-pressed', String(on));
      fill(btn, icon(on ? 'check' : 'plus', 14), h('span', {}, on ? 'Added' : 'Compare'));
    });
    const items = pins.map((x) => {
      const p = proteins.find((q) => q.key === x.proteinKey);
      const b = bindersFor(catalog, x.proteinKey).find((y) => y.seq === x.seq);
      return { p, b };
    }).filter((x) => x.b).sort((a, c) => a.b.deltaG - c.b.deltaG);
    tray.hidden = !items.length;
    if (!items.length) return;
    let verdict = 'Pin one more to see how they compare.';
    if (items.length >= 2) {
      const gap = items[1].b.deltaG - items[0].b.deltaG;
      verdict = gap < 0.05
        ? 'These grip about equally well.'
        : `${namedPeptide(items[0].b.seq)?.name || items[0].b.seq} on ${items[0].p.name} grips tighter, by ${fmt(gap, 1)} kcal/mol.`;
    }
    tray.replaceChildren(
      h('div', { class: 'tray-head' }, h('h3', {}, 'Side by side'), h('button', { class: 'btn sm ghost', type: 'button', onClick: () => { pins.length = 0; paintPins(); } }, 'Clear')),
      h('ul', { class: 'tray-rows' }, items.map(({ p, b }, i) => h('li', { class: i === 0 && items.length > 1 ? 'best' : '' },
        h('span', { class: 'tray-name' }, namedPeptide(b.seq)?.name || b.seq, h('small', {}, ` on ${p.name}`)),
        h('span', { class: 'meter wide', 'aria-hidden': 'true' }, h('i', { style: { width: `${Math.round(strength(b.deltaG) * 100)}%` } })),
        h('span', { class: 'tray-dg nums' }, fmt(b.deltaG))))),
      h('p', { class: 'tray-verdict', role: 'status' }, verdict));
  }

  // start on the card for the example protein when the page asks for one; otherwise the first
  queueMicrotask(() => show(0, { instant: true }));
  return {
    el,
    show,
    destroy() { alive = false; removeEventListener('resize', onResize); },
  };
}
