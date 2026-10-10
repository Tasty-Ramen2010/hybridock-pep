// results.js — a finished run. One large number, what it means, the 3D viewer, the ranked poses, downloads.
// Handles Predict binding ("dock") and Score a structure here; Compare has its own view in compare.js.

import { h, saveBlob } from './dom.mjs';
import { icon } from './icons.mjs';
import { compareResultView } from './compare.mjs';
import { loadProtein } from '../structures.mjs';
import { demoAdapter, adapterFor } from '../adapter.mjs';
import { TYPICAL_ERROR } from '../config.mjs';
import { peptideStats } from '../peptide.mjs';
import { errorFoldChange, fmt, fmtKd, kdFromDG, meaningOf, roughGuide } from '../interpret.mjs';
import { toast } from './toast.mjs';
import { freshSetup, freshScore } from '../state.mjs';

/** The big ΔG number block shared by Predict and Score. */
export function bigNumber(result, { label = 'Binding strength' } = {}) {
  const dg = result.deltaG;
  const guide = roughGuide(dg);
  return h('div', { class: 'stack', style: { display: 'flex', flexDirection: 'column', gap: '10px' } },
    h('div', { class: 'result-head' },
      h('span', { class: 'eyebrow' }, label, h('span', { class: 'tech' }, 'ΔG')),
      result.demo && h('span', { class: 'badge-demo', title: 'This result is simulated' }, 'Demo')),
    h('div', { class: 'big-number', 'aria-label': `${fmt(dg)} kilocalories per mole` }, h('span', { class: 'n nums' }, fmt(dg)), h('span', { class: 'unit' }, 'kcal/mol')),
    h('p', { class: 'err-line' }, 'Typical error ', h('b', {}, `±${TYPICAL_ERROR} kcal/mol`)),
    h('p', {}, meaningOf(dg)),
    h('p', { class: 'guide-line' }, h('span', { class: 'chip' }, guide.label), 'Our own rough guide, not a result from the program.'),
    Math.abs(peptideStats(result.peptide).charge) >= 2 && h('p', { class: 'small muted' }, `This peptide carries charge (net ${peptideStats(result.peptide).charge > 0 ? '+' : '−'}${Math.abs(peptideStats(result.peptide).charge)}). The fast scorer is blind to charged residues, so read this ΔG as a direction rather than an exact value. In Expert mode, Ultra mode adds a correction for charged residues.`),
    h('p', { class: 'small muted expert-only' }, `≈ ${fmtKd(kdFromDG(dg))} dissociation constant (converted from ΔG at 25 °C). One typical error is about a ×${Math.round(errorFoldChange())} change in binding.`),
    result.demo && h('p', { class: 'small muted' }, 'Demo: this number and the pose are simulated from your inputs, not predicted by the docking program.'));
}

export function commandBlock(result) {
  const pre = h('pre', { class: 'cmd', tabindex: '0', 'aria-label': 'Command used' }, result.command);
  const copy = h('button', { class: 'btn sm', type: 'button', onClick: async () => { try { await navigator.clipboard.writeText(result.command); toast('Command copied.'); } catch { toast('Couldn’t copy.'); } } }, icon('copy', 15), 'Copy');
  return h('div', { class: 'expert-only' },
    h('div', { class: 'row', style: { justifyContent: 'space-between', marginBottom: '6px' } }, h('span', { class: 'field-label', style: { margin: 0 } }, 'Command used', h('span', { class: 'tech' }, 'Reproduce this run')), copy), pre);
}

export function mountResults(ctx) {
  const { store, stage, go, param } = ctx;
  let alive = true;
  const entry = (store.get().history || []).find((e) => e.id === param);
  const el = h('section', { class: 'screen results' });

  if (!entry) {
    el.append(h('div', { class: 'glass panel', style: { maxWidth: '520px', margin: '40px auto' } },
      h('h1', { class: 'panel-title' }, 'We couldn’t find that run'),
      h('p', { class: 'muted' }, 'It may have been cleared from History.'),
      h('button', { class: 'btn primary', type: 'button', onClick: () => go('/') }, 'Back to home')));
    return { el, destroy() {} };
  }

  // Seeded demo entries keep only their inputs; rebuild the simulated poses when opened.
  async function ensureResult() {
    if (entry.result) return entry.result;
    if (!entry.demo) throw new Error('This run’s details were not saved.');
    const r = await demoAdapter.runDock(entry.job, { instant: true });
    r.id = entry.id; r.createdAt = entry.createdAt;
    store.updateHistory(entry.id, { result: r });
    return r;
  }

  el.append(h('div', { class: 'glass panel', style: { maxWidth: '520px', margin: '40px auto' } }, h('p', { class: 'muted' }, 'Loading your result…')));
  ensureResult().then((result) => { if (alive) render(result); }).catch((err) => {
    if (alive) el.replaceChildren(h('div', { class: 'glass panel error-card', style: { maxWidth: '560px', margin: '40px auto' } }, h('h1', { class: 'panel-title' }, 'Couldn’t open this result'), h('p', {}, err.message), h('button', { class: 'btn', type: 'button', onClick: () => go('/') }, 'Back to home')));
  });

  function render(result) {
    if (result.kind === 'compare') { compareResultView(ctx, result, el, () => alive); return; }
    const isScore = result.kind === 'score';
    const slot = h('div', { class: 'stage-slot', 'data-stage-slot': '' },
      h('p', { class: 'hint pose-label', 'aria-live': 'polite' }),
      h('div', { class: 'legend' }, h('span', {}, h('i', { class: 'dot protein' }), 'Protein'), h('span', {}, h('i', { class: 'dot peptide' }), 'Peptide pose')));
    const panel = h('aside', { class: 'glass panel', 'aria-label': 'Result' });
    el.replaceChildren(h('div', { class: 'row', style: { marginBottom: '14px' } }, h('button', { class: 'btn ghost sm', type: 'button', onClick: () => go('/') }, icon('chevronL', 15), 'Home'), h('span', { class: 'mono muted small' }, result.name)),
      h('div', { class: 'split' }, slot, panel));
    stage.setSlot(slot); // the slot is new, so tell the stage where to float

    const kids =[h('h1', { class: 'visually-hidden', tabindex: '-1' }, `Result: ${result.name}`), bigNumber(result, { label: isScore ? 'Binding strength of this pose' : 'Binding strength' })];

    if (!isScore) {
      const rows = result.poses.map((p) => h('tr', { 'data-rank': p.rank, tabindex: '0', 'aria-selected': 'false' },
        h('td', {}, String(p.rank)), h('td', { class: 'num' }, fmt(p.deltaG)),
        h('td', { class: 'num expert-only' }, p.rankScore == null ? '—' : p.rankScore.toFixed(3)), h('td', { class: 'num expert-only' }, String(p.nClash)), h('td', { class: 'num expert-only' }, String(p.cluster))));
      const table = h('table', { class: 'poses', role: 'grid', 'aria-label': 'Ranked poses' },
        h('thead', {}, h('tr', {}, h('th', {}, 'Pose'), h('th', { class: 'num' }, 'ΔG (kcal/mol)'),
          h('th', { class: 'num expert-only', title: 'Ranking score: only compares poses for this protein (lower = stronger). Not a ΔG.' }, 'Ranking score*'),
          h('th', { class: 'num expert-only' }, 'Clashes'), h('th', { class: 'num expert-only' }, 'Cluster'))),
        h('tbody', {}, rows));
      kids.push(h('div', {},
        h('div', { class: 'pose-picker-title', style: { marginBottom: '8px' } }, h('span', { class: 'field-label', style: { margin: 0 } }, 'Ranked poses', h('span', { class: 'tech' }, 'Select one to see it in 3D')),
          h('span', { class: 'small muted' }, result.nPoses > result.poses.length ? `Top ${result.poses.length} of ${result.nPoses}` : `${result.poses.length} poses`)),
        result.poses.length ? h('div', { class: 'table-wrap' }, table) : h('p', { class: 'pose-note' }, 'Pose files aren’t available for this run, so there is nothing to show in 3D.'),
        result.poses.length > 1 && h('p', { class: 'small muted guided-only', style: { marginTop: '6px' } }, 'The poses are ordered by the program’s own ranking, so their ΔG values are not always in step. The big number above is the best-ranked pose.'),
        h('p', { class: 'small muted expert-only', style: { marginTop: '6px' } }, '*Ranking score only compares poses for this protein (lower = stronger). It is not a ΔG and shouldn’t be compared across proteins.')));

      const dl = (what, name, sub) => h('button', { class: 'btn sm', type: 'button', onClick: () => download(result, what) }, icon('download', 15), h('span', {}, name, h('span', { class: 'mono small muted' }, ` ${sub}`)));
      kids.push(h('div', {}, h('p', { class: 'field-label' }, 'Downloads'), h('div', { class: 'downloads' },
        dl('best_pose', 'Best pose', 'best_pose.pdb'), dl('ranked_csv', 'Ranked list', 'ranked_poses.csv'), dl('run_folder', 'Run folder', 'everything'))));

      table.addEventListener('click', (e) => { const tr = e.target.closest('tr[data-rank]'); if (tr) selectPose(Number(tr.dataset.rank)); });
      table.addEventListener('keydown', (e) => { const tr = e.target.closest('tr[data-rank]'); if (tr && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); selectPose(Number(tr.dataset.rank)); } });
      el._table = table;
    }
    if (result.outputDir) kids.push(h('p', { class: 'small muted expert-only' }, 'Saved in ', h('span', { class: 'mono' }, result.outputDir)));
    kids.push(commandBlock(result));
    kids.push(h('div', { class: 'panel-foot' },
      h('button', { class: 'btn', type: 'button', onClick: () => { store.set(isScore ? { score: freshScore() } : { setup: freshSetup() }); go(isScore ? '/score' : '/predict'); } }, isScore ? 'Score another' : 'Run another'),
      h('button', { class: 'btn primary', type: 'button', onClick: () => go('/') }, 'Done')));
    panel.replaceChildren(...kids);

    stageFor(result.protein, result.peptide.length, isScore ? result.ca : result.poses[0]?.ca, result);
    function selectPose(rank) {
      const poseLabel = el.querySelector('.pose-label');
      adapterFor(result).loadPose(result, rank).then((pts) => {
        if (!alive) return;
        if (!pts) { toast('That pose file isn’t available.'); return; }
        stage.setPeptideMode('dock');
        stage.setDockPose(pts);
        el._table.querySelectorAll('tr[data-rank]').forEach((tr) => tr.setAttribute('aria-selected', String(Number(tr.dataset.rank) === rank)));
        poseLabel.textContent = `Pose ${rank} of ${result.poses.length}`;
      });
    }
    if (!isScore && result.poses.length) selectPose(1);
    if (isScore) el.querySelector('.pose-label').textContent = 'Your pose, scored';
  }

  /** Put the protein, the search box and the peptide pose on the stage. */
  function stageFor(proteinRef, n, ca, result) {
    stage.setPeptide(n);
    if (ca) { stage.setPeptideMode('dock'); stage.setDockPose(Float32Array.from(ca)); }
    loadProtein(proteinRef).then(({ structure }) => {
      if (!alive) return;
      stage.setProtein(structure);
      if (result?.kind === 'dock' && !result.blind) stage.setBox({ center: [result.site.x, result.site.y, result.site.z], size: result.box, level: 'ok' });
    }).catch(() => {
      if (alive) toast('The protein file for this run isn’t stored in History, so the 3D view is empty.', 6000);
    });
  }

  async function download(result, what) {
    try {
      const file = await adapterFor(result).download(result, what);
      if (file.copyText) {
        try { await navigator.clipboard.writeText(file.copyText); toast(`Run folder path copied: ${file.copyText}`, 6000); }
        catch { toast(`The run folder is: ${file.copyText}`, 8000); }
        return;
      }
      saveBlob(file.blob, file.filename);
    } catch (err) { toast(err.message, 6000); }
  }

  return { el, destroy() { alive = false; } };
}
