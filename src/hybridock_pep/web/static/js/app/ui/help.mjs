// help.js — a plain-language glossary and a few honest notes about what the numbers mean.

import { h } from './dom.mjs';
import { icon } from './icons.mjs';
import { TYPICAL_ERROR } from '../config.mjs';
import { adapter } from '../adapter.mjs';

const TERMS = [
  ['Protein', 'The big molecule your peptide tries to stick to. HybriDock-Pep uses its 3D shape from the Protein Data Bank.'],
  ['Peptide', 'A short chain of amino acids, typed as letters, such as LIYKWVNK.'],
  ['Binding strength (ΔG)', `How tightly the peptide sticks, in kcal/mol. More negative means stronger. A single prediction is typically off by about ±${TYPICAL_ERROR} kcal/mol.`],
  ['Pose', 'One way the peptide could sit on the protein. HybriDock-Pep makes many poses and ranks them.'],
  ['Binding site and box', 'Where on the protein to look. The box is the search area, with its size in ångströms (Å); 1 Å is a ten-billionth of a metre.'],
  ['Find the pocket for me', 'Searches the whole protein for likely binding pockets instead of using a box you place. Slower, but handy when you don’t know where to aim.'],
  ['Selectivity (ΔΔG)', 'The difference between two binding strengths: target minus off-target. Negative means the peptide prefers the target. The 95% interval shows how sure we are; if it crosses zero, there is no clear preference.'],
  ['Ranking score', 'A number used only to order the poses of one protein (lower is stronger). It is not a ΔG, so don’t compare it with one or across proteins.'],
  ['“Strong / moderate / weak”', 'Our own rough wording for a ΔG, to help you read it. The program doesn’t produce it.'],
  ['Demo', 'Anything marked Demo is simulated so the app works without the server. It is not a real prediction.'],
  ['Guided and Expert', 'Guided hides technical settings and adds short explanations. Expert shows coordinates, advanced options and the exact command. Same screens, same results.'],
];

export function openHelp() {
  const dlg = h('dialog', { class: 'modal', 'aria-labelledby': 'help-title' });
  const close = () => { dlg.close(); dlg.remove(); };
  dlg.append(
    h('div', { class: 'dialog-head' }, h('h2', { class: 'serif', id: 'help-title', style: { fontSize: '30px' } }, 'Help'),
      h('button', { class: 'btn icon ghost', type: 'button', 'aria-label': 'Close help', onClick: close }, icon('x'))),
    h('div', { class: 'dialog-body' },
      h('p', { class: 'muted', style: { marginBottom: '16px' } }, 'HybriDock-Pep predicts how tightly a short protein piece (a peptide) sticks to a protein, and shows you where. Here is what the words mean.'),
      h('dl', { class: 'glossary' }, TERMS.flatMap(([t, d]) => [h('dt', {}, t), h('dd', {}, d)])),
      adapter.kind === 'live' && h('p', { class: 'small', style: { marginTop: '14px' } }, 'Prefer the original layout? ', h('a', { href: '/static/studio.html' }, 'Open the classic studio'), '.'),
      h('p', { class: 'small muted', style: { marginTop: '18px' } }, 'Tip: drag the floating protein to turn it. Keyboard: Tab moves between controls, Esc closes this window.')));
  dlg.addEventListener('click', (e) => { if (e.target === dlg) close(); });
  dlg.addEventListener('close', () => dlg.remove());
  document.getElementById('overlays').append(dlg);
  dlg.showModal();
}
