// history.js — a drawer listing past runs (name, date, ΔG) that can be reopened.
// Built on the native <dialog>, so Esc closes it and keyboard focus stays inside.

import { h, fmtDate } from './dom.mjs';
import { icon } from './icons.mjs';
import { fmt, fmtSigned } from '../interpret.mjs';

export function openHistory({ store, go }) {
  const dlg = h('dialog', { class: 'drawer', 'aria-labelledby': 'hist-title' });
  const body = h('div', { class: 'dialog-body', tabindex: '0', role: 'region', 'aria-label': 'Contents' });
  const close = () => { dlg.close(); dlg.remove(); };

  function render() {
    const hist = store.get().history || [];
    const clear = h('button', { class: 'btn sm ghost danger', type: 'button', onClick: () => { store.clearHistory(); render(); } }, icon('trash', 15), 'Clear all');
    body.replaceChildren(
      hist.length
        ? h('ul', { class: 'hist-list' }, hist.map((e) => h('li', {},
          h('button', { class: 'hist-item', type: 'button', onClick: () => { close(); go(`/results/${e.id}`); } },
            h('span', { class: 'name' }, e.name),
            h('span', { class: 'when' }, fmtDate(e.createdAt), e.demo && ' · ', e.demo && h('span', { class: 'badge-demo' }, 'Demo')),
            h('span', { class: 'val nums' }, e.headline.label === 'ΔΔG' ? fmtSigned(e.headline.value) : fmt(e.headline.value), h('small', {}, e.headline.label + ' kcal/mol'))))))
        : h('div', { class: 'empty-state' }, h('p', {}, 'No runs yet.'), h('p', { class: 'small' }, 'Your predictions will show up here so you can reopen them.')),
      hist.length ? h('div', { class: 'row', style: { justifyContent: 'flex-end', marginTop: '14px' } }, clear) : null);
  }

  dlg.append(
    h('div', { class: 'dialog-head' }, h('h2', { id: 'hist-title' }, 'History'),
      h('button', { class: 'btn icon ghost', type: 'button', 'aria-label': 'Close history', onClick: close }, icon('x'))),
    body);
  dlg.addEventListener('click', (e) => { if (e.target === dlg) close(); });
  dlg.addEventListener('close', () => dlg.remove());
  document.getElementById('overlays').append(dlg);
  render();
  dlg.showModal();
}
