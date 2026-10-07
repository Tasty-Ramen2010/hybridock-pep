// topbar.js — logo, Guided/Expert switch, History, Help, accent colour, dark mode, avatar.
// Built once; store changes update the pieces in place (so keyboard focus is never lost).

import { h } from './dom.mjs';
import { icon, logo } from './icons.mjs';
import { ACCENTS } from '../config.mjs';
import { initials } from '../jobs.mjs';
import { adapter } from '../adapter.mjs';
import { openHistory } from './history.mjs';
import { openHelp } from './help.mjs';

export function mountTopbar(ctx) {
  const { store, go } = ctx;
  const root = document.getElementById('topbar');

  const modeBtn = (id, label) => h('button', { type: 'button', onClick: () => store.set({ mode: id }) }, label);
  const guided = modeBtn('guided', 'Guided');
  const expert = modeBtn('expert', 'Expert');
  const modeSeg = h('div', { class: 'seg', role: 'group', 'aria-label': 'Detail level' }, guided, expert);

  const themeBtn = h('button', { class: 'btn icon', type: 'button', onClick: () => store.set({ theme: store.get().theme === 'dark' ? 'light' : 'dark' }) });
  const runPill = h('button', { class: 'btn sm primary', type: 'button', hidden: true, onClick: () => go('/running') }, icon('play', 14), h('span', { class: 'label' }, 'Run in progress'));

  // accent popover
  const swatchBtns = ACCENTS.map((a) => h('button', { class: 'swatch', type: 'button', 'aria-label': a.name, title: a.name, onClick: () => store.set({ accent: a.id }) }));
  const accentPop = h('div', { class: 'popover glass', hidden: true, role: 'group', 'aria-label': 'Accent colour' },
    h('span', { class: 'small muted' }, 'Accent colour'), h('div', { class: 'swatches' }, swatchBtns));
  const accentBtn = h('button', { class: 'btn icon', type: 'button', 'aria-label': 'Accent colour', 'aria-expanded': 'false', onClick: () => togglePop(accentPop, accentBtn) }, icon('palette'));

  // avatar popover (rename)
  const nameInput = h('input', { class: 'field', type: 'text', 'aria-label': 'Your name', maxlength: '40', onInput: (e) => store.set({ userName: e.target.value || 'there' }) });
  const avatar = h('button', { class: 'avatar', type: 'button', 'aria-label': 'Your name', 'aria-expanded': 'false', onClick: () => { togglePop(namePop, avatar); nameInput.focus(); } });
  const namePop = h('div', { class: 'popover glass', hidden: true }, h('label', { class: 'small muted' }, 'Your name (used in the greeting)'), nameInput);

  let envPop = null, envBtn = null; // set below when the live server is connected
  // status: "Demo mode" badge, or the live server's readiness (from /api/env) with the details in a popover
  const statusEl = (() => {
    if (adapter.kind === 'demo') return h('span', { class: 'badge-demo', title: 'Results in this build are simulated' }, 'Demo mode');
    const env = adapter.env;
    if (!env) return h('span', { class: 'chip' }, h('i', { class: 'dot', style: { background: 'var(--warn)' } }), 'Live · server not reached');
    const gpu = env.checks?.gpu;
    const label = !env.ready ? 'Setup needed' : gpu?.ok ? `Live · ${gpu.detail}` : 'Live · CPU only';
    const rows = Object.entries(env.checks || {}).map(([key, c]) => h('li', { class: 'env-row' },
      h('i', { class: 'dot', style: { background: c.ok ? 'var(--ok)' : 'var(--warn)' } }),
      h('span', {}, h('b', {}, c.detail), !c.ok && c.fix ? h('small', { class: 'tech' }, c.fix) : null)));
    envPop = h('div', { class: 'popover glass', hidden: true, role: 'group', 'aria-label': 'What this machine can do', style: { minWidth: '260px' } },
      h('span', { class: 'small muted' }, 'This machine'), h('ul', { class: 'env-list' }, rows));
    envBtn = h('button', { class: 'chip', type: 'button', 'aria-expanded': 'false', onClick: () => togglePop(envPop, envBtn) },
      h('i', { class: 'dot', style: { background: env.ready ? 'var(--ok)' : 'var(--warn)' } }), label);
    return h('div', { class: 'rel' }, envBtn, envPop);
  })();
  const allPops = () => [[accentPop, accentBtn], [namePop, avatar], ...(envPop ? [[envPop, envBtn]] : [])];

  function togglePop(pop, btn) {
    const open = pop.hidden;
    for (const [p, b] of allPops()) { p.hidden = true; b.setAttribute('aria-expanded', 'false'); }
    pop.hidden = !open;
    btn.setAttribute('aria-expanded', String(open));
  }
  document.addEventListener('click', (e) => { if (!root.contains(e.target) || !e.target.closest('.rel')) for (const [p, b] of allPops()) { p.hidden = true; b.setAttribute('aria-expanded', 'false'); } });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') for (const [p, b] of allPops()) { p.hidden = true; b.setAttribute('aria-expanded', 'false'); } });

  root.replaceChildren(
    h('button', { class: 'brand', type: 'button', onClick: () => go('/'), 'aria-label': 'HybriDock-Pep home' }, logo(28), 'HybriDock-Pep'),
    h('div', { class: 'topbar-actions' },
      runPill,
      modeSeg,
      statusEl,
      h('button', { class: 'btn sm', type: 'button', onClick: () => openHistory(ctx) }, icon('history', 16), h('span', { class: 'label' }, 'History')),
      h('button', { class: 'btn sm', type: 'button', onClick: () => openHelp() }, icon('help', 16), h('span', { class: 'label' }, 'Help')),
      h('div', { class: 'rel' }, accentBtn, accentPop),
      themeBtn,
      h('div', { class: 'rel' }, avatar, namePop)),
  );

  function update() {
    const s = store.get();
    guided.setAttribute('aria-pressed', String(s.mode === 'guided'));
    expert.setAttribute('aria-pressed', String(s.mode === 'expert'));
    themeBtn.replaceChildren(icon(s.theme === 'dark' ? 'sun' : 'moon'));
    themeBtn.setAttribute('aria-label', s.theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
    swatchBtns.forEach((b, i) => {
      const a = ACCENTS[i];
      b.style.background = a[s.theme];
      b.setAttribute('aria-pressed', String(s.accent === a.id));
    });
    avatar.textContent = initials(s.userName);
    if (document.activeElement !== nameInput) nameInput.value = s.userName === 'there' ? '' : s.userName;
    runPill.hidden = s.run.status !== 'running';
  }
  store.subscribe(update);
  update();
}
