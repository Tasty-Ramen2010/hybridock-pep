// topbar.js — the navigation bar: name, Predict / Compare / Score, Guided / Expert, status, History, Help, and one
// avatar menu for appearance, accent colour and your name. On a phone the three destinations become a tab bar.
// Built once; store changes update the pieces in place (so keyboard focus is never lost).

import { h } from './dom.mjs';
import { icon, logo } from './icons.mjs';
import { ACCENTS } from '../config.mjs';
import { initials } from '../jobs.mjs';
import { adapter } from '../adapter.mjs';
import { openHistory } from './history.mjs';
import { openHelp } from './help.mjs';

const DESTINATIONS = [
  { id: 'predict', label: 'Predict', icon: 'zap' },
  { id: 'compare', label: 'Compare', icon: 'columns' },
  { id: 'score', label: 'Score', icon: 'badge' },
  { id: 'guide', label: 'Guide', icon: 'book' },
];

export function mountTopbar(ctx) {
  const { store, go } = ctx;
  const root = document.getElementById('topbar');

  const modeBtn = (id, label) => h('button', { type: 'button', onClick: () => store.set({ mode: id }) }, label);
  const guided = modeBtn('guided', 'Guided');
  const expert = modeBtn('expert', 'Expert');
  const modeSeg = h('div', { class: 'seg', role: 'group', 'aria-label': 'Detail level' }, guided, expert);

  const runPill = h('button', { class: 'btn sm primary', type: 'button', hidden: true, 'aria-label': 'Run in progress', onClick: () => go('/running') }, icon('play', 14), h('span', { class: 'label' }, 'Run in progress'));

  // ---- the three destinations: links in the bar on a desktop, a tab bar on a phone ----
  const navLink = (d, withIcon) => h('a', { href: `#/${d.id}`, 'data-nav': d.id }, withIcon ? icon(d.icon, 24) : null, d.label);
  const links = h('nav', { class: 'nav-links', 'aria-label': 'Sections' }, DESTINATIONS.map((d) => navLink(d, false)));
  document.getElementById('tabbar')?.remove();
  const tabbar = h('nav', { class: 'tabbar', id: 'tabbar', 'aria-label': 'Sections (tab bar)' }, DESTINATIONS.map((d) => navLink(d, true)));
  document.body.append(tabbar);
  function markCurrent() {
    const name = location.hash.replace(/^#\/?/, '').split('/')[0];
    for (const a of document.querySelectorAll('[data-nav]')) {
      if (a.dataset.nav === name) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
    }
  }
  addEventListener('hashchange', markCurrent);

  // ---- the avatar menu: appearance, accent colour, name ----
  const lightBtn = h('button', { type: 'button', 'aria-label': 'Light appearance', onClick: () => store.set({ theme: 'light' }) }, 'Light');
  const darkBtn = h('button', { type: 'button', 'aria-label': 'Dark appearance', onClick: () => store.set({ theme: 'dark' }) }, 'Dark');
  const themeSeg = h('div', { class: 'seg', role: 'group', 'aria-label': 'Appearance' }, lightBtn, darkBtn);
  const swatchBtns = ACCENTS.map((a) => h('button', { class: 'swatch', type: 'button', 'aria-label': a.name, title: a.name, onClick: () => store.set({ accent: a.id }) }));
  const nameInput = h('input', { class: 'field', id: 'user-name', type: 'text', 'aria-label': 'Your name', placeholder: 'Used in the greeting', maxlength: '40', onInput: (e) => store.set({ userName: e.target.value.trim() }) });
  const menuPop = h('div', { class: 'popover glass', hidden: true, role: 'group', 'aria-label': 'Appearance and name' },
    h('div', { class: 'menu-row' }, h('span', { class: 'small' }, 'Appearance'), themeSeg),
    h('div', { class: 'menu-row' }, h('span', { class: 'small' }, 'Accent colour'), h('div', { class: 'swatches', role: 'group', 'aria-label': 'Accent colour' }, swatchBtns)),
    h('div', { class: 'menu-row' }, h('label', { class: 'small', for: 'user-name' }, 'Your name'), nameInput),
    h('div', { class: 'menu-links' },
      h('button', { class: 'btn', type: 'button', onClick: () => { closePops(); openHistory(ctx); } }, icon('history', 16), 'History'),
      h('button', { class: 'btn', type: 'button', onClick: () => { closePops(); openHelp(); } }, icon('help', 16), 'Help')));
  const avatar = h('button', { class: 'avatar', type: 'button', 'aria-label': 'Appearance and name', 'aria-expanded': 'false', 'aria-haspopup': 'true', onClick: () => togglePop(menuPop, avatar) });

  let envPop = null, envBtn = null; // set below when the live server is connected
  // status: "Demo mode" badge, or the live server's readiness (from /api/env) with the details in a popover
  const statusEl = (() => {
    if (adapter.kind === 'demo') return h('span', { class: 'badge-demo', title: 'Results in this build are simulated' }, 'Demo mode');
    const env = adapter.env;
    const dot = (colour) => h('i', { class: 'dot', style: { background: colour, marginRight: 0 } });
    // "Live" stays in the text for screen readers; what people see is the part that matters
    const label = (text) => [h('span', { class: 'visually-hidden' }, 'Live, '), h('span', { class: 'chip-text' }, text)];
    if (!env) return h('span', { class: 'chip' }, dot('var(--warn)'), ...label('Server not reached'));
    const gpu = env.checks?.gpu;
    const text = !env.ready ? 'Setup needed' : gpu?.ok ? gpu.detail : 'CPU only';
    const rows = Object.entries(env.checks || {}).map(([key, c]) => h('li', { class: 'env-row' },
      h('i', { class: 'dot', style: { background: c.ok ? 'var(--ok)' : 'var(--warn)' } }),
      h('span', {}, h('b', {}, c.detail), !c.ok && c.fix ? h('small', { class: 'tech' }, c.fix) : null)));
    envPop = h('div', { class: 'popover glass', hidden: true, role: 'group', 'aria-label': 'What this machine can do', style: { minWidth: '280px' } },
      h('span', { class: 'small muted' }, 'This machine'), h('ul', { class: 'env-list' }, rows));
    envBtn = h('button', { class: 'chip', type: 'button', 'aria-expanded': 'false', onClick: () => togglePop(envPop, envBtn) },
      dot(env.ready ? 'var(--ok)' : 'var(--warn)'), ...label(text));
    return h('div', { class: 'rel' }, envBtn, envPop);
  })();
  const allPops = () => [[menuPop, avatar], ...(envPop ? [[envPop, envBtn]] : [])];
  const closePops = () => { for (const [p, b] of allPops()) { p.hidden = true; b.setAttribute('aria-expanded', 'false'); } };

  function togglePop(pop, btn) {
    const open = pop.hidden;
    for (const [p, b] of allPops()) { p.hidden = true; b.setAttribute('aria-expanded', 'false'); }
    pop.hidden = !open;
    btn.setAttribute('aria-expanded', String(open));
    if (open && pop === menuPop) nameInput.focus({ preventScroll: true });
  }
  document.addEventListener('click', (e) => { if (!root.contains(e.target) || !e.target.closest('.rel')) for (const [p, b] of allPops()) { p.hidden = true; b.setAttribute('aria-expanded', 'false'); } });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') for (const [p, b] of allPops()) { p.hidden = true; b.setAttribute('aria-expanded', 'false'); } });

  root.replaceChildren(
    h('button', { class: 'brand', type: 'button', onClick: () => go('/'), 'aria-label': 'HybriDock-Pep home' }, logo(20), 'HybriDock-Pep'),
    links,
    h('div', { class: 'topbar-actions' },
      runPill,
      modeSeg,
      statusEl,
      h('button', { class: 'btn icon bar-only', type: 'button', 'aria-label': 'History', title: 'History', onClick: () => openHistory(ctx) }, icon('history', 18)),
      h('button', { class: 'btn icon bar-only', type: 'button', 'aria-label': 'Help', title: 'Help', onClick: () => openHelp() }, icon('help', 18)),
      h('div', { class: 'rel' }, avatar, menuPop)),
  );
  addEventListener('scroll', () => root.classList.toggle('scrolled', scrollY > 4), { passive: true });
  markCurrent();

  function update() {
    const s = store.get();
    guided.setAttribute('aria-pressed', String(s.mode === 'guided'));
    expert.setAttribute('aria-pressed', String(s.mode === 'expert'));
    lightBtn.setAttribute('aria-pressed', String(s.theme !== 'dark'));
    darkBtn.setAttribute('aria-pressed', String(s.theme === 'dark'));
    swatchBtns.forEach((b, i) => {
      const a = ACCENTS[i];
      b.style.background = a[s.theme];
      b.setAttribute('aria-pressed', String(s.accent === a.id));
    });
    avatar.replaceChildren(s.userName ? initials(s.userName) : icon('user', 16));
    if (document.activeElement !== nameInput) nameInput.value = s.userName;
    runPill.hidden = s.run.status !== 'running';
  }
  store.subscribe(update);
  update();
}
