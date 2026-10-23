// guide.mjs — the in-app guide, built like an app and not like a manual: a grid of short lessons, and inside each one
// a card per step with a picture, one headline and one sentence. The words live in guide-content.mjs.
//
//   #/guide                    the lessons
//   #/guide/<lesson>           a lesson, from its first step
//   #/guide/<lesson>/<n>       a lesson, at step n

import { h } from './dom.mjs';
import { icon } from './icons.mjs';
import { assetUrl } from '../config.mjs';
import { adapter } from '../adapter.mjs';
import { backend } from '../config.mjs';
import { toast } from './toast.mjs';
import { COLAB_URL, findTopic, richParts, sizeOf, topicsFor } from '../guide-content.mjs';

const DONE_KEY = 'hp-guide-done';
const readDone = () => { try { return new Set(JSON.parse(localStorage.getItem(DONE_KEY) || '[]')); } catch { return new Set(); } };
const markDone = (id) => { try { const d = readDone(); d.add(id); localStorage.setItem(DONE_KEY, JSON.stringify([...d])); } catch { /* private window: it just will not be remembered */ } };

/** **bold** and `code` → DOM. Text only, never HTML. */
export function rich(text) {
  return richParts(text).map((p) => (p.t === 'strong' ? h('strong', {}, p.v) : p.t === 'code' ? h('code', { class: 'mono' }, p.v) : document.createTextNode(p.v)));
}

function command(text) {
  const copy = h('button', { class: 'btn sm', type: 'button', 'aria-label': 'Copy this command', onClick: async () => {
    try { await navigator.clipboard.writeText(text); toast('Copied. Paste it into a terminal.'); } catch { toast('Couldn’t copy. Select the text and copy it by hand.'); }
  } }, icon('copy', 15), 'Copy');
  return h('div', { class: 'cmd-wrap' }, h('pre', { class: 'cmd', tabindex: '0', 'aria-label': 'Command' }, text), copy);
}

const goTo = (path) => { location.hash = '#' + path; };

export function mountGuide(ctx) {
  const { stage, param, sub } = ctx;
  const kind = adapter.kind !== 'live' ? 'demo' : backend.remote ? 'remote' : 'local';
  const topics = topicsFor(kind);
  const topic = param ? findTopic(topics, param) : null;
  stage.setProtein(null);
  stage.setPeptideMode('hidden');
  let onKey = null;

  const el = h('section', { class: 'screen guide', 'aria-labelledby': 'guide-heading' });
  if (topic && topic.kind === 'steps') el.append(stepsView(topic));
  else if (topic) el.append(listView(topic));
  else el.append(lessonsView());

  // ---------------------------------------------------------------- the lessons
  function lessonsView() {
    const done = readDone();
    const tiles = topics.map((t, i) => {
      const hero = i === 0;
      const dark = hero || t.dark;
      const cover = hero && t.cover
        ? t.cover === 'terminal'
          ? h('span', { class: 'lesson-cover term', 'aria-hidden': 'true' }, h('span', { class: 'term-bar' }, h('i'), h('i'), h('i')), h('code', {}, 'git clone --recurse-submodules …'), h('code', {}, './install.sh'), h('code', { class: 'ok' }, 'HybriDock-Pep is ready.'))
          : h('span', { class: 'lesson-cover shot', 'aria-hidden': 'true' }, h('img', { src: assetUrl(`guide/guide-img/${t.cover}`), alt: '', decoding: 'async' }))
        : null;
      return h('a', { class: `lesson${hero ? ' hero' : ''}${dark ? ' dark' : ''}`, href: `#/guide/${t.id}`, 'data-topic': t.id },
        h('span', { class: 'lesson-num', 'aria-label': `Lesson ${i + 1}` }, String(i + 1)),
        h('span', { class: 'lesson-icon', 'aria-hidden': 'true' }, icon(t.icon, hero ? 28 : 24)),
        h('span', { class: 'lesson-title' }, t.title),
        h('span', { class: 'lesson-blurb' }, t.blurb),
        cover,
        h('span', { class: 'lesson-foot' },
          h('span', { class: 'lesson-meta' }, sizeOf(t)),
          done.has(t.id)
            ? h('span', { class: 'lesson-done', role: 'img', 'aria-label': 'Done' }, icon('check', 16))
            : h('span', { class: 'lesson-go', 'aria-hidden': 'true' }, icon('chevron', 16))));
    });
    const seen = topics.filter((t) => done.has(t.id)).length;
    return h('div', { class: 'guide-home' },
      h('header', { class: 'guide-head' },
        h('h1', { id: 'guide-heading', tabindex: '-1' }, 'How to use it.', ' ', h('span', { class: 'dim' }, 'Follow the steps, in order.')),
        seen ? h('p', { class: 'guide-progress' }, `${seen} of ${topics.length} done`) : null),
      h('div', { class: 'lessons' }, tiles),
      h('p', { class: 'small muted guide-foot' }, 'No terminal? ',
        h('a', { href: COLAB_URL, target: '_blank', rel: 'noopener noreferrer' }, 'Run it in Google Colab'), ' instead.'));
  }

  // ---------------------------------------------------------------- a lesson made of steps
  function stepsView(t) {
    const total = t.steps.length;
    let i = Math.min(Math.max((parseInt(sub, 10) || 1) - 1, 0), total - 1);
    const card = h('div', { class: 'step-card', 'aria-live': 'polite' });
    const bars = h('div', { class: 'step-bars', role: 'group', 'aria-label': 'Steps' });
    const count = h('span', { class: 'step-count' });
    const back = h('button', { class: 'btn', type: 'button', onClick: () => show(i - 1) }, icon('back', 16), 'Back');
    const next = h('button', { class: 'btn primary', type: 'button', onClick: () => (i === total - 1 ? finish() : show(i + 1)) });
    const finish = () => { markDone(t.id); goTo('/guide'); };

    function show(n, { animate = true } = {}) {
      i = Math.min(Math.max(n, 0), total - 1);
      const s = t.steps[i];
      card.replaceChildren(
        s.image
          ? h('div', { class: 'step-shot' }, h('img', { src: assetUrl(`guide/guide-img/${s.image}`), alt: s.alt || `The ${s.title.toLowerCase()} screen`, decoding: 'async' }))
          : s.cmd
            ? h('div', { class: 'step-shot terminal' }, h('div', { class: 'term-bar', 'aria-hidden': 'true' }, h('i'), h('i'), h('i'), h('span', {}, 'Terminal')), command(s.cmd))
            : h('div', { class: 'step-shot plain', 'aria-hidden': 'true' }, h('span', { class: 'plain-icon' }, icon(s.icon || t.icon, 56))),
        h('div', { class: 'step-text' },
          h('span', { class: 'step-num', 'aria-hidden': 'true' }, String(i + 1)),
          h('h2', { id: 'guide-heading', tabindex: '-1' }, s.title),
          h('p', { class: 'step-body' }, ...rich(s.body)),
          s.tip ? h('p', { class: 'step-tip' }, icon('info', 16), h('span', {}, ...rich(s.tip))) : null));
      if (animate) { card.classList.remove('swap'); void card.offsetWidth; card.classList.add('swap'); }
      count.textContent = `Step ${i + 1} of ${total}`;
      [...bars.children].forEach((b, k) => { b.classList.toggle('on', k <= i); if (k === i) b.setAttribute('aria-current', 'step'); else b.removeAttribute('aria-current'); });
      back.hidden = i === 0;
      next.replaceChildren(...(i === total - 1 ? ['Done', icon('check', 16)] : ['Next', icon('arrow', 16)]));
      try { history.replaceState(null, '', `#/guide/${t.id}/${i + 1}`); } catch { /* the page still shows the right step */ }
    }

    t.steps.forEach((s, k) => bars.append(h('button', { class: 'step-bar', type: 'button', 'aria-label': `Step ${k + 1}: ${s.title}`, onClick: () => show(k) })));
    onKey = (e) => {
      if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey || e.target.closest('input, textarea, select, pre')) return;
      if (e.key === 'ArrowRight') { if (i < total - 1) { e.preventDefault(); show(i + 1); } }
      else if (e.key === 'ArrowLeft') { if (i > 0) { e.preventDefault(); show(i - 1); } }
    };
    addEventListener('keydown', onKey);
    show(i, { animate: false });
    return h('div', { class: 'guide-lesson' },
      h('div', { class: 'guide-bar' }, h('a', { class: 'btn ghost back-link', href: '#/guide' }, icon('back', 16), 'All lessons'), count),
      bars, card,
      h('div', { class: 'step-nav' }, back, next));
  }

  // ---------------------------------------------------------------- a lesson made of a list
  function listView(t) {
    const accordion = t.kind === 'accordion';
    const rows = t.rows.map((r) => accordion
      ? h('details', { class: 'qa' }, h('summary', {}, ...rich(r.q)), h('p', {}, ...rich(r.text)))
      : h('div', { class: 'info-row' },
        r.icon ? h('span', { class: 'info-icon', 'aria-hidden': 'true' }, icon(r.icon, 20)) : null,
        h('div', {}, h('h3', {}, r.term), h('p', {}, ...rich(r.text)))));
    return h('div', { class: 'guide-lesson' },
      h('div', { class: 'guide-bar' }, h('a', { class: 'btn ghost back-link', href: '#/guide' }, icon('back', 16), 'Guide')),
      h('header', { class: 'guide-head compact' },
        h('h1', { id: 'guide-heading', tabindex: '-1' }, t.title),
        h('p', { class: 'lede' }, t.blurb)),
      h('div', { class: accordion ? 'qa-list' : 'info-list' }, rows),
      h('div', { class: 'step-nav' }, h('a', { class: 'btn primary', href: '#/guide', onClick: () => markDone(t.id) }, 'Done', icon('check', 16))));
  }

  return { el, destroy() { if (onKey) removeEventListener('keydown', onKey); } };
}
