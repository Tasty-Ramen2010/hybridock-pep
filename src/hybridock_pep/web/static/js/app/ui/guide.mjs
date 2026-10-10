// guide.mjs — the in-app guide: a long, searchable-by-eye page with a table of contents, rendered from
// static/guide/guide.md (the same file is docs/GUIDE.md on GitHub). The text is data; this file only draws it.

import { h } from './dom.mjs';
import { icon } from './icons.mjs';
import { assetUrl } from '../config.mjs';
import { fetchRetry } from '../structures.mjs';
import { outline, parseMarkdown, safeHref } from '../markdown.mjs';
import { toast } from './toast.mjs';

/** Inline nodes → DOM (text only, never HTML). */
function inlines(nodes) {
  return nodes.map((n) => {
    switch (n.t) {
      case 'text': return document.createTextNode(n.v);
      case 'code': return h('code', { class: 'mono' }, n.v);
      case 'strong': return h('strong', {}, ...inlines(n.c));
      case 'em': return h('em', {}, ...inlines(n.c));
      case 'img': return image(n.src, n.alt);
      case 'link': {
        const href = safeHref(n.href);
        if (href == null) return h('span', {}, ...inlines(n.c));
        const external = /^https?:/i.test(href);
        const attrs = external ? { href, target: '_blank', rel: 'noopener noreferrer' } : { href: href.startsWith('#') && !href.startsWith('#/') ? `#/guide/${href.slice(1)}` : href, 'data-jump': href.startsWith('#') ? href.slice(1) : null };
        return h('a', attrs, ...inlines(n.c));
      }
      default: return document.createTextNode('');
    }
  });
}

const image = (src, alt) => h('img', { src: /^(https?:|data:)/i.test(src) ? src : assetUrl(`guide/${src}`), alt, loading: 'lazy', decoding: 'async' });

function codeBlock(b) {
  const pre = h('pre', { class: 'cmd', tabindex: '0', 'aria-label': b.lang ? `${b.lang} example` : 'Code example' }, b.text);
  const copy = h('button', { class: 'btn sm', type: 'button', 'aria-label': 'Copy this example', onClick: async () => {
    try { await navigator.clipboard.writeText(b.text); toast('Copied.'); } catch { toast('Couldn’t copy. Select the text and copy it by hand.'); }
  } }, icon('copy', 15), 'Copy');
  return h('div', { class: 'cmd-wrap' }, pre, copy);
}

function block(b) {
  switch (b.type) {
    case 'heading': return h(`h${Math.min(b.level + 0, 4)}`, { id: b.id, tabindex: '-1', class: b.level === 1 ? 'guide-title' : null }, ...inlines(b.inlines));
    case 'p': {
      // a paragraph that is only an image is a figure with a caption
      if (b.inlines.length === 1 && b.inlines[0].t === 'img') {
        const n = b.inlines[0];
        return h('figure', {}, image(n.src, n.alt), n.alt ? h('figcaption', {}, n.alt) : null);
      }
      return h('p', {}, ...inlines(b.inlines));
    }
    case 'ul': case 'ol': return h(b.type, {}, b.items.map((it) => h('li', {}, ...inlines(it.inlines), ...it.children.map(block))));
    case 'code': return codeBlock(b);
    case 'hr': return h('hr');
    case 'quote': return h('div', { class: 'callout', role: 'note' }, ...b.blocks.map(block));
    case 'table': return h('div', { class: 'table-scroll' }, h('table', { class: 'guide-table' },
      h('thead', {}, h('tr', {}, b.head.map((c) => h('th', {}, ...inlines(c))))),
      h('tbody', {}, b.rows.map((r) => h('tr', {}, r.map((c) => h('td', {}, ...inlines(c))))))));
    default: return document.createTextNode('');
  }
}

export function mountGuide(ctx) {
  const { stage, param } = ctx;
  let alive = true, spy = null;
  stage.setProtein(null);
  stage.setPeptideMode('hidden');

  const toc = h('nav', { class: 'guide-toc', 'aria-label': 'On this page' });
  const body = h('article', { class: 'guide-body', 'aria-live': 'polite' }, h('p', { class: 'muted' }, 'Loading the guide…'));
  const el = h('section', { class: 'screen guide', 'aria-labelledby': 'guide-heading' },
    h('div', { class: 'guide-layout' }, toc, body));

  const jump = (id, { smooth = true } = {}) => {
    const target = document.getElementById(id);
    if (!target) return false;
    target.scrollIntoView({ behavior: smooth ? 'smooth' : 'auto', block: 'start' });
    target.focus({ preventScroll: true });
    try { history.replaceState(null, '', `#/guide/${id}`); } catch { /* not allowed here: the page still scrolled */ }
    return true;
  };
  // links inside the guide scroll instead of re-opening the screen
  el.addEventListener('click', (e) => {
    const a = e.target.closest('a[data-jump]');
    if (!a) return;
    if (jump(a.dataset.jump)) e.preventDefault();
  });

  async function load() {
    try {
      const res = await fetchRetry(assetUrl('guide/guide.md'));
      if (!res.ok) throw new Error(String(res.status));
      const { blocks } = parseMarkdown(await res.text());
      if (!alive) return;
      const title = blocks.find((b) => b.type === 'heading' && b.level === 1);
      const items = outline(blocks);
      const rest = blocks.filter((b) => b !== title);
      body.replaceChildren(
        h('h1', { id: 'guide-heading', tabindex: '-1' }, title ? title.text : 'Guide'),
        ...rest.map(block));
      toc.replaceChildren(
        h('details', { class: 'toc-details', open: window.matchMedia('(min-width: 901px)').matches },
          h('summary', {}, 'On this page'),
          h('ul', {}, items.map((it) => h('li', { class: it.level === 3 ? 'sub' : '' }, h('a', { href: `#/guide/${it.id}`, 'data-jump': it.id, 'data-toc': it.id }, it.text))))));
      if (param) requestAnimationFrame(() => jump(param, { smooth: false }));
      watchHeadings(items);
    } catch {
      body.replaceChildren(h('h1', { id: 'guide-heading', tabindex: '-1' }, 'The guide didn’t load'),
        h('p', {}, 'Check your connection and reload the page. The same guide is in the project on GitHub: ',
          h('a', { href: 'https://github.com/Tasty-Ramen2010/hybridock-pep/blob/master/docs/GUIDE.md', target: '_blank', rel: 'noopener noreferrer' }, 'docs/GUIDE.md'), '.'));
    }
  }

  /** Mark the section being read in the table of contents. */
  function watchHeadings(items) {
    if (typeof IntersectionObserver === 'undefined') return;
    spy = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        for (const a of toc.querySelectorAll('a[data-toc]')) a.removeAttribute('aria-current');
        const link = toc.querySelector(`a[data-toc="${e.target.id}"]`);
        link?.setAttribute('aria-current', 'true');
        if (link && getComputedStyle(toc).position === 'sticky') { // keep the current section visible in the long, scrollable contents list
          const r = link.getBoundingClientRect(), t = toc.getBoundingClientRect();
          if (r.top < t.top + 24 || r.bottom > t.bottom - 24) toc.scrollTop += r.top - t.top - t.height / 3;
        }
      }
    }, { rootMargin: '-72px 0px -70% 0px' });
    for (const it of items) { const node = document.getElementById(it.id); if (node) spy.observe(node); }
  }

  load();
  return { el, destroy() { alive = false; spy?.disconnect(); } };
}
