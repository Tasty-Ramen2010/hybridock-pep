// dom.js — a tiny helper for building elements, so screens read like the page they make.
//   h('button', { class: 'btn', onClick: go }, 'Continue', icon('arrow'))
// Text always goes in as text (never as HTML), so nothing a user types can inject markup.

const PROPS = new Set(['value', 'checked', 'disabled', 'selected', 'indeterminate']);

export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'text') el.textContent = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (PROPS.has(k)) el[k] = v;
    else el.setAttribute(k, v === true ? '' : String(v));
  }
  append(el, children);
  return el;
}

export function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

/** Replace all children of `el`. */
export function fill(el, ...children) {
  el.replaceChildren();
  return append(el, children);
}

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

/** Save text/blob as a file download. */
export function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = h('a', { href: url, download: filename });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

/** Read a File as text. */
export const readText = (file) => new Promise((resolve, reject) => {
  const r = new FileReader();
  r.onload = () => resolve(String(r.result));
  r.onerror = () => reject(new Error(`Couldn't read ${file.name}.`));
  r.readAsText(file);
});

export const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

/** 85 -> "about 1.5 minutes", 3600 -> "about 1 hour" */
export function fmtDuration(seconds) {
  if (seconds < 90) return `${Math.max(1, Math.round(seconds / 5) * 5)} seconds`;
  if (seconds < 5400) return `${Math.round(seconds / 60)} minutes`;
  return `${(seconds / 3600).toFixed(1)} hours`;
}

export const fmtDate = (iso) => new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
