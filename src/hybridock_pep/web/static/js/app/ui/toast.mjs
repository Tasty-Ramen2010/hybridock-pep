// toast.js — a brief message at the bottom of the screen (announced to screen readers).
import { h } from './dom.mjs';

export function toast(message, ms = 3200) {
  const host = document.getElementById('toasts');
  if (!host) return;
  const el = h('div', { class: 'toast', text: message });
  host.append(el);
  setTimeout(() => el.remove(), ms);
}
