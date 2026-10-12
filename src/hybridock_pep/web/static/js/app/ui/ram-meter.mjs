// ram-meter.mjs — the little liquid tank in the top bar. A dot while nothing runs; while a prediction runs it widens into a
// tank whose liquid shows how full the computer's memory is, going from green (plenty free) through amber to red, and
// spilling over when memory is nearly gone. A speech bubble warns when memory is low. Pure DOM: topbar.mjs does the polling.

import { h } from './dom.mjs';
import { icon } from './icons.mjs';
import { memoryLevel } from '../config.mjs';

const GB = (n) => (n >= 10 ? Math.round(n) : Math.round(n * 10) / 10);

export function createRamMeter() {
  const liquid = h('span', { class: 'ram-liquid' });
  const glass = h('span', { class: 'ram-glass' }, liquid);
  const foam = h('span', { class: 'ram-foam', 'aria-hidden': 'true' }, h('i'), h('i'), h('i'));
  const tank = h('span', { class: 'ram', 'aria-hidden': 'true', 'data-level': 'unknown' }, glass, foam);
  const pct = h('span', { class: 'ram-pct' });

  const bubbleText = h('span', { class: 'ram-bubble-text' });
  const dismiss = h('button', { class: 'ram-bubble-x', type: 'button', 'aria-label': 'Dismiss the memory warning', onClick: () => { dismissed = true; bubble.hidden = true; } }, icon('x', 14));
  const bubble = h('div', { class: 'ram-bubble', role: 'status', hidden: true }, icon('alert', 18), bubbleText, dismiss);

  let dismissed = false, running = false, last = null, popOpen = false;

  function render() {
    const m = memoryLevel(last?.percent, last?.available_gb);
    tank.dataset.level = m.id;
    tank.dataset.on = String(running && m.id !== 'unknown');
    tank.style.setProperty('--fill', `${m.fill}%`);
    tank.style.setProperty('--liq', `hsl(${m.hue} 78% 48%)`);
    pct.textContent = m.percent == null ? '' : `${m.percent}%`;
    const warn = m.id === 'low' || m.id === 'full';
    if (!warn) dismissed = false; // it warns again the next time memory runs low
    bubble.hidden = !warn || dismissed || popOpen;
    if (warn) {
      bubbleText.replaceChildren(
        h('b', {}, m.id === 'full' ? 'Memory is full.' : 'Memory is running low.'),
        running ? ' If the run stops, close other apps and try Quick.' : ' Close other apps before you start a run.');
    }
    return m;
  }

  return {
    el: h('span', { class: 'ram-wrap' }, tank, pct),
    bubble,
    /** New numbers from /api/stats (or null when the server could not say). */
    apply(stats) { last = stats && stats.ok ? stats : null; return render(); },
    setRunning(on) { running = !!on; return render(); },
    setPopOpen(on) { popOpen = !!on; return render(); },
    /** "62% used, 10 of 16 GB" for the popover and the screen-reader label; null when unknown. */
    describe() {
      if (!last) return null;
      const used = last.total_gb - last.available_gb;
      return `${Math.round(last.percent)}% used, ${GB(used)} of ${GB(last.total_gb)} GB`;
    },
    runGb: () => last?.run_gb ?? null,
    level: () => memoryLevel(last?.percent, last?.available_gb),
  };
}
