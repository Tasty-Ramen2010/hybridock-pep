// sitepicker.js — the "where does it bind?" controls that go with the draggable 3D box.
// Used by the Predict setup (step 3) and by each side of Compare.
//
// The 3D box itself lives in the stage; this file is the panel next to it: a size slider,
// the x/y/z centre (filled in automatically), the "is the box on the protein?" check, and a
// "use the suggested site" button. Whatever the user does in 3D or in the panel stays in sync.

import { h } from './dom.mjs';
import { icon } from './icons.mjs';
import { assessBox } from '../pdb.mjs';
import { placeHelixAt } from '../placement.mjs';
import { mulberry32, hashString } from '../stage/math.mjs';
import { BOX_MIN, BOX_MAX } from '../config.mjs';

const r1 = (v) => Math.round(v * 10) / 10;

/** A starting site for proteins with no known site: the backbone atom nearest the middle. */
export function defaultSiteFor(structure) {
  let best = structure.ca[0], bd = Infinity;
  for (const a of structure.ca) {
    const d = Math.hypot(a.x - structure.center[0], a.y - structure.center[1], a.z - structure.center[2]);
    if (d < bd) { bd = d; best = a; }
  }
  return { x: r1(best.x), y: r1(best.y), z: r1(best.z) };
}

/**
 * @param o.ctx            app context (stage, store)
 * @param o.getStructure   () => parsed structure or null while loading
 * @param o.getRef         () => the protein reference (for the suggested site)
 * @param o.getValue       () => { site:{x,y,z}, box }
 * @param o.setValue       ({ site?, box? }) => void   (persist the change)
 * @param o.getPepLen      () => number of residues, to size the settled peptide
 */
export function createSitePicker(o) {
  const { stage, store } = o.ctx;
  const pickMsg = h('p', { class: 'small muted', 'aria-live': 'polite' });
  const assessEl = h('div', { class: 'msg empty', role: 'status' });
  const atomsEl = h('span', { class: 'coords-note' });

  const sizeOut = h('output', { class: 'mono' });
  const slider = h('input', {
    type: 'range', min: BOX_MIN, max: BOX_MAX, step: 1, 'aria-label': 'Box size in ångströms',
    onInput: (e) => { o.setValue({ box: Number(e.target.value) }); push(false); },
    onChange: () => settle(),
  });

  const coord = (axis) => h('input', {
    class: 'field mono', type: 'number', step: '0.1', 'aria-label': `Box centre ${axis.toUpperCase()}`,
    onInput: (e) => {
      const v = parseFloat(e.target.value);
      if (!Number.isFinite(v)) return;
      o.setValue({ site: { ...o.getValue().site, [axis]: v } });
      push(true);
    },
  });
  const cx = coord('x'), cy = coord('y'), cz = coord('z');

  const suggestNote = h('span', { class: 'small muted' });
  const suggestBtn = h('button', { class: 'btn sm', type: 'button', onClick: useSuggested }, icon('check', 16), 'Use the suggested site');

  const coordsBlock = h('details', { class: 'adv' },
    h('summary', {}, h('span', {}, 'Coordinates ', h('span', { class: 'tech' }, 'Box centre in ångströms (x, y, z)'))),
    h('div', { class: 'stack' },
      h('div', { class: 'grid-3' }, cx, cy, cz),
      h('p', { class: 'coords-note' }, 'Filled in for you as you move the box. You can also type exact numbers.')));

  const el = h('div', { class: 'stack', style: { display: 'flex', flexDirection: 'column', gap: '14px' } },
    h('p', { class: 'small muted' }, 'Drag the box to move it, drag a corner to resize it, or click the protein to jump there. Drag empty space to rotate.'),
    h('div', {}, h('div', { class: 'row', style: { justifyContent: 'space-between' } }, h('label', { class: 'field-label' }, 'Box size ', h('span', { class: 'tech' }, 'Search-box edge length')), sizeOut), slider),
    assessEl, atomsEl, pickMsg,
    h('div', { class: 'row' }, suggestBtn, suggestNote),
    coordsBlock,
    h('p', { class: 'coords-note' }, 'Keyboard: click the 3D view, then arrow keys move the box, Page Up/Down move it in depth, and + / − resize it.'));

  function center() { const s = o.getValue().site; return [s.x, s.y, s.z]; }

  function assess() {
    const st = o.getStructure();
    if (!st) return null;
    return assessBox(st, center(), o.getValue().box);
  }

  function readout(a) {
    const v = o.getValue();
    sizeOut.textContent = `${v.box} Å`;
    slider.value = v.box;
    if (document.activeElement !== cx) cx.value = v.site.x;
    if (document.activeElement !== cy) cy.value = v.site.y;
    if (document.activeElement !== cz) cz.value = v.site.z;
    if (!a) { assessEl.className = 'msg empty'; assessEl.replaceChildren('Loading the protein…'); atomsEl.textContent = ''; return; }
    assessEl.className = `msg ${a.level}`;
    assessEl.replaceChildren(icon(a.level === 'ok' ? 'check' : 'alert', 18), h('span', {}, a.message));
    atomsEl.textContent = a.atomsInside ? `${a.atomsInside.toLocaleString()} protein atoms inside the box` : '';
    const ref = o.getRef();
    suggestBtn.hidden = !ref?.site;
    suggestNote.textContent = ref?.site ? `Suggested: ${ref.siteNote}, from the crystal structure. A starting point, not a prediction.` : '';
    o.onAssess?.(a);
  }

  /** Send the current numbers to the 3D box. */
  function push(movePeptide) {
    const st = o.getStructure();
    if (!st) return;
    const a = assess();
    stage.setBox({ center: center(), size: o.getValue().box, level: a.level });
    if (movePeptide) stage.moveDock(center());
    readout(a);
  }

  /** Let the peptide settle beside the box centre (pushed out of the protein). */
  function settle() {
    const st = o.getStructure();
    const n = o.getPepLen();
    if (!st || !n) return;
    const seed = hashString(`${center().map((v) => v.toFixed(0)).join(',')}|${n}`);
    stage.setDockPose(placeHelixAt(st, n, center(), mulberry32(seed)));
  }

  function useSuggested() {
    const ref = o.getRef();
    if (!ref?.site) return;
    o.setValue({ site: { ...ref.site }, box: ref.box || o.getValue().box });
    push(true);
    settle();
    stage.pingAt(center());
    pickMsg.textContent = `Box moved to ${ref.siteNote}.`;
  }

  const handlers = {
    onBoxChange({ center: c, size }, { final }) {
      o.setValue({ site: { x: r1(c[0]), y: r1(c[1]), z: r1(c[2]) }, box: Math.round(size) });
      const a = assess();
      stage.setBoxLevel(a.level);
      readout(a);
      if (final) settle();
    },
    onPick(hit) {
      o.setValue({ site: { x: r1(hit.local[0]), y: r1(hit.local[1]), z: r1(hit.local[2]) } });
      push(true);
      settle();
      pickMsg.textContent = `Box moved to chain ${hit.chain}, residue ${hit.res} (${hit.resName}).`;
    },
  };

  return {
    el,
    /** Show the box in 3D and start listening for drags. */
    activate() { stage.editBox(handlers); stage.setPeptideMode('dock'); push(false); settle(); },
    /** Show the box without letting it be edited (Review step). */
    show() { stage.editBox(null); stage.setPeptideMode('dock'); push(false); },
    deactivate() { stage.editBox(null); },
    /** Call after the protein loads or the value changes from outside. */
    refresh() { readout(assess()); },
    setExpertOpen(open) { coordsBlock.open = open; },
  };
}
