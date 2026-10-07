// compare.js — "Compare two proteins": one peptide, two proteins, one answer.
//   ΔΔG = ΔG(target) − ΔG(off-target), shown with its 95% interval and a three-way verdict.
// Also exports compareResultView(), which results.js uses to show a finished comparison.

import { h, readText } from './dom.mjs';
import { icon } from './icons.mjs';
import { createSitePicker, defaultSiteFor } from './sitepicker.mjs';
import { loadProtein, registerUpload } from '../structures.mjs';
import { validatePeptide } from '../peptide.mjs';
import { compareJob, findProtein } from '../jobs.mjs';
import { fmt, fmtSigned, selectivityVerdict } from '../interpret.mjs';
import { adapter } from '../adapter.mjs';
import { BOX_DEFAULT, EXAMPLE, THOROUGHNESS } from '../config.mjs';
import { commandBlock } from './results.mjs';
import { toast } from './toast.mjs';

const SIDES = [
  { id: 'target', label: 'Target', hint: 'The protein you want the peptide to bind' },
  { id: 'offTarget', label: 'Off-target', hint: 'A look-alike you want it to ignore' },
];
const SCORE_NAMES = { mmgbsa_dg: 'MM-GBSA ΔG', vina_score: 'Vina physics score', hybrid_score: 'hybrid score' };
const sideName = (id) => (id === 'target' ? 'Target' : 'Off-target');

export function mountCompare(ctx) {
  const { store, stage, go, runner } = ctx;
  let alive = true, view = 'main';
  const structs = { target: null, offTarget: null };
  const C = () => store.get().compare;
  const setC = (patch) => store.merge('compare', patch);
  const setSide = (side, patch) => setC({ [side]: { ...C()[side], ...patch } });
  const pep = () => validatePeptide(C().peptide);

  // pre-filled defaults: Tau vs α-synuclein with the example peptide
  const { proteins } = store.get();
  if (!C().target.proteinRef && proteins.length) {
    const t = findProtein(proteins, 'tau'), o = findProtein(proteins, 'asyn');
    setC({
      peptide: C().peptide || EXAMPLE.peptide,
      target: { proteinRef: t, site: { ...t.site }, box: t.box || BOX_DEFAULT },
      offTarget: { proteinRef: o, site: { ...o.site }, box: o.box || BOX_DEFAULT },
    });
  }

  const el = h('section', { class: 'screen compare' });

  // ---- loading structures -----------------------------------------------------------------------
  async function loadSide(side) {
    const ref = C()[side].proteinRef;
    if (!ref) return;
    try {
      const { structure } = await loadProtein(ref);
      if (!alive || C()[side].proteinRef !== ref) return;
      structs[side] = structure;
      if (!C()[side].site) setSide(side, { site: defaultSiteFor(structure) });
      if (C().focus === side && view === 'main') stage.setProtein(structure);
      updateRun();
    } catch (err) { toast(err.message, 6000); }
  }

  // ---- main view --------------------------------------------------------------------------------
  const runBtn = h('button', { class: 'btn primary', type: 'button', onClick: run }, 'Compare', icon('arrow', 16));
  const runMsg = h('p', { class: 'small muted' });
  const pepMsg = h('div', { class: 'msg empty', role: 'status' });
  let cards = {};

  function pepInput() {
    const input = h('input', { class: 'field mono', id: 'cmp-pep', type: 'text', autocomplete: 'off', spellcheck: 'false', placeholder: 'e.g. LIYKWVNK', style: { textTransform: 'uppercase' }, 'aria-describedby': 'cmp-pep-msg', onInput: () => { setC({ peptide: input.value }); checkPep(); } });
    input.value = C().peptide;
    pepMsg.id = 'cmp-pep-msg';
    function checkPep() {
      const v = pep();
      pepMsg.className = `msg ${v.level}`;
      pepMsg.replaceChildren(v.level === 'ok' || v.level === 'warn' ? icon(v.level === 'ok' ? 'check' : 'alert', 18) : v.level === 'error' ? icon('alert', 18) : '', v.message);
      if (v.ok) stage.setPeptide(v.seq.length);
      updateRun();
    }
    queueMicrotask(checkPep);
    return input;
  }

  function proteinCard({ id, label, hint }) {
    const side = () => C()[id];
    const select = h('select', { class: 'field', 'aria-label': `${label} protein`, onChange: onSelect });
    const file = h('input', { type: 'file', accept: '.pdb,.ent,.txt', hidden: true, onChange: onUpload });
    const about = h('p', { class: 'small muted' });
    const siteLine = h('p', { class: 'small' });
    const adjust = h('button', { class: 'btn sm', type: 'button', onClick: () => editSite(id) }, icon('columns', 15), 'Adjust the site in 3D');

    function fillOptions() {
      const cur = side().proteinRef;
      select.replaceChildren(
        ...ctx.store.get().proteins.map((p) => h('option', { value: p.key, selected: cur?.key === p.key }, `${p.name} (${p.pdb})`)),
        cur && !cur.key ? h('option', { value: '__cur', selected: true }, `${cur.name} (your file)`) : null,
        h('option', { value: '__upload' }, 'Upload my own PDB file…'));
    }
    function refresh() {
      const s = side(), ref = s.proteinRef;
      fillOptions();
      about.textContent = ref?.about || (ref?.custom ? 'Your uploaded structure.' : '');
      const where = ref?.site && s.site && ref.site.x === s.site.x && ref.site.y === s.site.y && ref.site.z === s.site.z ? `Suggested site: ${ref.siteNote}.` : 'Custom site.';
      siteLine.replaceChildren(`${where} Box ${s.box} Å`, h('span', { class: 'expert-only mono' }, s.site ? `  at (${s.site.x}, ${s.site.y}, ${s.site.z})` : ''));
    }
    function onSelect() {
      if (select.value === '__upload') { file.click(); refresh(); return; }
      if (select.value === '__cur') return;
      const p = ctx.store.get().proteins.find((x) => x.key === select.value);
      setSide(id, { proteinRef: p, site: { ...p.site }, box: p.box || BOX_DEFAULT });
      structs[id] = null;
      refresh(); loadSide(id); updateRun();
    }
    async function onUpload(e) {
      const f = e.target.files[0];
      if (!f) return;
      try {
        const ref = registerUpload(f.name, await readText(f));
        await loadProtein(ref);
        setSide(id, { proteinRef: ref, site: null, box: BOX_DEFAULT });
        structs[id] = null;
        refresh(); await loadSide(id); refresh();
      } catch (err) { toast(err.message, 6000); }
      e.target.value = '';
    }
    refresh();
    const node = h('article', { class: `card pcard ${id === 'target' ? 'target' : ''}`, onFocusin: () => focusSide(id), onPointerenter: () => focusSide(id) },
      h('span', { class: 'tag' }, label),
      h('p', { class: 'small muted' }, hint),
      h('div', {}, h('label', { class: 'field-label' }, 'Protein', h('span', { class: 'tech' }, 'Receptor structure')), select, file),
      about, siteLine, adjust);
    return { el: node, refresh };
  }

  function focusSide(id) {
    if (C().focus === id || view !== 'main') return;
    setC({ focus: id });
    if (structs[id]) stage.setProtein(structs[id]);
  }

  function updateRun() {
    const v = pep(), t = C().target, o = C().offTarget;
    const same = t.proteinRef && o.proteinRef && (t.proteinRef.id || t.proteinRef.key) === (o.proteinRef.id || o.proteinRef.key);
    const ready = v.ok && structs.target && structs.offTarget && !same && t.site && o.site;
    runBtn.disabled = !ready;
    runMsg.textContent = same ? 'Pick two different proteins to compare.' : !v.ok ? 'Enter a valid peptide to continue.' : !(structs.target && structs.offTarget) ? 'Loading the structures…' : '';
  }

  function run() {
    runner.start('compare', compareJob({ ...C(), peptide: pep().seq }), { backTo: '/compare' });
  }

  function renderMain() {
    view = 'main';
    stage.editBox(null); stage.setBox(null);
    cards = Object.fromEntries(SIDES.map((s) => [s.id, proteinCard(s)]));
    const slot = h('div', { class: 'stage-slot vs-slot', 'data-stage-slot': '' }, h('div', { class: 'legend' }, h('span', {}, h('i', { class: 'dot protein' }), 'Protein'), h('span', {}, h('i', { class: 'dot peptide' }), 'Peptide')));
    el.replaceChildren(
      h('div', { class: 'page-head' },
        h('h1', {}, 'Compare two proteins'),
        h('p', { class: 'lede' }, 'Does your peptide prefer one protein over another? Give it a target and a look-alike, and we’ll tell you.'),
        h('span', { class: 'tech' }, 'Selectivity, ΔΔG')),
      h('div', { class: 'note guided-only' }, icon('info', 18), h('div', {}, h('b', {}, 'Why compare? '), 'Comparing two proteins is more trustworthy than a single number, because error they share cancels out.')),
      h('div', { class: 'card panel', style: { marginTop: '14px' } },
        h('label', { class: 'field-label', for: 'cmp-pep' }, 'Peptide', h('span', { class: 'tech' }, 'Amino acid sequence')), pepInput(), pepMsg),
      h('div', { class: 'vs-grid' }, cards.target.el, slot, cards.offTarget.el),
      h('div', { class: 'card panel', style: { maxWidth: '520px', margin: '0 auto 18px' } },
        h('span', { class: 'field-label', style: { margin: 0 } }, 'How thorough? ', h('span', { class: 'tech' }, 'Number of poses per protein; a comparison runs both')),
        h('div', { class: 'choice-grid three', role: 'radiogroup', 'aria-label': 'How thorough?' },
          THOROUGHNESS.map((t) => h('button', { class: 'choice', type: 'button', role: 'radio', 'aria-checked': String(C().thorough === t.id),
            onClick: (e) => { setC({ thorough: t.id }); e.currentTarget.parentElement.querySelectorAll('.choice').forEach((b, i) => b.setAttribute('aria-checked', String(THOROUGHNESS[i].id === t.id))); } },
          t.label, h('small', {}, `${t.poses} poses`))))),
      h('div', { class: 'row', style: { justifyContent: 'center', flexDirection: 'column', gap: '8px' } }, runBtn, runMsg));
    stage.setSlot(slot);
    stage.setPeptideMode('float');
    const s = structs[C().focus];
    if (s) stage.setProtein(s);
    updateRun();
  }

  // ---- adjusting one side's site in 3D ------------------------------------------------------------
  function editSite(id) {
    if (!structs[id]) { toast('Still loading that protein. Try again in a moment.'); return; }
    view = 'edit';
    setC({ focus: id });
    stage.setProtein(structs[id]);
    stage.setPeptide(pep().ok ? pep().seq.length : 8);
    const picker = createSitePicker({
      ctx, getStructure: () => structs[id], getRef: () => C()[id].proteinRef,
      getValue: () => ({ site: C()[id].site, box: C()[id].box }),
      setValue: (v) => setSide(id, v), getPepLen: () => (pep().ok ? pep().seq.length : 8),
    });
    const slot = h('div', { class: 'stage-slot', 'data-stage-slot': '' }, h('p', { class: 'hint' }, `Drag the box over the ${sideName(id).toLowerCase()} binding spot.`),
      h('div', { class: 'legend' }, h('span', {}, h('i', { class: 'dot protein' }), 'Protein'), h('span', {}, h('i', { class: 'dot peptide' }), 'Peptide'), h('span', {}, h('i', { class: 'dot box' }), 'Search box')));
    const done = () => { picker.deactivate(); renderMain(); };
    el.replaceChildren(h('div', { class: 'split' }, slot,
      h('aside', { class: 'glass panel setup-panel', 'aria-label': 'Adjust site' },
        h('h1', { class: 'panel-title', tabindex: '-1' }, `Adjust the ${sideName(id).toLowerCase()} site`, h('span', { class: 'tech' }, C()[id].proteinRef.name)),
        picker.el,
        h('div', { class: 'panel-foot' }, h('span'), h('button', { class: 'btn primary', type: 'button', onClick: done }, icon('check', 16), 'Done')))));
    stage.setSlot(slot);
    picker.activate();
    picker.setExpertOpen(store.get().mode === 'expert');
    picker.refresh();
  }

  SIDES.forEach((s) => loadSide(s.id));
  renderMain();
  return { el, destroy() { alive = false; stage.editBox(null); } };
}

// ================================================================================================
// A finished comparison
// ================================================================================================

/** The ΔΔG bar: the 95% interval as a band, a thick mark for the estimate, and a line at zero. */
function ddgBar(ddg, ci) {
  const m = Math.max(3, Math.ceil(Math.max(Math.abs(ci[0]), Math.abs(ci[1]), Math.abs(ddg)) + 0.5));
  const pct = (v) => ((v + m) / (2 * m)) * 100;
  return h('div', {},
    h('div', { class: 'ddg-bar', role: 'img', 'aria-label': `ΔΔG ${fmt(ddg)}, 95% interval from ${fmt(ci[0])} to ${fmt(ci[1])} kcal/mol, with a line at zero` },
      h('span', { class: 'axis' }),
      h('span', { class: 'ci', style: { left: `${pct(ci[0])}%`, width: `${pct(ci[1]) - pct(ci[0])}%` } }),
      h('span', { class: 'est', style: { left: `${pct(ddg)}%` } }),
      h('span', { class: 'zero', style: { left: '50%' } }),
      h('span', { class: 'zero-label', style: { left: '50%' } }, 'No difference (0)')),
    h('div', { class: 'ddg-scale' }, h('span', {}, '← prefers the target'), h('span', {}, 'prefers the off-target →')));
}

/** Fill `host` with the comparison result and put the matching 3D view on the stage. */
export function compareResultView(ctx, result, host, isAlive) {
  const { stage, go, store } = ctx;
  const v = selectivityVerdict(result.ci[0], result.ci[1]);
  const hint = h('p', { class: 'hint', 'aria-live': 'polite' });
  const slot = h('div', { class: 'stage-slot', 'data-stage-slot': '' }, hint,
    h('div', { class: 'legend' }, h('span', {}, h('i', { class: 'dot protein' }), 'Protein'), h('span', {}, h('i', { class: 'dot peptide' }), 'Peptide pose'), h('span', {}, h('i', { class: 'dot box' }), 'Search box')));

  const tabs = SIDES.map((s) => h('button', { type: 'button', 'aria-pressed': 'false', onClick: () => show(s.id) }, s.label));
  function show(id) {
    const side = id === 'target' ? result.target : result.offTarget;
    tabs.forEach((b, i) => b.setAttribute('aria-pressed', String(SIDES[i].id === id)));
    hint.textContent = `${sideName(id)}: ${side.protein.name} (ΔG ${fmt(side.deltaG)})`;
    stage.setPeptide(result.peptide.length);
    if (side.ca) { stage.setPeptideMode('dock'); stage.setDockPose(Float32Array.from(side.ca)); }
    else { stage.setPeptideMode('float'); stage.setDockPose(null); } // pose files not available: no pose to show
    loadProtein(side.protein).then(({ structure }) => {
      if (!isAlive()) return;
      stage.setProtein(structure);
      stage.setBox({ center: [side.site.x, side.site.y, side.site.z], size: side.box, level: 'ok' });
    }).catch(() => toast('The protein file isn’t stored in History, so the 3D view is empty.', 6000));
  }

  const dgCard = (label, side, cls) => h('div', { class: `card ${cls}` }, h('span', { class: 'small muted' }, label), h('div', { class: 'v nums' }, fmt(side.deltaG)), h('span', { class: 'small muted' }, `${side.protein.name} · kcal/mol`));

  const panel = h('aside', { class: 'glass panel', 'aria-label': 'Comparison result' },
    h('h1', { class: 'visually-hidden', tabindex: '-1' }, `Result: ${result.name}`),
    h('div', { class: 'result-head' },
      h('span', { class: 'eyebrow' }, 'Selectivity', h('span', { class: 'tech' }, 'ΔΔG = ΔG(target) − ΔG(off-target)')),
      result.demo && h('span', { class: 'badge-demo', title: 'This result is simulated' }, 'Demo')),
    h('div', { class: 'big-number' }, h('span', { class: 'n nums' }, fmtSigned(result.ddg)), h('span', { class: 'unit' }, 'kcal/mol')),
    h('p', { class: 'err-line' }, '95% interval: ', h('b', { class: 'nums' }, `${fmt(result.ci[0])} to ${fmt(result.ci[1])}`)),
    h('div', { class: 'ddg-card' }, ddgBar(result.ddg, result.ci)),
    h('div', { class: `verdict ${v.id}`, role: 'status' }, icon(v.id === 'none' ? 'info' : 'check', 22), h('div', {}, h('h3', {}, v.title), h('p', { class: 'small' }, v.text))),
    h('p', { class: 'guide-line' }, 'The verdict follows from where the interval sits relative to zero.'),
    h('div', { class: 'two-dg' }, dgCard('Target ΔG', result.target, 'target-card'), dgCard('Off-target ΔG', result.offTarget, '')),
    h('p', { class: 'small muted' }, 'Comparing two proteins is more trustworthy than a single number, because error they share cancels out.'),
    result.scoreField && result.scoreField !== 'demo' && h('p', { class: 'small muted' }, `Both ΔG values here are the mean of the top ${result.topK ?? 'few'} poses, using the ${SCORE_NAMES[result.scoreField] || result.scoreField}. Comparisons use this physics-based score because the entropy correction in the full ΔG model adds noise to a difference.`),
    Math.abs(result.ddg) < 1 && !result.demo && h('p', { class: 'small muted' }, 'Below about 1 kcal/mol this is a direction, not a measurement: the “charged floor” is the same size.'),
    result.demo && h('p', { class: 'small muted' }, 'Demo: these numbers are simulated from your inputs, not predicted by the docking program.'),
    commandBlock(result),
    h('div', { class: 'panel-foot' },
      h('button', { class: 'btn', type: 'button', onClick: () => { store.merge('compare', { editing: null }); go('/compare'); } }, 'Compare again'),
      h('button', { class: 'btn primary', type: 'button', onClick: () => go('/') }, 'Done')));

  host.replaceChildren(
    h('div', { class: 'row', style: { marginBottom: '14px', justifyContent: 'space-between' } },
      h('div', { class: 'row' }, h('button', { class: 'btn ghost sm', type: 'button', onClick: () => go('/') }, icon('back', 15), 'Home'), h('span', { class: 'mono muted small' }, result.name)),
      h('div', { class: 'seg', role: 'group', 'aria-label': 'Which protein to show in 3D' }, tabs)),
    h('div', { class: 'split' }, slot, panel));
  stage.setSlot(slot);
  show('target');
}
