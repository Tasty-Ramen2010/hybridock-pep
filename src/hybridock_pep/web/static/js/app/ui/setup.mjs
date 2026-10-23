// setup.js — "Predict binding": three steps (Protein, Peptide, Run). The binding site is already known for every protein in
// the catalog, so it is set quietly; only a protein with no known site (an uploaded file) gets a fourth step to mark one.
// The 3D model sits on the left (the stage), a frosted-glass panel on the right.
// On a narrow screen the panel drops below the model.

import { h, readText } from './dom.mjs';
import { icon } from './icons.mjs';
import { createSitePicker, defaultSiteFor } from './sitepicker.mjs';
import { loadProtein, registerUpload, fetchFromRCSB } from '../structures.mjs';
import { cleanSequence, peptideStats, validatePeptide } from '../peptide.mjs';
import { namedPeptide } from '../peptides.mjs';
import { buildDockCommand } from '../command.mjs';
import { dockJob, findProtein, poseCount } from '../jobs.mjs';
import { adapter, estimateTrusted, noGpu } from '../adapter.mjs';
import { BOX_DEFAULT, EXPERT_DEFAULTS, LIMITS, THOROUGHNESS, ad4Available, longModelAvailable, poseCap, settingProblems, sharedLimits, timingFor } from '../config.mjs';
import { toast } from './toast.mjs';
import { fmtDuration } from './dom.mjs';

const STEP_LABELS = { protein: 'Protein', peptide: 'Peptide', site: 'Binding site', review: 'Run' };
// offered after the chosen protein's own example; each has a name and a one-line description (see peptides.mjs)
const EXAMPLE_PEPTIDES = ['ETFSDLWKLLPE', 'LSECLKRIGDELDS', 'LVVVGACGV'];

const note = (...kids) => h('div', { class: 'note guided-only' }, icon('info', 18), h('div', {}, ...kids));

export function mountSetup(ctx) {
  const { store, stage, go, runner } = ctx;
  let alive = true, structure = null, loadToken = 0, lastAssess = null, lastEstimate = null;

  const S = () => store.get().setup;
  const setS = (patch) => store.merge('setup', patch);
  const pepCheck = () => validatePeptide(S().peptide);
  const pepLen = () => (pepCheck().ok ? pepCheck().seq.length : 8);
  /** The steps for this protein: a protein with a known site skips "Binding site" unless Expert asked to adjust it. */
  const stepIds = () => (S().proteinRef?.site && !S().adjustSite ? ['protein', 'peptide', 'review'] : ['protein', 'peptide', 'site', 'review']);
  const stepId = () => stepIds()[Math.min(S().step, stepIds().length) - 1];

  // ---- the site picker (step 3) -----------------------------------------------------------------
  const picker = createSitePicker({
    ctx,
    getStructure: () => structure,
    getRef: () => S().proteinRef,
    getValue: () => ({ site: S().site, box: S().box }),
    setValue: (v) => setS(v),
    getPepLen: pepLen,
    onAssess: (a) => { lastAssess = a; updateNav(); },
  });

  // ---- frame ----------------------------------------------------------------------------------
  const stepsEl = h('ol', { class: 'steps', 'aria-label': 'Progress' });
  const hint = h('p', { class: 'hint' });
  const slot = h('div', { class: 'stage-slot', 'data-stage-slot': '' }, hint,
    h('div', { class: 'legend' },
      h('span', {}, h('i', { class: 'dot protein' }), 'Protein'),
      h('span', {}, h('i', { class: 'dot peptide' }), 'Peptide'),
      h('span', { class: 'site-legend', hidden: true }, h('i', { class: 'dot box' }), 'Search box')));
  const panel = h('aside', { class: 'glass panel setup-panel', 'aria-label': 'Setup' });
  const el = h('section', { class: 'screen setup' }, stepsEl, h('div', { class: 'split' }, slot, panel));
  let nextBtn = null;
  let reviewProblems = []; // what the Review step currently objects to; Run stays off while there is any

  function renderSteps() {
    const cur = S().step;
    stepsEl.replaceChildren(...stepIds().map((id, i) => h('li', { class: i + 1 < cur ? 'done' : '', 'aria-current': i + 1 === cur ? 'step' : null },
      h('span', { class: 'num' }, i + 1 < cur ? icon('check', 14) : String(i + 1)), h('span', { class: 'name' }, STEP_LABELS[id]))));
  }

  function goStep(n) {
    setS({ step: n });
    renderSteps();
    renderPanel();
    applyStage();
    scrollTo({ top: 0 }); // each step starts at the top, so its title is never hidden under the nav bar
    panel.querySelector('h1')?.focus({ preventScroll: true });
  }

  // ---- what the 3D stage shows at each step -----------------------------------------------------
  function applyStage() {
    const { siteMode } = S();
    const id = stepId();
    const marking = stepIds().includes('site'); // a protein with no known site: the box is marked by hand
    stage.setPeptide(pepLen());
    const legend = slot.querySelector('.site-legend');
    legend.hidden = !(marking && (id === 'site' || id === 'review') && siteMode === 'known');
    if (marking && id === 'site' && siteMode === 'known') {
      hint.textContent = 'Drag the box over the spot where the peptide should bind.';
      picker.activate();
    } else if (marking && id === 'review' && siteMode === 'known') {
      hint.textContent = 'The peptide is settled inside your box. Drag to look around.';
      picker.show();
    } else {
      picker.deactivate();
      stage.setBox(null);
      stage.setPeptideMode('float');
      hint.textContent = id === 'site' ? 'HybriDock-Pep will search the whole protein.' : 'Drag to turn the protein.';
    }
  }

  // ---- loading a protein ------------------------------------------------------------------------
  const status = h('div', { class: 'msg empty', role: 'status' });
  async function selectProtein(ref, { keep = false } = {}) {
    const token = ++loadToken;
    // `keep` = coming back to a half-finished setup: leave the user's site and box alone.
    setS(keep ? { proteinRef: ref } : { proteinRef: ref, adjustSite: false, site: ref.site ? { ...ref.site } : null, box: ref.box || BOX_DEFAULT });
    status.className = 'msg empty';
    status.replaceChildren('Loading the structure…');
    renderSteps(); // a protein with no known site (an upload) has a fourth step, so the progress list changes with the choice
    updateNav();
    try {
      const { structure: st } = await loadProtein(ref);
      if (!alive || token !== loadToken) return;
      structure = st;
      if (!S().site) setS({ site: defaultSiteFor(st) });
      stage.setProtein(st);
      status.className = 'msg ok';
      status.replaceChildren(icon('check', 18), `${st.ca.length.toLocaleString()} residues loaded.`);
      picker.refresh();
      if (S().step >= 2) applyStage();
    } catch (err) {
      if (!alive || token !== loadToken) return;
      structure = null;
      status.className = 'msg error';
      status.replaceChildren(icon('alert', 18), err.message);
    }
    updateNav();
  }

  // ---- panel: one builder per step -------------------------------------------------------------
  function title(text, tech) {
    return h('h1', { class: 'panel-title', tabindex: '-1' }, text, h('span', { class: 'tech' }, tech));
  }

  function stepProtein() {
    const proteins = store.get().proteins;
    const search = h('input', { class: 'field', type: 'search', placeholder: 'Search any protein or PDB ID…', 'aria-label': 'Search for a protein or PDB ID', autocomplete: 'off', onInput: renderList });
    const list = h('div', { class: 'pick-list', role: 'radiogroup', 'aria-label': 'Proteins' });
    const file = h('input', { type: 'file', accept: '.pdb,.ent,.txt', hidden: true, onChange: onUpload });

    function row(ref, about, id) {
      const on = S().proteinRef && (S().proteinRef.id || S().proteinRef.key) === (ref.id || ref.key);
      return h('button', { class: 'pick', type: 'button', role: 'radio', 'aria-checked': String(!!on), onClick: () => { selectProtein(ref); renderList(); } },
        h('span', { class: 'pick-main' }, ref.name, h('small', {}, about)),
        h('span', { class: 'pick-meta' }, h('span', { class: 'mono' }, id), on && icon('check', 18)));
    }
    function renderList() {
      const q = search.value.trim().toLowerCase();
      const rows = proteins
        .filter((p) => !q || `${p.name} ${p.pdb} ${p.about}`.toLowerCase().includes(q))
        .map((p) => row(p, p.about, `PDB ${p.pdb}`));
      const cur = S().proteinRef;
      if (cur && !cur.key && !q) rows.unshift(row(cur, 'Loaded', cur.pdb ? `PDB ${cur.pdb}` : 'Your file'));
      if (/^[0-9][a-z0-9]{3}$/i.test(q) && !proteins.some((p) => p.pdb.toLowerCase() === q)) {
        rows.push(h('button', { class: 'pick', type: 'button', onClick: async () => {
          try { const ref = await fetchFromRCSB(q); selectProtein(ref); search.value = ''; renderList(); toast(`Loaded ${ref.name} from the Protein Data Bank.`); }
          catch (err) { toast(err.message, 6000); }
        } }, h('span', {}, `Load PDB ${q.toUpperCase()} from the Protein Data Bank`, h('small', {}, 'Downloads the structure so you can use it')), icon('download', 18)));
      }
      list.replaceChildren(...(rows.length ? rows : [h('p', { class: 'muted small' }, 'No match. Try a PDB ID like 1YCR, or upload a file.')]));
    }
    async function onUpload(e) {
      const f = e.target.files[0];
      if (!f) return;
      try {
        const ref = registerUpload(f.name, await readText(f));
        await loadProtein(ref); // validates the file before we accept it
        selectProtein(ref);
        renderList();
      } catch (err) { toast(err.message, 6000); }
      e.target.value = '';
    }
    renderList();
    if (S().proteinRef) status.replaceChildren(structure ? icon('check', 18) : '', structure ? `${structure.ca.length.toLocaleString()} residues loaded.` : '');
    return [
      title('Choose a protein', 'Receptor structure (PDB)'),
      note('The protein is what your peptide tries to stick to. Pick one below, search the Protein Data Bank, or upload your own structure.'),
      search, list,
      h('div', { class: 'row' }, h('button', { class: 'btn sm', type: 'button', onClick: () => file.click() }, icon('upload', 16), 'Upload my own PDB file'), file),
      status,
    ];
  }

  function stepPeptide() {
    const ta = h('textarea', {
      class: 'field mono', rows: 2, spellcheck: 'false', autocomplete: 'off', autocapitalize: 'characters', style: { textTransform: 'uppercase', resize: 'vertical' },
      id: 'pep-input', placeholder: 'e.g. LIYKWVNK', 'aria-describedby': 'pep-msg', onInput: check,
      onBlur: () => { const v = pepCheck(); if (v.seq) ta.value = v.seq; },
    });
    ta.value = S().peptide;
    const msg = h('div', { class: 'msg empty', id: 'pep-msg', role: 'status' });
    const stats = h('p', { class: 'coords-note' });
    const nameLine = h('p', { class: 'small pep-name' });
    function check() {
      setS({ peptide: ta.value });
      const v = pepCheck();
      ta.setAttribute('aria-invalid', String(v.level === 'error'));
      const ic = { ok: 'check', warn: 'alert', error: 'alert' }[v.level];
      msg.className = `msg ${v.level}`;
      msg.replaceChildren(ic ? icon(ic, 18) : '', v.message);
      if (v.ok) {
        const st = peptideStats(v.seq);
        const long = v.seq.length >= Number(S().expert.longCheckpointThreshold);
        stats.replaceChildren(`${st.length} amino acids, about ${Math.round(st.mass).toLocaleString()} Da, net charge ${st.charge > 0 ? '+' : st.charge < 0 ? '−' : ''}${Math.abs(st.charge)}`,
          h('span', { class: 'expert-only' }, long ? (longModelAvailable(adapter.env) ? '. Uses the long-peptide model.' : '. The long-peptide model isn’t installed here, so the standard model is used.') : ''));
        stage.setPeptide(v.seq.length);
        const known = namedPeptide(v.seq);
        nameLine.replaceChildren(...(known ? [h('b', {}, known.name), ` — ${known.about}`] : []));
      } else { stats.textContent = ''; nameLine.replaceChildren(); }
      updateNav();
    }
    const examples = [...new Set([S().proteinRef?.examplePeptide, ...EXAMPLE_PEPTIDES].filter(Boolean))].slice(0, 4);
    queueMicrotask(check);
    return [
      title('Enter a peptide', 'Peptide sequence'),
      note('A peptide is a short chain of amino acids, one letter each (3 to 30). Longer ones take longer and are harder to predict.'),
      h('div', {}, h('label', { class: 'field-label', for: 'pep-input' }, 'Amino acid sequence'), ta),
      msg, stats, nameLine,
      h('div', {}, h('p', { class: 'small muted', style: { marginBottom: '6px' } }, 'Or start from a real example'),
        h('div', { class: 'pep-list' }, examples.map((seq) => {
          const known = namedPeptide(seq);
          return h('button', { class: 'pep-pick', type: 'button', onClick: () => { ta.value = seq; check(); } },
            h('span', { class: 'pep-pick-name' }, known?.name || 'Example peptide'),
            known && h('span', { class: 'pep-pick-about' }, known.about),
            h('span', { class: 'pep-pick-seq mono' }, seq));
        }))),
    ];
  }

  function stepSite() {
    if (sharedLimits(adapter.env) && S().siteMode === 'find') setS({ siteMode: 'known' }); // not offered on a shared computer
    const mode = S().siteMode;
    const shared = sharedLimits(adapter.env);
    const choice = (id, label, sub) => h('button', { class: 'choice', type: 'button', role: 'radio', 'aria-checked': String(mode === id), disabled: !!(shared && id === 'find'), onClick: () => { setS({ siteMode: id }); renderPanel(); applyStage(); } }, label, h('small', {}, sub));
    return [
      title('Choose where it binds', 'Search box (grid box)'),
      note('The box tells the program where to look. A small box around the real binding spot is faster and more accurate.'),
      h('div', { class: 'choice-grid two', role: 'radiogroup', 'aria-label': 'How to pick the binding site' },
        choice('known', 'I know where it binds', 'Place a box on the protein'),
        choice('find', 'Find the pocket for me', sharedLimits(adapter.env) ? 'Not on the shared computer' : 'Search the whole protein')),
      mode === 'known' ? picker.el : h('div', { class: 'stack', style: { display: 'flex', flexDirection: 'column', gap: '10px' } },
        h('p', {}, 'HybriDock-Pep will look across the whole protein, group the best spots, and dock your peptide into each one.'),
        h('p', { class: 'small muted' }, 'This takes longer than a known site and is a good first step when you aren’t sure where to aim. ',
          h('span', { class: 'expert-only' }, 'No --site or --box is sent; the command uses --blind.'))),
    ];
  }

  function stepReview() {
    const s = S();
    const v = pepCheck();
    const cap = poseCap(adapter.env);
    if (poseCount(s.thorough) > cap) { const ok = THOROUGHNESS.filter((t) => t.poses <= cap).pop(); setS({ thorough: ok ? ok.id : 'quick' }); }
    const poses = poseCount(S().thorough);
    const cmd = h('pre', { class: 'cmd', tabindex: '0', 'aria-label': 'Command preview' });
    const eta = h('p', { class: 'small muted' });
    const problemsEl = h('div', { class: 'msg error', role: 'alert', hidden: true });
    const jobNow = () => dockJob({ ...S(), peptide: v.seq });
    let previewToken = 0, previewTimer = null;
    /** Show what the settings get wrong, in words, and keep "Run prediction" off until it is fixed. */
    function showProblems(list) {
      reviewProblems = list;
      problemsEl.hidden = !list.length;
      problemsEl.replaceChildren(...(list.length ? [icon('alert', 18), h('div', {}, list.map((m) => h('div', {}, m)))] : []));
      if (nextBtn) { nextBtn.disabled = list.length > 0; nextBtn.title = list.length ? 'Fix the setting above first' : ''; }
    }
    function refreshCmd() {
      const job = jobNow();
      cmd.textContent = buildDockCommand(job); // instant local version, replaced by the server's exact one below
      const local = settingProblems({ box: job.box, expert: job.expert });
      showProblems(local);
      const token = ++previewToken;
      const early = timingFor(job.poses, job.peptide.length);
      eta.textContent = early == null ? 'Working out how long this will take…' : `About ${fmtDuration(early)}.`;
      clearTimeout(previewTimer); // typing in a field fires this on every key: ask the server once typing pauses
      previewTimer = setTimeout(() => askServer(job, local, token), 220);
    }
    function askServer(job, local, token) {
      adapter.preview(job).then(({ command, estimateSeconds, problems = [] }) => {
        if (token !== previewToken) return; // a newer change superseded this one
        showProblems(local.length ? local : problems); // our wording first; the server's own messages when the local check found nothing
        if (command) cmd.textContent = command.replace(/ --/g, ' \\\n  --');
        // The measured table is the source of truth for the demo computer; the server's own figure is used on any other machine.
        // On the measured computer the server's figure is the best one: it is the table, corrected by how long this very peptide
        // really took last time. Anywhere else the table in the page, or the server's own model where that can be trusted.
        const measured = timingFor(job.poses, job.peptide.length);
        const serverKnows = adapter.kind === 'live' && !!adapter.env?.timing && estimateSeconds != null;
        lastEstimate = serverKnows ? estimateSeconds : measured ?? (estimateTrusted(adapter.env) ? estimateSeconds : null);
        eta.textContent = adapter.kind === 'demo'
          ? `Demo run: about ${estimateSeconds} seconds.`
          : lastEstimate == null ? (noGpu(adapter.env) ? 'This computer has no GPU, so a run usually takes tens of minutes. You can leave this tab open.' : 'This takes a few minutes. You can leave this tab open.')
            : `About ${fmtDuration(lastEstimate)}. You can leave this tab open.`;
      }).catch(() => { if (token === previewToken) { const m = timingFor(job.poses, job.peptide.length); lastEstimate = m; eta.textContent = m == null ? 'This takes a few minutes.' : `About ${fmtDuration(m)}.`; } });
    }

    const thorough = h('div', { class: 'choice-grid three', role: 'radiogroup', 'aria-label': 'How many poses?' },
      THOROUGHNESS.map((t) => h('button', { class: 'choice', type: 'button', role: 'radio', 'aria-checked': String(S().thorough === t.id), disabled: t.poses > cap,
        onClick: () => { setS({ thorough: t.id }); thorough.querySelectorAll('.choice').forEach((b, i) => b.setAttribute('aria-checked', String(THOROUGHNESS[i].id === t.id))); poseLine.textContent = t.hint; refreshCmd(); } },
      t.label, h('small', {}, (() => { const m = timingFor(t.poses, v.seq.length); return m == null ? t.hint : `≈ ${fmtDuration(m)}`; })()))));
    const poseLine = h('span', { class: 'small muted' }, THOROUGHNESS.find((t) => t.id === S().thorough)?.hint || '');

    const preset = !!(s.proteinRef?.site && !s.adjustSite);
    const site = s.siteMode === 'find'
      ? 'Find the pocket for me (searches the whole protein)'
      : preset
        ? [s.proteinRef.siteNote ? `Ready: ${s.proteinRef.siteNote}` : 'Ready', h('span', { class: 'expert-only' }, ' ', h('button', { class: 'btn sm ghost', type: 'button', onClick: () => { setS({ adjustSite: true }); goStep(3); } }, 'Adjust'))]
        : [`Box ${s.box} Å`, h('span', { class: 'expert-only' }, h('span', { class: 'mono' }, ` at (${s.site.x}, ${s.site.y}, ${s.site.z})`))];

    const copyBtn = h('button', { class: 'btn sm', type: 'button', onClick: async () => {
      try { await navigator.clipboard.writeText(cmd.textContent); toast('Command copied.'); } catch { toast('Couldn’t copy. Select the text and copy it by hand.'); }
    } }, icon('copy', 15), 'Copy');

    const ex = () => S().expert;
    const setEx = (patch) => { setS({ expert: { ...ex(), ...patch } }); refreshCmd(); };
    const num = (key, label, tech, props = {}) => h('div', {}, h('label', { class: 'field-label' }, label, h('span', { class: 'tech' }, tech)),
      h('input', { class: 'field mono', type: 'number', value: ex()[key], ...props, onInput: (e) => setEx({ [key]: e.target.value === '' ? '' : Number(e.target.value) }) }));
    const txt = (key, label, tech, ph = '') => h('div', {}, h('label', { class: 'field-label' }, label, h('span', { class: 'tech' }, tech)),
      h('input', { class: 'field mono', type: 'text', value: ex()[key], placeholder: ph, onInput: (e) => setEx({ [key]: e.target.value }) }));
    const tick = (key, label, tech) => h('label', { class: 'check' }, h('input', { type: 'checkbox', checked: !!ex()[key], onChange: (e) => setEx({ [key]: e.target.checked }) }), h('span', {}, label, h('small', {}, tech)));

    refreshCmd();
    return [
      title('Review and run', 'Run settings'),
      note('More poses is a more thorough search and a longer wait. All three finish in minutes.'),
      h('dl', { class: 'summary' },
        h('dt', {}, 'Protein'), h('dd', {}, s.proteinRef.name, s.proteinRef.pdb ? h('span', { class: 'mono muted' }, `  ${s.proteinRef.pdb}`) : ''),
        h('dt', {}, 'Peptide'), h('dd', {}, h('span', { class: 'mono' }, v.seq), h('span', { class: 'muted' }, `  ${v.seq.length} amino acids`)),
        h('dt', {}, 'Binding site'), h('dd', {}, site)),
      h('div', {}, h('div', { class: 'row', style: { justifyContent: 'space-between', marginBottom: '6px' } },
        h('span', { class: 'field-label', style: { margin: 0 } }, 'How many poses? ', h('span', { class: 'tech' }, 'Ways the peptide is tried on the protein')), poseLine), thorough),
      eta,
      problemsEl,
      sharedLimits(adapter.env) ? null : h('details', { class: 'adv expert-only' }, h('summary', {}, h('span', {}, 'Advanced settings ', h('span', { class: 'tech' }, 'Expert options for the docking run'))),
        h('div', { class: 'stack' },
          num('longCheckpointThreshold', 'Long-peptide model starts at', longModelAvailable(adapter.env) ? '--long-checkpoint-threshold (residues)' : '--long-checkpoint-threshold. The long-peptide model isn’t installed here, so this has no effect.', { min: LIMITS.longCheckpointThreshold[0], max: LIMITS.longCheckpointThreshold[1] }),
          h('div', {}, h('label', { class: 'field-label' }, 'Scoring mode', h('span', { class: 'tech' }, '--scoring')),
            h('select', { class: 'field', onChange: (e) => setEx({ scoring: e.target.value }) },
              ['vina', 'vina,ad4'].map((o) => h('option', { value: o, selected: ex().scoring === o, disabled: o === 'vina,ad4' && !ad4Available(adapter.env) }, o === 'vina' ? 'vina (default)' : ad4Available(adapter.env) ? 'vina + AD4 (telemetry)' : 'vina + AD4 (needs autogrid4: not installed here)')))),
          num('refineTopK', 'Refine the top poses (MM-GBSA)', '--refine-topk K (blank = off)', { min: 1, max: LIMITS.refineTopK[1], placeholder: 'off' }),
          tick('ultra', 'Ultra mode', '--ultra: the slow, high-certainty stack'),
          num('ultraK', 'Ultra mode K', '--ultra smoothing depth (used when Ultra is on)', { min: 1, max: LIMITS.ultraK[1] }),
          num('seed', 'Random seed', '--seed (blank = random). The same seed gives similar, not identical, results.', { min: 0, max: LIMITS.seed[1], placeholder: 'random' }),
          txt('inputPoses', 'Input-poses folder', '--input-poses (skips pose generation)', 'optional folder path'),
          tick('noMinimize', 'Skip pre-minimization', '--no-minimize'),
          tick('ensemble', 'Add the ensemble ΔG column', '--ensemble'),
          txt('calibration', 'Calibration file', '--calibration'),
          txt('outputDir', 'Output folder', '--output-dir', 'automatic: runs/studio/…'))),
      h('div', { class: 'expert-only' }, h('div', { class: 'row', style: { justifyContent: 'space-between', marginBottom: '6px' } }, h('span', { class: 'field-label', style: { margin: 0 } }, 'Exact command', h('span', { class: 'tech' }, 'What HybriDock-Pep will run')), copyBtn),
        h('div', { class: 'cmd-wrap' }, cmd)),
    ];
  }

  // ---- panel frame: step content + Back / Continue ----------------------------------------------
  function renderPanel() {
    const step = S().step;
    const id = stepId();
    const body = { protein: stepProtein, peptide: stepPeptide, site: stepSite, review: stepReview }[id]();
    const back = step > 1 ? h('button', { class: 'btn ghost', type: 'button', onClick: () => goStep(step - 1) }, icon('chevronL', 16), 'Back') : h('span');
    nextBtn = h('button', { class: 'btn primary', type: 'button', onClick: () => (id === 'review' ? run() : goStep(step + 1)) }, id === 'review' ? 'Run prediction' : 'Continue');
    panel.replaceChildren(...body.filter(Boolean), h('div', { class: 'panel-foot' }, back, nextBtn)); // (replaceChildren would print an empty slot as the word "null")
    updateNav();
    picker.setExpertOpen(store.get().mode === 'expert');
  }

  /** Enable Continue only when this step is complete. */
  function updateNav() {
    if (!nextBtn) return;
    const s = S();
    let ok = true;
    const id = stepId();
    if (id === 'protein') ok = !!(s.proteinRef && structure);
    if (id === 'peptide') ok = pepCheck().ok;
    if (id === 'site' && s.siteMode === 'known') ok = !!structure && !(lastAssess?.level === 'bad' && store.get().mode === 'guided');
    if (id === 'review') ok = reviewProblems.length === 0;
    nextBtn.disabled = !ok;
    nextBtn.title = ok ? '' : id === 'review' ? 'Fix the setting above first' : id === 'site' ? 'Move the box onto the protein first' : 'Finish this step to continue';
  }

  function run() {
    const s = S();
    runner.start('dock', { ...dockJob({ ...s, peptide: pepCheck().seq }), estimateSeconds: lastEstimate }, { backTo: '/predict' });
  }

  // keep Expert-only bits in sync when the mode flips mid-flow
  const unsub = store.subscribe((state, patch) => {
    if ('mode' in patch) { picker.setExpertOpen(state.mode === 'expert'); updateNav(); }
  });

  // ---- start ---------------------------------------------------------------------------------------
  // Pre-filled default: the first protein is already chosen (selectProtein sets it synchronously).
  const startRef = S().proteinRef || findProtein(store.get().proteins, 'mdm2') || store.get().proteins[0];
  if (startRef) selectProtein(startRef, { keep: !!S().proteinRef });
  renderSteps();
  renderPanel();
  applyStage();

  return {
    el,
    destroy() { alive = false; unsub(); picker.deactivate(); },
  };
}
