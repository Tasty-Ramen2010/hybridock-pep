// score.js — "Score a structure": upload a protein and an existing bound peptide pose, get one ΔG.
// Backend equivalent: `hybridock-pep crystal-score`.

import { h, readText } from './dom.mjs';
import { icon } from './icons.mjs';
import { loadProtein, registerUpload } from '../structures.mjs';
import { parsePDB, sequenceOf } from '../pdb.mjs';
import { validatePeptide } from '../peptide.mjs';
import { toast } from './toast.mjs';
import { adapter } from '../adapter.mjs';

export function mountScore(ctx) {
  const { store, stage, runner } = ctx;
  let alive = true;
  const S = () => store.get().score;
  const setS = (patch) => store.merge('score', patch);

  const msgProtein = h('div', { class: 'msg empty', role: 'status' });
  const msgPose = h('div', { class: 'msg empty', role: 'status' });
  const pepMsg = h('div', { class: 'msg empty', role: 'status' });
  const runBtn = h('button', { class: 'btn primary', type: 'button', onClick: run }, 'Score it');

  function dropZone({ title, sub, accept, onFile, current }) {
    const input = h('input', { type: 'file', accept, hidden: true, onChange: (e) => { if (e.target.files[0]) onFile(e.target.files[0]); e.target.value = ''; } });
    const zone = h('button', { class: `drop ${current ? 'has-file' : ''}`, type: 'button', onClick: () => input.click() },
      icon('upload', 22), h('span', { class: 'drop-text' }, h('b', {}, current || title), h('span', { class: 'small muted' }, current ? 'Click to choose a different file' : sub)), input);
    zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('over'));
    zone.addEventListener('drop', (e) => { e.preventDefault(); zone.classList.remove('over'); if (e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0]); });
    return zone;
  }

  async function chooseProtein(file) {
    try {
      const ref = registerUpload(file.name, await readText(file));
      const { structure } = await loadProtein(ref);
      setS({ protein: ref });
      stage.setProtein(structure);
      msgProtein.className = 'msg ok';
      msgProtein.replaceChildren(icon('check', 18), `${structure.ca.length.toLocaleString()} residues loaded.`);
      render();
    } catch (err) { toast(err.message, 6000); }
  }

  async function choosePose(file) {
    try {
      const text = await readText(file);
      const pepStruct = parsePDB(text); // throws a friendly error if there are no atoms
      const seq = sequenceOf(pepStruct);
      setS({ peptidePdb: { name: file.name, text }, peptide: validatePeptide(S().peptide).ok ? S().peptide : seq.replace(/X/g, '') });
      msgPose.className = 'msg ok';
      msgPose.replaceChildren(icon('check', 18), `${pepStruct.ca.length} residues found in the pose.`);
      render();
    } catch (err) { toast(err.message, 6000); }
  }

  function checkPep() {
    const v = validatePeptide(S().peptide);
    pepMsg.className = `msg ${v.level}`;
    pepMsg.replaceChildren(v.level === 'ok' ? icon('check', 18) : v.level === 'empty' ? '' : icon('alert', 18), v.message);
    const pose = S().peptidePdb;
    if (v.ok && pose) {
      const n = parsePDB(pose.text).ca.length;
      if (n && n !== v.seq.length) {
        pepMsg.className = 'msg warn';
        pepMsg.replaceChildren(icon('alert', 18), `The sequence has ${v.seq.length} letters but the pose file has ${n} residues. Check that they match.`);
      }
    }
    runBtn.disabled = !(v.ok && S().protein && pose);
  }

  function run() {
    const v = validatePeptide(S().peptide);
    runner.start('score', { kind: 'score', protein: S().protein, receptor: S().protein.file || `${S().protein.name}.pdb`, peptidePdb: S().peptidePdb, peptidePdbName: S().peptidePdb.name, peptide: v.seq, allowClashes: S().allowClashes }, { backTo: '/score' });
  }

  const body = h('div', { class: 'glass panel', style: { maxWidth: '640px' } });
  const slot = h('div', { class: 'stage-slot', 'data-stage-slot': '', style: { minHeight: '260px' } });
  const el = h('section', { class: 'screen score' },
    h('div', { class: 'page-head' }, h('h1', {}, 'Score a structure'), h('p', { class: 'lede' }, 'Already have a protein with a peptide bound to it? Upload both and get a binding strength straight away.')),
    h('div', { class: 'split', style: { gridTemplateColumns: 'minmax(0,1fr) minmax(340px, 480px)' } }, slot, body));

  function render() {
    const s = S();
    const input = h('input', { class: 'field mono', id: 'sc-pep', type: 'text', autocomplete: 'off', spellcheck: 'false', style: { textTransform: 'uppercase' }, placeholder: 'e.g. LIYKWVNK', 'aria-describedby': 'sc-pep-msg', onInput: () => { setS({ peptide: input.value }); checkPep(); } });
    input.value = s.peptide;
    pepMsg.id = 'sc-pep-msg';
    body.replaceChildren(
      h('div', { class: 'note guided-only' }, icon('info', 18), h('div', {}, h('b', {}, 'Why this matters. '), 'Scoring needs a pose that is already bound, like one from a crystal structure or another program. HybriDock-Pep won’t move it; it only measures how well it fits.')),
      h('div', {}, h('p', { class: 'field-label' }, '1. The protein', h('span', { class: 'tech' }, 'Receptor PDB, protein only')),
        dropZone({ title: 'Drop a protein file here', sub: 'A .pdb file, or click to browse', accept: '.pdb,.ent,.txt', onFile: chooseProtein, current: s.protein?.file }), msgProtein),
      h('div', {}, h('p', { class: 'field-label' }, '2. The bound peptide', h('span', { class: 'tech' }, 'Peptide pose PDB')),
        dropZone({ title: 'Drop the peptide pose here', sub: 'A .pdb file with just the peptide', accept: '.pdb,.ent,.txt', onFile: choosePose, current: s.peptidePdb?.name }), msgPose),
      h('div', {}, h('label', { class: 'field-label', for: 'sc-pep' }, '3. Its sequence', h('span', { class: 'tech' }, 'One-letter amino acid codes')), input, pepMsg),
      adapter.supports?.allowClashes && h('label', { class: 'check expert-only' }, h('input', { type: 'checkbox', checked: s.allowClashes, onChange: (e) => setS({ allowClashes: e.target.checked }) }), h('span', {}, 'Allow clashes', h('small', {}, '--allow-clashes: score even if the pose overlaps the protein'))),
      h('div', { class: 'panel-foot' }, h('span'), runBtn));
    checkPep();
  }

  render();
  if (S().protein) loadProtein(S().protein).then(({ structure }) => { if (alive) stage.setProtein(structure); }).catch(() => {});
  stage.setPeptide(8);
  stage.setPeptideMode('float');
  return { el, destroy() { alive = false; } };
}
