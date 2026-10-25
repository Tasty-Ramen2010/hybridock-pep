// Unit tests for the browser UI's pure logic (no browser needed). Run from the repo root:
//   node --test "tests/web_js/*.test.mjs"
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { parsePDB, assessBox, atomsInBox, nearestAtom, sequenceOf } from '../../src/hybridock_pep/web/static/js/app/pdb.mjs';
import { validatePeptide, cleanSequence, peptideStats } from '../../src/hybridock_pep/web/static/js/app/peptide.mjs';
import { buildDockCommand, buildCompareCommand, buildScoreCommand } from '../../src/hybridock_pep/web/static/js/app/command.mjs';
import { selectivityVerdict, fmt, roughGuide, kdFromDG } from '../../src/hybridock_pep/web/static/js/app/interpret.mjs';
import { demoDeltaG } from '../../src/hybridock_pep/web/static/js/app/adapter.mjs';
import { placeHelixAt } from '../../src/hybridock_pep/web/static/js/app/placement.mjs';
import { mulberry32 } from '../../src/hybridock_pep/web/static/js/app/stage/math.mjs';
import { initials, poseCount } from '../../src/hybridock_pep/web/static/js/app/jobs.mjs';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'src', 'hybridock_pep', 'web', 'static');
const proteins = JSON.parse(readFileSync(join(root, 'data/proteins.json'), 'utf8'));
const load = (p) => parsePDB(readFileSync(join(root, p.file), 'utf8'));

test('all bundled proteins parse and have a site inside the protein', () => {
  assert.equal(proteins.length, 8);
  for (const p of proteins) {
    const s = load(p);
    assert.ok(s.ca.length > 50, `${p.name} has a backbone`);
    const a = assessBox(s, [p.site.x, p.site.y, p.site.z], p.box);
    assert.notEqual(a.level, 'bad', `${p.name} suggested site is on the protein (${a.message})`);
  }
});

test('assessBox: empty space is flagged in plain language', () => {
  const s = load(proteins[1]);
  const far = [s.center[0] + 500, s.center[1], s.center[2]];
  const a = assessBox(s, far, 30);
  assert.equal(a.level, 'bad');
  assert.match(a.message, /empty space/);
  assert.equal(atomsInBox(s, far, 30).length, 0);
});

test('assessBox: tiny and oversized boxes get a warning', () => {
  const p = proteins[1], s = load(p), c = [p.site.x, p.site.y, p.site.z];
  assert.equal(assessBox(s, c, 10).level, 'warn');
  assert.equal(assessBox(s, c, 80).level, 'warn');
});

test('nearestAtom finds an atom at distance 0 and none far away', () => {
  const s = load(proteins[1]);
  const hit = nearestAtom(s, s.x[5], s.y[5], s.z[5], 2);
  assert.ok(hit && hit.distance < 1e-3);
  assert.equal(nearestAtom(s, s.x[5] + 400, s.y[5], s.z[5], 8), null);
});

test('parsePDB rejects text with no atoms and skips hydrogens / extra models', () => {
  assert.throws(() => parsePDB('hello world'), /No protein atoms/);
  const txt = [
    'ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N',
    'ATOM      2  CA  ALA A   1       1.500   0.000   0.000  1.00  0.00           C',
    'ATOM      3  H   ALA A   1       1.500   1.000   0.000  1.00  0.00           H',
    'ENDMDL',
    'ATOM      4  CA  ALA A   2       9.000   9.000   9.000  1.00  0.00           C',
  ].join('\n');
  const s = parsePDB(txt);
  assert.equal(s.n, 2);
  assert.equal(s.ca.length, 1);
  assert.equal(sequenceOf(s), 'A');
});

test('validatePeptide: friendly messages', () => {
  assert.equal(validatePeptide('').level, 'empty');
  const bad = validatePeptide('LIYKBWVNK');
  assert.equal(bad.ok, false);
  assert.match(bad.message, /“B”/);
  assert.match(bad.message, /20 standard amino acids/);
  assert.equal(validatePeptide('A').ok, false);
  assert.equal(validatePeptide('AC').ok, false); // the backend needs 3-30 residues
  assert.equal(validatePeptide('ACD').ok, true);
  assert.equal(validatePeptide('liyk wvnk').seq, 'LIYKWVNK');
  assert.equal(validatePeptide('LIYKWVNK').level, 'ok');
  assert.equal(validatePeptide('A'.repeat(25)).level, 'warn'); // valid but slow
  assert.equal(validatePeptide('A'.repeat(30)).ok, true);
  const tooLong = validatePeptide('A'.repeat(31));
  assert.equal(tooLong.ok, false); // the backend would reject it on Run, so say so now
  assert.match(tooLong.message, /up to 30 amino acids/);
  assert.equal(validatePeptide('A'.repeat(5000)).ok, false);
  const numbered = validatePeptide('LIYKWVN1');
  assert.equal(numbered.ok, true); // digits are ignored (numbered sequences)...
  assert.match(numbered.message, /ignored/); // ...but never silently
  assert.equal(numbered.seq, 'LIYKWVN');
  assert.equal(validatePeptide('LIYK WVNK').message, 'Looks good.');
  assert.equal(cleanSequence('>seq1\nLIY\nKWV 12 NK'), 'LIYKWVNK');
});

test('peptideStats: length, mass, charge', () => {
  const s = peptideStats('GKDE');
  assert.equal(s.length, 4);
  assert.equal(s.charge, -1); // K +1, D −1, E −1
  assert.ok(s.mass > 400 && s.mass < 500);
});

test('dock command: defaults are omitted, expert flags appear, site vs blind', () => {
  const base = { peptide: 'LIYKWVNK', receptor: 'data/pdb/5O3L.pdb', blind: false, site: { x: 1, y: 2.5, z: -3 }, box: 30, poses: 100, expert: {} };
  const cmd = buildDockCommand(base);
  assert.match(cmd, /^hybridock-pep dock/);
  assert.match(cmd, /--site 1 2.5 -3/);
  assert.match(cmd, /--box 30/);
  assert.match(cmd, /--n-samples 100/);
  assert.match(cmd, /--output-dir runs\/studio\/run/);
  assert.doesNotMatch(cmd, /--seed|--ultra|--refine-topk|--no-minimize|--ensemble|--scoring|--calibration|--long-checkpoint/);
  const full = buildDockCommand({ ...base, expert: { seed: 7, refineTopK: 5, ultra: true, ultraK: 16, noMinimize: true, ensemble: true, scoring: 'vina,ad4', longCheckpointThreshold: 10, inputPoses: 'my poses', outputDir: 'out' } });
  for (const frag of ['--seed 7', '--refine-topk 5', '--ultra 16', '--no-minimize', '--ensemble', '--scoring vina,ad4', '--long-checkpoint-threshold 10', "--input-poses 'my poses'", '--output-dir out']) assert.ok(full.includes(frag), frag);
  const blind = buildDockCommand({ ...base, blind: true });
  assert.match(blind, /--blind/);
  assert.doesNotMatch(blind, /--site|--box/);
});

test('compare and score commands use the real subcommands and flags', () => {
  const side = (n) => ({ receptor: `${n}.pdb`, site: { x: 1, y: 2, z: 3 }, box: 30 });
  const c = buildCompareCommand({ peptide: 'AAA', target: side('t'), offTarget: side('o'), poses: 100 });
  for (const f of ['hybridock-pep selectivity', '--target-receptor t.pdb', '--target-site 1 2 3', '--target-box 30', '--offtarget-receptor o.pdb', '--offtarget-site 1 2 3', '--offtarget-box 30']) assert.ok(c.includes(f), f);
  const s = buildScoreCommand({ receptor: 'r.pdb', peptidePdbName: 'pep.pdb', peptide: 'AAA', allowClashes: true });
  for (const f of ['hybridock-pep crystal-score', '--receptor r.pdb', '--peptide-pdb pep.pdb', '--peptide AAA', '--allow-clashes']) assert.ok(s.includes(f), f);
});

test('selectivity verdict matches the backend rule exactly', () => {
  assert.equal(selectivityVerdict(-2, -0.4).id, 'target');
  assert.equal(selectivityVerdict(0.4, 2).id, 'offtarget');
  assert.equal(selectivityVerdict(-1, 1).id, 'none');
  assert.equal(selectivityVerdict(-1, 0).id, 'none'); // touching zero is NOT clearly selective
});

test('formatting and rough guide', () => {
  assert.equal(fmt(-7.4), '−7.40');
  assert.equal(fmt(0), '0.00');
  assert.equal(fmt(null), '—');
  assert.equal(roughGuide(-10).id, 'strong');
  assert.equal(roughGuide(-7).id, 'moderate');
  assert.equal(roughGuide(-4).id, 'weak');
  assert.ok(kdFromDG(-10) < kdFromDG(-8));
});

test('demo ΔG is repeatable, negative and bounded', () => {
  const a = demoDeltaG('tau', 'LIYKWVNK');
  assert.equal(a, demoDeltaG('tau', 'LIYKWVNK'));
  assert.ok(a < 0 && a >= -11.8);
  assert.notEqual(a, demoDeltaG('mdm2', 'LIYKWVNK'));
});

test('placeHelixAt keeps the peptide out of the protein', () => {
  const p = proteins[1], s = load(p);
  const pts = placeHelixAt(s, 10, [p.site.x, p.site.y, p.site.z], mulberry32(3));
  assert.equal(pts.length, 30);
  let closest = Infinity;
  for (let i = 0; i < 10; i++) { const h = nearestAtom(s, pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2], 6); if (h) closest = Math.min(closest, h.distance); }
  assert.ok(closest > 2.0, `closest approach ${closest.toFixed(2)} Å`);
});

test('small helpers', () => {
  assert.equal(initials('Ada Lovelace'), 'AL');
  assert.equal(initials('Ram'), 'R');
  assert.deepEqual(['quick', 'half', 'full'].map(poseCount), [5, 10, 25]); // the three sizes the live demo offers
});

import { build as buildRequests } from './regen_fixtures.mjs';
import { progressFrom } from '../../src/hybridock_pep/web/static/js/app/adapter.mjs';

test('the requests the UI sends match the shared fixture (the Python side validates the same file)', () => {
  const fixture = JSON.parse(readFileSync(join(dirname(fileURLToPath(import.meta.url)), '..', 'fixtures', 'web_requests.json'), 'utf8'));
  assert.deepEqual(buildRequests(), fixture, 'UI requests changed: run `node tests/web_js/regen_fixtures.mjs` and re-run the Python tests');
});

test('server progress maps onto the four plain stages and never invents an ETA', () => {
  const snap = (stage, fraction, elapsed, counter = '') => ({ stage, fraction, elapsed, counter, lines: [] });
  assert.equal(progressFrom(snap('sample', 0.2, 20), 100).stageIndex, 0);
  assert.equal(progressFrom(snap('min', 0.52, 50)).stageIndex, 1);
  assert.equal(progressFrom(snap('score', 0.7, 60)).stageIndex, 2);
  assert.equal(progressFrom(snap('cluster', 0.9, 80)).stageIndex, 2);
  assert.equal(progressFrom(snap('affinity', 0.95, 90)).stageIndex, 3);
  assert.equal(progressFrom(snap(null, 0, 1)).stageIndex, 0);
  // ETA comes only from the server's own estimate, and only until the run is 1.5x over it
  assert.equal(progressFrom(snap('sample', 0.1, 40), 100).etaSeconds, 60);
  assert.equal(progressFrom(snap('sample', 0.1, 40), 100).overdue, false);
  const late = progressFrom(snap('sample', 0.1, 150), 100);
  assert.equal(late.etaSeconds, null);
  assert.equal(late.overdue, true);
  assert.equal(progressFrom(snap('sample', 0.5, 50)).etaSeconds, null); // no estimate: say nothing rather than extrapolate
  // "1/5" is the pipeline's step number, not a pose count
  assert.equal(progressFrom(snap('sample', 0.1, 5, '1/5')).counter, '');
  assert.equal(progressFrom(snap('sample', 0.3, 5, '43/100')).counter, '43/100');
});

import { execFileSync } from 'node:child_process';

test('the shipped bundle is valid JavaScript (this is what the page actually loads)', () => {
  const bundle = join(root, 'dist', 'app.mjs');
  execFileSync(process.execPath, ['--check', bundle], { stdio: 'pipe' }); // throws with the syntax error if it is broken
  assert.match(readFileSync(bundle, 'utf8').split('\n', 1)[0], /source-hash: [0-9a-f]{16} \*\/$/, 'header comment must be a complete comment');
});

test('friendlyError: known failures get a plain title and keep the raw message as detail', async () => {
  const { friendlyError } = await import('../../src/hybridock_pep/web/static/js/app/errors.mjs');
  const cases = [
    ["RuntimeError: Cannot locate Python 3 in conda env 'rapidock'. Set RAPIDOCK_PYTHON", /docking engine isn.t installed/],
    ['hybridock-pep: error: Crystal scoring failed for 1YCR_peptide.pdb.', /couldn.t be scored/],
    ['A run is already going. Wait for it, or stop it first.', /Another run is already going/],
    ['The run finished but no ΔG was reported. Look in the run folder: runs/studio/x', /No pose fit at this site/],
    ['Docking complete. 0 poses scored.', /No pose fit at this site/],
    ['The run was stopped from outside (it was terminated). Start it again if that was not intended.', /stopped from outside/],
    ['Killed', /ran out of memory/],
    ['concurrent.futures.process.BrokenProcessPool: A process in the process pool was terminated abruptly while the future was running', /ran out of memory/],
    ['process exited with exit code -9', /ran out of memory/],
    ['FileNotFoundError: [Errno 2] No such file or directory: runs/x.pdb', /file for this run went missing/],
  ];
  for (const [msg, title] of cases) {
    const e = friendlyError(new Error(msg));
    assert.match(e.title, title, msg);
    assert.equal(e.detail, msg);
    assert.ok(e.body.length > 30);
  }
  // a server that forgot the job (HTTP 404) and a dead network
  assert.match(friendlyError(Object.assign(new Error('The server answered 404.'), { status: 404 })).title, /server restarted/);
  assert.match(friendlyError(new TypeError('Failed to fetch')).title, /reach the server/);
  const net = friendlyError(Object.assign(new Error('Couldn’t reach the HybriDock-Pep server. Is `hybridock-pep serve` still running?'), { name: 'NetworkError' }));
  assert.match(net.title, /reach the server/); // the adapter's own wording for a dead connection
  assert.match(net.detail, /serve/);
  // anything else: generic title, never an empty or "undefined" body
  const g = friendlyError(new Error('weird'));
  assert.equal(g.title, 'That run didn’t finish');
  assert.equal(g.detail, 'weird');
  assert.ok(!/undefined|\[object/.test(friendlyError({}).body + friendlyError({}).detail));
});

test('settingProblems: the backend limits (box 10-60, threshold 3-30, top-K 0-50, ultra K 0-256, seed >= 0)', async () => {
  const { settingProblems, LIMITS, BOX_MAX } = await import('../../src/hybridock_pep/web/static/js/app/config.mjs');
  assert.equal(BOX_MAX, 60); // the slider must not offer a size the server rejects
  const ok = { box: 30, expert: { longCheckpointThreshold: 13, refineTopK: '', ultra: false, ultraK: 32, seed: '' } };
  assert.deepEqual(settingProblems(ok), []);
  const bad = (patch, box = 30) => settingProblems({ box, expert: { ...ok.expert, ...patch } });
  assert.match(bad({}, 61)[0], /Box size must be between 10 and 60/);
  assert.match(bad({}, 9)[0], /Box size/);
  assert.match(bad({ longCheckpointThreshold: 2 })[0], /between 3 and 30/);
  assert.match(bad({ longCheckpointThreshold: 31 })[0], /between 3 and 30/);
  assert.match(bad({ longCheckpointThreshold: 4.5 })[0], /whole number/);
  assert.match(bad({ refineTopK: 51 })[0], /between 0 and 50/);
  assert.deepEqual(bad({ refineTopK: 0 }), []); // 0 = off is allowed
  assert.match(bad({ ultra: true, ultraK: 300 })[0], /Ultra mode K/);
  assert.deepEqual(bad({ ultra: false, ultraK: 300 }), []); // K is ignored while Ultra is off
  assert.match(bad({ seed: -1 })[0], /Random seed/);
  assert.deepEqual(bad({ seed: 2147483647 }), []);
  assert.deepEqual(LIMITS.box, [10, 60]);
});

test('whole-protein search scales with How thorough: Full is the CLI default, Quick and Half are smaller', async () => {
  const { blindSearchSize } = await import('../../src/hybridock_pep/web/static/js/app/adapter.mjs');
  assert.deepEqual(blindSearchSize({ blind: false, poses: 100 }), {});
  assert.deepEqual(blindSearchSize({ blind: true, poses: 100 }), { n_pocket_search: '300', n_per_pocket: '150' });
  const quick = blindSearchSize({ blind: true, poses: 25 }), half = blindSearchSize({ blind: true, poses: 50 });
  assert.ok(Number(quick.n_pocket_search) < Number(half.n_pocket_search) && Number(half.n_pocket_search) < 300);
  assert.ok(Number(quick.n_per_pocket) >= 10, 'never below what the CLI accepts');
});

test('named peptides: a name for the famous ones, just letters for the rest', async () => {
  const { namedPeptide, peptideTitle, peptideWithLetters, NAMED_PEPTIDES } = await import('../../src/hybridock_pep/web/static/js/app/peptides.mjs');
  assert.equal(namedPeptide('ETFSDLWKLLPE').name, 'p53 peptide');
  assert.equal(namedPeptide('etfsdlwkllpe').name, 'p53 peptide');
  assert.equal(peptideTitle('ACDEFGH'), 'ACDEFGH');
  assert.equal(peptideWithLetters('ETFSDLWKLLPE'), 'p53 peptide (ETFSDLWKLLPE)');
  assert.equal(peptideWithLetters('ACDEFGH'), 'ACDEFGH');
  for (const p of NAMED_PEPTIDES) { // every entry is a valid peptide with a real description
    assert.equal(validatePeptide(p.seq).ok, true, p.seq);
    assert.ok(p.name.length > 3 && p.about.length > 20, p.seq);
  }
  assert.equal(new Set(NAMED_PEPTIDES.map((p) => p.seq)).size, NAMED_PEPTIDES.length, 'no duplicate sequences');
});

test('the guide: every lesson is short, plain and complete, for both audiences', async () => {
  const { TOPICS, topicsFor, richParts, sizeOf, findTopic } = await import('../../src/hybridock_pep/web/static/js/app/guide-content.mjs');
  const fs = await import('node:fs');
  const img = new URL('../../docs/guide-img/', import.meta.url);
  assert.equal(new Set(TOPICS.map((t) => t.id)).size, TOPICS.length, 'unique lesson ids');
  for (const kind of ['demo', 'local', 'remote']) {
    const list = topicsFor(kind);
    assert.ok(list.length >= 8 && list.length <= 10, `${kind}: a short list of lessons`);
    assert.equal(list[0].id, 'live-demo', `${kind}: the live demo lesson comes first`);
    for (const t of list) {
      assert.ok(t.title && t.blurb && t.icon, t.id);
      const items = t.kind === 'steps' ? t.steps : t.rows;
      assert.ok(items.length >= 3 && items.length <= 9, `${t.id}: ${items.length} items`);
      assert.ok(sizeOf(t).length > 3, t.id);
      for (const s of items) {
        const text = [s.title, s.body, s.text, s.q, s.term].filter(Boolean).join(' ');
        assert.ok(text.length > 10, `${t.id}: empty item`);
        assert.ok((s.body || s.text).length <= 260, `${t.id}: "${(s.title || s.term || s.q)}" is a wall of text (${(s.body || s.text).length} chars)`);
        assert.ok(!/—/.test(text), `${t.id}: no em dashes`);
        assert.ok(!/\b(GPU|CUDA|VRAM|RAM|latency|leverage|seamless)\b/i.test(text), `${t.id}: jargon in "${text.slice(0, 40)}"`);
        if (s.image) assert.ok(fs.existsSync(new URL(s.image, img)), `${t.id}: missing screenshot ${s.image}`);
      }
    }
  }
  assert.ok(findTopic(topicsFor('local'), 'troubleshooting') && findTopic(topicsFor('remote'), 'troubleshooting') && findTopic(topicsFor('demo'), 'live-demo'), 'the lessons the README links to exist');
  // the demo never promises a real result, and the real app never says it is a demo
  const words = (kind) => JSON.stringify(topicsFor(kind));
  assert.ok(!/preview website|simulated/.test(words('local')), 'local: no demo wording');
  const remoteText = JSON.stringify(topicsFor('remote').filter((t) => t.id !== 'own-computer'));
  assert.ok(!/launch_web|\.\/install|data\/pdbs|on this computer/.test(remoteText), 'remote: nothing that only makes sense on your own computer');
  assert.deepEqual(richParts('Press **Run** or `./x`.').map((p) => p.t), ['text', 'strong', 'text', 'code', 'text']);
});

test('memory meter: levels and colours follow how full memory is', async () => {
  const { memoryLevel } = await import('../../src/hybridock_pep/web/static/js/app/config.mjs');
  assert.equal(memoryLevel(null).id, 'unknown');
  assert.equal(memoryLevel(NaN).id, 'unknown');
  assert.equal(memoryLevel(40).id, 'ok');
  assert.equal(memoryLevel(80).id, 'busy');
  assert.equal(memoryLevel(90).id, 'low');
  assert.equal(memoryLevel(99).id, 'full');
  assert.equal(memoryLevel(80, 0.5).id, 'low', 'under 1 GB free is low even at 80 %');
  assert.equal(memoryLevel(-5).fill, 0); assert.equal(memoryLevel(250).fill, 100);
  const hues = [10, 50, 70, 80, 90, 100].map((p) => memoryLevel(p).hue);
  for (let i = 1; i < hues.length; i++) assert.ok(hues[i] < hues[i - 1], `the colour keeps moving from green to red: ${hues}`);
  assert.ok(memoryLevel(10).hue > 120 && memoryLevel(100).hue < 10, 'green when empty, red when full');
});

test('catalog: preloaded binders become results the results screen can open, ordered strongest first', async () => {
  const { bindersFor, binderResult, binderKey, strength, useTiming } = await import('../../src/hybridock_pep/web/static/js/app/catalog.mjs');
  const mdm2 = proteins.find((p) => p.key === 'mdm2');
  const catalog = { generated: '2026-10-23', poses: 25, binders: { mdm2: [
    { seq: 'SQETFSDLWKLLP', deltaG: -8.1, known: true, poses: [{ rank: 1, deltaG: -8.1, ca: new Array(39).fill(1) }] },
    { seq: 'ETFSDLWKLLPE', deltaG: -9.3, known: true, poses: [{ rank: 1, deltaG: -9.3, ca: new Array(36).fill(2) }, { rank: 2, deltaG: -9.0, ca: new Array(36).fill(3) }] },
  ] } };
  assert.deepEqual(bindersFor(catalog, 'mdm2').map((b) => b.deltaG), [-9.3, -8.1]);
  assert.deepEqual(bindersFor(catalog, 'nothing'), []);
  const r = binderResult(mdm2, catalog.binders.mdm2[1], catalog);
  assert.equal(r.kind, 'dock'); assert.equal(r.preloaded, true); assert.equal(r.demo, false, 'a preloaded run is a real run');
  assert.equal(r.deltaG, -9.3); assert.equal(r.poses.length, 2); assert.equal(r.poses[0].ca.length, r.peptide.length * 3);
  assert.match(r.command, /--n-samples 25/); assert.match(r.command, /--peptide ETFSDLWKLLPE/);
  assert.equal(binderKey('mdm2', 'AAA'), 'mdm2:AAA');
  assert.ok(strength(-12) === 1 && strength(-1) > 0 && strength(-9) > strength(-6), 'the bar grows with binding strength');
  useTiming(null);
});

test('timing: "about N minutes" comes only from a measured grid, and the JS and Python estimators agree', async () => {
  const { useTiming } = await import('../../src/hybridock_pep/web/static/js/app/catalog.mjs');
  const { timingFor, TIMING } = await import('../../src/hybridock_pep/web/static/js/app/config.mjs');
  const fixture = JSON.parse(readFileSync(new URL('../fixtures/timing_cases.json', import.meta.url), 'utf8'));
  useTiming(null);
  assert.equal(timingFor(5), null, 'no table, no claim');
  useTiming({ machine: 'Test board', refLength: fixture.refLength, grid: fixture.grid });
  assert.equal(TIMING.machine, 'Test board');
  assert.deepEqual([5, 10, 25].map((n) => TIMING.seconds[n]), [68, 82, 134], 'the three sizes at the reference length are the measured ones');
  for (const c of fixture.cases) assert.equal(timingFor(c.poses, c.length), c.seconds, `${c.length} letters, ${c.poses} poses`);
  assert.ok(timingFor(25, 16) > 1.7 * timingFor(25, 12), 'from 13 residues length matters more at 25 poses than a single multiplier would say');
  useTiming(null);
  assert.equal(timingFor(10), null, 'clearing the table clears the claim');
});

test('the home page carries no error figures and the catalog data names a disease for every protein', () => {
  for (const p of proteins) {
    assert.ok(p.disease && p.plain && Array.isArray(p.panel) && p.panel.length >= 1, `${p.key} has a disease, a plain line and a panel`);
    assert.ok(!/±|typical error|off by/i.test(`${p.disease} ${p.plain}`), p.key);
  }
  const home = readFileSync(join(root, 'js/app/ui/home.mjs'), 'utf8');
  assert.ok(!/TYPICAL_ERROR|±/.test(home), 'no error figure on the home page');
});

// ---- the 3D view: the floating, tumbling and settling peptide never passes into the protein ----------------------------
import { Stage } from '../../src/hybridock_pep/web/static/js/app/stage/stage.mjs';
import { closestApproach, clearShift, makeGrid } from '../../src/hybridock_pep/web/static/js/app/stage/collide.mjs';
import { helixPoints } from '../../src/hybridock_pep/web/static/js/app/peptide.mjs';
import { qAxisAngle, qToMat3, mat3ApplyInverse } from '../../src/hybridock_pep/web/static/js/app/stage/math.mjs';

test('clearShift moves a peptide that sits inside a protein until it is clear, and leaves a clear one alone', () => {
  const pts = [];
  for (let x = -20; x <= 20; x += 1.5) for (let y = -20; y <= 20; y += 1.5) pts.push(x, y, 0);
  const grid = makeGrid(Float32Array.from(pts), 6);
  const pep = Float32Array.from(helixPoints(10));
  assert.ok(closestApproach(grid, pep, 10) < 2, 'the helix starts inside the wall');
  const s = clearShift(grid, pep, 10, 4.2);
  const moved = pep.map((v, i) => v + s[i % 3]);
  assert.ok(closestApproach(grid, moved, 10) >= 4.2 - 0.05, `still ${closestApproach(grid, moved, 10).toFixed(2)} Å away`);
  const far = pep.map((v, i) => v + (i % 3 === 2 ? 30 : 0));
  assert.deepEqual(clearShift(grid, far, 10, 4.2), [0, 0, 0]);
});

function fakeStage(structure, mode, synthetic = false) {
  const st = Object.create(Stage.prototype);
  Object.assign(st, { reduced: false, t: 0, pep: { n: 12, shape: helixPoints(12), mode, dockT: 0, prev: null, next: null, tp: 1, alpha: 1, synthetic }, invalidate() {}, _updateInside() {} });
  Stage.prototype._applyProtein.call(st, structure);
  return st;
}

test('the floating and tumbling peptide stays clear of every bundled protein at every moment and angle', () => {
  const rng = mulberry32(7);
  for (const p of proteins) {
    const structure = load(p);
    for (const mode of ['float', 'tumble']) {
      const st = fakeStage(structure, mode);
      let worst = Infinity;
      for (let step = 0; step < 240; step++) {
        st.t = step * 0.5;
        const M = qToMat3(qAxisAngle([rng() - 0.5, rng() - 0.5, rng() - 0.5], rng() * 6.28));
        Stage.prototype._stepPeptide.call(st, 1 / 32, M); // the spring moves the peptide (and keeps it out) once a frame
        const { v } = Stage.prototype._peptideView.call(st, M, st.P.center, st.P.R);
        const local = new Float32Array(v.length);
        for (let i = 0; i < 12; i++) {
          const q = mat3ApplyInverse(M, v[i * 3], v[i * 3 + 1], v[i * 3 + 2]);
          local[i * 3] = q[0] + st.P.center[0]; local[i * 3 + 1] = q[1] + st.P.center[1]; local[i * 3 + 2] = q[2] + st.P.center[2];
        }
        worst = Math.min(worst, closestApproach(st._grid, local, 12));
      }
      assert.ok(worst >= 4.2 - 0.1, `${p.name} (${mode}): closest approach ${worst.toFixed(2)} Å`);
    }
  }
});

test('a stand-in peptide in the search box is kept out of the protein too', () => {
  for (const p of proteins) {
    const structure = load(p);
    const st = fakeStage(structure, 'dock', true);
    st.pep.dockT = 1;
    st.pep.next = Float32Array.from(helixPoints(12), (v, i) => v + p.site[['x', 'y', 'z'][i % 3]]);
    const M = qToMat3([0, 0, 0, 1]);
    const { v } = Stage.prototype._peptideView.call(st, M, st.P.center, st.P.R);
    const local = v.map((x, i) => x + st.P.center[i % 3]);
    assert.ok(closestApproach(st._grid, local, 12) >= 3.4 - 0.1, `${p.name}: ${closestApproach(st._grid, local, 12).toFixed(2)} Å`);
  }
});

test('settling onto a result never cuts through the protein on the way', () => {
  const catalog = JSON.parse(readFileSync(join(root, 'data/catalog.json'), 'utf8'));
  for (const p of proteins) {
    const binder = (catalog.binders[p.key] || [])[0];
    if (!binder) continue;
    const pose = binder.poses[0].ca, n = binder.seq.length;
    const structure = load(p);
    const st = Object.create(Stage.prototype);
    Object.assign(st, { reduced: false, t: 3, pep: { n, shape: helixPoints(n), mode: 'dock', dockT: 0, prev: null, next: Float32Array.from(pose), tp: 1, alpha: 1, synthetic: false }, invalidate() {}, _updateInside() {} });
    Stage.prototype._applyProtein.call(st, structure);
    const M = qToMat3(qAxisAngle([0.2, 1, 0.1], 1.1));
    for (const dockT of [0, 0.1, 0.25, 0.5, 0.75]) {
      st.pep.dockT = dockT;
      const { v } = Stage.prototype._peptideView.call(st, M, st.P.center, st.P.R);
      const local = new Float32Array(v.length);
      for (let i = 0; i < n; i++) {
        const q = mat3ApplyInverse(M, v[i * 3], v[i * 3 + 1], v[i * 3 + 2]);
        local[i * 3] = q[0] + st.P.center[0]; local[i * 3 + 1] = q[1] + st.P.center[1]; local[i * 3 + 2] = q[2] + st.P.center[2];
      }
      const e = dockT < 0.5 ? 4 * dockT ** 3 : 1 - (-2 * dockT + 2) ** 3 / 2; // easeInOut
      const want = 4.2 * (1 - e);
      assert.ok(closestApproach(st._grid, local, n) >= want - 0.1, `${p.name} at ${dockT}: ${closestApproach(st._grid, local, n).toFixed(2)} Å, needs ${want.toFixed(2)}`);
    }
  }
});
