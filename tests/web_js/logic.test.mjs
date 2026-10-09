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
import { greeting, initials, poseCount } from '../../src/hybridock_pep/web/static/js/app/jobs.mjs';

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
  assert.equal(greeting(new Date(2026, 0, 1, 14)), 'Good afternoon');
  assert.equal(greeting(new Date(2026, 0, 1, 9)), 'Good morning');
  assert.equal(poseCount('full'), 100);
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
    ['Killed', /ran out of memory/],
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
  // anything else: generic title, never an empty or "undefined" body
  const g = friendlyError(new Error('weird'));
  assert.equal(g.title, 'That run didn’t finish');
  assert.equal(g.detail, 'weird');
  assert.ok(!/undefined|\[object/.test(friendlyError({}).body + friendlyError({}).detail));
});
