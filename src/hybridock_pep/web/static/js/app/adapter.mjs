// adapter.mjs — THE ONLY FILE THAT TALKS TO THE BACKEND.
//
// Every screen calls `adapter.something(...)` and nothing else. Two adapters live here:
//
//   mockAdapter  — a pretend backend so the whole app works with no server (Demo mode). EVERYTHING it
//                  returns is simulated, is flagged `demo: true`, and is labelled "Demo" in the UI.
//   liveAdapter  — the real thing: it drives `hybridock-pep serve` (src/hybridock_pep/web/server.py)
//                  through its /api/* endpoints. It adds no backend features: every run is the same CLI
//                  command the terminal UI would build, started by the server as a subprocess.
//
// Which one is used (see initAdapter at the bottom):
//   served by `hybridock-pep serve`  -> live (it answers /api/env)
//   anywhere else / opened as files / the static site  -> demo
//   ?demo  forces demo (handy for screen recordings),  ?live  forces live.
//
// The shapes below (DockResult, CompareResult, ScoreResult) are what the screens expect.

import { parsePDB, nearestAtom, caPoints } from './pdb.mjs';
import { loadProtein } from './structures.mjs';
import { placeHelixAt } from './placement.mjs';
import { mulberry32, hashString } from './stage/math.mjs';
import { buildDockCommand, buildCompareCommand, buildScoreCommand } from './command.mjs';
import { EXPERT_DEFAULTS } from './config.mjs';

/* ------------------------------------------------------------------------------------------------
 * Types (documentation only)
 *
 * Job (dock):     { peptide, protein:{key?,name,file?,custom?}, receptor:'path/to.pdb', blind, site:{x,y,z},
 *                   box, poses, expert:{…} }
 * Progress:       { stageIndex:0-3, fraction:0-1, etaSeconds:number|null }
 *
 * DockResult:     { id, kind:'dock', demo, createdAt, name, protein, peptide, site, box, blind, nPoses,
 *                   deltaG,                       // headline binding strength, kcal/mol (delta_g)
 *                   poses:[{ rank, file, deltaG,  // delta_g of that pose
 *                            rankScore,           // rank_score: NOT a ΔG; only compares poses of this protein
 *                            nClash, cluster, ca:number[]|null }],  // ca = peptide C-alpha coords, flat
 *                   command, job }
 * CompareResult:  { id, kind:'compare', demo, createdAt, name, peptide, ddg, ci:[low, high],
 *                   target:{protein, deltaG, site, box, ca}, offTarget:{…}, command, job }
 * ScoreResult:    { id, kind:'score', demo, createdAt, name, protein, peptide, deltaG, ca, command, job }
 * ---------------------------------------------------------------------------------------------- */

export class NotWiredError extends Error {
  constructor(what) {
    super(`The live backend isn't connected yet (${what}). It is waiting for the hybridock-pep serve code to be wired into js/adapter.js.`);
    this.name = 'NotWiredError';
  }
}

const sleep = (ms, signal) => new Promise((resolve, reject) => {
  if (signal?.aborted) return reject(new DOMException('Stopped', 'AbortError'));
  const t = setTimeout(resolve, ms);
  signal?.addEventListener('abort', () => { clearTimeout(t); reject(new DOMException('Stopped', 'AbortError')); }, { once: true });
});

const newId = (prefix) => `${prefix}_${Date.now().toString(36)}${Math.floor(Math.random() * 1296).toString(36)}`;
const round1 = (v) => Math.round(v * 10) / 10;

/* ================================================================================================
 * DEMO ADAPTER — pretend results, repeatable from the inputs, clearly labelled "Demo".
 * ============================================================================================== */

const HYDROPHOBIC = 'AVILMFWY';

/** Pretend ΔG: more negative for longer, greasier peptides, plus a repeatable wobble. */
export function demoDeltaG(proteinKey, peptide) {
  const rng = mulberry32(hashString(`${proteinKey}|${peptide}`));
  const h = [...peptide].filter((c) => HYDROPHOBIC.includes(c)).length / peptide.length;
  const dg = -(3.6 + 0.42 * Math.min(peptide.length, 16) + 2.2 * h) + (rng() - 0.5) * 2.0;
  return Math.round(Math.min(-3.6, Math.max(-11.8, dg)) * 100) / 100;
}

const STAGE_WEIGHTS = [0.45, 0.2, 0.15, 0.2];

/** Walk through the four stages over `seconds`, reporting progress ~8 times a second. */
async function playTimeline(seconds, { onProgress, signal, instant }) {
  if (instant) return;
  const step = 120;
  const total = Math.max(1, Math.round((seconds * 1000) / step));
  for (let i = 0; i <= total; i++) {
    const fraction = i / total;
    let acc = 0, stageIndex = 0;
    for (let s = 0; s < STAGE_WEIGHTS.length; s++) {
      if (fraction >= acc + STAGE_WEIGHTS[s] && s < STAGE_WEIGHTS.length - 1) { acc += STAGE_WEIGHTS[s]; stageIndex = s + 1; } else break;
    }
    onProgress?.({ stageIndex, fraction, etaSeconds: Math.max(0, Math.round(seconds * (1 - fraction))) });
    if (i < total) await sleep(step, signal);
  }
}

const centerOfSite = (job, structure) => {
  if (!job.blind && job.site) return [job.site.x, job.site.y, job.site.z];
  // Blind demo: pretend the "pocket search" landed on the preset's known site, else the backbone atom nearest the middle.
  if (job.protein.site) return [job.protein.site.x, job.protein.site.y, job.protein.site.z];
  let best = structure.ca[0], bd = Infinity;
  for (const a of structure.ca) {
    const d = Math.hypot(a.x - structure.center[0], a.y - structure.center[1], a.z - structure.center[2]);
    if (d < bd) { bd = d; best = a; }
  }
  return [best.x, best.y, best.z];
};

function demoPoses(job, structure, center, headline, count) {
  const poses = [];
  for (let i = 0; i < count; i++) {
    const rng = mulberry32(hashString(`${job.protein.key || job.protein.name}|${job.peptide}|pose${i}`));
    const jitter = 0.5 + i * 0.45;
    const c = [center[0] + (rng() - 0.5) * jitter * 2, center[1] + (rng() - 0.5) * jitter * 2, center[2] + (rng() - 0.5) * jitter * 2];
    const ca = placeHelixAt(structure, job.peptide.length, c, rng);
    poses.push({
      rank: i + 1,
      file: `pose_${i + 1}.pdb`,
      deltaG: Math.round((headline + (i === 0 ? 0 : 0.12 * i + rng() * 0.3)) * 100) / 100,
      // Deliberately on a different scale from ΔG, so the two can't be mistaken for each other.
      rankScore: Math.round((-1.4 + i * 0.22 + rng() * 0.1) * 1000) / 1000,
      nClash: Math.floor(rng() * 4),
      cluster: Math.floor(i / 4),
      ca: Array.from(ca, round1),
    });
  }
  return poses;
}

const mockAdapter = {
  kind: 'demo',
  label: 'Demo',
  supports: { allowClashes: true },

  /** Same shape as the live adapter's preview(): the command and a time estimate. */
  async preview(job) { return { command: buildDockCommand(job), estimateSeconds: this.estimate({ ...job, kind: 'dock' }), problems: [] }; },

  /** Rough seconds a run takes. Demo runs are short on purpose. */
  estimate(job) {
    if (job.kind === 'compare') return 12;
    if (job.kind === 'score') return 3;
    return Math.round(5 + job.poses * 0.06);
  },

  async runDock(job, opts = {}) {
    const { structure } = await loadProtein(job.protein);
    await playTimeline(this.estimate({ ...job, kind: 'dock' }), opts);
    const headline = demoDeltaG(job.protein.key || job.protein.name, job.peptide);
    const center = centerOfSite(job, structure);
    const poses = demoPoses(job, structure, center, headline, Math.min(20, job.poses));
    return {
      id: newId('run'), kind: 'dock', demo: true, createdAt: new Date().toISOString(),
      name: `${job.peptide} → ${job.protein.name}`,
      protein: job.protein, peptide: job.peptide,
      site: { x: round1(center[0]), y: round1(center[1]), z: round1(center[2]) }, box: job.box, blind: !!job.blind,
      nPoses: job.poses, deltaG: headline, poses,
      command: buildDockCommand(job), job,
    };
  },

  async runCompare(job, opts = {}) {
    const [t, o] = await Promise.all([loadProtein(job.target.protein), loadProtein(job.offTarget.protein)]);
    await playTimeline(this.estimate({ kind: 'compare' }), opts);
    const dgT = demoDeltaG(job.target.protein.key || job.target.protein.name, job.peptide);
    const dgO = demoDeltaG(job.offTarget.protein.key || job.offTarget.protein.name, job.peptide);
    const ddg = Math.round((dgT - dgO) * 100) / 100;
    const rng = mulberry32(hashString(`${job.peptide}|${job.target.protein.name}|${job.offTarget.protein.name}`));
    const half = Math.round((0.9 + rng() * 1.1) * 100) / 100;
    const side = (s, structure, dg) => {
      const c = [s.site.x, s.site.y, s.site.z];
      const ca = placeHelixAt(structure, job.peptide.length, c, mulberry32(hashString(s.protein.name + job.peptide)));
      return { protein: s.protein, deltaG: dg, site: s.site, box: s.box, ca: Array.from(ca, round1) };
    };
    return {
      id: newId('cmp'), kind: 'compare', demo: true, createdAt: new Date().toISOString(),
      name: `${job.peptide}: ${job.target.protein.name} vs ${job.offTarget.protein.name}`,
      peptide: job.peptide, ddg, ci: [Math.round((ddg - half) * 100) / 100, Math.round((ddg + half) * 100) / 100],
      target: side(job.target, t.structure, dgT), offTarget: side(job.offTarget, o.structure, dgO),
      command: buildCompareCommand(job), job,
    };
  },

  async runScore(job, opts = {}) {
    const { structure } = await loadProtein(job.protein);
    const pep = parsePDB(job.peptidePdb.text);
    await playTimeline(this.estimate({ kind: 'score' }), opts);
    // Count contacts and clashes between the two structures (simple distance checks).
    let contacts = 0, clashes = 0;
    for (let i = 0; i < pep.n; i++) {
      const hit = nearestAtom(structure, pep.x[i], pep.y[i], pep.z[i], 4.5);
      if (hit) { contacts++; if (hit.distance < 2.2) clashes++; }
    }
    // Assumption: like the real `crystal-score`, refuse clashing poses unless told otherwise.
    if (clashes > 0 && !job.allowClashes) {
      throw new Error(`The peptide overlaps the protein in ${clashes} place${clashes > 1 ? 's' : ''}. Turn on “Allow clashes” (Expert) if you still want a score.`);
    }
    const deltaG = Math.round(Math.min(-2.5, Math.max(-12, -(2.5 + 0.045 * contacts))) * 100) / 100;
    const ca = new Float32Array(pep.ca.length * 3);
    pep.ca.forEach((a, i) => { ca[i * 3] = a.x; ca[i * 3 + 1] = a.y; ca[i * 3 + 2] = a.z; });
    return {
      id: newId('scr'), kind: 'score', demo: true, createdAt: new Date().toISOString(),
      name: `${job.peptide} on ${job.protein.name} (scored)`,
      protein: job.protein, peptide: job.peptide, deltaG, ca: Array.from(ca, round1),
      command: buildScoreCommand(job), job: { ...job, protein: job.protein, peptidePdb: { name: job.peptidePdb.name } },
    };
  },

  /** Peptide C-alpha coordinates of one pose (flat Float32Array in the protein's frame), or null if unavailable. */
  async loadPose(result, rank) {
    const pose = result.poses?.find((p) => p.rank === rank);
    return pose?.ca ? Float32Array.from(pose.ca) : null;
  },

  /** Build a downloadable file. Demo files are named *_DEMO so they can't be mistaken for real output. */
  async download(result, what) {
    if (what === 'ranked_csv') {
      const rows = ['rank,pose_filename,delta_g,rank_score,n_clash,cluster_id',
        ...result.poses.map((p) => [p.rank, p.file, p.deltaG, p.rankScore, p.nClash, p.cluster].join(','))];
      return { filename: 'ranked_poses_DEMO.csv', blob: new Blob([rows.join('\n') + '\n'], { type: 'text/csv' }) };
    }
    if (what === 'best_pose') {
      const { text } = await loadProtein(result.protein);
      const ca = result.poses[0].ca;
      const lines = ['REMARK   DEMO: synthetic pose from the HybriDock-Pep demo adapter. Not a real prediction.'];
      lines.push(...text.split('\n').filter((l) => l.startsWith('ATOM')));
      lines.push('TER');
      for (let i = 0; i < result.peptide.length; i++) {
        const [x, y, z] = [ca[i * 3], ca[i * 3 + 1], ca[i * 3 + 2]];
        lines.push(`ATOM  ${String(90000 + i).padStart(5)}  CA  ALA Z${String(i + 1).padStart(4)}    ${x.toFixed(3).padStart(8)}${y.toFixed(3).padStart(8)}${z.toFixed(3).padStart(8)}  1.00  0.00           C`);
      }
      lines.push('TER', 'END');
      return { filename: 'best_pose_DEMO.pdb', blob: new Blob([lines.join('\n') + '\n'], { type: 'chemical/x-pdb' }) };
    }
    throw new Error("Demo runs don't create a run folder. With the live backend this downloads the whole folder.");
  },
};

/* ================================================================================================
 * LIVE ADAPTER — the real `hybridock-pep serve` API.
 *
 *   POST /api/upload            { name, content }              -> { path }   (PDB text the CLI can read)
 *   POST /api/preview | /api/validate  { mode, values, scoring } -> { command } | { estimate_seconds, … }
 *   POST /api/run               { mode, values, scoring }      -> { job: snapshot }
 *   GET  /api/jobs/<id>?since=N                                 -> { state, stage, fraction, counter, lines, … }
 *   GET  /api/jobs/<id>/results                                 -> { headline, poses[], files[], … }
 *   GET  /api/jobs/<id>/pose?name=…   |  /file?name=…&download=1 -> the file
 *   POST /api/jobs/<id>/cancel
 *
 * `values` uses the terminal UI's own field keys (peptide, receptor, site "x y z", box, n_samples, …).
 * ============================================================================================== */

class ApiError extends Error {
  constructor(message, status) { super(message); this.name = 'ApiError'; this.status = status; }
}

let fieldLabels = null; // key -> human label, from /api/fields (used to word validation errors)

async function labelFor(key) {
  if (!fieldLabels) {
    try { fieldLabels = Object.fromEntries((await (await fetch('/api/fields')).json()).fields.map((f) => [f.key, f.label])); }
    catch { fieldLabels = {}; }
  }
  return fieldLabels[key] || key;
}

async function failure(res) {
  let data = null;
  try { data = await res.json(); } catch { /* not JSON */ }
  let message = data?.error?.message || `The server answered ${res.status}.`;
  const fields = data?.error?.fields;
  if (fields && Object.keys(fields).length) {
    const parts = await Promise.all(Object.entries(fields).map(async ([k, v]) => `${await labelFor(k)}: ${v}`));
    message += ' ' + parts.join('; ');
  }
  return new ApiError(message, res.status);
}

const API = {
  async request(method, path, body, signal) {
    let res;
    try {
      res = await fetch(path, {
        method, signal, cache: 'no-store',
        headers: body ? { 'Content-Type': 'application/json' } : undefined,
        body: body ? JSON.stringify(body) : undefined,
      });
    } catch (err) {
      if (err?.name === 'AbortError') throw err;
      throw Object.assign(new Error('Couldn’t reach the HybriDock-Pep server. Is `hybridock-pep serve` still running?'), { name: 'NetworkError' });
    }
    if (!res.ok) throw await failure(res);
    return res;
  },
  async json(method, path, body, signal) { return (await this.request(method, path, body, signal)).json(); },
  async text(path, signal) { return (await this.request('GET', path, null, signal)).text(); },
  get: (path, signal) => API.json('GET', path, null, signal),
  post: (path, body, signal) => API.json('POST', path, body, signal),
};

const STAGE_INDEX = { sample: 0, min: 1, score: 2, cluster: 2, affinity: 3 }; // server stages -> the 4 plain labels
const yn = (b) => (b ? 'y' : 'n');
const num = (v) => { const x = parseFloat(v); return Number.isFinite(x) ? x : null; };
const int = (v) => { const x = parseInt(v, 10); return Number.isFinite(x) ? x : 0; };
const safePdbName = (name, fallback) => {
  const base = String(name || fallback).replace(/[^\w.-]+/g, '_');
  return /\.pdb$/i.test(base) ? base : base.replace(/\.[^.]*$/, '') + '.pdb';
};

const uploaded = new Map(); // protein id -> path on the server's disk

/** Make sure the CLI can read this protein: bundled examples already live on the server, the rest are uploaded. */
async function receptorOnServer(ref, signal) {
  if (ref.receptorPath) return ref.receptorPath; // a file the server already has (from /api/examples)
  const id = ref.id || ref.key || ref.name;
  if (uploaded.has(id)) return uploaded.get(id);
  const { text } = await loadProtein(ref);
  const name = `${safePdbName(ref.pdb || ref.name, 'protein')}`.replace(/\.pdb$/i, `_${hashString(text).toString(36)}.pdb`);
  const res = await API.post('/api/upload', { name, content: text }, signal);
  uploaded.set(id, res.path);
  return res.path;
}

async function uploadText(name, content, signal) {
  return (await API.post('/api/upload', { name: safePdbName(name, 'peptide'), content }, signal)).path;
}

const siteStr = (s) => `${s.x} ${s.y} ${s.z}`;

/** The terminal UI's field values for a dock job. Blank means "use the default". */
export function dockValues(job, receptor) {
  const e = { ...EXPERT_DEFAULTS, ...(job.expert || {}) };
  return {
    peptide: job.peptide, receptor,
    blind: yn(job.blind),
    site: job.blind ? '' : siteStr(job.site), box: String(job.box),
    n_samples: String(job.poses),
    long_checkpoint_threshold: String(e.longCheckpointThreshold),
    refine_topk: e.refineTopK === '' || e.refineTopK == null ? '0' : String(e.refineTopK),
    ultra: e.ultra ? String(e.ultraK || EXPERT_DEFAULTS.ultraK) : '0',
    seed: e.seed === '' || e.seed == null ? '' : String(e.seed),
    input_poses: e.inputPoses || '',
    no_minimize: yn(e.noMinimize), ensemble: yn(e.ensemble),
    calibration: e.calibration && e.calibration !== EXPERT_DEFAULTS.calibration ? e.calibration : '',
    output_dir: job.outputDir,
  };
}

export function compareValues(job, target, offTarget) {
  return {
    peptide: job.peptide, blind: 'n',
    receptor: target, site: siteStr(job.target.site), box: String(job.target.box),
    offtarget_receptor: offTarget, offtarget_site: siteStr(job.offTarget.site), offtarget_box: String(job.offTarget.box),
    n_samples: String(job.poses), output_dir: job.outputDir,
  };
}

/** The terminal UI's field values for Score a structure (crystal-score). */
export const crystalValues = (job, receptor, pose) => ({ peptide: job.peptide, receptor, peptide_pdb: pose, mode: 'crystal' });

/**
 * The server's time estimate is a GPU figure. On a machine with no GPU it is wrong by an order of magnitude, and a
 * countdown that runs out while the run is still sampling ("Almost done" at 10%) is dishonest, so there is none there.
 */
export const noGpu = (env) => env?.checks?.gpu?.ok === false;
const liveEstimate = (job) => (noGpu(liveAdapter.env) ? null : job.estimateSeconds ?? null);

export function progressFrom(snap, estimateSeconds) {
  const fraction = Math.min(1, Math.max(0, snap.fraction || 0));
  // The server's progress is stage-weighted: inside the long sampling stage it has no finer detail, so the bar can
  // sit still for minutes. Extrapolating from it would give wild ETAs, so only the server's own rough estimate is used,
  // and only while the run is still within 1.5x of it. After that we say so instead of guessing.
  const known = estimateSeconds != null;
  const overdue = known && snap.elapsed >= estimateSeconds * 1.5;
  const etaSeconds = known && !overdue ? Math.max(0, estimateSeconds - snap.elapsed) : null;
  // a counter like "1/5" is the pipeline's own step number, not a pose count: only show per-item counts
  const m = /^(\d+)\/(\d+)$/.exec(snap.counter || '');
  const counter = m && Number(m[2]) > 5 ? snap.counter : '';
  return { stageIndex: STAGE_INDEX[snap.stage] ?? 0, fraction, etaSeconds, overdue, counter, lines: snap.lines || [] };
}

/** Poll a job until it finishes. Stop = cancel on the server. Survives short network hiccups. */
async function watch(jobId, { signal, onProgress, estimateSeconds }) {
  let since = 0, failures = 0;
  const high = { fraction: 0, stageIndex: 0 };
  const cancel = () => { API.post(`/api/jobs/${jobId}/cancel`, {}).catch(() => {}); };
  signal?.addEventListener('abort', cancel, { once: true });
  try {
    for (;;) {
      if (signal?.aborted) throw new DOMException('Stopped', 'AbortError');
      let snap;
      try {
        snap = await API.get(`/api/jobs/${jobId}?since=${since}`, signal);
        failures = 0;
      } catch (err) {
        if (err?.name === 'AbortError' || err?.status === 404 || ++failures >= 15) throw err;
        await sleep(1500, signal);
        continue;
      }
      since = snap.line_count;
      // The server's fraction and stage can step backwards (its parser also reads counters like "3/100" from any log
      // line), so what the user sees only ever moves forward.
      const p = progressFrom(snap, estimateSeconds);
      high.fraction = p.fraction = Math.max(high.fraction, p.fraction);
      high.stageIndex = p.stageIndex = Math.max(high.stageIndex, p.stageIndex);
      onProgress?.(p);
      if (snap.state === 'done') return snap;
      if (snap.state === 'cancelled') throw new DOMException('Stopped', 'AbortError');
      if (snap.state === 'failed') throw new Error(snap.error || 'The run failed.');
      await sleep(1000, signal);
    }
  } finally {
    signal?.removeEventListener('abort', cancel);
  }
}

/** Peptide C-alpha positions from a pose file (flat numbers, 1 decimal), keeping the last `n` residues. */
function peptideCA(text, n) {
  const all = caPoints(text);
  const start = Math.max(0, all.length / 3 - n) * 3;
  return Array.from(all.slice(start), round1);
}

const poseUrl = (jobId, file) => `/api/jobs/${jobId}/pose?name=${encodeURIComponent(file)}`;
const fileUrl = (jobId, name, download) => `/api/jobs/${jobId}/file?name=${encodeURIComponent(name)}${download ? '&download=1' : ''}`;

function dockResult(job, jobId, snap, res) {
  const dgOf = (r) => num(r.pooled_affinity_dg) ?? num(r.delta_g) ?? num(r.mmgbsa_dg);
  const poses = (res.poses || []).slice(0, 20).map((r, i) => ({
    rank: int(r.rank) || i + 1, file: r.pose_filename || '', deltaG: dgOf(r), rankScore: num(r.rank_score),
    nClash: int(r.n_clash), cluster: int(r.cluster_id), ca: null,
  }));
  const headline = poses[0]?.deltaG;
  if (headline == null) {
    throw new Error(`The run finished but no ΔG was reported. Look in the run folder: ${snap.output_dir}`);
  }
  return {
    id: newId('run'), kind: 'dock', demo: false, createdAt: new Date().toISOString(),
    name: `${job.peptide} → ${job.protein.name}`, protein: job.protein, peptide: job.peptide,
    site: job.site, box: job.box, blind: !!job.blind, nPoses: job.poses, deltaG: headline, poses,
    command: snap.command, outputDir: snap.output_dir, jobId, files: res.files || [], job: stripJob(job),
  };
}

/** Keep what is needed to re-run or describe a job, without big text blobs (history is saved in the browser). */
function stripJob(job) {
  const { peptidePdb, ...rest } = job;
  return peptidePdb ? { ...rest, peptidePdb: { name: peptidePdb.name } } : rest;
}

const liveAdapter = {
  kind: 'live',
  label: 'Live',
  supports: { allowClashes: false }, // the server's crystal-score command has no --allow-clashes
  env: null, //      filled by initAdapter from /api/env
  examples: [], //   filled by initAdapter from /api/examples

  estimate: (job) => liveEstimate(job),

  /** The exact command (from the server's own builder) and its rough time estimate. */
  async preview(job) {
    // Put the protein on the server first (cached per protein), so the server validates a real file and the command shows
    // its real path. Falling back to the plain name only if the upload itself fails.
    let receptor = job.protein.receptorPath || job.receptor, uploadedOk = !!job.protein.receptorPath;
    try { receptor = await receptorOnServer(job.protein); uploadedOk = true; } catch { /* keep the placeholder */ }
    const body = { mode: 'dock', values: dockValues(job, receptor), scoring: job.expert?.scoring || 'vina' };
    const [cmd, val] = await Promise.all([
      API.post('/api/preview', body).catch(() => null),
      API.post('/api/validate', body).catch(() => null),
    ]);
    const errors = Object.entries(val?.errors || {}).filter(([k]) => uploadedOk || k !== 'receptor'); // no file yet is not the user's mistake
    const problems = val && val.ok === false
      ? await Promise.all(errors.map(async ([k, m]) => `${await labelFor(k)}: ${m}`))
      : [];
    return { command: cmd?.command || null, estimateSeconds: val?.estimate_seconds ?? null, problems };
  },

  async runDock(job, opts = {}) {
    const receptor = await receptorOnServer(job.protein, opts.signal);
    const start = await API.post('/api/run', { mode: 'dock', values: dockValues(job, receptor), scoring: job.expert?.scoring || 'vina' }, opts.signal);
    const jobId = start.job.id;
    const snap = await watch(jobId, { ...opts, estimateSeconds: liveEstimate(job) });
    const res = await API.get(`/api/jobs/${jobId}/results`);
    const result = dockResult(job, jobId, snap, res);
    // Keep the best few poses' shapes with the saved result, so History can show them after a server restart.
    await Promise.all(result.poses.slice(0, 10).map((p) => this.loadPose(result, p.rank).catch(() => null)));
    return result;
  },

  async runCompare(job, opts = {}) {
    const [t, o] = [await receptorOnServer(job.target.protein, opts.signal), await receptorOnServer(job.offTarget.protein, opts.signal)];
    const start = await API.post('/api/run', { mode: 'selectivity', values: compareValues(job, t, o), scoring: 'vina' }, opts.signal);
    const jobId = start.job.id;
    const snap = await watch(jobId, { ...opts, estimateSeconds: liveEstimate(job) });
    let sel;
    try { sel = JSON.parse(await API.text(fileUrl(jobId, 'selectivity.json'))); }
    catch { throw new Error(`The comparison finished but its result file (selectivity.json) wasn't found in ${snap.output_dir}.`); }
    const side = async (key, s, dg) => ({ protein: s.protein, deltaG: dg, site: s.site, box: s.box, ca: await bestPoseCA(jobId, key, job.peptide.length) });
    return {
      id: newId('cmp'), kind: 'compare', demo: false, createdAt: new Date().toISOString(),
      name: `${job.peptide}: ${job.target.protein.name} vs ${job.offTarget.protein.name}`,
      peptide: job.peptide, ddg: sel.ddg_kcal_mol, ci: [sel.ddg_ci_95_low, sel.ddg_ci_95_high],
      scoreField: sel.score_field, topK: sel.top_k,
      target: await side('target', job.target, sel.target_dg_mean_kcal_mol),
      offTarget: await side('offtarget', job.offTarget, sel.offtarget_dg_mean_kcal_mol),
      command: snap.command, outputDir: snap.output_dir, jobId, job: stripJob(job),
    };
  },

  async runScore(job, opts = {}) {
    const receptor = await receptorOnServer(job.protein, opts.signal);
    const pose = await uploadText(job.peptidePdb.name, job.peptidePdb.text, opts.signal);
    const start = await API.post('/api/run', { mode: 'crystal', values: crystalValues(job, receptor, pose) }, opts.signal);
    const jobId = start.job.id;
    const snap = await watch(jobId, { ...opts, estimateSeconds: liveEstimate(job) });
    const res = await API.get(`/api/jobs/${jobId}/results`);
    const dg = res.headline?.delta_g;
    if (dg == null) throw new Error('The scoring finished but no ΔG could be read from its output.');
    return {
      id: newId('scr'), kind: 'score', demo: false, createdAt: new Date().toISOString(),
      name: `${job.peptide} on ${job.protein.name} (scored)`, protein: job.protein, peptide: job.peptide, deltaG: dg,
      ca: peptideCA(job.peptidePdb.text, job.peptide.length), command: snap.command, outputDir: snap.output_dir, jobId, job: stripJob(job),
    };
  },

  /** Peptide C-alpha coordinates of one pose, from the saved copy or the server (null if unavailable). */
  async loadPose(result, rank) {
    const pose = result.poses?.find((p) => p.rank === rank);
    if (!pose) return null;
    if (!pose.ca) {
      if (!result.jobId || !pose.file) return null;
      try { pose.ca = peptideCA(await API.text(poseUrl(result.jobId, pose.file)), result.peptide.length); }
      catch { return null; }
    }
    return pose.ca.length === result.peptide.length * 3 ? Float32Array.from(pose.ca) : null;
  },

  async download(result, what) {
    if (what === 'run_folder') return { copyText: result.outputDir };
    const name = what === 'best_pose' ? 'best_pose.pdb' : 'ranked_poses.csv';
    try {
      return { filename: name, blob: new Blob([await API.text(fileUrl(result.jobId, name, true))]) };
    } catch (err) {
      throw new Error(err.status === 404
        ? `The server no longer has this run (it may have been restarted). The files are still on disk in ${result.outputDir}.`
        : err.message);
    }
  },
};

/** First pose file of one side of a comparison ("target" / "offtarget"), as peptide C-alpha coordinates. */
async function bestPoseCA(jobId, side, n) {
  try {
    const csv = (await API.text(fileUrl(jobId, `${side}/ranked_poses.csv`))).trim().split(/\r?\n/);
    const header = csv[0].split(',');
    const file = csv[1].split(',')[header.indexOf('pose_filename')];
    for (const sub of ['poses_scored', 'poses_minimized', 'poses', 'poses_raw']) {
      try { return peptideCA(await API.text(fileUrl(jobId, `${side}/${sub}/${file}`)), n); } catch { /* try the next folder */ }
    }
  } catch { /* the pose files may have been cleaned up */ }
  return null;
}

/* ================================================================================================
 * Choosing the adapter
 * ============================================================================================== */

export let adapter = mockAdapter;

/** Demo results (and demo History entries) are always handled by the demo adapter, even when the live backend is active. */
export const demoAdapter = mockAdapter;
export const adapterFor = (result) => (result?.demo ? mockAdapter : adapter);

/** Pick live when this page is served by `hybridock-pep serve`, otherwise demo. Call once at start-up. */
export async function initAdapter() {
  const q = new URLSearchParams(globalThis.location?.search || '');
  // A static copy (GitHub Pages, scripts/build_pages.py) has no server behind it: go straight to the demo.
  if (q.has('demo') || globalThis.location?.protocol === 'file:' || globalThis.HYBRIDOCK_STATIC) return (adapter = mockAdapter);
  try {
    const res = await fetch('/api/env', { cache: 'no-store', signal: AbortSignal.timeout(2500) });
    if (!res.ok) throw new Error('no api');
    liveAdapter.env = await res.json();
    liveAdapter.examples = (await (await fetch('/api/examples', { cache: 'no-store' })).json()).examples || [];
    return (adapter = liveAdapter);
  } catch {
    if (q.has('live')) { liveAdapter.env = null; return (adapter = liveAdapter); } // forced: errors will say why
    return (adapter = mockAdapter);
  }
}
