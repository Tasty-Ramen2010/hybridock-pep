// state.js — one small store for the whole app, saved to the browser's localStorage.
// Only a few things are saved (mode, theme, accent, name, history). Everything else resets on reload.

import { BOX_DEFAULT, DEFAULT_THOROUGHNESS, EXPERT_DEFAULTS } from './config.mjs';

const KEY = 'hybridock-web:v1';
/** The system light/dark setting, used until the person picks one with the toggle. */
function systemTheme() {
  try { return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'; } catch { return 'light'; }
}

const SAVED = ['mode', 'theme', 'accent', 'userName', 'history', 'runsTotal', 'activeRun'];
const MAX_HISTORY = 30;

function load() {
  try { return JSON.parse(localStorage.getItem(KEY) || '{}'); } catch { return {}; }
}

function save(state) {
  const out = {};
  for (const k of SAVED) out[k] = state[k];
  const attempt = (hist) => localStorage.setItem(KEY, JSON.stringify({ ...out, history: hist }));
  try {
    attempt(state.history);
  } catch {
    try { attempt(state.history.map((e, i) => (i < 5 ? e : { ...e, result: undefined }))); } // storage full: keep only recent poses
    catch { /* storage unavailable: the app still works, it just won't remember */ }
  }
}

const stamp = () => Date.now().toString(36);

export const freshSetup = () => ({
  step: 1,
  runStamp: stamp(),
  proteinRef: null, //        { key?, name, file?, … } chosen protein
  peptide: '',
  siteMode: 'known', //       'known' | 'find'
  site: null, //              { x, y, z }
  box: BOX_DEFAULT,
  thorough: DEFAULT_THOROUGHNESS,
  expert: { ...EXPERT_DEFAULTS },
});

export const freshCompare = () => ({
  runStamp: stamp(),
  thorough: DEFAULT_THOROUGHNESS,
  peptide: '',
  target: { proteinRef: null, site: null, box: BOX_DEFAULT },
  offTarget: { proteinRef: null, site: null, box: BOX_DEFAULT },
  editing: null, // 'target' | 'offTarget' while adjusting a site in 3D
  focus: 'target',
});

export const freshScore = () => ({ protein: null, peptidePdb: null, peptide: '', allowClashes: false });

export function createStore() {
  const saved = load();
  let state = {
    mode: saved.mode || 'guided',
    theme: saved.theme || systemTheme(),
    accent: saved.accent || 'teal',
    userName: saved.userName && saved.userName !== 'there' ? saved.userName : '', // empty until the visitor adds one
    runsTotal: Number.isFinite(saved.runsTotal) ? saved.runsTotal : (Array.isArray(saved.history) ? saved.history.length : 0), // lifetime count: History keeps only the last 30
    history: Array.isArray(saved.history) ? saved.history : null, // null = never used: main.js seeds demo entries
    historySeeded: Array.isArray(saved.history),
    activeRun: saved.activeRun && saved.activeRun.jobId ? saved.activeRun : null, // a run the server may still be working on: resumed after a reload
    proteins: [],
    setup: freshSetup(),
    compare: freshCompare(),
    score: freshScore(),
    run: { status: 'idle' },
  };
  const subs = new Set();
  return {
    get: () => state,
    set(patch) {
      state = { ...state, ...patch };
      if (SAVED.some((k) => k in patch)) save({ ...state, history: state.history || [] });
      subs.forEach((fn) => fn(state, patch));
    },
    /** Merge into a nested object: store.merge('setup', { step: 2 }) */
    merge(key, patch) { this.set({ [key]: { ...state[key], ...patch } }); },
    subscribe(fn) { subs.add(fn); return () => subs.delete(fn); },
    addHistory(entry) {
      const had = (state.history || []).some((e) => e.id === entry.id); // a resumed run can arrive twice (two tabs)
      const history = [entry, ...(state.history || []).filter((e) => e.id !== entry.id)].slice(0, MAX_HISTORY);
      this.set({ history, runsTotal: had ? state.runsTotal : Math.max(state.runsTotal || 0, (state.history || []).length) + 1 });
    },
    updateHistory(id, patch) {
      this.set({ history: (state.history || []).map((e) => (e.id === id ? { ...e, ...patch } : e)) });
    },
    clearHistory() { this.set({ history: [], runsTotal: 0 }); },
  };
}

/** What a history row shows: a short label and a headline number. */
export function historyEntryFor(result) {
  const headline = result.kind === 'compare'
    ? { label: 'ΔΔG', value: result.ddg }
    : { label: 'ΔG', value: result.deltaG };
  const keys = result.kind === 'compare'
    ? [result.target.protein.name, result.offTarget.protein.name]
    : [result.protein.name];
  return { id: result.id, kind: result.kind, name: result.name, createdAt: result.createdAt, demo: !!result.demo, headline, proteinNames: keys, result };
}
