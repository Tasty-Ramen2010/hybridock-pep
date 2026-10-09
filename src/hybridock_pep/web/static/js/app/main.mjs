// main.js — start-up and the tiny router. Each screen is a module that returns { el, destroy }.
//
//   #/            Home            #/predict   Predict binding (4-step setup)
//   #/compare     Compare two proteins        #/score     Score a structure
//   #/running     a run in progress           #/results/<id>   a finished run

import { Stage } from './stage/stage.mjs';
import { adapter, initAdapter } from './adapter.mjs';
import { createStore } from './state.mjs';
import { createRunner } from './runner.mjs';
import { ACCENTS, assetUrl } from './config.mjs';
import { mountTopbar } from './ui/topbar.mjs';
import { toast } from './ui/toast.mjs';
import { fetchRetry } from './structures.mjs';
import { seedHistory } from './seed.mjs';
import { mountHome } from './ui/home.mjs';
import { mountSetup } from './ui/setup.mjs';
import { mountRunning } from './ui/running.mjs';
import { mountResults } from './ui/results.mjs';
import { mountCompare } from './ui/compare.mjs';
import { mountScore } from './ui/score.mjs';

const store = createStore();
const stage = new Stage(document.getElementById('stage'));
const main = document.getElementById('main');

const go = (path) => { location.hash = '#' + path; };
const runner = createRunner({ store, go });
const ctx = { store, stage, adapter, go, toast, runner, main };

const SCREENS = {
  '': { mount: mountHome, title: 'HybriDock-Pep' },
  predict: { mount: mountSetup, title: 'Predict binding' },
  compare: { mount: mountCompare, title: 'Compare two proteins' },
  score: { mount: mountScore, title: 'Score a structure' },
  running: { mount: mountRunning, title: 'Running…' },
  results: { mount: mountResults, title: 'Results' },
};

// ---- light/dark, Guided/Expert and the accent colour all come from three attributes/variables ----
function applyAppearance() {
  const { theme, mode, accent } = store.get();
  const root = document.documentElement;
  root.dataset.theme = theme;
  root.dataset.mode = mode;
  const a = ACCENTS.find((x) => x.id === accent) || ACCENTS[0];
  root.style.setProperty('--accent', a[theme]);
  stage.refreshTheme();
}
let lastLook = '';
store.subscribe((s) => {
  const look = `${s.theme}|${s.mode}|${s.accent}`;
  if (look !== lastLook) { lastLook = look; applyAppearance(); }
});

// ---- router ----
let current = null;
let routeToken = 0;
let firstRender = true; // on page load, leave focus alone so the first Tab reaches the skip link and the top bar

function parseHash() {
  const [name = '', param] = location.hash.replace(/^#\/?/, '').split('/');
  return { name, param };
}

async function render() {
  const { name, param } = parseHash();
  const def = SCREENS[name] || SCREENS[''];
  const token = ++routeToken;
  current?.destroy?.();
  stage.editBox(null);
  stage.setBox(null);
  current = def.mount({ ...ctx, param });
  main.replaceChildren(current.el);
  document.title = name in SCREENS && name !== '' ? `${def.title} · HybriDock-Pep` : 'HybriDock-Pep: see how a peptide binds a protein';
  stage.setSlot(current.el.querySelector('[data-stage-slot]'));
  window.scrollTo(0, 0);
  if (!firstRender) main.focus({ preventScroll: true }); // after in-app navigation, move focus to the new screen
  firstRender = false;
  void token;
}

/** The server's /api/examples carry validated sites and the real receptor file; prefer them over my derived defaults. */
function applyBackendExamples(proteins) {
  for (const ex of adapter.examples || []) {
    const p = proteins.find((x) => x.backendExample === ex.id);
    if (!p) continue;
    Object.assign(p, {
      receptorPath: ex.receptor_path, site: { x: ex.site[0], y: ex.site[1], z: ex.site[2] }, box: ex.box,
      examplePeptide: ex.peptide, siteNote: p.siteNote, expect: ex.expect, blurb: ex.blurb, exampleNote: ex.note,
    });
  }
}

async function boot() {
  try { sessionStorage.removeItem('hp-retry'); } catch { /* storage unavailable */ } // the page started: re-arm the one-time retry
  await initAdapter();
  ctx.adapter = adapter;
  try {
    const res = await fetchRetry(assetUrl('data/proteins.json'));
    const proteins = await res.json();
    if (adapter.kind === 'live') applyBackendExamples(proteins);
    store.set({ proteins });
  } catch {
    toast('Couldn’t load the protein list. Run HybriDock-Pep from a web server (see README).', 8000);
  }
  if (!store.get().historySeeded) store.set({ history: adapter.kind === 'demo' ? seedHistory(store.get().proteins) : [], historySeeded: true });
  applyAppearance();
  mountTopbar(ctx);
  addEventListener('hashchange', render);
  render();
}

// Handy for debugging in the browser console.
window.hybridock = ctx;
boot();
