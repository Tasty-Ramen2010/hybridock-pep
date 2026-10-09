// runner.js — starts a run, reports progress to the Running screen, and handles stop / errors.
// It only ever calls the adapter; it never knows how the backend works.

import { adapter } from './adapter.mjs';
import { friendlyError } from './errors.mjs';
import { historyEntryFor } from './state.mjs';
import { toast } from './ui/toast.mjs';

export function createRunner({ store, go }) {
  let controller = null;

  async function start(kind, job, { backTo = '/' } = {}) {
    controller = new AbortController();
    const startedAt = Date.now();
    store.set({
      run: {
        status: 'running', kind, job, startedAt, backTo, error: null,
        progress: { stageIndex: 0, fraction: 0, etaSeconds: adapter.estimate({ ...job, kind }) },
      },
    });
    go('/running');
    const call = { dock: 'runDock', compare: 'runCompare', score: 'runScore' }[kind];
    try {
      const result = await adapter[call](job, {
        signal: controller.signal,
        onProgress: (progress) => {
          const run = store.get().run;
          if (run.status === 'running') store.set({ run: { ...run, progress } });
        },
      });
      store.addHistory(historyEntryFor(result));
      store.set({ run: { status: 'idle' } });
      go(`/results/${result.id}`);
    } catch (err) {
      if (err?.name === 'AbortError') {
        store.set({ run: { status: 'idle' } });
        toast('Stopped. Nothing was saved.');
        go(backTo);
      } else {
        store.set({ run: { ...store.get().run, status: 'error', error: friendlyError(err) } });
      }
    } finally {
      controller = null;
    }
  }

  return { start, stop: () => controller?.abort() };
}
