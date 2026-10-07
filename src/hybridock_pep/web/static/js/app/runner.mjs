// runner.js — starts a run, reports progress to the Running screen, and handles stop / errors.
// It only ever calls the adapter; it never knows how the backend works.

import { adapter, NotWiredError } from './adapter.mjs';
import { historyEntryFor } from './state.mjs';
import { toast } from './ui/toast.mjs';

/** Turn any error into a sentence a student can act on. */
export function friendlyError(err) {
  if (err instanceof NotWiredError) return { title: 'The live backend isn’t connected yet', body: err.message };
  if (err?.name === 'TypeError' && /fetch|network/i.test(err.message)) {
    return { title: 'Couldn’t reach the server', body: 'Check that it is running and your connection is working, then try again.' };
  }
  return { title: 'Something went wrong', body: err?.message || 'The run stopped unexpectedly.' };
}

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
