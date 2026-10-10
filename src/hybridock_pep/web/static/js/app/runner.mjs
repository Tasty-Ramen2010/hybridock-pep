// runner.js — starts a run, reports progress to the Running screen, and handles stop / errors.
// It only ever calls the adapter; it never knows how the backend works.

import { adapter } from './adapter.mjs';
import { friendlyError } from './errors.mjs';
import { historyEntryFor } from './state.mjs';
import { toast } from './ui/toast.mjs';

export function createRunner({ store, go }) {
  let controller = null;

  /**
   * Start a run, or (with `resume`) carry on watching one the server already has: after a page reload, or after a
   * dropped connection. The job id is saved the moment the server accepts the run, so closing the tab never loses it.
   */
  async function start(kind, job, { backTo = '/', resume = null } = {}) {
    // One run at a time, like the server. Starting a second one used to overwrite the first run's state: the server then
    // refused it, and the person lost the way back to the run that was still going (the pill vanished, Stop hit the wrong job).
    if (store.get().run.status === 'running') {
      toast('A run is already going. Wait for it to finish (use “Run in progress” at the top), or stop it first.', 7000);
      return;
    }
    controller = new AbortController();
    const startedAt = resume?.startedAt || Date.now();
    const clearActive = () => { if (store.get().activeRun) store.set({ activeRun: null }); };
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
        jobId: resume?.jobId,
        onJobId: (jobId) => store.set({ activeRun: { kind, job, jobId, startedAt, backTo } }),
        onProgress: (progress) => {
          const run = store.get().run;
          if (run.status === 'running') store.set({ run: { ...run, progress } });
        },
      });
      clearActive();
      adapter.refreshEnv?.(); // the first real run has downloaded the model files: the "first prediction" notes can go
      store.addHistory(historyEntryFor(result));
      store.set({ run: { status: 'idle' } });
      go(`/results/${result.id}`);
    } catch (err) {
      if (err?.name === 'AbortError') {
        clearActive();
        store.set({ run: { status: 'idle' } });
        toast('Stopped. Nothing was saved.');
        go(backTo);
      } else {
        // A dead connection does not end the run on the server: keep its id so "Try again" picks it up instead of starting over.
        const active = store.get().activeRun;
        const resumable = err?.name === 'NetworkError' && active?.jobId ? { jobId: active.jobId, startedAt } : null;
        if (!resumable) clearActive();
        store.set({ run: { ...store.get().run, status: 'error', error: { ...friendlyError(err), resume: resumable } } });
      }
    } finally {
      controller = null;
    }
  }

  return { start, stop: () => controller?.abort() };
}
