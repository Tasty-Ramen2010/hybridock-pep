// errors.mjs — turn what went wrong into a sentence a student can act on.
// The raw message is always kept as `detail` (shown under "Technical details"), so nothing is hidden from an expert.
// Pure logic, no DOM: it is unit-tested in tests/web_js/.

const RULES = [
  { // the server runs one job at a time (see web/server.py); a second request is refused
    test: /run is already going|already running/i,
    title: 'Another run is already going',
    body: 'This computer runs one prediction at a time. Wait for the current one to finish (or stop it), then try again.',
  },
  { // the sampling environment isn't there (a machine set up only for scoring)
    test: /Cannot locate Python 3 in conda env|RAPIDOCK_PYTHON|rapidock.*(not found|missing|doesn.t exist)|No module named '?(torch|e3nn|torch_geometric)/i,
    title: 'The docking engine isn’t installed on this computer',
    body: 'Predicting binding needs the sampling environment (rapidock), and it isn’t set up here. You can still use Score a structure on this computer, or run ./install.sh to add the docking engine.',
  },
  { // crystal-score could not read or featurise the pose
    test: /Crystal scoring failed|geometry features/i,
    title: 'That pose couldn’t be scored',
    body: 'Check that the peptide file is the pose bound to the protein you uploaded, that it has only the peptide, and that the sequence has the same number of letters as the pose.',
  },
  { // out of memory (the kernel kills the run: exit 137 / "Killed")
    test: /out of memory|MemoryError|\bKilled\b|exit(ed)? (code|status) (-9|137)|cannot allocate memory|CUDA out of memory/i,
    title: 'The computer ran out of memory',
    body: 'Close other programs and try again. A smaller job (a shorter peptide, or Quick thoroughness) needs less memory.',
  },
  { // the server forgot the run (it was restarted while the run was going)
    test: /unknown job|no such job|job .* not found/i,
    title: 'The server restarted during your run',
    body: 'The run was stopped when the server restarted. Start it again to continue.',
  },
  { // the run's input file vanished or is unreadable
    test: /No such file or directory|FileNotFoundError/i,
    title: 'A file for this run went missing',
    body: 'The protein or pose file is no longer where the run expected it. Choose the files again and start over.',
  },
];

export function friendlyError(err) {
  const message = String(err?.message || '').trim();
  if (err?.name === 'NotWiredError') return { title: 'The live backend isn’t connected yet', body: message, detail: '' };
  if (err?.name === 'NetworkError' || (err?.name === 'TypeError' && /fetch|network|load failed/i.test(message))) {
    return { title: 'Couldn’t reach the server', body: 'Check that it is running and your connection is working, then try again.', detail: message };
  }
  if (err?.status === 404 && !/pose|file/i.test(message)) {
    return { title: 'The server restarted during your run', body: 'The run was stopped when the server restarted. Start it again to continue.', detail: message };
  }
  const rule = RULES.find((r) => r.test.test(message));
  if (rule) return { title: rule.title, body: rule.body, detail: message };
  return {
    title: 'That run didn’t finish',
    body: 'The program stopped before it could produce a result. Try again; if it keeps happening, the technical details below say why.',
    detail: message || 'The run stopped unexpectedly.',
  };
}
