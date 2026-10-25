// guide-content.mjs — the words of the in-app guide, as data. ui/guide.mjs draws it; nothing here touches the page.
//
// The guide is about using the app and nothing else: how to get it, how to run a prediction, how to read what comes back,
// and what to do when something goes wrong. Keep every line short and plain. A step is one idea, one sentence of body.
//
//   **bold** and `code` are the only marks the text may use.
//   `only` shows a topic, step or row to one audience: 'demo' (the preview website), 'live' (the real app, run here or on the
//   team's shared computer), 'local' (the real app on your own computer) or 'remote' (the team's shared computer).
//   `image` is a file in docs/guide-img/ (copied to static/guide/guide-img/ by scripts/build_web.py).

export const COLAB_URL = 'https://colab.research.google.com/github/Tasty-Ramen2010/hybridock-pep/blob/master/notebooks/HybriDock_Pep_Colab.ipynb';

/**
 * @typedef {{title: string, body: string, image?: string, icon?: string, cmd?: string, tip?: string, only?: 'demo'|'live'|'local'|'remote'}} Step
 * @typedef {{q?: string, term?: string, icon?: string, text: string, only?: 'demo'|'live'|'local'|'remote'}} Row
 * @typedef {{id: string, title: string, blurb: string, icon: string, kind: 'steps'|'rows'|'accordion', minutes: number,
 *            cover?: string, dark?: boolean, steps?: Step[], rows?: Row[], only?: 'demo'|'live'|'local'|'remote', first?: 'demo'|'live'}} Topic
 */

/** @type {Topic[]} */
export const TOPICS = [
  {
    id: 'live-demo', title: 'Try the live demo', blurb: 'See a real result in two minutes.', icon: 'play', kind: 'steps', minutes: 2,
    steps: [
      { title: 'Open the website', image: 'home.png', body: 'Press **Try the live demo** at the top of the GitHub page. The home screen opens with a row of diseases we have already worked on.' },
      { title: 'Swipe through the diseases', image: 'carousel.png', body: 'Use the arrows, or swipe, to move from one disease to the next. Each card names the protein behind the illness and lists its best peptide matches.' },
      { title: 'Open a result', image: 'result.png', body: 'Tap a match to see how the peptide fits the protein in 3D. **More negative means a tighter grip.**' },
      { title: 'Compare two matches', image: 'tray.png', body: 'Press **Compare** on any matches you like. They line up side by side, with the tightest grip highlighted.' },
      { title: 'Run it live', image: 'run-row.png', body: 'Pick **5, 10 or 25** poses and press **Run it live**. The time shown beside the button is how long it takes on our computer.' },
      { title: 'Watch it finish', image: 'running.png', body: 'You can leave the page while it works. **Run in progress** in the top bar brings you back to the result.' },
    ],
  },
  {
    id: 'your-own', title: 'Test your own peptide', blurb: 'Three screens from idea to result.', icon: 'zap', kind: 'steps', minutes: 2,
    steps: [
      { title: 'Choose a protein', image: 'protein.png', body: 'Press **Test a peptide** on Home and pick the protein you care about. The place where a peptide would attach is already set for you.',
        tip: 'Not on the list? Type its four-letter PDB code, or upload your own `.pdb` file.' },
      { title: 'Type a peptide', image: 'peptide.png', body: 'A peptide is a short chain of amino acids, written as letters. Type 3 to 25 of them, or tap one of the named examples.' },
      { title: 'Choose 5, 10 or 25 poses', image: 'review.png', body: 'More poses is a more thorough search. The time under each choice is how long it takes. Then press **Run prediction**.' },
      { title: 'Read the result', image: 'result.png', body: 'The big number is how tightly the peptide sticks. **More negative means tighter.** Click any row in the list to see that pose in 3D.' },
    ],
  },
  {
    id: 'results', title: 'Read a result', blurb: 'What the big number means.', icon: 'badge', kind: 'steps', minutes: 1,
    steps: [
      { title: 'The big number', image: 'result.png', body: 'It is the binding strength, in kcal/mol. **More negative means the peptide holds on harder.**' },
      { title: 'Strong, Good or Light', icon: 'info', body: 'The coloured label puts the number in words so you can read it at a glance.' },
      { title: 'See each pose', image: 'poses.png', body: 'A pose is one way the peptide can sit on the protein. Click a row to see it in 3D, and drag to turn the view.' },
      { title: 'Keep your files', image: 'poses.png', body: '**Best pose** opens in PyMOL or ChimeraX. **Ranked list** is a spreadsheet of every pose. Earlier runs stay in **History**.' },
    ],
  },
  {
    id: 'compare', title: 'Compare two proteins', blurb: 'Does a peptide prefer one over another?', icon: 'columns', kind: 'steps', minutes: 1,
    steps: [
      { title: 'Choose a peptide and two proteins', image: 'compare.png', body: 'Pick the protein you want the peptide to grab (the target) and a look-alike you want it to leave alone.' },
      { title: 'Press Compare', icon: 'play', body: 'It runs both proteins, so it takes about twice as long as one. The time is shown under each choice.' },
      { title: 'Read the verdict', image: 'compare-result.png', body: '**Below zero** means the peptide prefers your target. A medicine that ignores the look-alike is a safer medicine.' },
    ],
  },
  {
    id: 'score', title: 'Score a structure', blurb: 'Rate a pose you already have.', icon: 'file', kind: 'steps', minutes: 1,
    steps: [
      { title: 'Add your two files', image: 'score.png', body: 'Drop in the protein as a `.pdb`, then the peptide in its bound position. The sequence fills in for you.' },
      { title: 'Check the sequence', icon: 'check', body: 'It must have the same number of letters as the peptide file. The page tells you if it does not.' },
      { title: 'Press Score it', icon: 'zap', body: 'You get the binding strength in a few seconds. Nothing is moved. It only measures the pose you gave it.' },
    ],
  },
  {
    id: 'own-computer', title: 'Run it on your own computer', blurb: 'Three commands, once.', icon: 'download', kind: 'steps', minutes: 2, cover: 'terminal',
    steps: [
      { title: 'Download it', icon: 'download', body: 'Open a terminal and paste these two lines. This copies the app into a new folder.',
        cmd: 'git clone --recurse-submodules https://github.com/Tasty-Ramen2010/hybridock-pep.git\ncd hybridock-pep',
        tip: 'On Windows, download the project and double-click `install.bat` instead of the next step.' },
      { title: 'Install it', icon: 'zap', body: 'This sets up everything the app needs and opens it when it finishes.', cmd: './install.sh' },
      { title: 'Start it any time', icon: 'play', body: 'Run this from the app folder. Your browser opens the app. Press Ctrl-C in the terminal to stop it.', cmd: './launch_web.sh',
        tip: 'On a Mac you can also double-click `HybriDock-Pep.command`. On Windows, double-click `launch_web.bat`.' },
    ],
  },
  {
    id: 'troubleshooting', title: 'If something goes wrong', blurb: 'Quick fixes for common messages.', icon: 'alert', kind: 'accordion', minutes: 1,
    rows: [
      { q: 'Setup isn’t finished on this computer', text: 'Something the app needs is missing. The message names it and gives the command that fixes it, usually `./install.sh`.', only: 'local' },
      { q: 'Another run is already going', text: 'The app does one prediction at a time. Wait for it, or open **Run in progress** in the top bar.' },
      { q: 'The lab computer is busy', text: 'It does one prediction at a time, for everyone. Wait a few minutes and press the button again.', only: 'remote' },
      { q: 'The lab computer is offline', text: 'The results in the carousel are still there to open. New runs here are simulated and marked Demo until it is back. Reload in a few minutes.', only: 'remote' },
      { q: 'No pose fit at this site', text: 'Every pose overlapped the protein, so there is no result. Try a shorter peptide.' },
      { q: 'The computer ran out of memory', text: 'Close other programs, choose **5 poses**, or use a shorter peptide.', only: 'local' },
      { q: 'Couldn’t reach the server', text: 'The app has stopped. Start it again with `./launch_web.sh`, reload this page, and press **Check again**. A run already going keeps going.', only: 'local' },
      { q: '`command not found: hybridock-pep`', text: 'Start the app with `./launch_web.sh` from the app folder. It uses the right environment for you.', only: 'local' },
      { q: 'The 3D view is empty', text: 'Your browser may have blocked it. Try another browser.' },
      { q: 'Where do I get the real app?', text: 'Open **Run it on your own computer** in this guide. It is three commands.', only: 'demo' },
      { q: 'The numbers say Demo', text: 'This page could not reach a computer to run on, so the run was simulated. The results in the carousel are real. Reload to try again.', only: 'demo' },
    ],
  },
  {
    id: 'words', title: 'Words you’ll see', blurb: 'Plain meanings.', icon: 'book', kind: 'rows', minutes: 1,
    rows: [
      { term: 'Protein', text: 'A large molecule in the body. Many diseases start with one that misbehaves.' },
      { term: 'Peptide', text: 'A short chain of amino acids, typed as letters. Think of it as a key that can fit the protein’s lock.' },
      { term: 'Binding strength (ΔG)', text: 'How tightly the peptide sticks, in kcal/mol. More negative means tighter.' },
      { term: 'Pose', text: 'One way the peptide could sit on the protein.' },
      { term: 'Binding site', text: 'The place on the protein where the peptide attaches. Every protein in the catalog already has it set.' },
      { term: 'ΔΔG', text: 'The difference between two binding strengths. Below zero means the first protein is preferred.' },
      { term: 'PDB', text: 'The free public library of protein structures, and the file type they use.' },
    ],
  },
];

/** What a person sees: the topics (and inside them the steps and rows) meant for this audience, in the right order. */
export function topicsFor(kind) {
  const keep = (x) => !x.only || x.only === kind || (x.only === 'live' && kind !== 'demo');
  const out = TOPICS.filter(keep).map((t) => ({
    ...t,
    steps: t.steps?.filter(keep),
    rows: t.rows?.filter(keep),
  }));
  return out; // the order above is the order of the numbers on the tiles
}

export const findTopic = (list, id) => list.find((t) => t.id === id) || null;

/** The size of a topic, as the tile shows it: "6 steps", "5 items". */
export function sizeOf(t) {
  const n = t.kind === 'steps' ? t.steps.length : t.rows.length;
  return t.kind === 'steps' ? `${n} step${n === 1 ? '' : 's'}` : `${n} item${n === 1 ? '' : 's'}`;
}

/** Split **bold** and `code` into parts: [{t:'text'|'strong'|'code', v}]. Anything else is plain text. */
export function richParts(text) {
  const parts = [];
  const re = /\*\*([^*]+)\*\*|`([^`]+)`/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) parts.push({ t: 'text', v: text.slice(last, m.index) });
    parts.push(m[1] != null ? { t: 'strong', v: m[1] } : { t: 'code', v: m[2] });
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push({ t: 'text', v: text.slice(last) });
  return parts;
}
