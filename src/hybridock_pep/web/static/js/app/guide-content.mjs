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
    id: 'get-started', title: 'Get the app', blurb: 'Install it once. Start it any time.', icon: 'download', kind: 'steps', minutes: 2, first: 'demo', cover: 'terminal',
    steps: [
      { title: 'Download it', icon: 'download', body: 'Open a terminal and paste these two lines. This copies the app into a new folder.',
        cmd: 'git clone --recurse-submodules https://github.com/Tasty-Ramen2010/hybridock-pep.git\ncd hybridock-pep',
        tip: 'On Windows, download the project and double-click `install.bat` instead of the next step.' },
      { title: 'Install it', icon: 'zap', body: 'This sets up everything the app needs. It takes about 30 minutes, once, and opens the app when it finishes.',
        cmd: './install.sh' },
      { title: 'Start it again later', icon: 'play', body: 'Run this from the app folder. Your browser opens the app. Press Ctrl-C in the terminal to stop it.',
        cmd: './launch_web.sh',
        tip: 'On a Mac you can also double-click `HybriDock-Pep.command`. On Windows, double-click `launch_web.bat`.' },
      { title: 'Expect a slow first run', icon: 'info', body: 'The first prediction downloads about 2.5 GB of model files, which adds around 10 minutes. After that it is quick to start.',
        tip: 'No terminal? You can also run it in Google Colab. The link is on the Home page of the preview.', only: 'demo' },
      { title: 'Expect a slow first run', icon: 'info', body: 'The first prediction downloads about 2.5 GB of model files, which adds around 10 minutes. The app tells you when this is happening.', only: 'live' },
    ],
  },
  {
    id: 'first-prediction', title: 'Your first prediction', blurb: 'Pick a protein and a peptide. Press run.', icon: 'play', kind: 'steps', minutes: 2, first: 'live', cover: 'result.png',
    steps: [
      { title: 'Start from Home', image: 'home.png', body: 'Press **Run example** to see a finished result in one click. It uses a p53 peptide and its partner protein MDM2.',
        tip: 'This is the preview, so the numbers are simulated and marked Demo. Install the app for real predictions.', only: 'demo' },
      { title: 'Start from Home', image: 'home.png', body: 'Press **Run example** to see a finished result in one click. It uses a p53 peptide and its partner protein MDM2. To use your own, press **Predict**.', only: 'live' },
      { title: 'Choose a protein', image: 'protein.png', body: 'Pick one from the list, type a four-letter PDB code, or upload your own `.pdb` file.',
        tip: 'Use a clean file with just the protein. Remove ligands and water first.' },
      { title: 'Type a peptide', image: 'peptide.png', body: 'A peptide is a short chain of amino acids. Type it as letters, 3 to 30 of them, or tap one of the named examples.' },
      { title: 'Mark the binding site', image: 'site.png', body: 'Press **Use the suggested site**, or drag the box over the part of the protein you care about. Not sure? Choose **Find the pocket for me**. It is slower.' },
      { title: 'Choose how thorough', image: 'review.png', body: '**Quick** is a first look. **Full** is the most reliable. Then press **Run prediction**.' },
      { title: 'Wait for it', image: 'running.png', body: 'You can leave this page. The run keeps going, and **Run in progress** in the top bar brings you back.',
        tip: 'The app runs one prediction at a time.' },
    ],
  },
  {
    id: 'results', title: 'Read your result', blurb: 'What the big number means.', icon: 'badge', kind: 'steps', minutes: 2,
    steps: [
      { title: 'The big number is ΔG', image: 'result.png', body: 'It is how tightly the peptide sticks. **More negative means tighter.** One number is off by about ±1.6 kcal/mol, so treat it as a good estimate, not a measurement.' },
      { title: 'Strong, Moderate or Weak', icon: 'info', body: 'This label is our own rough wording for the number, so you can read it at a glance. Below −9.0 is Strong, above −6.5 is Weak.',
        tip: 'The program does not produce the label. Only the number is a result.' },
      { title: 'See each pose', image: 'poses.png', body: 'A pose is one way the peptide can sit on the protein. Click a row to see it in 3D, and drag to turn the view.' },
      { title: 'Keep your files', icon: 'download', body: '**Best pose** is a structure file for PyMOL or ChimeraX. **Ranked list** is a spreadsheet of every pose.',
        tip: 'Earlier runs stay in **History** in the top bar, saved in your browser.' },
    ],
  },
  {
    id: 'compare', title: 'Compare two proteins', blurb: 'Does a peptide prefer one over another?', icon: 'columns', kind: 'steps', minutes: 1,
    steps: [
      { title: 'Choose a peptide and two proteins', image: 'compare.png', body: 'Pick the protein you want it to bind (the target) and a look-alike you want it to ignore (the off-target).' },
      { title: 'Press Compare', icon: 'play', body: 'It runs both predictions, so it takes about twice as long as one.' },
      { title: 'Read the verdict', image: 'compare-result.png', body: 'ΔΔG is the difference between the two. **Below zero** means the peptide prefers your target. If the band crosses zero, there is no clear preference.',
        tip: 'A quick comparison can swing between runs. Repeat it, or use **Full**, before you trust it.' },
    ],
  },
  {
    id: 'score', title: 'Score a structure', blurb: 'Rate a pose you already have.', icon: 'file', kind: 'steps', minutes: 1, dark: true,
    steps: [
      { title: 'Add your two files', image: 'score.png', body: 'Drop in the protein as a `.pdb`, then the peptide in its bound position. The sequence fills in for you.' },
      { title: 'Check the sequence', icon: 'check', body: 'It must have the same number of letters as the peptide file. The page warns you if it does not.' },
      { title: 'Press Score it', icon: 'zap', body: 'You get a ΔG in a few seconds. Nothing is moved. It only measures the pose you gave it.', only: 'demo' },
      { title: 'Press Score it', icon: 'zap', body: 'You get a ΔG in a few seconds. Nothing is moved. It only measures the pose you gave it.',
        tip: 'To try it, score `data/pdbs/1YCR_mdm2.pdb` with `data/pdbs/1YCR_peptide.pdb` and the sequence `ETFSDLWKLLPE`. Expect about −9.3.', only: 'local' },
      { title: 'Press Score it', icon: 'zap', body: 'You get a ΔG in a few seconds. Nothing is moved. It only measures the pose you gave it.', only: 'remote' },
    ],
  },
  {
    id: 'around', title: 'Around the app', blurb: 'History, detail level, night mode.', icon: 'user', kind: 'rows', minutes: 1,
    rows: [
      { icon: 'history', term: 'History', text: 'Every finished run is saved in your browser. Open **History** in the top bar to see the last 30 and reopen any of them.' },
      { icon: 'book', term: 'Guided and Expert', text: '**Guided** keeps things simple. **Expert** adds coordinates, the full ranked table and advanced settings. The results are the same.' },
      { icon: 'moon', term: 'Night mode', text: 'Use the sun or moon button in the top bar to switch between light and dark.' },
      { icon: 'user', term: 'Your avatar', text: 'The round button on the right holds your accent colour and the name that sets your initials.' },
      { icon: 'stop', term: 'Stopping a run', text: 'Press **Stop** on the running screen. The run is thrown away, so you cannot keep a partial result.' },
    ],
  },
  {
    id: 'tips', title: 'Better predictions', blurb: 'Five habits that help.', icon: 'zap', kind: 'rows', minutes: 1,
    rows: [
      { icon: 'badge', term: 'Aim the box', text: 'A box on the real pocket beats a big box on the whole protein. Not sure where it is? Search the whole protein once, then run again with a tight box.' },
      { icon: 'zap', term: 'Match effort to the decision', text: 'Use **Quick** to explore and **Full** for anything you will act on.' },
      { icon: 'columns', term: 'Compare, then trust', text: 'Two proteins side by side cancel out shared error. Repeat the runs that matter.' },
      { icon: 'info', term: 'Keep peptides short', text: 'Under about 12 letters is the most reliable. Past 20 it is slow and harder to predict.' },
      { icon: 'alert', term: 'Watch the charge', text: 'Peptides rich in D, E, K or R need extra care. The result tells you when it matters.' },
    ],
  },
  {
    id: 'troubleshooting', title: 'If something goes wrong', blurb: 'Quick fixes for common messages.', icon: 'alert', kind: 'accordion', minutes: 1,
    rows: [
      { q: 'Setup isn’t finished on this computer', text: 'Something the app needs is missing. The message names it and gives the command that fixes it, usually `./install.sh`.', only: 'local' },
      { q: 'Another run is already going', text: 'The app does one prediction at a time. Wait for it, or open **Run in progress** in the top bar and stop it.' },
      { q: 'No pose fit at this site', text: 'Every pose overlapped the protein, so there is no ΔG. Move the box to a more open spot, make it bigger, or try a shorter peptide.' },
      { q: 'The computer ran out of memory', text: 'Close other programs, choose **Quick**, or use a shorter peptide. Searching the whole protein needs the most.' },
      { q: 'The first run seems stuck', text: 'It is downloading about 2.5 GB of model files. This happens once and can take around 10 minutes.', only: 'local' },
      { q: 'Couldn’t reach the server', text: 'The app has stopped. Start it again with `./launch_web.sh`, reload this page, and press **Check again**. A run already going keeps going.', only: 'local' },
      { q: 'The shared computer is busy', text: 'It does one prediction at a time, for everyone. Wait a few minutes and press the button again.', only: 'remote' },
      { q: 'The shared computer went offline', text: 'This page then shows simulated results marked Demo. Reload in a few minutes, or install the app to run it yourself (see **Get the app**).', only: 'remote' },
      { q: '`command not found: hybridock-pep`', text: 'Start the app with `./launch_web.sh` from the app folder. It uses the right environment for you.', only: 'local' },
      { q: 'The 3D view is empty', text: 'Your browser may have blocked it, or the structure is very large. Try another browser or a smaller file.' },
      { q: 'The numbers aren’t real', text: 'This is the preview website, so every result is simulated and marked Demo. Install the app to get real ones (see **Get the app**).', only: 'demo' },
    ],
  },
  {
    id: 'words', title: 'Words you’ll see', blurb: 'Plain meanings.', icon: 'book', kind: 'rows', minutes: 1,
    rows: [
      { term: 'Protein', text: 'The large molecule your peptide tries to stick to. Its 3D shape comes from a structure file.' },
      { term: 'Peptide', text: 'A short chain of amino acids, typed as letters.' },
      { term: 'PDB', text: 'The free public library of protein structures, and the file type they use.' },
      { term: 'ΔG', text: 'The binding strength in kcal/mol. More negative means tighter.' },
      { term: 'Pose', text: 'One way the peptide could sit on the protein.' },
      { term: 'Binding site', text: 'The place on the protein to look. The box you draw marks it.' },
      { term: 'ΔΔG', text: 'The difference between two ΔG values. Below zero means the first protein is preferred.' },
      { term: 'Ranking score', text: 'A number that only puts the poses of one protein in order. It is not a ΔG, so never compare the two.' },
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
  // the topic that suits this audience best goes first
  const lead = kind === 'demo' ? 'demo' : 'live';
  out.sort((a, b) => (b.first === lead) - (a.first === lead));
  return out;
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
