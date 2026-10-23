# HybriDock-Pep guide

HybriDock-Pep helps researchers find new medicines faster. It predicts how tightly a short chain of amino acids (a peptide)
grabs onto a protein that causes a disease, and shows where it sits. The same lessons are built into the page: open **Guide**
in the top bar.

## 1. Try the live demo

1. **Open the website.** Press **Open the live demo** at the top of the GitHub page.
2. **Swipe through the diseases.** Each card names the protein behind the illness and lists its best peptide matches.
3. **Open a result.** Tap a match to see how the peptide fits the protein in 3D. **More negative means a tighter grip.**
4. **Compare two matches.** Press **Compare** on any matches you like. They line up side by side.
5. **Run it live.** Pick **5, 10 or 25** poses and press **Run it live**. The time beside the button is how long it takes.
6. **Watch it finish.** You can leave the page. **Run in progress** in the top bar brings you back.

## 2. Test your own peptide

1. Press **Test a peptide** on Home and **choose a protein**. The place where a peptide attaches is already set for you.
2. **Type a peptide**, 3 to 25 letters, or tap a named example.
3. **Choose 5, 10 or 25 poses.** The time under each choice is how long it takes. Press **Run prediction**.
4. **Read the result.** Click any row in the list to see that pose in 3D.

## 3. Read a result

1. **The big number** is the binding strength in kcal/mol. More negative means the peptide holds on harder.
2. **The coloured label** puts the number in words.
3. **Poses.** A pose is one way the peptide can sit on the protein. Click a row to see it in 3D.
4. **Files.** *Best pose* opens in PyMOL or ChimeraX. *Ranked list* is a spreadsheet of every pose.

## 4. Compare two proteins

1. Choose a peptide and two proteins: the one you want it to grab (the target) and a look-alike you want it to leave alone.
2. Press **Compare**. It runs both, so it takes about twice as long as one.
3. **Below zero** means the peptide prefers your target.

## 5. Score a structure

1. Add the protein as a `.pdb`, then the peptide in its bound position.
2. Check that the sequence has the same number of letters as the peptide file.
3. Press **Score it**. You get the binding strength in a few seconds.

## 6. Run it on your own computer

```bash
git clone --recurse-submodules https://github.com/Tasty-Ramen2010/hybridock-pep.git
cd hybridock-pep
./install.sh          # sets everything up and opens the app
./launch_web.sh       # start it again any time
```

On Windows, double-click `install.bat`, then `launch_web.bat`. Nothing to install? Use the
[Colab notebook](https://colab.research.google.com/github/Tasty-Ramen2010/hybridock-pep/blob/master/notebooks/HybriDock_Pep_Colab.ipynb).

## If something goes wrong

* **Another run is already going.** The app does one prediction at a time. Wait for it, or open **Run in progress**.
* **The lab computer is busy.** It does one prediction at a time, for everyone. Wait a few minutes and try again.
* **The lab computer is offline.** The results in the carousel are still there to open. New runs are simulated and marked *Demo* until it is back.
* **No pose fit at this site.** Try a shorter peptide.

## Words you'll see

* **Protein**: a large molecule in the body. Many diseases start with one that misbehaves.
* **Peptide**: a short chain of amino acids, typed as letters. Think of it as a key for the protein's lock.
* **Binding strength (ΔG)**: how tightly the peptide sticks, in kcal/mol. More negative means tighter.
* **Pose**: one way the peptide could sit on the protein.
* **ΔΔG**: the difference between two binding strengths. Below zero means the first protein is preferred.
* **PDB**: the free public library of protein structures, and the file type they use.
