"""Guards for the Colab path — scripts/colab_setup.sh + the notebook.

Neither artifact can be exercised in CI: colab_setup.sh only runs to completion
on a Linux runtime with an NVIDIA GPU and spends 20 minutes building conda
environments, and the notebook is executed by Colab, not pytest. What CI *can*
do is catch the failure modes that would waste a user's whole session before
they ever reach a dock:

* a shell syntax error, or a `--help` that no longer prints the flag list;
* the notebook drifting out of sync with the repo — a bundled receptor that was
  renamed, or an env prefix that no longer matches what the script creates;
* the checkpoint checksums diverging from install.sh's, which would mean one
  installer verifies bytes the other rejects.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COLAB_SH = REPO_ROOT / "scripts" / "colab_setup.sh"
INSTALL_SH = REPO_ROOT / "install.sh"
NOTEBOOK = REPO_ROOT / "notebooks" / "HybriDock_Pep_Colab.ipynb"


def _notebook_source() -> str:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell["source"]) for cell in nb["cells"])


class TestColabSetupScript:
    def test_is_syntactically_valid(self):
        result = subprocess.run(
            ["bash", "-n", str(COLAB_SH)], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr

    def test_help_lists_every_flag(self):
        """--help slices a fixed line range out of the header comment. Insert a
        line above it and the help silently truncates mid-sentence, which is
        exactly the kind of drift nobody notices."""
        result = subprocess.run(
            ["bash", str(COLAB_SH), "--help"], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        for flag in ("--cache-dir", "--backend", "--skip-rapidock", "--force", "--help"):
            assert flag in result.stdout, f"{flag} missing from --help"
        assert "set -euo" not in result.stdout, "help range ran past the header comment"

    def test_unknown_flag_is_rejected(self):
        result = subprocess.run(
            ["bash", str(COLAB_SH), "--nope"], capture_output=True, text=True
        )
        assert result.returncode == 2
        assert "unknown flag" in result.stderr

    def test_checkpoint_checksums_match_install_sh(self):
        """Both installers fetch the same two files from the same Zenodo record.
        If the checksums drift apart, one of them rejects a file the other just
        verified."""
        pattern = re.compile(r"fetch_ckpt (rapidock_\w+\.pt) \\\s*\n\s*([0-9a-f]{64})")
        colab = dict(pattern.findall(COLAB_SH.read_text(encoding="utf-8")))
        install = dict(pattern.findall(INSTALL_SH.read_text(encoding="utf-8")))
        assert colab, "no fetch_ckpt calls found in colab_setup.sh"
        assert colab == install, f"checksum drift: colab={colab} install={install}"

    def test_both_checkpoints_are_fetched(self):
        text = COLAB_SH.read_text(encoding="utf-8")
        # rapidock_global.pt is what --blind needs; omitting it leaves blind
        # docking dead on arrival with an opaque torch.load failure.
        assert "rapidock_local.pt" in text
        assert "rapidock_global.pt" in text


class TestColabNotebook:
    def test_is_valid_notebook_json(self):
        nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        assert nb["nbformat"] == 4
        assert nb["cells"], "notebook has no cells"
        assert nb["metadata"]["accelerator"] == "GPU", "Colab must open this with a GPU"

    def test_code_cells_are_valid_python(self):
        nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            # Strip IPython bang-escapes; they are not Python syntax.
            source = "\n".join(
                line for line in source.splitlines() if not line.lstrip().startswith("!")
            )
            compile(source, f"cell {i}", "exec")

    def test_env_prefixes_match_the_setup_script(self):
        """The notebook hard-codes /opt/conda/envs/... paths. That is only
        correct while colab_setup.sh keeps its micromamba root prefix there —
        and that prefix is also what rapidock_runner.py probes."""
        assert 'MAMBA_ROOT_PREFIX:-/opt/conda' in COLAB_SH.read_text(encoding="utf-8")
        source = _notebook_source()
        assert "/opt/conda/envs/score-env" in source
        assert "/opt/conda/envs/rapidock" in source

    def test_referenced_bundled_pdbs_exist(self):
        """The receptor dropdown and the crystal-score check name files in
        data/pdbs/. A rename there turns the first cell a user runs into a
        FileNotFoundError."""
        referenced = set(re.findall(r"data/pdbs/[\w.]+\.pdb", _notebook_source()))
        assert referenced, "notebook no longer references any bundled PDB"
        missing = [p for p in sorted(referenced) if not (REPO_ROOT / p).exists()]
        assert not missing, f"notebook references missing PDBs: {missing}"

    def test_invokes_the_colab_setup_script(self):
        assert "scripts/colab_setup.sh" in _notebook_source()
