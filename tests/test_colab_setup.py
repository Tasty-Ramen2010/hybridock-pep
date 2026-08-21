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

    def test_torch_specs_carry_a_local_version(self):
        """Every torch spec must pin "+cuXXX"/"+cpu", not a bare version.

        conda-forge::e3nn drags conda-forge's own pytorch into the rapidock env
        during `micromamba create`. A bare `pip install torch==2.6.0` compares
        equal to it, reports "Requirement already satisfied", and leaves that
        torch in place — built with a different C++ ABI than the PyG wheels,
        which then die on import with `undefined symbol:
        _ZN5torch3jit17parseSchemaOrNameERKSsb` ~10 s into Stage 1. A local
        version can never compare equal, so pip performs the replacement.
        """
        text = COLAB_SH.read_text(encoding="utf-8")
        specs = re.findall(r'"(torch==[^"]+)"', text)
        assert specs, "no torch specs found in colab_setup.sh"
        bare = [s for s in specs if "+" not in s]
        assert not bare, f"torch specs missing a local version: {bare}"

    def test_pyg_extensions_installed_without_deps(self):
        """The compiled extensions must not be allowed to resolve torch
        themselves, and must be force-reinstalled over a mismatched build that
        carries the same version number."""
        text = COLAB_SH.read_text(encoding="utf-8")
        match = re.search(
            r"pip\" install[^\n]*--no-deps[^\n]*--force-reinstall[^\n]*\\\n\s*"
            r"torch-scatter torch-sparse torch-cluster torch-spline-conv",
            text,
        )
        assert match, "PyG extensions are not installed with --no-deps --force-reinstall"

    def test_verification_imports_the_pyg_extensions(self):
        """Verifying with a CUDA matmul alone passed on an env whose
        torch_scatter could not import — torch itself was healthy. The check has
        to import the extensions too."""
        text = COLAB_SH.read_text(encoding="utf-8")
        assert "stack_works()" in text, "verification helper is missing"
        verify = text[text.index("stack_works()"):]
        verify = verify[: verify.index("PY\n")]
        for mod in ("torch_scatter", "torch_sparse", "torch_cluster"):
            assert mod in verify, f"verification never imports {mod}"
        assert "torch.mm" in verify, "verification no longer launches a real CUDA kernel"

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
            # Strip IPython bang-escapes and %magics — legitimate notebook
            # syntax, but not Python, so they would fail compile() on their own.
            source = "\n".join(
                line for line in source.splitlines()
                if not line.lstrip().startswith(("!", "%"))
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


class TestTerminalPath:
    """The notebook is one way in; a shell is the other. Someone who would
    rather type commands should not have to source an env file first or
    discover that `hybridock-pep` is only reachable by absolute path."""

    def test_both_console_scripts_are_shimmed_onto_path(self):
        text = COLAB_SH.read_text(encoding="utf-8")
        assert "/usr/local/bin/$_cmd" in text, "no PATH shim is installed"
        for cmd in ("hybridock-pep", "hybridock-tui"):
            assert cmd in text, f"{cmd} is never shimmed"

    def test_shim_sets_the_sampling_env_vars(self):
        """A bare symlink would resolve the sampling env by luck (via the
        /opt/conda probe). Setting them explicitly means a shell started any
        which way still finds it."""
        text = COLAB_SH.read_text(encoding="utf-8")
        shim = text[text.index("Putting hybridock-pep on PATH"):]
        shim = shim[: shim.index("done")]
        assert "RAPIDOCK_PYTHON" in shim
        assert "RAPIDOCK_DIR" in shim
        assert 'exec "$SCORE_PREFIX/bin/$_cmd" "\\$@"' in shim, "shim drops its arguments"

    def test_bashrc_wiring_is_idempotent(self):
        """Re-running setup is normal (a flaky download, a --force). Appending
        the same source line every time would leave a .bashrc full of them."""
        text = COLAB_SH.read_text(encoding="utf-8")
        assert 'grep -qs "hybridock_env.sh" "$HOME/.bashrc"' in text

    def test_notebook_offers_a_terminal(self):
        source = _notebook_source()
        assert "colab-xterm" in source, "no terminal option for free-tier users"
        assert "hybridock-tui" in source, "the TUI is never mentioned"
