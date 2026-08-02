"""Guards on what the README promises an outside reader can reproduce.

The README's "Datasets — download and test for yourself" section says
"Everything above is reproducible from data shipped in this repo" and links each
file. Nothing checked that, and one of the links —
`data/e180_protdcal3d.jsonl`, the feature file behind the flagship
ours-vs-PPI-clone head-to-head — pointed at a path that is in .gitignore and
has never been committed. A reader following the README got a 404 on the one
number the project leads with.

These tests make that class of failure loud at test time instead of at
reader time.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"


def _tracked_paths() -> set[str] | None:
    """Paths git actually tracks, or None outside a checkout.

    Existence on disk is not the question — a file can exist locally and still
    be absent from every clone. Only tracked files ship.
    """
    git = shutil.which("git")
    if git is None or not (REPO / ".git").exists():
        return None
    out = subprocess.run(
        [git, "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    return set(out.splitlines())


def _dataset_table_links() -> list[str]:
    """Repo-relative paths linked from the datasets section of the README."""
    text = README.read_text()
    start = text.find("## Datasets")
    assert start != -1, "README lost its Datasets section"
    end = text.find("\n## ", start + 1)
    section = text[start:end if end != -1 else len(text)]
    return [
        m.group(1)
        for m in re.finditer(r"\]\((data/[^)\s#]+)\)", section)
    ]


class TestReadmeDatasetsShip:
    def test_the_section_still_links_datasets(self):
        links = _dataset_table_links()
        assert len(links) >= 5, f"datasets table looks truncated: {links}"

    @pytest.mark.parametrize("rel", _dataset_table_links())
    def test_linked_dataset_is_committed(self, rel):
        tracked = _tracked_paths()
        if tracked is None:
            pytest.skip("not a git checkout")
        assert rel in tracked, (
            f"README links {rel} as shipped data, but git does not track it — "
            "a reader who clones this repo gets a broken link and cannot "
            "reproduce whatever it backs. Either commit the file or stop "
            "presenting it as shipped."
        )


class TestReproScriptsImportsResolve:
    """A script on the documented reproduce path must be runnable from a clone.

    `experiments/e180_protdcal_925.py` -> `e179_protdcal_3d` ->
    `e158_overfit_failure_analysis`, and that last module was never committed,
    so the chain dies with ModuleNotFoundError on any fresh clone.
    """

    # The scripts the README tells the reader to run.
    REPRO_SCRIPTS = [
        "experiments/e330_ours_pdbbind.py",
        "experiments/e331_ours_vs_ppiclone_clustered.py",
        "experiments/e366_identity_threshold_trend.py",
        "experiments/e367_gap_penalized_trend.py",
    ]

    @pytest.mark.parametrize("rel", REPRO_SCRIPTS)
    def test_script_is_committed(self, rel):
        tracked = _tracked_paths()
        if tracked is None:
            pytest.skip("not a git checkout")
        assert rel in tracked, f"{rel} is referenced but not committed"

    @pytest.mark.parametrize("rel", REPRO_SCRIPTS)
    def test_local_module_imports_resolve(self, rel):
        """Every `import eNNN_...` in the script resolves to a committed file.

        These scripts put experiments/ and scripts/ on sys.path and import each
        other by bare module name, so a missing sibling is invisible until the
        script is run.
        """
        tracked = _tracked_paths()
        if tracked is None:
            pytest.skip("not a git checkout")
        text = (REPO / rel).read_text()
        siblings = re.findall(r"^\s*import\s+(e\d+_\w+)", text, re.MULTILINE)
        siblings += re.findall(r"^\s*from\s+(e\d+_\w+)\s+import", text, re.MULTILINE)
        missing = [
            mod for mod in set(siblings)
            if f"experiments/{mod}.py" not in tracked and f"scripts/{mod}.py" not in tracked
        ]
        assert not missing, (
            f"{rel} imports {missing}, which are not committed anywhere in the "
            "repo — the script cannot run from a fresh clone"
        )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN GAP, tracked deliberately. e158_overfit_failure_analysis.py was "
        "never committed, and 48 experiment scripts import it for "
        "greedy_cluster() and pocket_seq(). That breaks the regeneration chain "
        "for data/e180_protdcal3d.jsonl (e180 -> e179 -> e158), the ProtDCal-3D "
        "feature file behind the ours-vs-PPI-clone head-to-head. Recover the "
        "original from whichever machine ran E158 — do NOT reimplement it, "
        "since pocket_seq's cutoff choices define what the published numbers "
        "mean. This flips to a pass the moment it is committed."
    ),
)
def test_protdcal_feature_pipeline_is_reconstructable():
    tracked = _tracked_paths()
    if tracked is None:
        pytest.skip("not a git checkout")
    assert "scripts/e158_overfit_failure_analysis.py" in tracked or \
           "experiments/e158_overfit_failure_analysis.py" in tracked
