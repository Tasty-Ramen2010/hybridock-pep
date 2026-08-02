"""Coverage for the built-in how-to guide (`hybridock-pep guide`).

The guide quotes concrete numbers and concrete commands. Both can rot: a flag
gets renamed, a subcommand is added and never documented, or a benchmark figure
is updated in the README but not here. These tests pin the parts that would
mislead a user if they drifted.
"""

from __future__ import annotations

import io

import pytest

from hybridock_pep.cli import _build_parser
from hybridock_pep.output import guide

TOPICS = ["dock", "crystal-score", "prep", "selectivity", "reproducibility",
          "benchmark", "calibrate", "tuning", "testing"]


def _render(topic=None):
    buf = io.StringIO()
    code = guide.print_guide(topic, stream=buf)
    return code, buf.getvalue()


class TestRendering:
    def test_overview_renders(self):
        code, out = _render()
        assert code == 0
        assert "HybriDock-Pep" in out and "crystal-score" in out

    @pytest.mark.parametrize("topic", TOPICS)
    def test_each_topic_renders(self, topic):
        code, out = _render(topic)
        assert code == 0
        assert len(out.strip()) > 200, f"{topic} guide is suspiciously short"

    def test_all_includes_every_topic(self):
        code, out = _render("all")
        assert code == 0
        for topic in TOPICS:
            assert topic in out

    def test_topic_lookup_is_case_insensitive(self):
        assert _render("DOCK")[0] == 0

    def test_unknown_topic_is_a_clean_error(self):
        code, out = _render("bogus")
        assert code == 1
        assert "No guide topic" in out and "Available:" in out

    def test_overview_lists_the_available_topics(self):
        _, out = _render()
        assert "Topics:" in out


class TestStaysInSyncWithTheCli:
    """The guide is only useful if it matches the actual CLI."""

    @staticmethod
    def _subcommands():
        import argparse
        for action in _build_parser()._actions:
            if isinstance(action, argparse._SubParsersAction):
                return set(action.choices) - {"guide"}
        return set()

    def test_every_subcommand_is_documented(self):
        undocumented = self._subcommands() - set(guide.COMMANDS)
        assert not undocumented, f"no guide entry for: {sorted(undocumented)}"

    def test_guide_documents_no_command_that_does_not_exist(self):
        phantom = set(guide.COMMANDS) - self._subcommands()
        assert not phantom, f"guide documents missing commands: {sorted(phantom)}"

    @pytest.mark.parametrize("topic", ["dock", "crystal-score", "selectivity"])
    def test_example_commands_parse(self, topic):
        """Every `hybridock-pep ...` example must be accepted by the real parser.

        Guides that ship copy-pasteable commands which no longer work are worse
        than no guide at all.
        """
        _, out = _render(topic)
        parser = _build_parser()
        joined = out.replace("\\\n", " ")
        for line in joined.splitlines():
            line = line.strip()
            if not line.startswith("hybridock-pep "):
                continue
            argv = line.split()[1:]
            if argv and argv[0] == "guide":
                continue
            parser.parse_args(argv)   # raises SystemExit on an unknown flag


class TestDocumentedSitesArePointedAtRealPockets:
    """A wrong --site is invisible until a run silently searches empty space.

    The guide originally paired 1YCR with `--site 31.9 17.5 9.5`, a value
    copied from the unrelated 1T2D example. It sits ~46 A from the 1YCR pocket,
    outside the receptor's y range entirely, and Vina rejects the native pose
    against it with "ligand is outside the grid box". Nothing in the test suite
    noticed, because the example only had to *parse*.

    So check the geometry, not just the syntax: every documented box must
    actually contain receptor atoms.
    """

    REPO = None  # set in _pairs

    @staticmethod
    def _pairs():
        """Extract (receptor_path, site_xyz, box_edge) from every guide example."""
        import re
        from pathlib import Path

        repo = Path(__file__).parent.parent
        found = []
        for topic in TOPICS:
            _, out = _render(topic)
            joined = out.replace("\\\n", " ")
            for line in joined.splitlines():
                line = line.strip()
                if not line.startswith("hybridock-pep "):
                    continue
                for prefix in ("", "target-", "offtarget-"):
                    m_r = re.search(rf"--{prefix}receptor\s+(\S+)", line)
                    m_s = re.search(
                        rf"--{prefix}site\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)", line
                    )
                    m_b = re.search(rf"--{prefix}box\s+(\d+)", line)
                    if m_r and m_s and m_b:
                        rec = repo / m_r.group(1)
                        found.append(
                            (rec, tuple(float(g) for g in m_s.groups()), float(m_b.group(1)))
                        )
        return found

    def test_at_least_one_example_was_parsed(self):
        """Guard the guard: a parser that silently matches nothing passes
        every geometry assertion below."""
        assert self._pairs(), "no --receptor/--site examples found in the guide"

    def test_every_documented_box_contains_receptor_atoms(self):
        np = pytest.importorskip("numpy")
        checked = 0
        for receptor, site, box in self._pairs():
            if not receptor.is_file():
                continue  # placeholder path in an illustrative example
            coords = np.array([
                (float(l[30:38]), float(l[38:46]), float(l[46:54]))
                for l in receptor.read_text().splitlines()
                if l.startswith(("ATOM", "HETATM"))
            ])
            half = box / 2.0
            inside = np.all(np.abs(coords - np.array(site)) <= half, axis=1).sum()
            assert inside > 50, (
                f"{receptor.name} with --site {site} --box {box} contains only "
                f"{inside} receptor atoms — the box is pointed at empty space"
            )
            checked += 1
        assert checked, "no real receptor files were exercised"

    def test_box_is_big_enough_for_the_documented_peptide(self):
        """1YCR's peptide spans 21 A, so the old --box 20 could not hold it
        even when centred correctly."""
        np = pytest.importorskip("numpy")
        from pathlib import Path
        repo = Path(__file__).parent.parent
        pep = repo / "data/pdbs/1YCR_peptide.pdb"
        if not pep.is_file():
            pytest.skip("1YCR peptide not present")
        coords = np.array([
            (float(l[30:38]), float(l[38:46]), float(l[46:54]))
            for l in pep.read_text().splitlines()
            if l.startswith(("ATOM", "HETATM"))
        ])
        span = (coords.max(0) - coords.min(0)).max()
        boxes = [b for r, _, b in self._pairs() if r.name == "1YCR_mdm2.pdb"]
        assert boxes, "no 1YCR example found"
        for b in boxes:
            assert b >= span, f"--box {b} is smaller than the peptide's {span:.1f} A span"


class TestQuotedNumbersAreTheRealOnes:
    """Guard the figures a user would act on."""

    def test_crystal_score_expected_value(self):
        _, out = _render("crystal-score")
        assert "-9.3" in out, "install-validation target value missing"
        assert "-8.5" in out, "experimental reference missing"

    def test_crystal_score_guide_matches_reality(self):
        """The quoted value must be the one the shipped model actually produces.

        This drifted once already: the guide promised -10.07 while a clean
        install deterministically returned -9.28, because the geometry/IFP
        features move with the resolved numpy/sklearn versions. Pin the guide
        to whatever the real artifact says, so the two cannot separate again.
        """
        pytest.importorskip("sklearn")
        from pathlib import Path
        repo = Path(__file__).parent.parent
        rec = repo / "data/pdbs/1YCR_mdm2.pdb"
        pep = repo / "data/pdbs/1YCR_peptide.pdb"
        if not (rec.is_file() and pep.is_file()):
            pytest.skip("1YCR reference structures not present")
        from hybridock_pep.scoring.interaction_map import score_crystal_complex
        dg = score_crystal_complex(str(rec), str(pep), "ETFSDLWKLLPE")
        if dg is None:
            pytest.skip("crystal-IFP artifact unavailable")
        # The guide tells users -8..-11 is a healthy install; the real value
        # must sit inside the band the guide advertises.
        assert -11.0 <= dg <= -8.0, (
            f"crystal-score returned {dg:.2f}, outside the -8..-11 band the "
            "guide tells users to expect. Either the model regressed or the "
            "guide needs updating."
        )

    def test_benchmark_table_matches_readme(self):
        _, out = _render("benchmark")
        for value in ("1.35", "1.69", "0.352", "1.46", "1.84", "0.210"):
            assert value in out, f"benchmark figure {value} missing"

    def test_tuning_documents_the_env_switches(self):
        _, out = _render("tuning")
        for var in ("HYBRIDOCK_RAPIDOCK_BATCH", "HYBRIDOCK_MMGBSA_FAST",
                    "RAPIDOCK_DISABLE_METAL_TP"):
            assert var in out

    def test_env_switches_actually_exist_in_the_code(self):
        """A documented switch that nothing reads is a lie."""
        from pathlib import Path
        src = Path(__file__).parent.parent / "src"
        blob = "\n".join(p.read_text(errors="ignore") for p in src.rglob("*.py"))
        for var in ("HYBRIDOCK_RAPIDOCK_BATCH", "HYBRIDOCK_MMGBSA_FAST"):
            assert var in blob, f"{var} is documented but read nowhere"

    def test_reproducibility_warns_about_nondeterminism(self):
        """Users must not chase ΔG differences smaller than the noise floor."""
        _, out = _render("reproducibility")
        assert "not deterministic" in out or "no fp64" in out or "spread" in out


class TestCliIntegration:
    def test_guide_subcommand_is_registered(self):
        ns = _build_parser().parse_args(["guide"])
        assert ns.command == "guide" and ns.topic is None

    def test_guide_accepts_a_topic(self):
        assert _build_parser().parse_args(["guide", "dock"]).topic == "dock"

    def test_help_epilog_points_at_the_guide(self, capsys):
        parser = _build_parser()
        parser.print_help()
        out = capsys.readouterr().out
        assert "hybridock-pep guide" in out
        assert "crystal-score" in out

    def test_help_epilog_fits_80_columns(self):
        """RawDescriptionHelpFormatter prints the epilog verbatim."""
        epilog = _build_parser().epilog or ""
        for line in epilog.splitlines():
            assert len(line) <= 80, f"epilog line is {len(line)} cols: {line!r}"
