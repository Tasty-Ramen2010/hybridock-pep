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


class TestQuotedNumbersAreTheRealOnes:
    """Guard the figures a user would act on."""

    def test_crystal_score_expected_value(self):
        _, out = _render("crystal-score")
        assert "-10.07" in out, "install-validation target value missing"
        assert "-8.5" in out, "experimental reference missing"

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
