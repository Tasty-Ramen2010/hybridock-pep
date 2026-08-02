"""The first-run walkthrough and the always-available help browser.

The UI was nearly deleted because a first-time, non-computational user opened
it and found a form of unexplained fields with no visible way to learn what
they meant. These tests pin the two things that fixed that:

  * a guided tour that appears by itself on first launch, and
  * a help browser covering the science (selectivity, crystal scoring, AI vs
    physics scoring, calibration), not just which key does what.

Content is asserted by meaning, not by exact wording, so the text can be
rewritten freely — but a topic cannot silently disappear, and the help cannot
quietly drift back into being keyboard-shortcuts-only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hybridock_pep.ui import help_content


class TestTopicSet:
    def test_every_topic_has_a_unique_hotkey(self):
        keys = [k for k, _, _ in help_content.TOPICS]
        assert len(keys) == len(set(keys)), f"duplicate hotkeys: {keys}"

    def test_hotkeys_are_single_characters(self):
        """The help view binds these directly as key presses."""
        for key, _title, _body in help_content.TOPICS:
            assert len(key) == 1, f"{key!r} cannot be a single key press"

    @pytest.mark.parametrize("subject", [
        "selectivity", "crystal", "calibrat", "reproducib", "troubleshoot",
    ])
    def test_the_topics_the_user_asked_for_exist(self, subject):
        """These were named explicitly as things a general user needs."""
        titles = " ".join(t for _, t, _ in help_content.TOPICS).lower()
        bodies = " ".join(b for _, _, b in help_content.TOPICS).lower()
        assert subject in titles or subject in bodies

    def test_ai_versus_physics_scoring_is_explained(self):
        """A user must be able to tell which of the several numbers is 'the'
        answer — quoting the Vina score as a binding energy is a real and
        easy mistake."""
        bodies = " ".join(b for _, _, b in help_content.TOPICS).lower()
        assert "vina" in bodies
        assert "not the binding energy" in bodies or "do not quote" in bodies

    def test_no_topic_is_a_stub(self):
        for key, title, body in help_content.TOPICS:
            assert len(body.strip()) > 300, f"topic {key} ({title}) is too thin to help anyone"

    def test_bodies_fit_a_standard_terminal(self):
        """The help pane is not horizontally scrollable; long lines wrap and
        wreck the layout."""
        for key, title, body in help_content.TOPICS:
            for line in body.splitlines():
                assert len(line) <= 76, f"topic {key} ({title}) has a {len(line)}-col line: {line!r}"


class TestRendering:
    def test_render_includes_the_body_and_the_menu(self):
        out = help_content.render("1")
        assert "GETTING STARTED" in out
        for _key, title in help_content.topic_titles():
            assert title in out, f"{title} missing from the topic menu"

    def test_render_marks_the_current_topic(self):
        assert ">5" in help_content.render("5").replace("> 5", ">5")

    def test_unknown_topic_falls_back_rather_than_crashing(self):
        """A stray key press must never blank the help pane."""
        out = help_content.render("zzz")
        assert len(out.strip()) > 300

    @pytest.mark.parametrize("key", [k for k, _, _ in help_content.TOPICS])
    def test_every_topic_renders(self, key):
        assert len(help_content.render(key).strip()) > 300

    def test_render_tells_the_user_how_to_leave(self):
        out = help_content.render("1")
        assert "Esc" in out

    def test_welcome_covers_the_five_steps_and_the_safe_start(self):
        w = help_content.WELCOME
        assert "Demo" in w, "the tour must point at the risk-free starting point"
        for field in ("PEPTIDE", "RECEPTOR", "SITE", "BOX"):
            assert field in w, f"walkthrough never mentions {field}"

    def test_welcome_explains_that_progress_bars_mean_it_is_working(self):
        """The specific confusion being fixed: 'is it frozen?'"""
        assert "%" in help_content.WELCOME
        assert "working" in help_content.WELCOME.lower()

    def test_welcome_fits_a_standard_terminal(self):
        for line in (help_content.LOGO + help_content.WELCOME).splitlines():
            assert len(line) <= 76, f"{len(line)}-col line: {line!r}"


class TestFirstRunMarker:
    def test_first_run_is_true_when_marker_absent(self, tmp_path):
        from hybridock_pep.ui import tui
        assert tui.is_first_run(tmp_path / "never_written") is True

    def test_first_run_is_false_after_marking(self, tmp_path):
        from hybridock_pep.ui import tui
        marker = tmp_path / "nested" / "ui_welcomed"
        tui.mark_welcomed(marker)
        assert marker.is_file()
        assert tui.is_first_run(marker) is False

    def test_marking_survives_an_unwritable_home(self, tmp_path):
        """A read-only or missing home directory must not stop the UI opening."""
        from hybridock_pep.ui import tui
        blocked = tmp_path / "file_not_dir"
        blocked.write_text("x")
        tui.mark_welcomed(blocked / "sub" / "marker")  # parent is a file -> OSError

    def test_marker_lives_outside_the_repo(self):
        """The tour is a property of the person, not of a checkout — a fresh
        clone must not re-trigger it."""
        from hybridock_pep.ui import tui
        repo = Path(__file__).parent.parent.resolve()
        assert repo not in tui.WELCOME_MARKER.resolve().parents
