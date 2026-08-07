"""Coverage for the terminal UI (hybridock_pep.ui.tui).

The TUI is 733 lines and had no tests at all. Everything here is the pure,
headless surface — validators, path cleaning, command building, the progress
model and the bar renderer — so it runs in CI with no terminal, no
prompt_toolkit event loop and no GPU.

The full-screen application itself still needs a human at a terminal; what these
tests pin down is that the command it builds and the input it accepts are right,
which is where a wrong answer would silently reach the user's shell.
"""

from pathlib import Path

import pytest

tui = pytest.importorskip("hybridock_pep.ui.tui", reason="prompt_toolkit not installed")


# --------------------------------------------------------------------------- #
# validators
# --------------------------------------------------------------------------- #

class TestPeptideValidator:
    @pytest.mark.parametrize("seq", ["LISDAELEAIFEADC", "ACD", "A" * 30])
    def test_accepts_valid(self, seq):
        assert tui._valid_peptide(seq) is None

    def test_accepts_lowercase_and_whitespace(self):
        """Users paste sequences from papers; case and stray spaces are normal."""
        assert tui._valid_peptide("  lisdaeleaifeadc  ") is None

    def test_rejects_empty(self):
        assert "required" in tui._valid_peptide("")

    def test_rejects_non_amino_acids(self):
        msg = tui._valid_peptide("ACDXZ")
        assert msg and "X" in msg and "Z" in msg

    @pytest.mark.parametrize("seq", ["AC", "A" * 31])
    def test_rejects_out_of_range_length(self, seq):
        assert "3–30" in tui._valid_peptide(seq)


class TestSiteValidator:
    @pytest.mark.parametrize("v", ["31.9 17.5 9.5", "0 0 0", "-1.5  2   3.25"])
    def test_accepts_three_numbers(self, v):
        assert tui._valid_site(v) is None

    @pytest.mark.parametrize("v", ["1 2", "1 2 3 4", ""])
    def test_rejects_wrong_count(self, v):
        assert "three numbers" in tui._valid_site(v)

    def test_rejects_non_numeric(self):
        assert "numbers" in tui._valid_site("x y z")


class TestPosInt:
    def test_in_range(self):
        assert tui._posint(10, 60)("20") is None

    @pytest.mark.parametrize("v", ["9", "61"])
    def test_out_of_range(self, v):
        assert "out of range" in tui._posint(10, 60)(v)

    @pytest.mark.parametrize("v", ["abc", "", "1.5"])
    def test_non_integer(self, v):
        assert "integer" in tui._posint(10, 60)(v)


class TestModeValidator:
    """Scoring mode is ai/crystal — NOT the vina/ad4 physics backends, which
    run automatically and are never a user-facing field (see build_dock_command)."""

    @pytest.mark.parametrize("v", ["ai", "crystal", " AI ", "Crystal"])
    def test_accepts(self, v):
        assert tui._valid_mode(v) is None

    @pytest.mark.parametrize("v", ["", "vina", "vina,ad4", "physics"])
    def test_rejects(self, v):
        assert tui._valid_mode(v)


class TestReceptorAndDirValidators:
    def test_receptor_accepts_existing_file(self, tmp_path):
        p = tmp_path / "r.pdb"
        p.write_text("ATOM\n")
        assert tui._valid_receptor(str(p)) is None

    def test_receptor_rejects_missing(self, tmp_path):
        assert tui._valid_receptor(str(tmp_path / "nope.pdb")) == "file not found"

    def test_receptor_rejects_empty(self):
        assert "required" in tui._valid_receptor("  ")

    def test_receptor_rejects_directory(self, tmp_path):
        """A folder is not a receptor; is_file() must be what's checked."""
        assert tui._valid_receptor(str(tmp_path)) == "file not found"

    def test_dir_requires_value(self):
        assert tui._valid_dir("") and tui._valid_dir("runs/x") is None

    def test_optional_wrapper_allows_blank(self):
        f = tui._optional(tui._valid_site)
        assert f("") is None and f("   ") is None
        assert f("1 2") is not None


# --------------------------------------------------------------------------- #
# dropped-path cleaning — macOS Terminal / iTerm drag-and-drop
# --------------------------------------------------------------------------- #

class TestCleanDroppedPath:
    @pytest.mark.parametrize("raw,expected", [
        ("'/Users/x/a.pdb'", "/Users/x/a.pdb"),
        ('"/Users/x/a.pdb"', "/Users/x/a.pdb"),
        ("file:///Users/x/a.pdb", "/Users/x/a.pdb"),
        # macOS Terminal escapes spaces and parens when you drop a file
        ("/Users/x/my\\ file.pdb", "/Users/x/my file.pdb"),
        ("/Users/x/a\\(1\\).pdb", "/Users/x/a(1).pdb"),
        ("  /Users/x/a.pdb  ", "/Users/x/a.pdb"),
    ])
    def test_cleans(self, raw, expected):
        assert tui.clean_dropped_path(raw) == expected

    def test_quote_and_scheme_together(self):
        """iTerm can produce both a file:// scheme and quotes."""
        assert tui.clean_dropped_path("'file:///Users/x/a.pdb'") == "/Users/x/a.pdb"

    def test_plain_path_untouched(self):
        assert tui.clean_dropped_path("/Users/x/a.pdb") == "/Users/x/a.pdb"


# --------------------------------------------------------------------------- #
# command construction — what actually reaches the user's shell
# --------------------------------------------------------------------------- #

def _values(**over):
    v = {
        "peptide": "lisdaeleaifeadc",
        "receptor": "data/pdbs/1T2D_receptor.pdb",
        "site": "31.9 17.5 9.5",
        "box": "20",
        "n_samples": "100",
        "mode": "ai",
        "peptide_pdb": "",
        "refine_topk": "0",
        "output_dir": "runs/tui_run",
        "offtarget_receptor": "data/pdbs/3LNJ_mdm2.pdb",
        "offtarget_site": "1 2 3",
        "offtarget_box": "20",
    }
    v.update(over)
    return v


class TestBuildDockCommand:
    def test_shape_and_uppercasing(self):
        cmd = tui.build_dock_command(_values())
        assert cmd[:2] == ["hybridock-pep", "dock"]
        assert "LISDAELEAIFEADC" in cmd, "peptide must be upper-cased for the CLI"
        assert cmd[cmd.index("--site") + 1: cmd.index("--site") + 4] == ["31.9", "17.5", "9.5"]

    def test_refine_topk_omitted_when_zero(self):
        assert "--refine-topk" not in tui.build_dock_command(_values(refine_topk="0"))

    def test_refine_topk_included_when_set(self):
        cmd = tui.build_dock_command(_values(refine_topk="10"))
        assert cmd[cmd.index("--refine-topk") + 1] == "10"

    def test_refine_topk_blank_is_treated_as_off(self):
        """The field is optional in the form; blank must not crash."""
        assert "--refine-topk" not in tui.build_dock_command(_values(refine_topk=""))

    def test_refine_topk_whitespace_is_treated_as_off(self):
        """A user who types a space then deletes it leaves ' ' behind."""
        assert "--refine-topk" not in tui.build_dock_command(_values(refine_topk="   "))

    def test_home_relative_receptor_is_expanded(self):
        cmd = tui.build_dock_command(_values(receptor="~/r.pdb"))
        assert "~" not in cmd[cmd.index("--receptor") + 1]

    def test_every_argument_is_a_string(self):
        """subprocess/shell join breaks on non-str entries."""
        assert all(isinstance(a, str) for a in tui.build_dock_command(_values()))


class TestBuildSelectivityCommand:
    def test_shape(self):
        cmd = tui.build_selectivity_command(_values())
        assert cmd[:2] == ["hybridock-pep", "selectivity"]
        for flag in ("--target-receptor", "--offtarget-receptor",
                     "--target-site", "--offtarget-site"):
            assert flag in cmd

    def test_target_and_offtarget_sites_are_distinct(self):
        cmd = tui.build_selectivity_command(_values())
        i, j = cmd.index("--target-site"), cmd.index("--offtarget-site")
        assert cmd[i + 1:i + 4] == ["31.9", "17.5", "9.5"]
        assert cmd[j + 1:j + 4] == ["1", "2", "3"]

    def test_every_argument_is_a_string(self):
        assert all(isinstance(a, str) for a in tui.build_selectivity_command(_values()))


class TestBuildCrystalCommand:
    """crystal mode: score an EXISTING bound pose, no sampling."""

    def test_shape(self):
        cmd = tui.build_crystal_command(_values(peptide_pdb="data/pdbs/1YCR_peptide.pdb"))
        assert cmd[:2] == ["hybridock-pep", "crystal-score"]
        assert "--peptide-pdb" in cmd
        assert "--site" not in cmd and "--n-samples" not in cmd

    def test_every_argument_is_a_string(self):
        cmd = tui.build_crystal_command(_values(peptide_pdb="data/pdbs/1YCR_peptide.pdb"))
        assert all(isinstance(a, str) for a in cmd)

    def test_home_relative_peptide_pdb_is_expanded(self):
        cmd = tui.build_crystal_command(_values(peptide_pdb="~/pose.pdb"))
        assert "~" not in cmd[cmd.index("--peptide-pdb") + 1]


class TestResolveExe:
    """Real bug, from a screenshot: right after a fresh install, every run
    failed with "'hybridock-pep' not on PATH" even though the install had
    succeeded. launch_ui.sh finds the correct score-env *Python interpreter*
    by full path (searching common conda locations) and execs it directly —
    but that alone never puts that environment's bin/ on PATH for the
    process it starts, and every real run shells out to the bare name
    "hybridock-pep". _resolve_exe() is the fallback: sys.executable is
    always correct (it's this very interpreter), so its sibling in the same
    bin/ directory must be the real executable, regardless of PATH."""

    def test_finds_it_on_path_first(self, monkeypatch):
        monkeypatch.setattr(tui.shutil, "which", lambda name: f"/usr/bin/{name}")
        assert tui._resolve_exe("hybridock-pep") == "/usr/bin/hybridock-pep"

    def test_falls_back_to_sys_executables_sibling_when_not_on_path(self, monkeypatch, tmp_path):
        monkeypatch.setattr(tui.shutil, "which", lambda name: None)
        fake_python = tmp_path / "python"
        fake_python.write_text("#!/bin/sh\n")
        fake_exe = tmp_path / "hybridock-pep"
        fake_exe.write_text("#!/bin/sh\n")
        fake_exe.chmod(0o755)
        monkeypatch.setattr(tui.sys, "executable", str(fake_python))
        assert tui._resolve_exe("hybridock-pep") == str(fake_exe)

    def test_returns_none_when_truly_not_installed(self, monkeypatch, tmp_path):
        monkeypatch.setattr(tui.shutil, "which", lambda name: None)
        monkeypatch.setattr(tui.sys, "executable", str(tmp_path / "python"))
        assert tui._resolve_exe("hybridock-pep") is None

    def test_sibling_must_actually_be_executable(self, monkeypatch, tmp_path):
        """A same-named non-executable file (or directory) next to the
        interpreter must not be handed to subprocess.Popen."""
        monkeypatch.setattr(tui.shutil, "which", lambda name: None)
        fake_python = tmp_path / "python"
        fake_python.write_text("#!/bin/sh\n")
        not_executable = tmp_path / "hybridock-pep"
        not_executable.write_text("not a real executable")  # no chmod +x
        monkeypatch.setattr(tui.sys, "executable", str(fake_python))
        assert tui._resolve_exe("hybridock-pep") is None


# --------------------------------------------------------------------------- #
# form validation
# --------------------------------------------------------------------------- #

class TestValidate:
    def test_clean_dock_form_passes(self, tmp_path):
        p = tmp_path / "r.pdb"
        p.write_text("ATOM\n")
        assert tui.validate(_values(receptor=str(p)), tui.DOCK_KEYS) == []

    def test_reports_every_bad_field_not_just_the_first(self, tmp_path):
        errs = tui.validate(_values(peptide="XXX", site="1 2", box="999"), tui.DOCK_KEYS)
        assert len(errs) >= 3

    def test_dock_mode_ignores_offtarget_fields(self, tmp_path):
        p = tmp_path / "r.pdb"
        p.write_text("ATOM\n")
        v = _values(receptor=str(p), offtarget_receptor="", offtarget_site="",
                    offtarget_box="")
        assert tui.validate(v, tui.DOCK_KEYS) == []

    def test_selectivity_mode_requires_offtarget_fields(self, tmp_path):
        p = tmp_path / "r.pdb"
        p.write_text("ATOM\n")
        v = _values(receptor=str(p), offtarget_receptor="", offtarget_site="",
                    offtarget_box="")
        errs = tui.validate(v, tui.SEL_KEYS)
        assert len(errs) == 3
        assert all("required for this mode" in e for e in errs)

    def test_error_messages_name_the_field(self):
        errs = tui.validate(_values(peptide=""), tui.DOCK_KEYS)
        assert any(tui.FIELD["peptide"].label in e for e in errs)

    def test_crystal_mode_requires_peptide_pdb(self):
        errs = tui.validate(_values(peptide_pdb=""), tui.CRYSTAL_KEYS)
        assert any("required for this mode" in e for e in errs)

    def test_crystal_mode_passes_with_peptide_pdb(self, tmp_path):
        p = tmp_path / "peptide.pdb"
        p.write_text("ATOM\n")
        assert tui.validate(_values(peptide_pdb=str(p)), tui.CRYSTAL_KEYS) == []

    def test_crystal_mode_ignores_dock_only_fields(self, tmp_path):
        """site/box/n_samples/output_dir are irrelevant to crystal-score."""
        p = tmp_path / "peptide.pdb"
        p.write_text("ATOM\n")
        v = _values(peptide_pdb=str(p), site="not a site", box="", output_dir="")
        assert tui.validate(v, tui.CRYSTAL_KEYS) == []


# --------------------------------------------------------------------------- #
# form definition integrity
# --------------------------------------------------------------------------- #

class TestFormDefinition:
    def test_keys_are_unique(self):
        keys = [f.key for f in tui.FIELDS]
        assert len(keys) == len(set(keys))

    def test_dock_and_sel_keys_all_exist(self):
        for k in set(tui.DOCK_KEYS) | set(tui.SEL_KEYS):
            assert k in tui.FIELD

    def test_sel_is_a_superset_of_dock(self):
        assert set(tui.DOCK_KEYS) <= set(tui.SEL_KEYS)

    def test_every_default_passes_its_own_validator(self):
        """Shipped defaults must not present the user with a pre-filled error.
        Path fields are exempt: they point at repo-relative sample data."""
        for f in tui.FIELDS:
            if f.is_path or not f.default:
                continue
            assert f.validate(f.default) is None, f"default for {f.key!r} is invalid"

    def test_every_field_has_help_text(self):
        for f in tui.FIELDS:
            assert f.help.strip(), f"{f.key} has no help text"


# --------------------------------------------------------------------------- #
# progress rendering
# --------------------------------------------------------------------------- #

class TestBar:
    def test_empty_and_full(self):
        assert tui._bar(0.0, 10) == "░" * 10
        assert tui._bar(1.0, 10) == "█" * 10

    def test_half(self):
        assert tui._bar(0.5, 10).count("█") == 5

    @pytest.mark.parametrize("frac", [-5.0, 1.5, float("inf")])
    def test_clamps_out_of_range(self, frac):
        b = tui._bar(frac, 10)
        assert len(b) == 10

    def test_width_is_respected(self):
        for w in (1, 8, 40):
            assert len(tui._bar(0.37, w)) == w


class TestDemoLines:
    def test_yields_pairs_of_text_and_delay(self):
        lines = list(tui.demo_lines(5))
        assert lines and all(isinstance(t, str) and isinstance(d, float)
                             for t, d in lines)

    def test_scales_with_n(self):
        assert len(list(tui.demo_lines(50))) > len(list(tui.demo_lines(5)))

    def test_covers_every_pipeline_stage(self):
        text = " ".join(t for t, _ in tui.demo_lines(3))
        for token in ("Stage 1", "Stage 1.5", "Stage 2", "Stage 3"):
            assert token in text

    def test_demo_is_finite(self):
        """A runaway generator would hang the demo UI forever."""
        assert len(list(tui.demo_lines(2))) < 100


class TestPipelineProgressModel:
    def test_starts_empty(self):
        p = tui.PipelineProgress()
        assert p.stage is None and p.done == set()

    def test_stage_weights_sum_to_one(self):
        """Otherwise the overall bar cannot reach 100%."""
        assert sum(w for _, _, w in tui.STAGES) == pytest.approx(1.0)

    def test_stage_tables_agree(self):
        assert set(tui._ORDER) == set(tui._WEIGHT) == set(tui._LABEL)


class TestPipelineProgressFeedRealRun:
    """feed() parses the actual subprocess output the TUI streams from a real
    `hybridock-pep dock` run — the "▶ ..." stage announcements
    output/progress.py's PipelineProgress.step() writes, plus per-item
    progress-tick lines and WARNING-level log lines that always show
    regardless of -v.

    Regression coverage for a real bug: driver.py logs
    "%d poses failed Vina scoring" as a WARNING whenever any pose fails
    scoring — not rare on a real receptor. The old feed() matched stage
    transitions with loose substrings ("pose" as a catch-all for "sample"),
    and "poses failed Vina scoring" contains "pose" but not a bare "scor"
    match (defeated by "failed" also containing "fail"), so it fell through
    to the "pose" fallback and yanked p.stage back to "sample" — mid Stage 2,
    well after real sampling had finished. That corrupted fraction() (which
    stops accounting extra progress for a stage already in `done`) and made
    the TUI's Stage-1 gallery animation (gated on `p.stage == "sample"`)
    spuriously reappear/restart partway through scoring or clustering.
    """

    def _run_to_score_stage(self):
        p = tui.PipelineProgress()
        p.feed("▶ [1/6] Generating poses…")
        p.feed("   45/100 poses minimized (45%)")
        p.feed("▶ [3/6] Scoring poses…")
        assert p.stage == "score"
        return p

    def test_vina_failure_warning_does_not_regress_to_sample(self):
        p = self._run_to_score_stage()
        p.feed("WARNING hybridock_pep.driver: 3 poses failed Vina scoring")
        assert p.stage == "score", (
            "a WARNING about failed poses must not be mistaken for a new "
            "stage announcement and reset progress back to Stage 1"
        )

    def test_ad4_failure_warning_does_not_regress_to_sample(self):
        p = self._run_to_score_stage()
        p.feed("WARNING hybridock_pep.driver: 2 poses failed AD4 scoring")
        assert p.stage == "score"

    def test_per_pose_affinity_failure_does_not_regress_to_sample(self):
        """"Pose %d: affinity scoring failed (%s)" — same class of false match
        (contains "pose", not "stage"), logged during the final affinity stage."""
        p = self._run_to_score_stage()
        p.feed("▶ [4/6] Clustering poses…")
        p.feed("▶ [7/7] Final ranking & ΔG…")
        assert p.stage == "affinity"
        p.feed("WARNING hybridock_pep.driver: Pose 12: affinity scoring failed (ValueError)")
        assert p.stage == "affinity"

    def test_real_step_announcements_advance_through_every_stage(self):
        p = tui.PipelineProgress()
        p.feed("▶ [1/6] Generating poses…")
        assert p.stage == "sample"
        p.feed("▶ [2/6] Preparing receptor & ligands…")
        assert p.stage == "sample", "no dedicated bucket for prep — must not regress either"
        p.feed("▶ [3/6] Scoring poses…")
        assert p.stage == "score"
        p.feed("▶ [4/6] Clustering poses…")
        assert p.stage == "cluster"
        p.feed("▶ [5/6] Refining top poses (MM-GBSA + entropy)…")
        assert p.stage == "affinity"
        assert {"sample", "score", "cluster"} <= p.done

    def test_loading_input_poses_enters_sample(self):
        """--input-poses bypass: driver.py's alternate Stage-1 label."""
        p = tui.PipelineProgress()
        p.feed("▶ Loading input poses…")
        assert p.stage == "sample"

    def test_ligand_prep_tick_does_not_move_stage(self):
        """prep/ligand.py's own progress tick ("N/100 ligands prepared") has
        no keyword this module tracks — must not silently regress or advance."""
        p = self._run_to_score_stage()
        p.feed("   80/100 poses minimized (80%)")  # would-be "min" tick, post-score
        assert p.stage == "score"

    def test_denoising_step_ticks_do_not_move_stage(self):
        """rapidock_runner.py's per-diffusion-step tick during real Stage 1
        sampling — no stage keyword, must not falsely exit "sample" mid-wait."""
        p = tui.PipelineProgress()
        p.feed("▶ [1/6] Generating poses…")
        for pct in (10, 20, 30, 90, 100):
            p.feed(f"   {pct}/100 denoising steps ({pct}%)")
            assert p.stage == "sample"

    def test_docking_complete_final_print_does_not_corrupt_finished_state(self):
        """cli.py's unconditional closing print() must not un-finish a run."""
        p = self._run_to_score_stage()
        p.feed("Best pose ΔG = -9.31 kcal/mol  (cluster 1, 34 members)")
        assert p.finished
        p.feed("Docking complete. 100 poses scored.")
        assert p.finished
        assert p.fraction() == 1.0


class TestPipelineProgressFeedDemo:
    """demo_lines()'s literal "Stage N[.M]:" markers — synthetic text this
    module fully controls — must still drive every stage transition."""

    def test_demo_transcript_advances_through_every_stage_in_order(self):
        p = tui.PipelineProgress()
        seen = []
        for text, _delay in tui.demo_lines(5):
            if text.startswith("__ART__:"):
                continue
            p.feed(text)
            if p.stage and (not seen or seen[-1] != p.stage):
                seen.append(p.stage)
        assert seen == ["sample", "min", "score", "cluster", "affinity"]
        assert p.finished

    def test_demo_per_pose_lines_do_not_disturb_the_current_stage(self):
        p = tui.PipelineProgress()
        p.feed("Stage 2: physics rescoring (vina + ad4)")
        for i in range(1, 6):
            p.feed(f"  scored {i}/5 poses")
        assert p.stage == "score"


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

class TestMain:
    def test_help_prints_docstring_and_exits_zero(self, capsys):
        assert tui.main(["--help"]) == 0
        assert "HybriDock-Pep terminal UI" in capsys.readouterr().out

    def test_print_mode_dispatches_without_running(self, monkeypatch):
        called = {}
        def fake(print_only=False):
            called["print_only"] = print_only
            return 0
        monkeypatch.setattr(tui, "run_cli_wizard", fake)
        assert tui.main(["--print"]) == 0
        assert called["print_only"] is True

    def test_cli_mode_dispatches(self, monkeypatch):
        called = {}
        def fake(print_only=False):
            called["print_only"] = print_only
            return 0
        monkeypatch.setattr(tui, "run_cli_wizard", fake)
        assert tui.main(["--cli"]) == 0
        assert called["print_only"] is False

    def test_fullscreen_failure_falls_back_to_wizard(self, monkeypatch, capsys):
        """A headless/dumb terminal must degrade, not traceback."""
        def boom(auto_demo=False):
            raise RuntimeError("no tty")
        monkeypatch.setattr(tui, "run_fullscreen", boom)
        monkeypatch.setattr(tui, "run_cli_wizard", lambda print_only=False: 0)
        assert tui.main([]) == 0
        assert "full-screen UI unavailable" in capsys.readouterr().out

    def test_demo_flag_reaches_fullscreen(self, monkeypatch):
        seen = {}
        def fake(auto_demo=False):
            seen["demo"] = auto_demo
            return 0
        monkeypatch.setattr(tui, "run_fullscreen", fake)
        assert tui.main(["--demo"]) == 0
        assert seen["demo"] is True


# --------------------------------------------------------------------------- #
# TUI <-> CLI contract
# --------------------------------------------------------------------------- #

class TestGeneratedCommandsParse:
    """The commands the TUI builds must actually be accepted by the CLI.

    Nothing else checks this. The TUI hard-codes flag spellings, so renaming a
    CLI flag would leave the UI silently emitting a command that only fails once
    the user pastes it into a shell.
    """

    @staticmethod
    def _parse(cmd):
        from hybridock_pep.cli import _build_parser
        # cmd[0] is the executable name; argparse takes everything after it
        return _build_parser().parse_args(cmd[1:])

    def test_dock_command_parses(self, tmp_path):
        r = tmp_path / "r.pdb"
        r.write_text("ATOM\n")
        ns = self._parse(tui.build_dock_command(_values(receptor=str(r))))
        assert ns.command == "dock"
        assert ns.peptide == "LISDAELEAIFEADC"

    def test_dock_command_with_refine_parses(self, tmp_path):
        r = tmp_path / "r.pdb"
        r.write_text("ATOM\n")
        ns = self._parse(
            tui.build_dock_command(_values(receptor=str(r), refine_topk="10")))
        assert ns.refine_topk == 10

    def test_selectivity_command_parses(self, tmp_path):
        r = tmp_path / "r.pdb"
        r.write_text("ATOM\n")
        o = tmp_path / "o.pdb"
        o.write_text("ATOM\n")
        ns = self._parse(tui.build_selectivity_command(
            _values(receptor=str(r), offtarget_receptor=str(o))))
        assert ns.command == "selectivity"

    def test_crystal_command_parses(self, tmp_path):
        r = tmp_path / "r.pdb"
        r.write_text("ATOM\n")
        pp = tmp_path / "pose.pdb"
        pp.write_text("ATOM\n")
        ns = self._parse(tui.build_crystal_command(
            _values(receptor=str(r), peptide_pdb=str(pp))))
        assert ns.command == "crystal-score"
        assert ns.peptide == "LISDAELEAIFEADC"

    def test_every_flag_the_tui_emits_is_known_to_the_cli(self, tmp_path):
        """Catches a renamed/removed CLI flag that the TUI still emits."""
        r = tmp_path / "r.pdb"
        r.write_text("ATOM\n")
        o = tmp_path / "o.pdb"
        o.write_text("ATOM\n")
        pp = tmp_path / "pose.pdb"
        pp.write_text("ATOM\n")
        for cmd in (tui.build_dock_command(_values(receptor=str(r), refine_topk="5")),
                    tui.build_selectivity_command(
                        _values(receptor=str(r), offtarget_receptor=str(o))),
                    tui.build_crystal_command(_values(receptor=str(r), peptide_pdb=str(pp)))):
            self._parse(cmd)   # argparse exits non-zero on an unknown flag
