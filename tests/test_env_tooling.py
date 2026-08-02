"""The environment spec must declare the tools the pipeline needs — and only
those.

Dropping ADFRsuite moved receptor prep onto meeko, which is a genuine hard
requirement: if it silently falls out of envs/score-env.yml nothing fails at
install time, and the user only finds out when `prep` dies on a real run, long
after the install they would otherwise trust.

autogrid (AD4 grid maps) and openbabel (fallback PDBQT conversion) are the
opposite case and these tests pin that distinction. conda-forge has no
linux-aarch64 autogrid build and only py39 openbabel builds, so listing either
as a hard yml dependency makes `conda env create` fail with
PackagesNotFoundError on ARM Linux — the entire install, lost for a tool that
is optional (AD4 is off by default; meeko is the primary converter). They move
to OPTIONAL_SCORE_ENV_TOOLS, installed best-effort and non-fatally.

There is also an upgrade hazard covered here: `install_score_env` skips
creation when the env already exists, so anything added after a user's first
install never reaches them. `_install_optional_score_env_tools` runs on both
paths and is pinned below.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
SCORE_ENV_YML = REPO / "envs" / "score-env.yml"


def _load_setup_environment():
    """Import scripts/setup_environment.py by path (it is not an installed module)."""
    path = REPO / "scripts" / "setup_environment.py"
    if not path.is_file():
        pytest.skip("scripts/setup_environment.py not present")
    spec = importlib.util.spec_from_file_location("_setup_environment_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _yml_dependency_lines() -> list[str]:
    """Dependency lines of score-env.yml, comments stripped.

    Comments must not be searched: the yml legitimately *names* autogrid and
    openbabel while explaining why they are not listed there.
    """
    return [
        ln.split("#", 1)[0].strip().lower()
        for ln in SCORE_ENV_YML.read_text().splitlines()
        if ln.split("#", 1)[0].strip()
    ]


class TestScoreEnvSpec:
    def test_yml_exists(self):
        assert SCORE_ENV_YML.is_file(), "envs/score-env.yml is missing"

    def test_meeko_is_a_hard_dependency(self):
        """meeko replaced ADFRsuite for receptor PDBQT — prep cannot run without
        it, and it has wheels on every platform, so it belongs in the yml."""
        assert any("meeko" in ln for ln in _yml_dependency_lines()), (
            "meeko is not declared in envs/score-env.yml — receptor prep will "
            "fail on a fresh install"
        )

    @pytest.mark.parametrize("package", ["autogrid", "openbabel"])
    def test_platform_limited_tools_are_not_hard_dependencies(self, package):
        """The ARM Linux regression: conda-forge has no linux-aarch64 autogrid
        build and only py39 openbabel builds. As yml dependencies they make
        `conda env create -f score-env.yml` fail outright on ARM Linux with
        PackagesNotFoundError, taking the whole install down for a tool the
        pipeline does not require."""
        offenders = [ln for ln in _yml_dependency_lines() if package in ln]
        assert not offenders, (
            f"{package} is a hard dependency again: {offenders} — this breaks "
            "`conda env create` on linux-aarch64. Put it in "
            "setup_environment.OPTIONAL_SCORE_ENV_TOOLS instead."
        )

    @pytest.mark.parametrize("binary", ["autogrid4", "obabel"])
    def test_platform_limited_tools_are_still_installed_best_effort(self, binary):
        """Not being a hard dependency must not mean "silently never installed"."""
        mod = _load_setup_environment()
        assert binary in mod.OPTIONAL_SCORE_ENV_TOOLS, (
            f"{binary} is neither a yml dependency nor an optional tool — "
            "nothing will ever install it"
        )

    def test_autogrid_is_pinned_to_a_working_version(self):
        """4.2.9 is the first conda-forge build with a native osx-arm64 target."""
        mod = _load_setup_environment()
        assert mod.OPTIONAL_SCORE_ENV_TOOLS["autogrid4"] == "autogrid>=4.2.9"

    def test_no_adfrsuite_requirement_reintroduced(self):
        """ADFRsuite has no arm64 build and needs a license click-through, so it
        must never become an install-time dependency again.

        Only dependency lines are checked — the yml's comments legitimately
        explain what replaced ADFRsuite and why, and must stay allowed.
        """
        deps = [
            ln.split("#", 1)[0].strip().lower()
            for ln in SCORE_ENV_YML.read_text().splitlines()
            if ln.split("#", 1)[0].strip()
        ]
        offenders = [d for d in deps if "adfrsuite" in d or "prepare_receptor" in d]
        assert not offenders, f"ADFRsuite reintroduced as a dependency: {offenders}"


class TestOptionalToolInstall:
    """Whatever the env is missing gets installed — without ever being fatal."""

    def _stub_run(self, calls, ok=True):
        def _run(cmd, dry, **kw):
            calls.append((cmd, kw))
            return ok
        return _run

    def test_installs_both_tools_when_missing(self, monkeypatch):
        mod = _load_setup_environment()
        calls = []
        monkeypatch.setattr(mod, "_run", self._stub_run(calls))
        monkeypatch.setattr(mod, "_env_has_binary", lambda env, binary: False)

        mod._install_optional_score_env_tools(dry_run=False)

        assert len(calls) == 1, f"expected one install call, got {calls}"
        cmd, _ = calls[0]
        assert "conda" in cmd[0] and "install" in cmd
        assert "score-env" in cmd
        assert any(a.startswith("autogrid") for a in cmd), cmd
        assert any(a.startswith("openbabel") for a in cmd), cmd

    def test_installs_only_the_missing_tool(self, monkeypatch):
        """A partial env is the common case (the two were added at different
        times), so this must not reinstall the tool that is already fine."""
        mod = _load_setup_environment()
        calls = []
        monkeypatch.setattr(mod, "_run", self._stub_run(calls))
        monkeypatch.setattr(
            mod, "_env_has_binary", lambda env, binary: binary == "autogrid4"
        )

        mod._install_optional_score_env_tools(dry_run=False)

        assert len(calls) == 1, f"expected one install call, got {calls}"
        specs = [a for a in calls[0][0] if a.startswith(("autogrid", "openbabel"))]
        assert specs == ["openbabel>=3.1"], f"touched a present tool: {specs}"

    def test_is_a_noop_when_both_present(self, monkeypatch):
        """Never reinstall into a working env — that is slow and can break a
        live environment for no reason."""
        mod = _load_setup_environment()
        calls = []
        monkeypatch.setattr(mod, "_run", self._stub_run(calls))
        monkeypatch.setattr(mod, "_env_has_binary", lambda env, binary: True)

        mod._install_optional_score_env_tools(dry_run=False)

        assert calls == [], f"touched a complete env: {calls}"

    def test_install_is_marked_optional_so_a_failure_is_not_fatal(self, monkeypatch):
        """The ARM Linux regression: conda has no autogrid build for
        linux-aarch64, so this conda install *will* fail there. It must not take
        the rest of the install with it."""
        mod = _load_setup_environment()
        calls = []
        monkeypatch.setattr(mod, "_run", self._stub_run(calls, ok=False))
        monkeypatch.setattr(mod, "_env_has_binary", lambda env, binary: False)

        mod._install_optional_score_env_tools(dry_run=False)  # must not raise/exit

        _, kwargs = calls[0]
        assert kwargs.get("optional") is True, (
            f"optional tooling installed with optional={kwargs.get('optional')} — "
            "a platform with no build for it will abort the whole install"
        )

    def test_run_with_optional_returns_false_instead_of_exiting(self, monkeypatch):
        """_run is the thing that actually enforces non-fatality."""
        mod = _load_setup_environment()

        class _Result:
            returncode = 1

        monkeypatch.setattr(mod.subprocess, "run", lambda *a, **kw: _Result())
        assert mod._run(["false"], dry_run=False, optional=True) is False

    def test_install_score_env_installs_tooling_for_an_existing_env(self, monkeypatch):
        """The regression that motivated this: an existing env is skipped for
        creation, so this step is the only thing that can backfill it."""
        mod = _load_setup_environment()
        monkeypatch.setattr(mod, "_env_exists", lambda name: True)
        monkeypatch.setattr(mod, "_run", lambda cmd, dry, **kw: True)
        monkeypatch.setattr(mod, "_pip_in", lambda env: ["python", "-m", "pip", "install"])

        called = []
        monkeypatch.setattr(
            mod, "_install_optional_score_env_tools", lambda dry: called.append(dry)
        )

        mod.install_score_env(dry_run=False, force=False)

        assert called, "install_score_env skipped the existing env without repairing it"


class TestInstalledToolingMatchesTheSpec:
    """If the tools are actually here, confirm they are the ones we claim."""

    def test_autogrid4_runs(self):
        import shutil
        import subprocess

        exe = shutil.which("autogrid4")
        if exe is None:
            pytest.skip("autogrid4 not on PATH (not a score-env shell)")
        # autogrid4 exits non-zero with no args but must at least be executable
        # and identify itself; a broken/wrong-arch binary fails to launch at all.
        proc = subprocess.run([exe], capture_output=True, text=True, timeout=30)
        combined = (proc.stdout + proc.stderr).lower()
        assert "autogrid" in combined or "usage" in combined, combined[:400]

    def test_meeko_receptor_script_is_importable(self):
        pytest.importorskip("meeko", reason="meeko not installed")
