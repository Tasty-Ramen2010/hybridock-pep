"""The environment spec must actually declare the tools the pipeline needs.

Dropping ADFRsuite moved two hard requirements into conda/pip packages: meeko
(receptor PDBQT) and autogrid (AD4 grid maps). If either silently falls out of
envs/score-env.yml, nothing fails at install time — the user only finds out
when `prep` or `--scoring ad4` dies on a real run, long after the install they
would otherwise trust.

There is also an upgrade hazard these tests cover: `install_score_env` skips
creation when the env already exists, so anything added to the yml after a
user's first install never reaches them. `_repair_score_env_tooling` backfills
that gap, and is pinned here.
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


class TestScoreEnvSpec:
    def test_yml_exists(self):
        assert SCORE_ENV_YML.is_file(), "envs/score-env.yml is missing"

    @pytest.mark.parametrize("package", ["autogrid", "meeko", "openbabel"])
    def test_receptor_prep_tooling_is_declared(self, package):
        """These replace ADFRsuite. If one is dropped, prep breaks at runtime."""
        text = SCORE_ENV_YML.read_text()
        assert package in text, (
            f"{package} is not declared in envs/score-env.yml — receptor prep "
            "or AD4 scoring will fail on a fresh install"
        )

    def test_autogrid_is_pinned_to_a_working_version(self):
        """4.2.9 is the first conda-forge build with a native osx-arm64 target."""
        assert "autogrid>=4.2.9" in SCORE_ENV_YML.read_text()

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


class TestExistingEnvRepair:
    """An env created before a tool was added to the yml must still get it."""

    def test_repair_installs_autogrid_when_missing(self, monkeypatch):
        mod = _load_setup_environment()
        calls = []
        monkeypatch.setattr(mod, "_run", lambda cmd, dry, **kw: calls.append(cmd))
        monkeypatch.setattr(mod, "_env_has_binary", lambda env, binary: False)

        mod._repair_score_env_tooling(dry_run=False)

        assert len(calls) == 1, f"expected one install call, got {calls}"
        cmd = calls[0]
        assert "conda" in cmd[0] and "install" in cmd
        assert "score-env" in cmd
        assert any(a.startswith("autogrid") for a in cmd), cmd

    def test_repair_is_a_noop_when_autogrid_present(self, monkeypatch):
        """Never reinstall into a working env — that is slow and can break a
        live environment for no reason."""
        mod = _load_setup_environment()
        calls = []
        monkeypatch.setattr(mod, "_run", lambda cmd, dry, **kw: calls.append(cmd))
        monkeypatch.setattr(mod, "_env_has_binary", lambda env, binary: True)

        mod._repair_score_env_tooling(dry_run=False)

        assert calls == [], f"repair touched a complete env: {calls}"

    def test_install_score_env_repairs_an_existing_env(self, monkeypatch):
        """The regression that motivated this: an existing env is skipped for
        creation, so the repair step is the only thing that can backfill it."""
        mod = _load_setup_environment()
        monkeypatch.setattr(mod, "_env_exists", lambda name: True)
        monkeypatch.setattr(mod, "_run", lambda cmd, dry, **kw: None)
        monkeypatch.setattr(mod, "_pip_in", lambda env: ["python", "-m", "pip", "install"])

        called = []
        monkeypatch.setattr(mod, "_repair_score_env_tooling", lambda dry: called.append(dry))

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
