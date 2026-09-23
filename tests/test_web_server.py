"""Tests for the studio web UI (``hybridock-pep serve``).

The important one is :data:`SETTING_MATRIX`: every field the UI exposes is
exercised three ways — default accepted, a bad value rejected, and a changed
value actually reaching the command line. A flag that silently stops being
passed is the failure mode this file exists to catch.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from hybridock_pep.ui import tui
from hybridock_pep.web import server

REPO = Path(__file__).resolve().parent.parent
RECEPTOR = REPO / "data/pdbs/1YCR_mdm2.pdb"
PEPTIDE_PDB = REPO / "data/pdbs/1YCR_peptide.pdb"
OFFTARGET = REPO / "data/pdbs/1I0Z.pdb"

pytestmark = pytest.mark.skipif(not RECEPTOR.exists(), reason="example PDBs not installed")


def base_values(**over: str) -> dict[str, str]:
    """A valid dock form, which each test then perturbs."""
    values = {
        "peptide": "ETFSDLWKLLPE",
        "receptor": str(RECEPTOR),
        "site": "25.20 -25.61 -7.97",
        "box": "30",
        "n_samples": "100",
        "output_dir": "runs/studio_test",
    }
    values.update(over)
    return values


def command_for(mode: str = "dock", **over: str) -> str:
    res = server.validate_request({"mode": mode, "values": base_values(**over)})
    assert res["ok"], res["errors"]
    return res["command"]


# --------------------------------------------------------------------------- #
#  The settings matrix
# --------------------------------------------------------------------------- #

#: key -> (good value, fragment expected in the command, bad value, extra context)
#: ``fragment`` None means the setting deliberately produces no flag by itself.
SETTING_MATRIX: dict[str, tuple[str, str | None, str | None, dict[str, str]]] = {
    "peptide": ("LISDAELEAIFEADC", "--peptide LISDAELEAIFEADC", "ZZZZ", {}),
    "receptor": (str(RECEPTOR), f"--receptor {RECEPTOR}", "/nope/missing.pdb", {}),
    "blind": ("y", "--blind", "maybe", {}),
    "n_pocket_search": ("250", "--n-pocket-search 250", "0", {"blind": "y"}),
    "n_pockets": ("4", "--n-pockets 4", "-1", {"blind": "y"}),
    "n_per_pocket": ("120", "--n-per-pocket 120", "abc", {"blind": "y"}),
    "site": ("10 11 12", "--site 10 11 12", "10 11", {}),
    "box": ("24", "--box 24", "-5", {}),
    "long_checkpoint_threshold": ("15", "--long-checkpoint-threshold 15", "0", {}),
    "n_samples": ("40", "--n-samples 40", "0", {}),
    "refine_topk": ("5", "--refine-topk 5", "-2", {}),
    "ultra": ("32", "--ultra 32", "-1", {}),
    "ultra_charged": ("y", "--ultra-charged", "perhaps", {"ultra": "32"}),
    "seed": ("7", "--seed 7", "-3", {}),
    "no_minimize": ("y", "--no-minimize", "sure", {}),
    "ensemble": ("y", "--ensemble", "yep!", {}),
    "free_entropy": ("y", "--free-entropy", "nah", {"ensemble": "y"}),
    "mmgbsa_ie": ("y", "--mmgbsa-ie", "ok", {"refine_topk": "5"}),
    "mmgbsa_3traj": ("y", "--mmgbsa-3traj", "ok", {"refine_topk": "5"}),
    "mmgbsa_dielectric": ("4.0", "--mmgbsa-dielectric 4.0", "0.2", {"refine_topk": "5"}),
    "mmgbsa_cpu_only": ("y", "--mmgbsa-cpu-only", "ok", {"refine_topk": "5"}),
    "output_dir": ("runs/studio_alt", "--output-dir runs/studio_alt", None, {}),
}


@pytest.mark.parametrize("key", sorted(SETTING_MATRIX))
def test_setting_reaches_the_command_line(key: str) -> None:
    good, fragment, _bad, context = SETTING_MATRIX[key]
    over = dict(context)
    over[key] = good
    cmd = command_for(**over)
    if fragment is not None:
        assert fragment in cmd, f"{key}={good} did not reach the command: {cmd}"


@pytest.mark.parametrize("key", sorted(SETTING_MATRIX))
def test_bad_setting_is_rejected(key: str) -> None:
    _good, _fragment, bad, context = SETTING_MATRIX[key]
    if bad is None:
        pytest.skip(f"{key} accepts any string")
    over = dict(context)
    over[key] = bad
    res = server.validate_request({"mode": "dock", "values": base_values(**over)})
    assert not res["ok"], f"{key}={bad!r} was accepted"
    assert key in res["errors"], f"{key}={bad!r} failed, but the error was not on {key}"
    assert res["command"] is None


def test_every_field_is_covered_by_the_matrix_or_a_dedicated_test() -> None:
    """Nobody adds a CLI flag to the TUI without it showing up here."""
    covered = set(SETTING_MATRIX) | {
        "mode",          # chooses dock vs crystal-score; covered by the mode tests
        "peptide_pdb",   # crystal mode; test_crystal_mode_command
        "input_poses",   # needs a real directory; test_input_poses_*
        "calibration",   # needs a real file; test_calibration_override
        "offtarget_receptor", "offtarget_site", "offtarget_box",  # selectivity tests
    }
    missing = {f.key for f in tui.FIELDS} - covered
    assert not missing, f"settings with no test: {sorted(missing)}"


# --------------------------------------------------------------------------- #
#  Mode-specific behaviour
# --------------------------------------------------------------------------- #

def test_blind_drops_site_and_adds_pocket_search() -> None:
    cmd = command_for(blind="y")
    assert "--blind" in cmd
    assert "--site" not in cmd
    assert "--box" not in cmd


def test_site_mode_has_no_blind_flag() -> None:
    assert "--blind" not in command_for(blind="n")


def test_crystal_mode_command() -> None:
    res = server.validate_request({
        "mode": "crystal",
        "values": base_values(peptide_pdb=str(PEPTIDE_PDB)),
    })
    assert res["ok"], res["errors"]
    assert "crystal-score" in res["command"]
    assert f"--peptide-pdb {PEPTIDE_PDB}" in res["command"]
    assert "--n-samples" not in res["command"]


def test_crystal_mode_needs_a_pose() -> None:
    res = server.validate_request({"mode": "crystal", "values": base_values(peptide_pdb="")})
    assert not res["ok"]
    assert "peptide_pdb" in res["errors"]


@pytest.mark.skipif(not OFFTARGET.exists(), reason="off-target example missing")
def test_selectivity_command_carries_both_receptors() -> None:
    res = server.validate_request({
        "mode": "selectivity",
        "values": base_values(
            offtarget_receptor=str(OFFTARGET),
            offtarget_site="1 2 3",
            offtarget_box="30",
        ),
    })
    assert res["ok"], res["errors"]
    cmd = res["command"]
    assert "selectivity" in cmd
    assert f"--target-receptor {RECEPTOR}" in cmd
    assert f"--offtarget-receptor {OFFTARGET}" in cmd
    assert "--offtarget-site 1 2 3" in cmd


def test_selectivity_without_offtarget_is_rejected() -> None:
    res = server.validate_request({"mode": "selectivity", "values": base_values()})
    assert not res["ok"]
    assert "offtarget_receptor" in res["errors"]


def test_selectivity_ignores_a_stale_blind_flag() -> None:
    """A two-receptor run has no blind path; a leftover blind=y must not break it."""
    res = server.validate_request({
        "mode": "selectivity",
        "values": base_values(blind="y", offtarget_receptor=str(OFFTARGET),
                              offtarget_site="1 2 3", offtarget_box="30"),
    })
    assert res["ok"], res["errors"]
    assert "--target-site" in res["command"]


def test_input_poses_replaces_sampling(tmp_path: Path) -> None:
    poses = tmp_path / "poses"
    poses.mkdir()
    cmd = command_for(input_poses=str(poses))
    assert f"--input-poses {poses}" in cmd
    assert "--n-samples" not in cmd, "sampling and input poses are mutually exclusive"


def test_input_poses_directory_must_exist() -> None:
    res = server.validate_request({
        "mode": "dock", "values": base_values(input_poses="/nope/not/here")})
    assert not res["ok"]
    assert "input_poses" in res["errors"]


def test_calibration_override(tmp_path: Path) -> None:
    cal = tmp_path / "cal.json"
    cal.write_text("{}", encoding="utf-8")
    assert f"--calibration {cal}" in command_for(calibration=str(cal))


def test_skipped_settings_follow_the_same_rules_as_the_terminal_ui() -> None:
    off = server.validate_request({"mode": "dock", "values": base_values()})["skipped"]
    assert {"mmgbsa_ie", "mmgbsa_3traj", "mmgbsa_cpu_only"} <= set(off)
    assert "n_pockets" in off  # not blind

    on = server.validate_request(
        {"mode": "dock", "values": base_values(refine_topk="5", blind="y")})["skipped"]
    assert "mmgbsa_ie" not in on
    assert "n_pockets" not in on


# --------------------------------------------------------------------------- #
#  Helpers the UI reads
# --------------------------------------------------------------------------- #

def test_peptide_stats_flags_the_caveats() -> None:
    stats = server.peptide_stats("LISDAELEAIFEADC")
    assert stats["length"] == 15
    assert stats["net_charge"] == -5
    assert stats["charged"] is True
    assert stats["ends_in_cys"] is True
    assert stats["band"] == "long"


def test_kd_conversion_is_human_scaled() -> None:
    assert server._kd_from_dg(-9.28).endswith("nM")
    assert server._kd_from_dg(-14.0).endswith(("pM", "fM"))
    assert server._kd_from_dg(None) is None


def test_estimate_scales_with_work() -> None:
    quick = server.estimate_seconds({"n_samples": "20"}, "dock")
    standard = server.estimate_seconds({"n_samples": "100"}, "dock")
    refined = server.estimate_seconds({"n_samples": "100", "refine_topk": "5"}, "dock")
    assert quick < standard < refined
    assert server.estimate_seconds({}, "crystal") < quick


def test_serialized_fields_match_the_terminal_ui() -> None:
    fields = server.serialize_fields()
    assert {f["key"] for f in fields} == {f.key for f in tui.FIELDS}
    assert all(f["kind"] for f in fields)


def test_examples_point_at_real_files() -> None:
    for example in server.available_examples():
        assert Path(example["receptor_path"]).exists()
        assert len(example["site"]) == 3


# --------------------------------------------------------------------------- #
#  HTTP layer
# --------------------------------------------------------------------------- #

@pytest.fixture()
def live_server():
    """A studio on an ephemeral port, torn down after the test."""
    manager = server.JobManager()
    handler = type("TestHandler", (server.StudioHandler,), {"manager": manager})
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def get(base: str, path: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(base + path, timeout=10) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def post(base: str, path: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


@pytest.mark.parametrize("path", [
    "/", "/index.html",
    "/static/css/studio.css", "/static/js/studio.js", "/static/js/cloud.js",
    "/api/env", "/api/fields", "/api/examples", "/api/jobs",
])
def test_every_page_and_endpoint_answers(live_server: str, path: str) -> None:
    status, body = get(live_server, path)
    assert status == 200
    assert body


def test_fields_endpoint_feeds_the_form(live_server: str) -> None:
    _, body = get(live_server, "/api/fields")
    data = json.loads(body)
    assert len(data["fields"]) == len(tui.FIELDS)
    assert data["dock_keys"] and data["crystal_keys"] and data["selectivity_keys"]
    assert [s["key"] for s in data["stages"]] == [k for k, _, _ in tui.STAGES]


def test_static_path_traversal_is_refused(live_server: str) -> None:
    status, _ = get(live_server, "/static/../../../../etc/passwd")
    assert status in (403, 404)


def test_unknown_route_is_a_clean_404(live_server: str) -> None:
    status, body = get(live_server, "/api/not-a-thing")
    assert status == 404
    assert b"error" in body


def test_upload_accepts_a_pdb_and_rejects_junk(live_server: str) -> None:
    pdb_text = RECEPTOR.read_text(encoding="utf-8")[:4000]
    status, data = post(live_server, "/api/upload", {"name": "demo.pdb", "content": pdb_text})
    assert status == 200
    assert Path(data["path"]).exists()
    assert data["atoms"] > 0

    status, data = post(live_server, "/api/upload",
                        {"name": "notes.txt", "content": "hello"})
    assert status == 400
    assert "pdb" in data["error"]["message"].lower()

    status, data = post(live_server, "/api/upload",
                        {"name": "empty.pdb", "content": "no atoms here"})
    assert status == 400


def test_validate_endpoint_reports_field_errors(live_server: str) -> None:
    status, data = post(live_server, "/api/validate",
                        {"mode": "dock", "values": base_values(peptide="ZZZ")})
    assert status == 200
    assert data["ok"] is False
    assert "peptide" in data["errors"]


def test_preview_returns_the_runnable_command(live_server: str) -> None:
    status, data = post(live_server, "/api/preview", {"mode": "dock", "values": base_values()})
    assert status == 200
    assert data["command"].startswith("hybridock-pep dock ")


def test_run_refuses_an_invalid_form(live_server: str) -> None:
    status, data = post(live_server, "/api/run",
                        {"mode": "dock", "values": base_values(box="-1")})
    assert status == 400
    assert "box" in data["error"]["fields"]


def test_job_lifecycle_with_a_harmless_command(live_server: str, tmp_path: Path) -> None:
    """Drive start → poll → finish → results without spending a GPU minute."""
    manager = None
    for thread_obj in threading.enumerate():  # the handler class holds the manager
        del thread_obj
    # reach the manager through a request instead: start a trivial job directly
    from hybridock_pep.web.server import JobManager

    manager = JobManager()
    job = manager.start([sys.executable, "-c", "print('Stage 1: generating poses'); print('done')"],
                        "dock", {}, tmp_path)
    for _ in range(100):
        if job.state in ("done", "failed"):
            break
        threading.Event().wait(0.1)
    assert job.state == "done", job.error
    snap = job.snapshot()
    assert snap["fraction"] == 1.0
    assert any("Stage 1" in line for line in job.lines)
    assert server.read_results(job)["poses"] == []


def test_a_failing_run_explains_itself(tmp_path: Path) -> None:
    manager = server.JobManager()
    job = manager.start([sys.executable, "-c", "raise SystemExit(3)"], "dock", {}, tmp_path)
    for _ in range(100):
        if job.state in ("done", "failed"):
            break
        threading.Event().wait(0.1)
    assert job.state == "failed"
    assert job.error
    assert "3" in job.error or "stopped" in job.error.lower()


def test_one_run_at_a_time(tmp_path: Path) -> None:
    manager = server.JobManager()
    manager.start([sys.executable, "-c", "import time; time.sleep(4)"], "dock", {}, tmp_path)
    with pytest.raises(RuntimeError, match="already"):
        manager.start([sys.executable, "-c", "print(1)"], "dock", {}, tmp_path)


def test_cancel_stops_a_run(tmp_path: Path) -> None:
    manager = server.JobManager()
    job = manager.start([sys.executable, "-c", "import time; time.sleep(30)"], "dock", {}, tmp_path)
    threading.Event().wait(0.5)
    assert manager.cancel(job.id) is True
    for _ in range(60):
        if job.state == "cancelled" and job.finished_at:
            break
        threading.Event().wait(0.1)
    assert job.state == "cancelled"


def test_results_reads_a_real_run_directory(tmp_path: Path) -> None:
    """Parse a ranked_poses.csv the way a finished run would produce it."""
    (tmp_path / "ranked_poses.csv").write_text(
        "rank,pooled_affinity_dg,cluster_id,pose_filename,charged_confidence\n"
        "1,-9.28,0,pose_1.pdb,low\n2,-8.10,1,pose_2.pdb,high\n",
        encoding="utf-8")
    job = server.Job("t", ["x"], "dock", {}, tmp_path)
    res = server.read_results(job)
    assert res["headline"]["delta_g"] == -9.28
    assert res["headline"]["n_poses"] == 2
    assert res["headline"]["n_clusters"] == 2
    assert res["headline"]["kd"].endswith("nM")
    assert "ranked_poses.csv" in res["files"]
