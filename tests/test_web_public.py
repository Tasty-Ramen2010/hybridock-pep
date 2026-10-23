"""``hybridock-pep serve --public``: what a stranger on the internet can and cannot do.

Each test plays the stranger. The server here is the real handler with a recording job manager in place of the real
pipeline, so no docking runs. Requests go through http.client (not urllib) so a system proxy cannot get in the way.
"""

from __future__ import annotations

import http.client
import json
import socket
import threading
import uuid
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from hybridock_pep.web import server
from hybridock_pep.web.public import PublicPolicy, scrub

REPO = Path(__file__).resolve().parent.parent
RECEPTOR = REPO / "data/pdbs/1YCR_mdm2.pdb"
PAGE = "https://tasty-ramen2010.github.io"

pytestmark = pytest.mark.skipif(not RECEPTOR.exists(), reason="example PDBs not installed")

PDB = "ATOM      1  N   MET A   1      27.340  24.430   2.614  1.00  0.00           N\n" * 3


class RecordingManager:
    """Stands in for JobManager: remembers what it was asked to run and never starts a process."""

    def __init__(self) -> None:
        self.jobs: dict[str, server.Job] = {}
        self.busy = False
        self.started: list[tuple[list[str], str, dict[str, str], Path]] = []

    def start(self, cmd, mode, values, output_dir):  # noqa: ANN001
        self.started.append((cmd, mode, values, output_dir))
        job = server.Job(uuid.uuid4().hex[:12], cmd, mode, values, output_dir)
        self.jobs[job.id] = job
        return job

    def cancel(self, job_id: str) -> bool:
        return job_id in self.jobs


class Running:
    def __init__(self, port: int, manager: RecordingManager, policy: PublicPolicy | None) -> None:
        self.port, self.manager, self.policy = port, manager, policy

    def call(self, method: str, path: str, body: dict | None = None, headers: dict[str, str] | None = None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        hdrs = dict(headers or {})
        payload = None
        if body is not None:
            payload = json.dumps(body)
            hdrs["Content-Type"] = "application/json"
        conn.request(method, path, body=payload, headers=hdrs)
        res = conn.getresponse()
        raw = res.read()
        conn.close()
        try:
            data = json.loads(raw)
        except ValueError:
            data = raw
        return res.status, data, dict(res.getheaders())


def _serve(tmp_path: Path, policy: PublicPolicy | None):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    manager = RecordingManager()
    handler = type("H", (server.StudioHandler,), {"manager": manager, "policy": policy})
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, Running(port, manager, policy)


@pytest.fixture()
def pub(tmp_path):
    policy = PublicPolicy(allowed_origins=(PAGE,), runs_dir=tmp_path / "runs_public")
    httpd, run = _serve(tmp_path, policy)
    yield run
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture()
def local(tmp_path):
    httpd, run = _serve(tmp_path, None)
    yield run
    httpd.shutdown()
    httpd.server_close()


def form(**over: str) -> dict:
    values = {"peptide": "ETFSDLWKLLPE", "receptor": "example:pdbs/1YCR_mdm2.pdb", "site": "25.20 -25.61 -7.97",
              "box": "30", "n_samples": "25"}
    values.update(over)
    return {"mode": "dock", "values": values}


# --------------------------------------------------------------------------- #
#  The normal use works
# --------------------------------------------------------------------------- #

def test_an_ordinary_request_becomes_a_safe_run(pub):
    status, data, _ = pub.call("POST", "/api/run", form(), {"Origin": PAGE})
    assert status == 200, data
    cmd, mode, values, out_dir = pub.manager.started[0]
    assert mode == "dock" and "--blind" not in cmd
    assert cmd[cmd.index("--receptor") + 1] == str(server.data_file("pdbs/1YCR_mdm2.pdb"))
    assert out_dir.is_relative_to(pub.policy.runs_dir), "runs are written where the server chose"
    assert cmd[cmd.index("--scoring") + 1] == "vina"


def test_an_uploaded_file_is_usable_and_never_shows_a_path(pub):
    status, data, _ = pub.call("POST", "/api/upload", {"name": "my_protein.pdb", "content": PDB}, {"Origin": PAGE})
    assert status == 200 and data["path"].startswith("upload:") and "/" not in data["path"].split(":", 1)[1], data
    status, _, _ = pub.call("POST", "/api/run", form(receptor=data["path"]), {"Origin": PAGE})
    assert status == 200
    used = pub.manager.started[-1][0]
    assert used[used.index("--receptor") + 1].startswith(str(pub.policy.uploads_dir.resolve()))


def test_env_and_examples_describe_limits_without_naming_this_computer(pub):
    for path in ("/api/env", "/api/examples"):
        status, data, _ = pub.call("GET", path)
        text = json.dumps(data)
        assert status == 200
        assert str(Path.home()) not in text and str(REPO) not in text, f"{path} leaks a folder"
    _, env, _ = pub.call("GET", "/api/env")
    assert env["public"]["blind"] is False and env["public"]["max_samples"] == 25
    assert env["public"]["allowed_samples"] == [5, 10, 25]
    _, ex, _ = pub.call("GET", "/api/examples")
    assert ex["examples"] and all(e["receptor_path"].startswith("example:") for e in ex["examples"])


# --------------------------------------------------------------------------- #
#  The stranger: files on this computer
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("token", [
    "/etc/passwd", "~/.ssh/id_rsa", "../../etc/passwd", "file:///etc/passwd", "C:\\Windows\\win.ini",
    "example:../../etc/passwd", "example:pdbs/../../../../etc/passwd", "upload:../../etc/passwd",
    "upload:0123456789ab_../../x.pdb", "upload:", "example:", "",
])
def test_a_visitor_cannot_name_a_file_on_this_computer(pub, token):
    for key in ("receptor", "offtarget_receptor", "peptide_pdb"):
        body = form(**{key: token})
        if key != "receptor":
            body["mode"] = "selectivity" if key == "offtarget_receptor" else "crystal"
            body["values"].update({"offtarget_site": "1 2 3", "offtarget_box": "30", "n_samples": "25"})
        if key == "receptor" and token == "":
            continue
        status, data, _ = pub.call("POST", "/api/run", body, {"Origin": PAGE})
        assert status in (400, 409), (key, token, status, data)
    assert pub.manager.started == [], "nothing was started"


def test_settings_that_reach_the_disk_are_ignored(pub):
    body = form(output_dir="/tmp/pwned", input_poses="/var/attacker_poses", calibration="/etc/passwd", refine_topk="20", ultra="64",
                no_minimize="y", ensemble="y", mmgbsa_cpu_only="y")
    status, _, _ = pub.call("POST", "/api/run", body, {"Origin": PAGE})
    assert status == 200
    cmd, _, values, out_dir = pub.manager.started[0]
    joined = " ".join(cmd)
    assert "/tmp/pwned" not in joined and "attacker_poses" not in joined and "/etc/passwd" not in joined
    assert "--refine-topk" not in joined and "--ultra" not in joined and "--input-poses" not in joined
    assert out_dir.is_relative_to(pub.policy.runs_dir)


def test_upload_rules(pub):
    bad = [
        ({"name": "x.txt", "content": PDB}, "pdb"),
        ({"name": "a b.pdb", "content": PDB}, "pdb"),
        ({"name": "x.pdb", "content": "hello"}, "ATOM"),
        ({"name": "x.pdb", "content": PDB + "X" * (7 * 1024 * 1024)}, "big"),
    ]
    for body, _ in bad:
        status, _, _ = pub.call("POST", "/api/upload", body, {"Origin": PAGE})
        assert status == 400, body["name"]
    status, data, _ = pub.call("POST", "/api/upload", {"name": "../../evil.pdb", "content": PDB}, {"Origin": PAGE})
    assert status == 200 and ".." not in data["path"], "a path in the name is reduced to a file name"
    assert not (pub.policy.runs_dir.parent / "evil.pdb").exists()


# --------------------------------------------------------------------------- #
#  The stranger: hogging the computer
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("over", [
    {"blind": "y"}, {"n_samples": "100"}, {"n_samples": "50"}, {"n_samples": "7"}, {"n_samples": "0"}, {"n_samples": "-5"}, {"n_samples": "abc"},
    {"peptide": "A" * 26}, {"peptide": "AB"}, {"peptide": "AAAXAAA"}, {"peptide": "--output-dir"},
    {"site": "1 2"}, {"site": "1 2 --evil"}, {"box": "30; rm -rf"}, {"seed": "-1"},
])
def test_limits_and_odd_values_are_refused(pub, over):
    status, data, _ = pub.call("POST", "/api/run", form(**over), {"Origin": PAGE})
    assert status == 400, (over, data)
    assert pub.manager.started == []


@pytest.mark.parametrize("n", ["5", "10", "25"])
def test_the_three_offered_sizes_are_accepted(pub, n):
    status, data, _ = pub.call("POST", "/api/run", form(n_samples=n), {"Origin": PAGE})
    assert status == 200, (n, data)


def test_a_comparison_is_capped_lower_than_a_single_run(pub):
    body = {"mode": "selectivity", "values": {**form()["values"], "offtarget_receptor": "example:pdbs/1I0Z.pdb",
                                             "offtarget_site": "24.84 22.73 41.69", "offtarget_box": "30", "n_samples": "50"}}
    status, _, _ = pub.call("POST", "/api/run", body, {"Origin": PAGE})
    assert status == 400
    body["values"]["n_samples"] = "25"
    status, data, _ = pub.call("POST", "/api/run", body, {"Origin": PAGE})
    assert status == 200, data


def test_a_busy_computer_says_so_plainly(pub):
    pub.manager.busy = True
    status, data, _ = pub.call("POST", "/api/run", form(), {"Origin": PAGE})
    assert status == 409 and "busy" in data["error"]["message"].lower()


def test_too_many_runs_from_one_visitor_are_slowed(tmp_path):
    policy = PublicPolicy(allowed_origins=(PAGE,), runs_dir=tmp_path / "r", runs_per_window=2)
    httpd, run = _serve(tmp_path, policy)
    try:
        codes = [run.call("POST", "/api/run", form(), {"Origin": PAGE})[0] for _ in range(3)]
        assert codes == [200, 200, 429]
        other = run.call("POST", "/api/run", form(), {"Origin": PAGE, "CF-Connecting-IP": "203.0.113.9"})[0]
        assert other == 200, "a different visitor is not punished for the first one's runs"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_the_memory_meter_works_for_visitors(pub):
    status, data, _ = pub.call("GET", "/api/stats", headers={"Origin": PAGE})
    assert status == 200 and 0 <= data["percent"] <= 100 and str(Path.home()) not in json.dumps(data)


def test_other_visitors_runs_cannot_be_listed(pub, local):
    assert pub.call("GET", "/api/jobs")[0] == 404
    assert local.call("GET", "/api/jobs")[0] == 200, "the ordinary local tool still lists them"


# --------------------------------------------------------------------------- #
#  The stranger: a page that is not ours
# --------------------------------------------------------------------------- #

def test_only_the_hosted_page_may_call_from_a_browser(pub):
    ok = pub.call("OPTIONS", "/api/run", headers={"Origin": PAGE, "Access-Control-Request-Method": "POST"})
    assert ok[0] == 204 and ok[2]["Access-Control-Allow-Origin"] == PAGE
    evil = pub.call("OPTIONS", "/api/run", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert evil[0] == 403 and "Access-Control-Allow-Origin" not in evil[2]
    status, _, hdrs = pub.call("POST", "/api/run", form(), {"Origin": "https://evil.example"})
    assert status == 403 and pub.manager.started == []
    _, _, hdrs = pub.call("GET", "/api/env", headers={"Origin": "https://evil.example"})
    assert "Access-Control-Allow-Origin" not in hdrs, "another page cannot read answers"
    _, _, hdrs = pub.call("GET", "/api/env", headers={"Origin": PAGE})
    assert hdrs["Access-Control-Allow-Origin"] == PAGE


def test_the_local_tool_sends_no_cross_site_permission(local):
    status, _, hdrs = local.call("GET", "/api/env", headers={"Origin": "https://evil.example"})
    assert status == 200 and "Access-Control-Allow-Origin" not in hdrs
    assert local.call("OPTIONS", "/api/run", headers={"Origin": "https://evil.example"})[0] == 403


def test_this_servers_own_page_may_call_it(pub):
    host = f"127.0.0.1:{pub.port}"
    status, _, _ = pub.call("POST", "/api/run", form(), {"Origin": f"http://{host}"})
    assert status == 200


# --------------------------------------------------------------------------- #
#  The stranger: reading results
# --------------------------------------------------------------------------- #

def _job_with_files(run: Running, tmp_path: Path):
    out = tmp_path / "run_a"
    out.mkdir()
    (out / "ranked_poses.csv").write_text(f"rank,note\n1,{Path.home()}/secret\n")
    (out / "best_pose.pdb").write_text(PDB)
    (out / "notes.txt").write_text("not on the list")
    (tmp_path / "run_a_sibling").mkdir()
    (tmp_path / "run_a_sibling" / "loot.txt").write_text("loot")
    (tmp_path / "outside.txt").write_text("outside")
    job = run.manager.start(["hybridock-pep", "dock"], "dock", {}, out)
    return job


def test_result_files_are_limited_to_the_ones_the_page_offers(pub, tmp_path):
    job = _job_with_files(pub, tmp_path)
    base = f"/api/jobs/{job.id}/file?name="
    assert pub.call("GET", base + "best_pose.pdb")[0] == 200
    assert pub.call("GET", base + "notes.txt")[0] == 404
    assert pub.call("GET", base + "../outside.txt")[0] == 403
    assert pub.call("GET", base + "../run_a_sibling/loot.txt")[0] == 403, "a folder that merely starts with the same letters"
    status, data, _ = pub.call("GET", base + "ranked_poses.csv")
    assert status == 200 and str(Path.home()).encode() not in data, "this computer's folder names are removed from text files"


def test_the_local_tool_also_refuses_a_look_alike_folder(local, tmp_path):
    job = _job_with_files(local, tmp_path)
    assert local.call("GET", f"/api/jobs/{job.id}/file?name=../run_a_sibling/loot.txt")[0] == 403
    assert local.call("GET", f"/api/jobs/{job.id}/file?name=notes.txt")[0] == 200


@pytest.mark.parametrize("path", ["/static/../server.py", "/static/%2e%2e/server.py", "/static/..%2fserver.py", "/static/../../pyproject.toml"])
def test_static_files_cannot_be_escaped(pub, path):
    assert pub.call("GET", path)[0] in (403, 404)


def test_commands_and_logs_do_not_name_this_computer(pub):
    status, data, _ = pub.call("POST", "/api/run", form(), {"Origin": PAGE})
    text = json.dumps(data)
    assert status == 200 and str(Path.home()) not in text and str(Path.cwd()) not in text
    status, data, _ = pub.call("POST", "/api/preview", form(), {"Origin": PAGE})
    assert status == 200 and str(Path.home()) not in data["command"]


def test_scrub_replaces_folder_names():
    assert scrub("/Users/x/work/runs/a and /Users/x/other", "/Users/x", "/Users/x/work") == "./runs/a and ~/other"
    assert scrub("nothing here", "/", ".") == "nothing here"


def test_old_runs_and_uploads_are_deleted(tmp_path):
    import os
    import time
    policy = PublicPolicy(runs_dir=tmp_path / "r", keep_hours=1)
    (policy.runs_dir / "old").mkdir(parents=True)
    (policy.runs_dir / "new").mkdir()
    policy.uploads_dir.mkdir()
    (policy.uploads_dir / "0123456789ab_old.pdb").write_text(PDB)
    long_ago = time.time() - 3 * 3600
    os.utime(policy.runs_dir / "old", (long_ago, long_ago))
    os.utime(policy.uploads_dir / "0123456789ab_old.pdb", (long_ago, long_ago))
    assert policy.prune() == 2
    assert (policy.runs_dir / "new").exists() and not (policy.runs_dir / "old").exists()
