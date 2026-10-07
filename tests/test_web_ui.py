"""The browser front end (web/index.html + web/static/{css,js}/app) against the real server code.

Three kinds of check:

* contract  - the requests the UI builds (tests/fixtures/web_requests.json, generated from the UI's own
              JS and asserted equal to it by tests/web_js/) are accepted by ``validate_request`` and turn
              into the commands the UI says they will;
* integrity - every file the page and its modules reference exists, and ``/`` really serves the app;
* packaging - everything under web/ is covered by a package-data glob, or ``serve`` 404s after pip install.
"""

from __future__ import annotations

import importlib.util
import json
import re
import socket
import threading
import tomllib
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path, PurePosixPath

import pytest

from hybridock_pep.web import server

REPO = Path(__file__).resolve().parent.parent
WEB = REPO / "src" / "hybridock_pep" / "web"
STATIC = WEB / "static"
APP_JS = STATIC / "js" / "app"
FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "web_requests.json").read_text())


def _real(body: dict) -> dict:
    """Swap the fixture's placeholder paths for files that exist on this machine."""
    receptor = str(server._example_receptor("pdbs/1YCR_mdm2.pdb"))
    offtarget = str(server._example_receptor("pdbs/1I0Z.pdb"))
    pose = str(server._example_receptor("pdbs/1YCR_peptide.pdb"))
    text = json.dumps(body).replace("__RECEPTOR__", receptor).replace("__OFFTARGET__", offtarget)
    return json.loads(text.replace("__PEPTIDE_POSE__", pose).replace("\\\\", "\\"))


# --------------------------------------------------------------------------- #
#  contract
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", sorted(FIXTURE))
def test_requests_the_ui_builds_are_accepted_by_the_server(name: str) -> None:
    check = server.validate_request(_real(FIXTURE[name]))
    assert check["ok"], check["errors"]
    assert check["command"]


def test_dock_request_becomes_the_expected_command() -> None:
    cmd = server.validate_request(_real(FIXTURE["dock"]))["command"]
    for fragment in ("dock", "--peptide ETFSDLWKLLPE", "--site 25.2 -25.61 -7.97", "--box 30",
                     "--n-samples 25", "--output-dir runs/studio/dock_test"):
        assert fragment in cmd, fragment
    assert "--blind" not in cmd


def test_expert_settings_reach_the_command_line() -> None:
    cmd = server.validate_request(_real(FIXTURE["dock_expert"]))["command"]
    for fragment in ("--refine-topk 3", "--seed 7", "--no-minimize", "--ensemble", "--long-checkpoint-threshold 10"):
        assert fragment in cmd, fragment


def test_blind_request_drops_the_site() -> None:
    cmd = server.validate_request(_real(FIXTURE["dock_blind"]))["command"]
    assert "--blind" in cmd
    assert "--site" not in cmd


def test_compare_and_score_requests_use_the_right_subcommands() -> None:
    compare = server.validate_request(_real(FIXTURE["compare"]))["command"]
    assert " selectivity " in compare and "--offtarget-receptor" in compare
    score = server.validate_request(_real(FIXTURE["crystal"]))["command"]
    assert " crystal-score " in score and "--peptide-pdb" in score


def test_the_ui_field_keys_exist_in_the_terminal_ui() -> None:
    """If a field is renamed in tui.FIELDS the browser would silently send a key nobody reads."""
    known = {f["key"] for f in server.serialize_fields()}
    for name, body in FIXTURE.items():
        assert set(body["values"]) <= known, (name, set(body["values"]) - known)


# --------------------------------------------------------------------------- #
#  integrity
# --------------------------------------------------------------------------- #

@pytest.fixture()
def live_server():
    manager = server.JobManager()
    handler = type("TestHandler", (server.StudioHandler,), {"manager": manager})
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _get(base: str, path: str) -> tuple[int, str, bytes]:
    try:
        with urllib.request.urlopen(base + path, timeout=10) as resp:
            return resp.status, resp.headers.get("Content-Type", ""), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, "", b""


def test_root_serves_the_new_app_and_the_classic_studio_is_still_reachable(live_server: str) -> None:
    status, _, body = _get(live_server, "/")
    assert status == 200 and b"/static/dist/app.mjs" in body
    status, _, body = _get(live_server, "/static/studio.html")
    assert status == 200 and b"studio.js" in body


def test_every_file_the_page_references_is_served_with_a_usable_type(live_server: str) -> None:
    html = (WEB / "index.html").read_text()
    refs = re.findall(r'(?:href|src)="(/static/[^"]+)"', html)
    assert refs, "index.html references no static files?"
    for ref in refs:
        status, ctype, body = _get(live_server, ref)
        assert status == 200 and body, ref
        if ref.endswith(".mjs"):
            assert "javascript" in ctype, f"{ref} served as {ctype!r}: browsers refuse to run it as a module"


def test_the_bundle_is_up_to_date_with_its_sources() -> None:
    """index.html loads static/dist/*, which scripts/build_web.py generates from the readable sources."""
    spec = importlib.util.spec_from_file_location("build_web", REPO / "scripts" / "build_web.py")
    build_web = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_web)
    assert build_web.is_fresh(), "static/dist/ is stale: run `python3 scripts/build_web.py` and commit the result"


def test_every_module_import_resolves() -> None:
    for path in APP_JS.rglob("*.mjs"):
        for spec in re.findall(r"from '(\.{1,2}/[^']+)'", path.read_text()):
            assert (path.parent / spec).resolve().exists(), f"{path.relative_to(STATIC)} imports missing {spec}"


def test_css_font_and_data_references_exist() -> None:
    for css in (STATIC / "css" / "app").glob("*.css"):
        for ref in re.findall(r"url\('?(/static/[^')]+)", css.read_text()):
            assert (WEB / ref.lstrip("/")).exists(), f"{css.name}: {ref}"
    proteins = json.loads((STATIC / "data" / "proteins.json").read_text())
    assert proteins
    for p in proteins:
        assert (STATIC / p["file"]).exists(), p["file"]
        assert set(p["site"]) == {"x", "y", "z"} and p["box"]


def test_the_ui_examples_match_the_servers_validated_examples() -> None:
    """Proteins tagged backendExample take their receptor and site from /api/examples at runtime."""
    ids = {e["id"] for e in server.EXAMPLES}
    proteins = json.loads((STATIC / "data" / "proteins.json").read_text())
    tagged = {p["backendExample"] for p in proteins if p.get("backendExample")}
    assert tagged and tagged <= ids


# --------------------------------------------------------------------------- #
#  packaging
# --------------------------------------------------------------------------- #

def test_every_shipped_web_file_is_in_the_wheel() -> None:
    cfg = tomllib.loads((REPO / "pyproject.toml").read_text())
    globs = cfg["tool"]["setuptools"]["package-data"]["hybridock_pep"]
    pkg = WEB.parent
    missing = []
    for f in WEB.rglob("*"):
        # code and docs are not data files
        if not f.is_file() or f.suffix in {".pyc", ".py", ".md"} or "__pycache__" in f.parts:
            continue
        rel = f.relative_to(pkg).as_posix()
        # setuptools globs: `*` never crosses a directory, so compare part by part
        if not any(len(PurePosixPath(g).parts) == len(PurePosixPath(rel).parts) and PurePosixPath(rel).match(g)
                   for g in globs):
            missing.append(rel)
    assert not missing, f"not covered by [tool.setuptools.package-data]: {missing}"
