#!/usr/bin/env python3
"""Let the hosted website run real predictions on THIS computer, until you stop it.

    python3 scripts/share_my_computer.py            # start sharing (keeps running; Ctrl-C to stop)
    python3 scripts/share_my_computer.py --stop     # stop a copy running in the background

What it does, and keeps doing every 30 seconds:
  1. starts `hybridock-pep serve --public` (the locked-down server, see docs/SHARING.md) on a local port;
  2. opens a Cloudflare quick tunnel to it, which gives this computer a public https address;
  3. publishes that address in a one-line file (branch `live-link` of the GitHub repo) that the hosted page reads;
  4. keeps the Mac awake, and restarts the server or the tunnel (and republishes the address) if either dies.

When it stops, it publishes "no address" so the website goes back to its demo on its own.
Needs: the score-env Python (this script is run with it), `cloudflared` (brew install cloudflared), and git push rights.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "runs" / "public"
PIDFILE = STATE / "share.pid"
STATUS = STATE / "share.json"
LOG = STATE / "share.log"
BRANCH = "live-link"
URL_RE = re.compile(r"https://[a-z0-9-]+(?:\.[a-z0-9-]+)*\.trycloudflare\.com")


def log(msg: str) -> None:
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    try:
        STATE.mkdir(parents=True, exist_ok=True)
        with LOG.open("a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def get(url: str, origin: str | None = None, timeout: float = 10) -> dict | None:
    req = urllib.request.Request(url, headers={"Origin": origin} if origin else {})
    try:
        # no system proxy: a local proxy that does not exempt 127.0.0.1 would answer instead of the server
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if "127.0.0.1" in url else urllib.request.build_opener()
        with opener.open(req, timeout=timeout) as res:
            return json.loads(res.read())
    except Exception:  # noqa: BLE001 - any failure means "not answering"
        return None


def publish(url: str | None) -> bool:
    """Write the address to the `live-link` branch (a branch holding only backend.json), replacing what was there."""
    try:
        remote = subprocess.run(["git", "-C", str(ROOT), "remote", "get-url", "origin"], capture_output=True, text=True, check=True).stdout.strip()
        name = subprocess.run(["git", "-C", str(ROOT), "config", "user.name"], capture_output=True, text=True).stdout.strip() or "hybridock-share"
        email = subprocess.run(["git", "-C", str(ROOT), "config", "user.email"], capture_output=True, text=True).stdout.strip() or "share@hybridock.invalid"
        with tempfile.TemporaryDirectory() as tmp:
            body = {"url": url, "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            (Path(tmp) / "backend.json").write_text(json.dumps(body) + "\n")
            run = lambda *a: subprocess.run(["git", "-C", tmp, "-c", f"user.name={name}", "-c", f"user.email={email}", *a], check=True, capture_output=True, text=True)  # noqa: E731
            run("init", "-q", "-b", BRANCH)
            run("add", "backend.json")
            run("commit", "-q", "-m", f"live link: {url or 'offline'}")
            run("push", "-q", "--force", remote, f"{BRANCH}:{BRANCH}")
        log(f"published {url or 'offline'} to branch {BRANCH}")
        return True
    except (subprocess.CalledProcessError, OSError) as exc:
        log(f"could not publish ({getattr(exc, 'stderr', exc)})")
        return False


class Sharing:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.server: subprocess.Popen | None = None
        self.tunnel: subprocess.Popen | None = None
        self.awake: subprocess.Popen | None = None
        self.url: str | None = None
        self.stopping = False
        self.local = f"http://127.0.0.1:{args.port}"

    # ---- the server ----
    def start_server(self) -> None:
        env = {**os.environ, "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")}
        cmd = [sys.executable, "-c", "from hybridock_pep.cli import main; main()", "serve", "--public", "--no-browser", "--port", str(self.args.port),
               "--max-samples", str(self.args.max_samples)]
        for origin in self.args.origin:
            cmd += ["--allow-origin", origin]
        STATE.mkdir(parents=True, exist_ok=True)
        self.server = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=LOG.open("a"), stderr=subprocess.STDOUT)
        for _ in range(60):
            env_json = get(f"{self.local}/api/env")
            if env_json and env_json.get("public"):
                log(f"server up on {self.local} (public mode, {self.args.max_samples} poses max)")
                return
            if self.server.poll() is not None:
                break
            time.sleep(1)
        raise RuntimeError("the server did not start; see " + str(LOG))

    # ---- the tunnel ----
    def start_tunnel(self) -> None:
        exe = shutil.which("cloudflared")
        if not exe:
            raise RuntimeError("cloudflared is not installed. Run: brew install cloudflared")
        self.tunnel = subprocess.Popen([exe, "tunnel", "--no-autoupdate", "--url", self.local], stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
        found: list[str] = []

        def read() -> None:
            assert self.tunnel and self.tunnel.stdout
            for line in self.tunnel.stdout:
                m = URL_RE.search(line)
                if m and not found:
                    found.append(m.group(0))

        threading.Thread(target=read, daemon=True).start()
        for _ in range(90):
            if found:
                break
            if self.tunnel.poll() is not None:
                raise RuntimeError("the tunnel stopped while starting")
            time.sleep(1)
        if not found:
            raise RuntimeError("the tunnel gave no address within 90 seconds")
        self.url = found[0]
        # the address can take a few seconds before the outside world resolves it
        for _ in range(45):
            env_json = get(f"{self.url}/api/env", origin=self.args.origin[0] if self.args.origin else None)
            if env_json and env_json.get("public"):
                log(f"tunnel up: {self.url}")
                return
            time.sleep(2)
        raise RuntimeError(f"the tunnel address {self.url} never answered")

    def stop_children(self, tunnel_only: bool = False) -> None:
        for name in ("tunnel",) if tunnel_only else ("tunnel", "server", "awake"):
            proc = getattr(self, name)
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(10)
                except subprocess.TimeoutExpired:
                    proc.kill()
            setattr(self, name, None)

    # ---- the loop ----
    def run(self) -> int:
        STATE.mkdir(parents=True, exist_ok=True)
        PIDFILE.write_text(str(os.getpid()))
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: setattr(self, "stopping", True))
        if sys.platform == "darwin" and shutil.which("caffeinate"):
            self.awake = subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(os.getpid())])  # no idle sleep while this runs
        try:
            self.start_server()
            self.start_tunnel()
            if not self.args.no_publish:
                publish(self.url)
            self.write_status("sharing")
            log(f"SHARING. Visitors of {', '.join(self.args.origin) or 'the tunnel address'} now run on this computer. Ctrl-C to stop.")
            misses = 0
            while not self.stopping:
                time.sleep(30)
                if self.stopping:
                    break
                if self.server is None or self.server.poll() is not None:
                    log("the server died; restarting it")
                    self.start_server()
                ok = get(f"{self.url}/api/env", origin=self.args.origin[0] if self.args.origin else None)
                misses = 0 if ok and ok.get("public") else misses + 1
                if misses >= 2 or (self.tunnel and self.tunnel.poll() is not None):
                    log("the tunnel stopped answering; opening a new one")
                    self.stop_children(tunnel_only=True)
                    self.start_tunnel()
                    if not self.args.no_publish:
                        publish(self.url)
                    self.write_status("sharing")
                    misses = 0
        except RuntimeError as exc:
            log(f"ERROR: {exc}")
            return 1
        finally:
            log("stopping")
            self.write_status("stopped")
            if not self.args.no_publish:
                publish(None)
            self.stop_children()
            PIDFILE.unlink(missing_ok=True)
        return 0

    def write_status(self, state: str) -> None:
        STATUS.write_text(json.dumps({"state": state, "url": self.url if state == "sharing" else None,
                                      "updated": datetime.now().isoformat(timespec="seconds")}))


def stop_running() -> int:
    try:
        pid = int(PIDFILE.read_text())
        os.kill(pid, signal.SIGTERM)
    except (OSError, ValueError):
        print("Nothing is being shared from this folder.")
        return 0
    for _ in range(40):
        time.sleep(1)
        try:
            os.kill(pid, 0)
        except OSError:
            print("Stopped. The website is back to its demo.")
            return 0
    print("Asked it to stop, but it is still shutting down. Check runs/public/share.log.")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--origin", action="append", default=None, help="a web page allowed to use this computer (default: the GitHub Pages site)")
    ap.add_argument("--max-samples", type=int, default=50, help="most poses one visitor's run may ask for (default 50)")
    ap.add_argument("--no-publish", action="store_true", help="do not write the address to GitHub (for testing)")
    ap.add_argument("--stop", action="store_true", help="stop a copy that is already running")
    args = ap.parse_args()
    if args.stop:
        return stop_running()
    args.origin = args.origin or ["https://tasty-ramen2010.github.io"]
    if PIDFILE.exists():
        try:
            os.kill(int(PIDFILE.read_text()), 0)
            print("Already sharing. Stop it first:  python3 scripts/share_my_computer.py --stop")
            return 1
        except (OSError, ValueError):
            PIDFILE.unlink(missing_ok=True)
    return Sharing(args).run()


if __name__ == "__main__":
    sys.exit(main())
