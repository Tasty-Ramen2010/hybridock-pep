#!/usr/bin/env python3
"""Assemble a static, demo-only copy of the browser UI that any free static host can serve.

    python3 scripts/build_pages.py [--out DIR]        # default: _site/

The copy has no backend: it flags itself with ``window.HYBRIDOCK_STATIC`` so the app goes straight to
its Demo adapter (every result is simulated and labelled "Demo"). It uses relative URLs, so it works at
the root of a domain or under a sub-path such as ``https://<user>.github.io/hybridock-pep/``.

Real predictions need the real tool: ``hybridock-pep serve`` (see src/hybridock_pep/web/README.md).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "src" / "hybridock_pep" / "web"
STATIC = WEB / "static"
# The only parts of static/ the app needs. (studio.html and the older studio.* files stay out.)
COPY = ("dist", "fonts", "data")


def build(out: Path) -> int:
    # Never publish a stale bundle: the generated files must match their sources.
    check = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_web.py"), "--check"], check=False)
    if check.returncode != 0:
        print("The bundle in static/dist is stale. Run: python3 scripts/build_web.py", file=sys.stderr)
        return 1

    if out.exists():
        shutil.rmtree(out)
    for name in COPY:
        shutil.copytree(STATIC / name, out / "static" / name)

    html = (WEB / "index.html").read_text(encoding="utf-8")
    html = html.replace('"/static/', '"static/')  # relative, so a sub-path works
    flag = "<script>window.HYBRIDOCK_STATIC = true; /* no server behind this copy: demo mode only */</script>\n"
    html = html.replace("  <!-- Apply the saved theme/mode", flag + "  <!-- Apply the saved theme/mode", 1)
    if "HYBRIDOCK_STATIC" not in html or '"/static/' in html:
        print("index.html no longer matches what build_pages.py expects.", file=sys.stderr)
        return 1
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")  # GitHub Pages: serve files as they are

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"built {out} ({total / 1e6:.1f} MB, {sum(1 for f in out.rglob('*') if f.is_file())} files)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / "_site", help="output folder (default: _site)")
    return build(parser.parse_args().out.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
