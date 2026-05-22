"""Entry point for the `hybridock-pep-ui` console script."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    app_path = Path(__file__).parent / "app.py"
    port = "8501"
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", port,
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    print(f"Starting HybriDock-Pep UI → http://localhost:{port}")
    print("Forward this port in VS Code (PORTS panel) for a browser-accessible URL.")
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
