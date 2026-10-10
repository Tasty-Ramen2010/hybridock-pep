#!/usr/bin/env bash
# macOS: double-click this file in Finder to start HybriDock-Pep (it opens in your browser).
cd "$(dirname "$0")" || exit 1
./launch_web.sh
echo
read -n 1 -s -r -p "HybriDock-Pep has stopped. Press any key to close this window."
