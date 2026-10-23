#!/bin/bash
# The HybriDock-Pep demo server, as the AGX runs it. Runs INSIDE the Ubuntu chroot (see hdp-up.sh, which starts it).
# Restarts the server if it ever exits. The settings live in hdp-env.sh (and /mnt/data/agx-setup/hdp.conf), which are read again at
# every restart: to apply a change, just stop the server (`pkill -f "hybridock-pep serve --public"`); it comes back with the new settings.
HDP=/mnt/hdp/hybridock-pep
cd "$HDP" || exit 1
while true; do
  # a fresh subshell each time, so a setting removed from hdp-env.sh / hdp.conf does not linger from the previous start
  (
    . "$HDP/deploy/agx/hdp-env.sh"
    echo "$(date '+%F %T') starting hybridock-pep serve on 127.0.0.1:$HDP_PORT (idle hook after ${HYBRIDOCK_IDLE_SECONDS}s)"
    exec hybridock-pep serve --public --no-browser --host 127.0.0.1 --port "$HDP_PORT" \
      --allow-origin https://tasty-ramen2010.github.io --max-samples 25
  )
  echo "$(date '+%F %T') server exited ($?); restarting in 3 s"
  sleep 3
done
