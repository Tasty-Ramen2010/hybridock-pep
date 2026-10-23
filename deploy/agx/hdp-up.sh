#!/bin/sh
# Brings the HybriDock-Pep demo up on the AGX and keeps it up. Idempotent: run it as often as you like. Started at boot from
# the board's start.sh. It is a small supervisor (one copy only):
#   * mounts the spare 14 GB partition (where the app lives) into the Ubuntu chroot
#   * starts the web server inside the chroot, and starts it again if it dies
#   * makes sure the public address (Tailscale Funnel) is switched on
#   * puts the server above the language model for the kernel's out-of-memory killer
# Pause it:  touch /mnt/persistent/data/agx-setup/hdp.disabled      Resume: rm that file.
D=/mnt/persistent/data/agx-setup
R=/mnt/persistent/data/ubuntu-focal
HDP_DIR=/home/lat/devspace/hdp
PORT=8765
TS="/etc/orin-persist/tailscale/tailscale --socket=/run/tailscale/sock"
LOG=/mnt/persistent/data/hdp-up.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

exec 8>/tmp/hdp-up.lock
flock -n 8 || exit 0
say "supervisor started (pid $$)"

# the board boots with its clock in 1971; HTTPS certificates need the real date
until [ "$(date +%Y)" -ge 2026 ]; do sleep 5; done

# keep the clock right: the board's own sync only fires when the YEAR is wrong, and a clock that is merely minutes slow breaks HTTPS
sync_clock() {
  T=$(curl -sI -m 8 http://www.google.com 2>/dev/null | sed -n 's/^[Dd]ate: //p' | tr -d '\r')
  [ -n "$T" ] || return 0
  want=$(date -u -d "$T" +%s 2>/dev/null) || return 0
  drift=$(( want - $(date +%s) )); [ "$drift" -lt 0 ] && drift=$(( -drift ))
  if [ "$drift" -gt 20 ]; then date -u -s "$T" >/dev/null 2>&1 && say "clock was off by ${drift}s, set from the web"; fi
}

funnel_ok=0
last_funnel=0
last_clock=0
while true; do
  if [ -e "$D/hdp.disabled" ]; then sleep 15; continue; fi
  [ -d "$HDP_DIR/hybridock-pep" ] || { sleep 30; continue; }

  now=$(date +%s)
  if [ $((now - last_clock)) -gt 600 ]; then last_clock=$now; sync_clock; fi

  # chroot + app disk
  sh $D/ub-enter.sh true >/dev/null 2>&1
  mkdir -p "$R/mnt/hdp"
  mountpoint -q "$R/mnt/hdp" || mount --bind "$HDP_DIR" "$R/mnt/hdp"

  # the server
  if ! pgrep -f "hybridock-pep serve --public" >/dev/null; then
    say "starting the server"
    (nohup sh $D/ub-enter.sh "exec bash /mnt/hdp/hybridock-pep/deploy/agx/hdp-serve.sh" >> /mnt/persistent/data/hdp-serve.log 2>&1 </dev/null &)
    sleep 8
  fi
  for p in $(pgrep -f "hybridock-pep serve --public"); do echo -500 > /proc/$p/oom_score_adj 2>/dev/null; done

  # the public address (Tailscale Funnel proxies https://<this board>.ts.net to the port on this machine)
  now=$(date +%s)
  if [ $((now - last_funnel)) -gt 300 ]; then
    last_funnel=$now
    if $TS funnel status 2>/dev/null | grep -q "127.0.0.1:$PORT"; then
      funnel_ok=1
    else
      out=$($TS funnel --bg "$PORT" 2>&1)
      say "funnel: $(echo "$out" | tr '\n' ' ' | cut -c1-300)"
      funnel_ok=0
    fi
  fi
  sleep 20
done
