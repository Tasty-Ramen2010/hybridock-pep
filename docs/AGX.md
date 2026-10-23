# The live demo computer (NVIDIA AGX Orin)

The live demo runs on the team's AGX Orin board. This page is for the people who look after it.

```
judge's browser ──► https://agx-orin.tail705d7c.ts.net  (Tailscale Funnel, https)
                        └─► 127.0.0.1:8765  hybridock-pep serve --public   (inside the board's Ubuntu 20.04 chroot)
                                └─► hybridock-pep dock ...  (one prediction at a time, 5 / 10 / 25 poses)
```

## What is installed where

| What | Where |
| --- | --- |
| The app, both conda environments, weights | `/home/lat/devspace/hdp/` (the spare 14 GB partition; `/mnt/hdp` inside the chroot) |
| Supervisor (starts everything, keeps it up) | `/mnt/persistent/data/agx-setup/hdp-up.sh`, started at boot from `start.sh` |
| Server launcher and its settings | `deploy/agx/hdp-serve.sh` (a restart loop) and `deploy/agx/hdp-env.sh` (the settings, re-read at every restart; `agx-setup/hdp.conf` overrides them) |
| Language-model hook | `deploy/agx/hdp-job-hook` |
| Logs | `/mnt/persistent/data/hdp-up.log`, `hdp-serve.log`, `hdp-hook.log` |

## A prediction comes first

The same board runs the Qwen language model, which holds about 26 of the 28 GB of memory. When a visitor starts a run the server
calls `hdp-job-hook start`, which stops the model (using the board's own `watch.paused` and `serve.disabled` switches) and
frees its memory, and the run starts at scheduling priority -10 (`HYBRIDOCK_NICE`). After 15 minutes with nothing running
(`HYBRIDOCK_IDLE_SECONDS`) it calls `hdp-job-hook idle`, which lets the board's watcher bring the model back within about a minute.
The model is only restarted if the hook was the one that stopped it. To keep it off while the demo is being judged:

```bash
ssh agx 'touch /mnt/persistent/data/agx-setup/hdp-keep-llm-off'     # rm that file to undo
```

## Everyday commands

```bash
ssh agx 'tail -n 20 /mnt/persistent/data/hdp-up.log'               # supervisor
ssh agx 'tail -n 20 /mnt/persistent/data/hdp-serve.log'            # web server
ssh agx 'curl -s http://127.0.0.1:8765/api/env | head -c 300'      # is it answering?
ssh agx 'touch /mnt/persistent/data/agx-setup/hdp.disabled'        # pause hosting (rm to resume)
```

## Updating the app

```bash
ssh agx 'sh /mnt/persistent/data/agx-setup/ub-enter.sh "cd /mnt/hdp/hybridock-pep && git pull --ff-only && git submodule update --init --recursive"'
ssh agx 'pkill -f "hybridock-pep serve --public"'                  # it comes back in a few seconds with the new code
```

## Re-measuring the times and the preloaded results

The "about N minutes" on the page are measured, not estimated. Re-measure after any change to the hardware or software:

```bash
ssh agx 'sh /mnt/persistent/data/agx-setup/ub-enter.sh "cd /mnt/hdp/hybridock-pep && PATH=/mnt/hdp/miniforge3/envs/score-env/bin:\$PATH \
  python3 scripts/benchmark_timing.py --machine \"NVIDIA Jetson AGX Orin\" --out src/hybridock_pep/web/static/data/timing.json"'
```

`scripts/precompute_catalog.py` re-runs the carousel's peptide panel and rewrites `static/data/catalog.json`.

## Public HTTPS address

`https://agx-orin.tail705d7c.ts.net` is a Tailscale Funnel: Tailscale's servers answer on the internet and forward to port 8765 on
the board, and the board holds the certificate. Two changes were made to the board's own scripts for this, each with a backup next
to it: `start.sh` starts `hdp-up.sh` at boot (`start.sh.bak-pre-hdp`), and `ts-main.sh` passes `--statedir` to `tailscaled`
(`ts-main.sh.bak-pre-hdp`), because without a state directory Tailscale cannot store HTTPS certificates ("no TailscaleVarRoot").
The supervisor re-checks the Funnel every five minutes and puts the clock right whenever it is more than 20 seconds out
(a clock that is minutes slow breaks HTTPS).

## Never

Only power-cycle the board; a warm reboot leaves the GPU dead (see the notes in `serve.sh`).
