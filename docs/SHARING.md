# Letting the hosted website use your computer

The website at `https://tasty-ramen2010.github.io/hybridock-pep/` is a demo: its numbers are simulated. While you run
`scripts/share_my_computer.py`, the same page becomes the real app: every prediction it starts runs on **your** computer, and
the page shows the real result. When you stop the script, the page goes back to its demo by itself.

```bash
conda activate score-env
python3 scripts/share_my_computer.py          # leave this window open
python3 scripts/share_my_computer.py --stop   # from any other window, or just press Ctrl-C
```

You need `cloudflared` (`brew install cloudflared`) and permission to push to the GitHub repo. Keep the computer plugged in;
the script keeps it from sleeping, but closing the lid or shutting down ends the sharing. A restart of the computer also
ends it, so start the script again afterwards.

## How it works

```
judge's browser ──► github.io page ──► https://<random>.trycloudflare.com ──► your computer, port 8765
                         ▲                                                   `hybridock-pep serve --public`
                         └── reads backend.json on the `live-link` branch (the current address)
```

1. `serve --public` is the ordinary studio server behind a policy (`src/hybridock_pep/web/public.py`).
2. A Cloudflare quick tunnel gives it a public https address. No account is needed, and your computer opens no port.
3. The address changes every time the tunnel starts, so the script writes it to `backend.json` on the `live-link` branch. The page
   reads that file when it loads. If the file says "no address", or the computer does not answer, the page is the demo and says so.

## What a visitor can and cannot do

The normal `hybridock-pep serve` trusts whoever talks to it: a request may name any file on the computer as the receptor, pick any
folder as the output, and stop anyone's run. That is fine on `127.0.0.1` and not on the internet, so public mode changes it:

| | Normal `serve` | `serve --public` |
| --- | --- | --- |
| Files | Any path on the computer | Only built-in proteins and files uploaded through the page (the page sends `example:…` or `upload:…`, never a path) |
| Settings | Every field, including output folder, input-poses folder and calibration file | A short allow-list. The rest are ignored |
| Size of a run | Anything | Up to 50 poses (25 for a comparison), peptides of 3 to 25 letters, no whole-protein search |
| Uploads | Any size | PDB text up to 6 MB, stored under a name the server chooses |
| Other people's runs | Listed, and anyone can stop them | Not listed. A run is reachable only by its random id |
| Which pages may call it | Any page on the computer | The hosted page (and the server's own page) |
| What comes back | Absolute paths and the computer's user name appear in commands and logs | Folder names are replaced by `~` and `.` |
| Result files | Anything under the run folder | Only the files the result screen offers |
| Rate | Unlimited | 8 runs and 40 uploads per visitor per 10 minutes |
| Housekeeping | Nothing is deleted | Runs and uploads older than 24 hours are deleted |

It still does one prediction at a time. A second visitor who presses Run while one is going is told the computer is busy.

`tests/test_web_public.py` plays the stranger: it tries file paths, `..`, other pages, huge uploads and the like, and checks each is
refused. Run it with `pytest tests/test_web_public.py` after changing anything in `web/`.

## What it does not protect against

* A visitor can use up your computer's time for a few minutes at a time. That is the point, and the limits keep it short.
* Anyone with the tunnel address can talk to the server directly, not just through the page. The policy above applies to them too.
* Predictions and uploaded structures are visible to you in `runs/public/` for 24 hours.
* Cloudflare carries the traffic, so treat anything a visitor sends as visible to Cloudflare. Do not share it for private data.

## Turning it off

`python3 scripts/share_my_computer.py --stop`, or Ctrl-C in its window. It publishes "no address" first, so the page goes back to
the demo within a few minutes (GitHub caches the file briefly). To remove the link for good, delete the `live-link` branch.
