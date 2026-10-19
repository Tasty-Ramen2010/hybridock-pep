# Security policy

HybriDock-Pep is a research tool maintained by a student team. We take security reports seriously
but cannot promise response times; we aim to acknowledge a report within a week.

## Supported versions

Only the latest release and the current `master` branch receive fixes.

## Reporting a vulnerability

Please do **not** put exploit details in a public issue.

1. Open a new issue titled **"Security: private contact request"** with *no* technical details in it —
   just that you have found a security problem and which part of the project it concerns.
2. We will reply on the issue with a private way to send the details.

(GitHub's private vulnerability reporting is not enabled on this repository yet. When the
repository owner turns it on in *Settings → Code security*, it becomes the preferred route and
this file will be updated.)

Include what you ran, what you expected, what happened, and the version or commit.

## What is in scope

HybriDock-Pep runs on your own computer, so the realistic risks are these:

- **The web UI (`hybridock-pep serve`).** By design it trusts whoever can reach it: a request may
  name any file on the computer as the receptor and any folder as the output. That is safe only
  because it binds to `127.0.0.1`. **Do not forward its port or put it behind a tunnel or proxy
  without `--public`.**
- **Public mode (`hybridock-pep serve --public`).** This is for sharing a machine with a hosted page
  ([docs/SHARING.md](docs/SHARING.md)). It rebuilds every request from an allow-list, never accepts
  file paths from visitors, caps run size and upload size, rate-limits per visitor address, deletes
  finished runs after 24 hours and keeps the computer's paths out of responses (see
  `src/hybridock_pep/web/public.py`). Bypasses of any of those are in scope and are the reports we
  most want.
- **Parsing untrusted structure files.** A crafted PDB or mmCIF that crashes, hangs or runs code in
  the parsing and preparation steps.
- **Command construction.** Anything that lets input reach a shell or a subprocess argument it
  should not control. Runs are launched as subprocesses of the CLI.
- **The install scripts** (`install.sh`, `install.bat`, `scripts/colab_setup.sh`) fetching or
  executing something other than what they say.
- **Shipped artifacts.** Model checkpoints and data files that do not match
  [ARTIFACTS.sha256](ARTIFACTS.sha256) (`python scripts/verify_artifacts.py`). The `.pt` and `.joblib`
  files are loaded with `torch`/`joblib`, which can execute code when a malicious file is loaded —
  only load artifacts from this repository's releases, and verify them first.

## What is out of scope

- Scientific disagreement or a wrong number in [RESULTS.md](RESULTS.md) — open an ordinary issue.
- Vulnerabilities in third-party tools we depend on (RAPiDock, AutoDock Vina, OpenMM, …); report
  those upstream. If our usage makes one exploitable, that part is ours.
- Denial of service by a user who already controls the machine.

## Safe use

- Run the tool as an ordinary user, not as root.
- Treat structure files from unknown sources the way you would treat any file you parse.
- Verify what you downloaded: `python scripts/verify_artifacts.py`.
