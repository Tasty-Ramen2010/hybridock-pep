"""Build one of RAPiDock's import-time lookup tables, without importing torch.

RAPiDock's utils/so3.py and utils/torus.py compute their tables in module-level
numpy code the first time they are imported (~8 min for so3, longer for torus),
saving ".*.npy" files into the current directory. colab_setup.sh runs this once
per module, in parallel with the PyTorch install, so the first dock does not pay.

    python3 prewarm_rapidock_tables.py <rapidock_dir> so3|torus

Run with cwd = <rapidock_dir>: the files land in, and are looked up from, cwd.
torch is stubbed because so3.py imports it but its table code is pure numpy, and
the torch wheel is not installed yet when this runs.
"""
import importlib
import sys
import time
from unittest import mock

rapidock_dir, which = sys.argv[1], sys.argv[2]
sys.modules.setdefault("torch", mock.MagicMock())
sys.path.insert(0, rapidock_dir)
started = time.time()
importlib.import_module(f"utils.{which}")
print(f"{which} tables ready in {time.time() - started:.0f}s", flush=True)
