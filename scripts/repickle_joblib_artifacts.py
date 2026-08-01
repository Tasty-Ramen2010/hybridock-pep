#!/usr/bin/env python3
"""scripts/repickle_joblib_artifacts.py — fix legacy-numpy joblib pickles in data/.

The shipped data/*.joblib artifacts were pickled with an old numpy whose
Generator/BitGenerator __reduce__ output isn't forward-compatible with modern
numpy: unpickling raises

    ValueError: <class 'numpy.random._pcg64.PCG64'> is not a known BitGenerator module.

(numpy.random._pickle.__bit_generator_ctor received the BitGenerator *class*
instead of its name string), and even after coercing that, the legacy state
tuple/dict shape no longer matches what BitGenerator.__setstate__ expects.

The affected object is always an incidental RNG (e.g. a fitted estimator's
`random_state`, or a bootstrap/CV splitter that got serialized alongside it)
that scikit-learn does not consult during `.predict()` — a fitted model's
predictions are deterministic. So instead of trying to faithfully reconstruct
the old RNG's exact draw sequence, this script patches unpickling to accept a
fresh, validly-constructed BitGenerator in its place, loads each artifact once,
and re-dumps it with the current joblib/numpy so future loads need no patching
at all (this script is a one-time migration, not a runtime dependency).

Usage:
    python3 scripts/repickle_joblib_artifacts.py [--check-only] [file ...]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import joblib
import numpy.random._pickle as np_pickle

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _install_compat_shim() -> None:
    """Make legacy-numpy RNG pickles loadable.

    These artifacts were pickled by a numpy old enough that its Generator/
    BitGenerator/RandomState __reduce__ output doesn't match what current
    numpy's unpickling constructors expect (wrong argument shape: a class or
    an already-reconstructed object where a name string is expected; a legacy
    state tuple/dict where the new state shape is expected). None of this
    matters for a fitted scikit-learn estimator doing `.predict()` — the RNG
    is never consulted after fitting — so every constructor here is made
    maximally tolerant: try the normal path, and on ANY failure fall back to
    a fresh, validly-constructed default instead of propagating the error.

    BitGenerator instances are C-extension objects: you can't monkeypatch a
    method onto an existing instance (`AttributeError: ... is read-only`) or
    onto the base class (`TypeError: ... immutable type`), but subclassing
    from Python *is* supported (numpy documents BitGenerator as subclassable),
    so __setstate__ tolerance is added via a per-class compat subclass.
    """
    import numpy as np

    original_bg_ctor = np_pickle.__bit_generator_ctor
    original_gen_ctor = np_pickle.__generator_ctor
    original_rs_ctor = np_pickle.__randomstate_ctor
    compat_classes: dict[type, type] = {}

    def _tolerant(cls: type) -> type:
        if cls not in compat_classes:
            def __setstate__(self, state, _orig=cls.__setstate__):
                try:
                    _orig(self, state)
                except (TypeError, ValueError):
                    pass  # incompatible legacy state; fresh default state is fine
            compat_classes[cls] = type(f"Compat{cls.__name__}", (cls,), {
                "__setstate__": __setstate__,
            })
        return compat_classes[cls]

    def patched_bg_ctor(bit_generator_name="MT19937"):
        try:
            if isinstance(bit_generator_name, type):
                bit_generator_name = bit_generator_name.__name__
            elif not isinstance(bit_generator_name, str):
                bit_generator_name = type(bit_generator_name).__name__
            real_cls = type(original_bg_ctor(bit_generator_name))
        except Exception:
            real_cls = np.random.PCG64  # unrecognized legacy form; sane default
        return _tolerant(real_cls)()

    def patched_gen_ctor(bit_generator_name="PCG64"):
        try:
            return np.random.Generator(patched_bg_ctor(bit_generator_name))
        except Exception:
            return np.random.Generator(np.random.PCG64())

    def patched_rs_ctor(bit_generator_name="MT19937"):
        try:
            return np.random.RandomState(patched_bg_ctor(bit_generator_name))
        except Exception:
            return np.random.RandomState()

    np_pickle.__bit_generator_ctor = patched_bg_ctor
    np_pickle.__generator_ctor = patched_gen_ctor
    np_pickle.__randomstate_ctor = patched_rs_ctor


def _dump_with_normalized_rngs(obj, path: Path) -> None:
    """joblib.dump(obj, path), but replacing any embedded RNG object (whether
    a plain numpy one or one of our locally-defined Compat* subclasses from
    _install_compat_shim) with a fresh instance of the real, plain, always-
    importable numpy class. Needed because:

    1. Our Compat* subclasses are defined inside a function, so pickle can't
       resolve them by name ("not found as __main__...") when dumping.
    2. Even a plain BitGenerator's own __reduce__ still points at the numpy
       module's __bit_generator_ctor — which we've monkeypatched in-process
       for the *load* side, and must not let leak into what gets dumped.

    Both are moot for correctness (§ the module docstring: the RNG's value
    is never consulted by `.predict()`), so we don't try to preserve state —
    just swap in a fresh, real, plain instance of the same broad kind.
    """
    import numpy as np
    from joblib.numpy_pickle import NumpyPickler

    def reducer_override(self, obj):
        if isinstance(obj, np.random.Generator):
            return (np.random.Generator, (np.random.PCG64(),))
        if isinstance(obj, np.random.RandomState):
            return (np.random.RandomState, ())
        if isinstance(obj, np.random.BitGenerator):
            real_cls = obj.__class__
            while real_cls.__module__.startswith("numpy.random") is False:
                real_cls = real_cls.__base__  # unwrap our Compat* subclass
            return (real_cls, ())
        return NotImplemented

    # reducer_override is an optional hook pickle looks up via getattr(), not a
    # real base-class method — NumpyPickler doesn't define one by default.
    had_original = "reducer_override" in NumpyPickler.__dict__
    original_method = NumpyPickler.__dict__.get("reducer_override")
    NumpyPickler.reducer_override = reducer_override
    try:
        joblib.dump(obj, path)
    finally:
        if had_original:
            NumpyPickler.reducer_override = original_method
        else:
            del NumpyPickler.reducer_override


def _verify_loads_cleanly(path: Path) -> bool:
    """Load path in a fresh, unpatched subprocess — proves the fix is real,
    not an artifact of the patch still being active in this process."""
    result = subprocess.run(
        [sys.executable, "-c", f"import joblib; joblib.load({str(path)!r})"],
        capture_output=True, text=True, timeout=60,
    )
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files", nargs="*",
        help="Specific .joblib files to fix (default: all of data/*.joblib)",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Report which files are affected without modifying anything",
    )
    args = parser.parse_args()

    targets = [Path(f) for f in args.files] if args.files else sorted(
        (_REPO_ROOT / "data").glob("*.joblib")
    )

    _install_compat_shim()

    n_fixed = n_already_ok = n_failed = 0
    for path in targets:
        if _verify_loads_cleanly(path):
            print(f"  [OK]      {path.name} — already loads cleanly")
            n_already_ok += 1
            continue

        if args.check_only:
            print(f"  [AFFECTED] {path.name} — needs re-pickling")
            continue

        try:
            obj = joblib.load(path)
        except Exception as exc:  # noqa: BLE001 - report and move on
            print(f"  [FAIL]    {path.name} — could not load even with compat shim: {exc}")
            n_failed += 1
            continue

        backup = path.with_suffix(path.suffix + ".bak")
        backup.write_bytes(path.read_bytes())
        _dump_with_normalized_rngs(obj, path)

        if _verify_loads_cleanly(path):
            backup.unlink()
            print(f"  [FIXED]   {path.name} — re-pickled, verified loads cleanly")
            n_fixed += 1
        else:
            path.write_bytes(backup.read_bytes())
            backup.unlink()
            print(f"  [FAIL]    {path.name} — re-pickle didn't fix it, reverted")
            n_failed += 1

    print(f"\n{n_fixed} fixed, {n_already_ok} already OK, {n_failed} failed "
          f"(of {len(targets)} checked)")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
