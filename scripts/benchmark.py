#!/usr/bin/env python3
"""Thin re-export shim.

The implementation moved to ``hybridock_pep._vendor.benchmark`` so
``hybridock-pep benchmark`` works from a real (non-editable) wheel install too
(it previously sys.path-hacked its way into this file, which doesn't exist in
an installed wheel). This shim keeps ``python scripts/benchmark.py`` working
unchanged for anyone invoking it directly from a source checkout.
"""
from __future__ import annotations

from hybridock_pep._vendor.benchmark import *  # noqa: F401,F403
from hybridock_pep._vendor.benchmark import main

if __name__ == "__main__":
    main()
