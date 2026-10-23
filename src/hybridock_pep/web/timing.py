"""Run-time estimates from a measured grid.

A prediction's time depends on how many poses are sampled AND on how long the peptide is, and the two interact: from 13 residues
the longer-peptide model is used, which costs several times more per pose. So the estimate is not one multiplier on a base time.
It is read from a table the benchmark measured (``scripts/benchmark_timing.py``): whole-run seconds for each peptide length and
each pose count, joined by straight lines between the measured points.

    grid = {8: {5: 58.0, 10: 65.4, 25: 98.0}, 12: {5: 67.6, 10: 81.9, 25: 134.3}, ...}
    estimate(grid, length=14, poses=10)

Past the measured ends a line carries on with its last slope, and an estimate is never allowed under half of the smallest
measurement. The same rule is written in ``static/js/app/config.mjs`` (``timingFor``); ``tests`` keep the two in step.
"""

from __future__ import annotations

from typing import Mapping

Grid = Mapping[int, Mapping[int, float]]


def _interp(points: list[tuple[float, float]], x: float) -> float:
    """Straight lines through ``points`` (sorted by x); the end segments continue past the ends."""
    if len(points) == 1:
        return points[0][1]
    if x <= points[0][0]:
        (x0, y0), (x1, y1) = points[0], points[1]
    elif x >= points[-1][0]:
        (x0, y0), (x1, y1) = points[-2], points[-1]
    else:
        i = next(k for k in range(len(points) - 1) if points[k][0] <= x <= points[k + 1][0])
        (x0, y0), (x1, y1) = points[i], points[i + 1]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def estimate(grid: Grid, length: int, poses: int) -> float:
    """Seconds for a whole prediction of a ``length``-letter peptide at ``poses`` poses."""
    rows = sorted((int(L), sorted((int(n), float(t)) for n, t in row.items())) for L, row in grid.items() if row)
    if not rows:
        raise ValueError("empty timing grid")
    at_length = [(float(L), _interp([(float(n), t) for n, t in row], float(poses))) for L, row in rows]
    floor = 0.5 * min(t for _, row in rows for _, t in row)
    return max(floor, _interp(at_length, float(length)))
