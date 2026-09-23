"""Two-way cancellation: strip the main effects, keep the interaction.

WHY THIS EXISTS.  A specificity panel is a complete two-way design -- every peptide against every
binder -- and that lets us write any score as

    S(i, j) = grand + peptide(i) + binder(j) + interaction(i, j)

Everything we systematically cannot model is a MAIN EFFECT.  A peptide's desolvation offset, its
length and charge, a binder's general stickiness: each is constant down a row or a column and
cancels in the interaction term.  Specificity is by definition the interaction -- binder j suits
peptide i beyond what either is worth on its own -- so the interaction residual is not a cosmetic
normalisation, it is the estimator that matches what the assay measures.

The evidence that this is the right frame, measured on the Coventry grid:

  * ~75% of affinity variance is a per-RECEPTOR baseline that no static representation recovers
    (pocket composition 0.049, ProtDCal 0.149, ESM-2 0.154).  It is a main effect; cancel it.
  * Plain mean double-centring already moves ref2015 from mean cognate rank 8.28 to 6.67 and
    AUC 0.584 to 0.667, with no fitting at all.
  * Our calibrated dG's per-binder COLUMN MEAN correlates with the true cognate affinity at
    +0.431, while the same model's single-cell prediction correlates at -0.268.  The main effects
    and the interaction carry different information and want different estimators.

WHY THE MEAN IS THE WRONG ESTIMATOR FOR IT.  A docked grid contains failed poses -- clashes worth
hundreds of REU -- and a row mean is dragged by them, so the "main effect" it removes is partly
an artefact of the worst cell in the row.  Tukey's median polish is the classical robust answer:
iterate, subtracting row and column MEDIANS, and a handful of catastrophic cells cannot move a
median.  `median_polish` below is that, and it is the default.

SCALE IS ALSO A MAIN EFFECT.  Rows differ not only in where they sit but in how far they spread:
a long peptide's scores swing more than a short one's, and that inflates its interaction residuals
for no physical reason.  `two_way` can divide each row and column by its own MAD after centring,
which puts every cell on a comparable footing before anything is ranked.
"""
from __future__ import annotations

import numpy as np

__all__ = ["median_polish", "mean_centre", "two_way", "cross_scorer"]


def mean_centre(M: np.ndarray) -> np.ndarray:
    """Classical additive double-centring: subtract row and column means, add back the grand mean.

    Args:
        M: (n_peptides, n_binders) score matrix; lower = predicted tighter binding.

    Returns:
        The interaction residual, same shape.
    """
    return M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()


def median_polish(M: np.ndarray, iters: int = 25, tol: float = 1e-9
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Tukey median polish — the robust two-way decomposition.

    Alternately sweeps row and column MEDIANS out of the table until the effects stop moving.
    Unlike the mean, a median is unmoved by a few catastrophic cells, which matters here because a
    docked grid always contains clashed poses worth hundreds of REU.

    Args:
        M: (n_rows, n_cols) score matrix.
        iters: Maximum sweeps. Convergence is usually reached in well under ten.
        tol: Stop when a full sweep moves the residual by less than this.

    Returns:
        (residual, row_effects, col_effects, grand). The residual is the interaction term;
        ``residual + row[:, None] + col[None, :] + grand`` reconstructs M exactly.
    """
    R = np.array(M, dtype=float, copy=True)
    n, m = R.shape
    row = np.zeros(n)
    col = np.zeros(m)
    grand = 0.0
    for _ in range(iters):
        before = R.copy()
        rm = np.median(R, axis=1)
        R -= rm[:, None]
        row += rm
        g = np.median(row)
        row -= g
        grand += g

        cm = np.median(R, axis=0)
        R -= cm[None, :]
        col += cm
        g = np.median(col)
        col -= g
        grand += g
        if np.abs(R - before).max() < tol:
            break
    return R, row, col, grand


def _mad(x: np.ndarray, axis: int) -> np.ndarray:
    med = np.median(x, axis=axis, keepdims=True)
    return np.median(np.abs(x - med), axis=axis, keepdims=True)


def two_way(M: np.ndarray, robust: bool = True, scale: bool = False,
            iters: int = 25) -> np.ndarray:
    """Remove the main effects and return the interaction.

    Args:
        M: (n_peptides, n_binders) score matrix; lower = predicted tighter.
        robust: Use median polish rather than mean centring. Default True — a docked panel
            always carries failed poses, and they move a mean but not a median.
        scale: Also divide each row and column by its MAD, so a peptide whose scores simply
            swing more does not get larger interaction residuals for free.
        iters: Sweeps for the scale-balancing loop.

    Returns:
        The interaction residual, same shape and orientation as M (lower = more specific).

    Raises:
        ValueError: If M is not two-dimensional.
    """
    A = np.asarray(M, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"expected a 2-D grid, got shape {A.shape}")
    R = median_polish(A, iters=iters)[0] if robust else mean_centre(A)
    if not scale:
        return R
    for _ in range(iters):
        before = R.copy()
        s = _mad(R, 1)
        R = R / np.where(s > 1e-12, s, 1.0)
        s = _mad(R, 0)
        R = R / np.where(s > 1e-12, s, 1.0)
        if np.abs(R - before).max() < 1e-9:
            break
    return R


def cross_scorer(interaction_from: np.ndarray, effects_from: np.ndarray,
                 robust: bool = True, weight: float = 1.0) -> np.ndarray:
    """Take the main effects from one scorer and the interaction from another.

    The two scorers on this grid are not redundant and not interchangeable. Our calibrated dG
    carries the receptor baseline (its per-binder column mean correlates with the true cognate
    affinity at +0.431) but almost nothing at the cell level (-0.268). ref2015 is the mirror
    image: no receptor baseline (-0.063) but a real peptide main effect (+0.293) and the usable
    interaction. So estimate each part from whichever scorer actually has it.

    Args:
        interaction_from: Score matrix whose interaction term is trusted (ref2015).
        effects_from: Score matrix whose main effects are trusted (our calibrated dG).
        robust: Use median polish for both decompositions.
        weight: How strongly the donor's main effects are added back, in units of the donor's
            own scale. 0 reduces this to plain cancellation on ``interaction_from``.

    Returns:
        Combined score, lower = predicted tighter.

    Raises:
        ValueError: If the two matrices differ in shape.
    """
    A = np.asarray(interaction_from, dtype=float)
    B = np.asarray(effects_from, dtype=float)
    if A.shape != B.shape:
        raise ValueError(f"shape mismatch: {A.shape} vs {B.shape}")
    inter = two_way(A, robust=robust)
    if robust:
        _, row, col, _ = median_polish(B)
    else:
        row = B.mean(1) - B.mean()
        col = B.mean(0) - B.mean()
    # put the donor's effects on the recipient's scale before mixing, so `weight` means the
    # same thing whatever units the two scorers happen to use
    sa = np.median(np.abs(inter - np.median(inter))) or 1.0
    eff = row[:, None] + col[None, :]
    sb = np.median(np.abs(eff - np.median(eff))) or 1.0
    return inter + weight * (sa / sb) * eff
