"""Double-difference (thermodynamic-cycle) absolute ΔG from three experimental anchors.

This is an ABSOLUTE ΔG predictor (output in kcal/mol), enabled by experimental anchors — not a pure
selectivity score. It predicts ``ΔG(P, R)`` for a query peptide P on receptor R from the three other
corners of a 2x2 grid, all measured experimentally::

    ΔG(P, R) ≈ ΔG(P, R_ref) + ΔG(P_ref, R) − ΔG(P_ref, R_ref)

Algebra (with G(p, r) = f(p) + g(r) + coupling): the double difference cancels BOTH the per-receptor
offset g(R) = b(R) AND the per-peptide bias f(P) = c(P), leaving only the (small) non-additive coupling.
On 31 real PPIKB 2x2 grids the coupling is mean 0.85 kcal/mol and the estimator predicts the held-out
corner at r=0.955 / MAE 0.85 (e282). It needs NO learned scorer — just three measured Kd values.

Scope (the constraint that makes it a *repurposing / selectivity* tool, not de-novo):
  * Needs ``ΔG(P, R_ref)`` — the QUERY peptide measured on a different receptor. Available for repurposing
    (a known binder of one target, scored on another) and cross-reactivity panels; NOT for a brand-new
    designed peptide.
  * Needs ``ΔG(P_ref, R)`` — a reference peptide on the QUERY receptor (a same-receptor anchor).
  * Needs ``ΔG(P_ref, R_ref)`` — the shared grid corner.

Selectivity use: compute ΔG(P, A) and ΔG(P, B) each via this routine, then ``ΔΔG = ΔG(P,A) − ΔG(P,B)``.
Because both share the reference peptide's terms, the selectivity difference is especially robust.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DoubleDiffResult:
    """Result of a double-difference ΔG estimate.

    Attributes:
        dg: Predicted absolute ΔG(P, R) in kcal/mol.
        residual_kind: Always ``"coupling"`` — the only error term left after b(R) and c(P) cancel.
    """

    dg: float
    residual_kind: str = "coupling"


def double_difference_dg(
    dg_query_on_ref_receptor: float,
    dg_ref_peptide_on_query_receptor: float,
    dg_ref_peptide_on_ref_receptor: float,
) -> DoubleDiffResult:
    """Predict absolute ΔG(P, R) from the three measured corners of a 2x2 anchor grid.

    Args:
        dg_query_on_ref_receptor: Experimental ΔG of the query peptide P on a reference receptor R_ref
            (kcal/mol). The corner that supplies the de-novo constraint (repurposing/cross-reactivity).
        dg_ref_peptide_on_query_receptor: Experimental ΔG of a reference peptide P_ref on the query
            receptor R (a same-receptor anchor), kcal/mol.
        dg_ref_peptide_on_ref_receptor: Experimental ΔG of P_ref on R_ref (the shared corner), kcal/mol.

    Returns:
        A :class:`DoubleDiffResult` with the predicted absolute ΔG(P, R).

    Notes:
        All three inputs must be on the same sign convention (negative = binding) and the same assay
        scale. The estimate is exact up to the non-additive coupling term (≈0.85 kcal/mol typical).
    """
    dg = (dg_query_on_ref_receptor + dg_ref_peptide_on_query_receptor
          - dg_ref_peptide_on_ref_receptor)
    return DoubleDiffResult(dg=float(dg))


def double_difference_selectivity(
    dg_query_on_A: float,
    dg_query_on_B: float,
) -> float:
    """Cross-target selectivity ΔΔG = ΔG(P, A) − ΔG(P, B) from two double-difference absolute estimates.

    Args:
        dg_query_on_A: Predicted (or measured) ΔG(P, A) in kcal/mol.
        dg_query_on_B: Predicted (or measured) ΔG(P, B) in kcal/mol.

    Returns:
        ΔΔG in kcal/mol; negative means P prefers target A over B.
    """
    return float(dg_query_on_A - dg_query_on_B)
